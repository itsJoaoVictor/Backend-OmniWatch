from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload, load_only
from sqlalchemy import text
from fastapi import BackgroundTasks
import logging
import uuid
from typing import Optional
from datetime import datetime, timezone, timedelta
import asyncio

logger = logging.getLogger(__name__)

from app.tracking.models import UserListItem, UserEpisodeProgress
from app.tracking.schemas import UserListItemCreate, UserListItemUpdate, UserEpisodeProgressCreate
from app.media.services import get_media_by_tmdb_id, create_media
from app.media.schemas import MediaCreate
from app.media.models import Media
from app.core.cache import tv_status_cache, stats_cache, details_cache

async def determine_tv_next_episode_status(tmdb_id: int, progress_list: list):
    """
    Retorna (is_up_to_date, next_episode_dict) para uma série de TV.
    is_up_to_date é True se todos os episódios lançados até o momento já foram assistidos
    ou se o próximo episódio ainda não estreou (futuro ou sem data).
    Usa tv_status_cache em memória para responder em < 1ms e evita travar em requisições externas.
    """
    max_season = 0
    max_ep = 0
    for ep in (progress_list or []):
        if ep.season_number > max_season:
            max_season = ep.season_number
            max_ep = ep.episode_number
        elif ep.season_number == max_season and ep.episode_number > max_ep:
            max_ep = ep.episode_number
            
    cache_key = f"tv_status:{tmdb_id}:{max_season}:{max_ep}"
    cached_data, is_stale = tv_status_cache.get_with_status(cache_key)
    if not is_stale and cached_data is not None:
        return cached_data.get("is_up_to_date", False), cached_data.get("next_episode")

    next_season = max_season if max_season > 0 else 1
    next_ep_num = (max_ep + 1) if max_season > 0 else 1

    try:
        from app.core.utils import is_date_released
        from app.details.services import fetch_season_details
        
        # Timeout rígido de 2.0s para nunca travar a resposta caso o TMDB demore
        async with asyncio.timeout(2.0):
            season_details = await fetch_season_details(tmdb_id, next_season)
            episodes = season_details.episodes or []
            
            # Se o episódio ultrapassar a contagem da temporada, checa a próxima
            if episodes and next_ep_num > len(episodes):
                next_season += 1
                next_ep_num = 1
                try:
                    season_details = await fetch_season_details(tmdb_id, next_season)
                    episodes = season_details.episodes or []
                except Exception:
                    episodes = []
                    
            ep_data = next((e for e in episodes if e.episode_number == next_ep_num), None)
            if not ep_data:
                # Próximo episódio ainda não existe no catálogo oficial -> Em dia!
                tv_status_cache.set(cache_key, {"is_up_to_date": True, "next_episode": None})
                return True, None
                
            is_released = is_date_released(ep_data.air_date)
            next_ep_dict = {
                "season_number": next_season,
                "episode_number": next_ep_num,
                "name": getattr(ep_data, "name", None) or f"Episódio {next_ep_num}",
                "air_date": ep_data.air_date,
                "is_released": is_released
            }
            
            is_up = not is_released
            tv_status_cache.set(cache_key, {"is_up_to_date": is_up, "next_episode": next_ep_dict})
            return is_up, next_ep_dict
    except Exception:
        # Fallback local imediato: não interrompe a exibição e calcula o próximo número matematicamente
        fallback_next = {
            "season_number": next_season,
            "episode_number": next_ep_num,
            "name": f"Episódio {next_ep_num}",
            "air_date": None,
            "is_released": True
        }
        return False, fallback_next

def invalidate_user_stats_cache(user_id: str):
    """Invalida o cache de estatísticas do usuário marcando como stale para o SWR funcionar."""
    stats_cache.invalidate_prefix(f"stats:{user_id}")

async def get_user_list(db: AsyncSession, user_id: str):
    target_uuid = user_id if isinstance(user_id, uuid.UUID) else uuid.UUID(str(user_id))
    # Carrega a lista com media em uma única consulta rápida no banco (sub-15ms)
    result = await db.execute(
        select(UserListItem)
        .where(UserListItem.user_id == target_uuid)
        .options(selectinload(UserListItem.media))
    )
    all_items = result.scalars().all()
    # Ignora itens órfãos (sem mídia associada), que quebrariam a serialização (HTTP 500)
    items = [item for item in all_items if item.media is not None]
    if len(items) != len(all_items):
        logger.warning(
            "get_user_list: %d item(ns) sem mídia ignorado(s) para o usuário %s",
            len(all_items) - len(items), user_id
        )

    # Autocorreção transparente e em tempo real dos status upcoming / plan_to_watch
    # Utiliza apenas dados já existentes no banco (zero requisições de rede)
    from app.core.utils import is_date_released
    needs_commit = False
    for item in items:
        media = item.media
        if not media or not media.release_date:
            continue
        
        if item.status in ["upcoming", "plan_to_watch"]:
            is_released = is_date_released(media.release_date)
            if is_released and item.status == "upcoming":
                item.status = "plan_to_watch"
                needs_commit = True
            elif not is_released and item.status == "plan_to_watch":
                item.status = "upcoming"
                needs_commit = True

    if needs_commit:
        await db.commit()

    # Determina o status de "em dia" e próximo episódio APENAS para séries em watching
    tv_watching_items = [
        item for item in items 
        if item.media and item.media.media_type == "tv" and item.status == "watching"
    ]

    if tv_watching_items:
        # Busca o progresso somente das séries que estão em watching (evita carregar milhares de episódios desnecessários)
        item_ids = [item.id for item in tv_watching_items]
        prog_result = await db.execute(
            select(UserEpisodeProgress)
            .where(UserEpisodeProgress.user_list_item_id.in_(item_ids))
        )
        prog_map = {}
        for p in prog_result.scalars().all():
            prog_map.setdefault(p.user_list_item_id, []).append(p)

        tv_watching_tasks = [
            determine_tv_next_episode_status(item.media.tmdb_id, prog_map.get(item.id, []))
            for item in tv_watching_items
        ]
        results = await asyncio.gather(*tv_watching_tasks, return_exceptions=True)
        for item, res in zip(tv_watching_items, results):
            if isinstance(res, tuple) and len(res) == 2:
                is_up_to_date, next_ep_dict = res
                item.is_up_to_date = is_up_to_date
                item.next_episode = next_ep_dict

    return items

async def add_to_list(db: AsyncSession, user_id: str, item: UserListItemCreate):
    media = await get_media_by_tmdb_id(db, item.tmdb_id)
    if not media:
        title = item.title
        poster_path = item.poster_path
        backdrop_path = item.backdrop_path
        release_date = item.release_date
        runtime = 0
        genres = item.genres or []
        
        # Se não mandou do front, busca do TMDB
        directors = item.directors or []
        main_cast = item.main_cast or []
        keywords = []
        original_language = None
        # Fase 4: person_ids (inicializado aqui caso o fetch seja pulado)
        cast_ids = []
        crew_ids = []
        
        collection_tmdb_id = None
        collection_name = None
        
        if not title or not runtime or not genres or not directors or not main_cast or not release_date:
            from app.details.services import fetch_movie_details, fetch_tv_details
            try:
                if item.media_type == "movie":
                    tmdb_data = await fetch_movie_details(item.tmdb_id)
                    title = title or tmdb_data.title
                    poster_path = poster_path or tmdb_data.poster_path
                    backdrop_path = backdrop_path or tmdb_data.backdrop_path
                    runtime = tmdb_data.runtime or 0
                    original_language = tmdb_data.original_language or None
                    release_date = release_date or tmdb_data.release_date
                    if hasattr(tmdb_data, 'belongs_to_collection') and tmdb_data.belongs_to_collection:
                        collection_tmdb_id = tmdb_data.belongs_to_collection.id
                        collection_name = tmdb_data.belongs_to_collection.name
                    else:
                        collection_tmdb_id = 0
                    if not genres and tmdb_data.genres:
                        genres = [{"id": g.id, "name": g.name} for g in tmdb_data.genres]
                    if not directors and tmdb_data.credits and tmdb_data.credits.crew:
                        directors = [c.name for c in tmdb_data.credits.crew]
                    if not main_cast and tmdb_data.credits and tmdb_data.credits.cast:
                        main_cast = [c.name for c in tmdb_data.credits.cast]
                    if hasattr(tmdb_data, 'keywords') and tmdb_data.keywords:
                        keywords = tmdb_data.keywords
                    # Fase 4: salvar person_ids estruturados
                    cast_ids = tmdb_data.cast_ids or []
                    crew_ids = tmdb_data.crew_ids or []
                else:
                    tmdb_data = await fetch_tv_details(item.tmdb_id)
                    title = title or tmdb_data.title
                    poster_path = poster_path or tmdb_data.poster_path
                    backdrop_path = backdrop_path or tmdb_data.backdrop_path
                    original_language = tmdb_data.original_language or None
                    release_date = release_date or tmdb_data.first_air_date
                    
                    if tmdb_data.episode_run_time and len(tmdb_data.episode_run_time) > 0:
                        runtime = tmdb_data.episode_run_time[0]
                    else:
                        runtime = 45 # Default estimate for TV shows
                        
                    if not genres and tmdb_data.genres:
                        genres = [{"id": g.id, "name": g.name} for g in tmdb_data.genres]
                    if not directors and tmdb_data.credits and tmdb_data.credits.crew:
                        directors = [c.name for c in tmdb_data.credits.crew]
                    if not main_cast and tmdb_data.credits and tmdb_data.credits.cast:
                        main_cast = [c.name for c in tmdb_data.credits.cast]
                    if hasattr(tmdb_data, 'keywords') and tmdb_data.keywords:
                        keywords = tmdb_data.keywords
                    # Fase 4: salvar person_ids estruturados
                    cast_ids = tmdb_data.cast_ids or []
                    crew_ids = tmdb_data.crew_ids or []
            except Exception:
                title = title or "Desconhecido"

        # Fase 6: Gera embedding semântico via OpenRouter
        embedding = None
        try:
            from app.recommendation.embeddings import build_text_for_embedding, generate_embedding
            dummy_dict = {
                "title": title,
                "media_type": item.media_type,
                "genres": genres,
                "crew": directors,
                "cast": main_cast,
                "keywords": keywords
            }
            emb_text = build_text_for_embedding(dummy_dict)
            embedding = await generate_embedding(emb_text)
        except Exception:
            embedding = None

        media_create = MediaCreate(
            tmdb_id=item.tmdb_id,
            media_type=item.media_type,
            title=title,
            poster_path=poster_path,
            backdrop_path=backdrop_path,
            original_language=original_language,
            genres=genres,
            directors=directors,
            main_cast=main_cast,
            keywords=keywords,
            cast_ids=cast_ids if cast_ids else None,
            crew_ids=crew_ids if crew_ids else None,
            embedding=embedding,
            release_date=release_date,
            runtime=runtime,
            collection_tmdb_id=collection_tmdb_id,
            collection_name=collection_name
        )
        media = await create_media(db, media_create)
    else:
        # Se a mídia já existe no banco mas não tem poster_path, preenche
        if not media.poster_path and item.poster_path:
            media.poster_path = item.poster_path
            if not media.backdrop_path and item.backdrop_path:
                media.backdrop_path = item.backdrop_path
            await db.commit()
            await db.refresh(media)
        elif not media.poster_path:
            from app.details.services import fetch_movie_details, fetch_tv_details
            try:
                if item.media_type == "movie":
                    tmdb_data = await fetch_movie_details(item.tmdb_id)
                else:
                    tmdb_data = await fetch_tv_details(item.tmdb_id)
                if tmdb_data and tmdb_data.poster_path:
                    media.poster_path = tmdb_data.poster_path
                    if not media.backdrop_path and tmdb_data.backdrop_path:
                        media.backdrop_path = tmdb_data.backdrop_path
                    await db.commit()
                    await db.refresh(media)
            except Exception:
                pass

        if not media.release_date:
            from app.details.services import fetch_movie_details, fetch_tv_details
            try:
                if media.media_type == "movie":
                    tmdb_data = await fetch_movie_details(media.tmdb_id)
                    media.release_date = tmdb_data.release_date
                else:
                    tmdb_data = await fetch_tv_details(media.tmdb_id)
                    media.release_date = tmdb_data.first_air_date
                if media.release_date:
                    await db.commit()
                    await db.refresh(media)
            except Exception:
                pass

        if media.media_type == "movie" and media.collection_tmdb_id is None:
            from app.details.services import fetch_movie_details
            try:
                tmdb_data = await fetch_movie_details(media.tmdb_id)
                if tmdb_data and getattr(tmdb_data, "belongs_to_collection", None):
                    media.collection_tmdb_id = tmdb_data.belongs_to_collection.id
                    media.collection_name = tmdb_data.belongs_to_collection.name
                else:
                    media.collection_tmdb_id = 0
                await db.commit()
                await db.refresh(media)
            except Exception:
                pass

    # Determinação e validação de status para títulos não lançados
    from app.core.utils import is_date_released
    from fastapi import HTTPException
    
    rel_date = media.release_date or item.release_date
    is_released = is_date_released(rel_date)
    
    final_status = item.status or "plan_to_watch"
    if not is_released:
        if final_status in ["completed", "watching"]:
            raise HTTPException(
                status_code=400,
                detail="Títulos que ainda não estrearam só podem ser adicionados com status 'Aguardando Estreia'."
            )
        final_status = "upcoming"
    elif final_status == "upcoming":
        # Se já foi lançado, regulariza para plan_to_watch
        final_status = "plan_to_watch"

    # Check if already in list
    user_uuid = uuid.UUID(str(user_id))
    result = await db.execute(
        select(UserListItem)
        .where(UserListItem.user_id == user_uuid, UserListItem.media_id == media.id)
        .options(selectinload(UserListItem.media))
    )
    existing_item = result.scalars().first()
    if existing_item:
        changed = False
        if existing_item.status != final_status:
            existing_item.status = final_status
            changed = True
            if existing_item.status not in ["watching", "completed"] and existing_item.is_favorite:
                existing_item.is_favorite = False
        if item.rating is not None and existing_item.rating != item.rating:
            existing_item.rating = item.rating
            changed = True
        if item.is_favorite is not None:
            new_fav = bool(item.is_favorite) and final_status in ["watching", "completed"]
            if existing_item.is_favorite != new_fav:
                existing_item.is_favorite = new_fav
                changed = True
        
        if changed:
            await db.commit()
            await db.refresh(existing_item)
            from app.recommendation.profile_manager import update_user_profile
            await update_user_profile(db, user_id, existing_item, existing_item.media)
        
        reloaded = await db.execute(
            select(UserListItem)
            .where(UserListItem.id == existing_item.id)
            .options(
                selectinload(UserListItem.media),
                selectinload(UserListItem.episodes)
            )
        )
        existing_item = reloaded.scalars().first() or existing_item
        if existing_item.media and existing_item.media.media_type == "tv" and existing_item.status == "watching":
            is_up_to_date, next_ep = await determine_tv_next_episode_status(existing_item.media.tmdb_id, existing_item.episodes)
            existing_item.is_up_to_date = is_up_to_date
            existing_item.next_episode = next_ep
        return existing_item

    initial_fav = bool(item.is_favorite) if final_status in ["watching", "completed"] else False
    new_item = UserListItem(
        user_id=user_uuid,
        media_id=media.id,
        status=final_status,
        rating=item.rating,
        rewatch_count=item.rewatch_count,
        is_favorite=initial_fav
    )
    db.add(new_item)
    await db.commit()
    await db.refresh(new_item)
    
    # Reload with media and episodes relationship
    result = await db.execute(
        select(UserListItem)
        .where(UserListItem.id == new_item.id)
        .options(
            selectinload(UserListItem.media),
            selectinload(UserListItem.episodes)
        )
    )
    saved_item = result.scalars().first()
    if saved_item and saved_item.media and saved_item.media.media_type == "tv" and saved_item.status == "watching":
        is_up_to_date, next_ep = await determine_tv_next_episode_status(saved_item.media.tmdb_id, saved_item.episodes)
        saved_item.is_up_to_date = is_up_to_date
        saved_item.next_episode = next_ep
    
    # Trigger Profile Update
    from app.recommendation.profile_manager import update_user_profile
    await update_user_profile(db, user_id, saved_item, saved_item.media)

    # Dispara o cacheamento local do pôster em segundo plano (LRU ~45KB)
    if saved_item and saved_item.media and saved_item.media.poster_path:
        import asyncio
        from app.images.router import ensure_image_cached
        asyncio.create_task(ensure_image_cached(saved_item.media.poster_path, "w342"))
    # Dispara sugestões de coleções em segundo plano se for filme
    if saved_item and saved_item.media and saved_item.media.media_type == "movie":
        try:
            from app.media_collections.services import schedule_collection_suggestions_scan
            schedule_collection_suggestions_scan(str(user_id))
        except Exception:
            pass

    # Dispara sugestões de universos em segundo plano (filmes e séries)
    if saved_item and saved_item.media:
        try:
            from app.universes.services import schedule_universe_suggestions_scan
            schedule_universe_suggestions_scan(str(user_id))
        except Exception:
            pass

    invalidate_user_stats_cache(user_id)
    return saved_item

async def _get_tv_released_episodes_and_status(tmdb_id: int, tv_data=None):
    """
    Retorna (released_episode_keys, has_unreleased_episodes, tv_data) para uma série de TV,
    onde released_episode_keys é a lista de tuplas (season_number, episode_number) já lançados.
    """
    from app.details.services import fetch_tv_details, fetch_season_details
    from app.core.utils import is_date_released

    if tv_data is None:
        tv_data = await fetch_tv_details(tmdb_id)

    has_unreleased_episodes = False
    released_keys = []

    for season in (tv_data.seasons or []):
        if season.season_number > 0:
            try:
                season_details = await fetch_season_details(tmdb_id, season.season_number)
                for ep in (season_details.episodes or []):
                    if is_date_released(ep.air_date):
                        released_keys.append((season.season_number, ep.episode_number))
                    else:
                        has_unreleased_episodes = True
            except Exception as e:
                logger.warning(f"Failed to fetch season {season.season_number} details: {e}")
                if season.air_date and not is_date_released(season.air_date):
                    has_unreleased_episodes = True
                else:
                    for ep_num in range(1, (season.episode_count or 0) + 1):
                        released_keys.append((season.season_number, ep_num))

    return released_keys, has_unreleased_episodes, tv_data


async def sync_tv_rewatch_from_episodes(db: AsyncSession, item: UserListItem, tv_data=None) -> int:
    """
    Calcula o rewatch_count da série inteira baseado nos episódios:
    Se todos os episódios lançados foram assistidos e possuem rewatch_count >= k (k >= 1),
    a obra toda é marcada como reassistida com rewatch_count = min(ep.rewatch_count).
    Caso contrário, se algum episódio não foi reassistido (ou não assistido), rewatch_count = 0.
    """
    if not item or not item.media or item.media.media_type != "tv":
        return item.rewatch_count if item else 0

    try:
        released_keys, has_unreleased, tv_data = await _get_tv_released_episodes_and_status(
            item.media.tmdb_id, tv_data=tv_data
        )
        if not released_keys:
            return item.rewatch_count or 0

        await db.flush()
        prog_res = await db.execute(
            select(UserEpisodeProgress).where(UserEpisodeProgress.user_list_item_id == item.id)
        )
        prog_map = {(p.season_number, p.episode_number): p for p in prog_res.scalars().all()}

        all_watched = all(k in prog_map for k in released_keys)
        if all_watched:
            min_rewatch = min((prog_map[k].rewatch_count or 0) for k in released_keys)
            item.rewatch_count = max(0, min_rewatch)
            is_series_ended = tv_data and tv_data.status in ("Ended", "Canceled")
            if is_series_ended and not has_unreleased:
                item.status = "completed"
            elif item.status in ("plan_to_watch", "upcoming"):
                item.status = "watching"
        else:
            item.rewatch_count = 0
    except Exception as e:
        logger.warning(f"Error syncing TV rewatch from episodes: {e}")

    return item.rewatch_count or 0


async def update_list_item(
    db: AsyncSession, 
    user_id: str, 
    item_id: str, 
    update_data: UserListItemUpdate,
    background_tasks: Optional[BackgroundTasks] = None
):
    try:
        item_uuid = uuid.UUID(item_id)
        user_uuid = uuid.UUID(user_id)
    except ValueError:
        return None

    result = await db.execute(
        select(UserListItem)
        .where(UserListItem.id == item_uuid, UserListItem.user_id == user_uuid)
        .options(selectinload(UserListItem.media))
    )
    item = result.scalars().first()
    if not item:
        return None

    from app.core.utils import is_date_released
    from fastapi import HTTPException

    rel_date = item.media.release_date if item.media else None
    if not rel_date and item.media:
        from app.core.cache import details_cache
        cached_details, _ = details_cache.get_with_status(f"details:{item.media.media_type}:{item.media.tmdb_id}")
        if cached_details and "release_date" in cached_details:
            rel_date = cached_details["release_date"]
            item.media.release_date = rel_date
        elif cached_details and "first_air_date" in cached_details:
            rel_date = cached_details["first_air_date"]
            item.media.release_date = rel_date
        else:
            from app.details.services import fetch_movie_details, fetch_tv_details
            try:
                if item.media.media_type == "movie":
                    tmdb_data = await fetch_movie_details(item.media.tmdb_id)
                    rel_date = getattr(tmdb_data, 'release_date', None)
                else:
                    tmdb_data = await fetch_tv_details(item.media.tmdb_id)
                    rel_date = getattr(tmdb_data, 'first_air_date', None)
                if rel_date:
                    item.media.release_date = rel_date
            except Exception:
                pass

    is_released = is_date_released(rel_date)

    if update_data.status is not None:
        # Regra de Negócio: 'upcoming' é 100% automático gerenciado pelo sistema
        if update_data.status == "upcoming" and is_released:
            raise HTTPException(
                status_code=400,
                detail="O status 'Aguardando Estreia' é gerenciado automaticamente pelo sistema."
            )

        # Títulos não lançados só podem ter status 'upcoming' ou 'dropped'
        if not is_released and update_data.status in ["plan_to_watch", "watching", "completed"]:
            raise HTTPException(
                status_code=400,
                detail="Títulos que ainda não estrearam permanecem automaticamente em 'Aguardando Estreia' até o lançamento."
            )

        if update_data.status == "completed":
            if item.media.media_type == "movie":
                rel_date = item.media.release_date
                if not is_date_released(rel_date):
                    raise HTTPException(
                        status_code=400,
                        detail="Filmes que ainda não estrearam não podem ser marcados como assistidos."
                    )
                item.status = "completed"
                item.last_watched_at = datetime.now(timezone.utc)
            elif item.media.media_type == "tv":
                try:
                    released_keys, has_unreleased_episodes, tv_data = await _get_tv_released_episodes_and_status(item.media.tmdb_id)
                    prog_res = await db.execute(
                        select(UserEpisodeProgress).where(UserEpisodeProgress.user_list_item_id == item.id)
                    )
                    existing_map = {(p.season_number, p.episode_number): p for p in prog_res.scalars().all()}

                    new_progresses = []
                    for s_num, e_num in released_keys:
                        if (s_num, e_num) not in existing_map:
                            new_progresses.append(
                                UserEpisodeProgress(
                                    user_list_item_id=item.id,
                                    season_number=s_num,
                                    episode_number=e_num
                                )
                            )
                    if new_progresses:
                        db.add_all(new_progresses)
                        item.last_watched_at = datetime.now(timezone.utc)
                    
                    is_series_ended = tv_data.status in ("Ended", "Canceled")
                    if is_series_ended and not has_unreleased_episodes:
                        item.status = "completed"
                    else:
                        item.status = "watching"
                except Exception as e:
                    logger.warning(f"Failed to auto-mark episodes: {e}")
        elif update_data.status == "watching" and item.media.media_type == "movie":
            rel_date = item.media.release_date
            if not is_date_released(rel_date):
                raise HTTPException(
                    status_code=400,
                    detail="Filmes que ainda não estrearam não podem ser marcados como em andamento."
                )
            item.status = "watching"
        else:
            item.status = update_data.status

        # Regra de negócio: se mudou status para fora de 'watching'/'completed', automaticamente desfavorita
        if item.status not in ["watching", "completed"] and item.is_favorite:
            item.is_favorite = False

        # Regra de negócio: se foi para plan_to_watch ou upcoming sem nota explícita, limpa nota e rewatches
        if item.status in ["plan_to_watch", "upcoming"]:
            if update_data.rating is None:
                item.rating = None
            if update_data.rewatch_count is None:
                item.rewatch_count = 0

    # Tratamento de Favoritos
    if update_data.is_favorite is not None:
        if update_data.is_favorite:
            if item.status not in ["watching", "completed"]:
                raise HTTPException(
                    status_code=400,
                    detail="Apenas mídias com status 'Assistindo' ou 'Assistido' podem ser favoritadas."
                )
            item.is_favorite = True
        else:
            item.is_favorite = False

    # Resolução de corrida e nota:
    # Se recebeu nota e o item estava em 'plan_to_watch' ou 'upcoming' (sem status explícito no payload),
    # gradua automaticamente para 'watching' para nunca descartar a nota enviada pelo usuário
    if update_data.rating is not None:
        if item.status in ["plan_to_watch", "upcoming"] and update_data.status is None:
            item.status = "watching"
        if item.status not in ["plan_to_watch", "upcoming"]:
            item.rating = update_data.rating

    if update_data.rewatch_count is not None:
        target_rewatch = max(0, update_data.rewatch_count)
        old_rewatch = item.rewatch_count or 0
        if target_rewatch > 0 and not is_released:
            raise HTTPException(
                status_code=400,
                detail="Títulos que ainda não estrearam não podem ser marcados como reassistidos."
            )

        if item.media and item.media.media_type == "movie":
            if target_rewatch > 0 and item.status != "completed" and update_data.status is None:
                item.status = "completed"
            item.rewatch_count = target_rewatch
            if target_rewatch > old_rewatch:
                item.last_watched_at = datetime.now(timezone.utc)
        elif item.media and item.media.media_type == "tv":
            try:
                released_keys, has_unreleased_episodes, tv_data = await _get_tv_released_episodes_and_status(item.media.tmdb_id)
                prog_res = await db.execute(
                    select(UserEpisodeProgress).where(UserEpisodeProgress.user_list_item_id == item.id)
                )
                existing_map = {(p.season_number, p.episode_number): p for p in prog_res.scalars().all()}

                if target_rewatch > 0:
                    new_progresses = []
                    for s_num, e_num in released_keys:
                        ep_prog = existing_map.get((s_num, e_num))
                        if ep_prog:
                            if target_rewatch > old_rewatch:
                                ep_prog.rewatch_count = max(ep_prog.rewatch_count or 0, target_rewatch)
                            else:
                                if (ep_prog.rewatch_count or 0) > target_rewatch:
                                    ep_prog.rewatch_count = target_rewatch
                        else:
                            new_progresses.append(
                                UserEpisodeProgress(
                                    user_list_item_id=item.id,
                                    season_number=s_num,
                                    episode_number=e_num,
                                    rewatch_count=target_rewatch
                                )
                            )
                    if new_progresses:
                        db.add_all(new_progresses)
                    is_series_ended = tv_data.status in ("Ended", "Canceled")
                    if is_series_ended and not has_unreleased_episodes:
                        item.status = "completed"
                    elif item.status in ("plan_to_watch", "upcoming"):
                        item.status = "watching"
                    if target_rewatch > old_rewatch:
                        item.last_watched_at = datetime.now(timezone.utc)
                else:
                    for ep_prog in existing_map.values():
                        ep_prog.rewatch_count = 0
                item.rewatch_count = target_rewatch
            except Exception as e:
                logger.warning(f"Failed to sync TV episodes on rewatch_count update: {e}")
                item.rewatch_count = target_rewatch
        else:
            item.rewatch_count = target_rewatch

    media_ref = item.media
    await db.commit()
    await db.refresh(item)
    
    # Fast-Path: Trigger Profile Update em background (0ms no request)
    if media_ref:
        if background_tasks:
            background_tasks.add_task(_async_post_process_list_item_update, user_id=user_id, item_id=str(item.id))
        else:
            asyncio.create_task(_async_post_process_list_item_update(user_id=user_id, item_id=str(item.id)))

    # Recalcula próximo episódio de TV se status ou rewatch_count foi alterado
    if (update_data.status is not None or update_data.rewatch_count is not None) and item.media and item.media.media_type == "tv":
        tv_status_cache.delete_prefix(f"tv_status:{item.media.tmdb_id}")
        if item.status == "watching":
            reloaded = await db.execute(
                select(UserListItem)
                .where(UserListItem.id == item.id)
                .options(selectinload(UserListItem.episodes))
            )
            item_with_episodes = reloaded.scalars().first() or item
            is_up_to_date, next_ep = await determine_tv_next_episode_status(item.media.tmdb_id, item_with_episodes.episodes)
            item.is_up_to_date = is_up_to_date
            item.next_episode = next_ep

    invalidate_user_stats_cache(user_id)
    return item

async def remove_from_list(db: AsyncSession, user_id: str, item_id: str):
    try:
        item_uuid = uuid.UUID(item_id)
        user_uuid = uuid.UUID(user_id)
    except ValueError:
        return False

    result = await db.execute(
        select(UserListItem)
        .where(UserListItem.id == item_uuid, UserListItem.user_id == user_uuid)
        .options(selectinload(UserListItem.media))
    )
    item = result.scalars().first()
    if item:
        is_movie = bool(item.media and item.media.media_type == "movie")
        # Fase 3: Feedback Implícito - sinal negativo (-0.5) ao remover item da lista
        if item.media:
            from app.recommendation.profile_manager import update_user_profile
            await update_user_profile(db, user_id, item, item.media, explicit_scale=-0.5)

        await db.delete(item)
        await db.commit()
        invalidate_user_stats_cache(user_id)
        if is_movie:
            try:
                from app.media_collections.services import schedule_collection_suggestions_scan
                schedule_collection_suggestions_scan(str(user_id))
            except Exception:
                pass
        try:
            from app.universes.services import schedule_universe_suggestions_scan
            schedule_universe_suggestions_scan(str(user_id))
        except Exception:
            pass
        return True
    return False

async def _async_post_process_episode(
    user_id: str,
    item_id: str,
    media_id: Optional[int],
    rating: Optional[float],
    is_completed: bool,
    needs_completion_check: bool
):
    from app.core.database import AsyncSessionLocal
    from app.tracking.models import UserListItem, UserEpisodeProgress
    from sqlalchemy.orm import selectinload
    
    async with AsyncSessionLocal() as session:
        # 1. Se precisa checar finalização da série em background
        if needs_completion_check and media_id:
            try:
                from app.details.services import fetch_tv_details
                tv_data = await fetch_tv_details(media_id)
                if tv_data.status in ("Ended", "Canceled") and tv_data.number_of_episodes:
                    all_prog = await session.execute(
                        select(UserEpisodeProgress).where(UserEpisodeProgress.user_list_item_id == uuid.UUID(item_id))
                    )
                    if len(all_prog.scalars().all()) >= tv_data.number_of_episodes:
                        item_res = await session.execute(
                            select(UserListItem).where(UserListItem.id == uuid.UUID(item_id))
                        )
                        it = item_res.scalars().first()
                        if it and it.status != "completed":
                            it.status = "completed"
                            is_completed = True
                            await session.commit()
            except Exception as e:
                logger.warning(f"Error in background completion check: {e}")

        # 2. Atualiza perfil do usuário e recomendações
        try:
            from app.recommendation.profile_manager import update_user_profile, calculate_episode_rating_scale
            item_res = await session.execute(
                select(UserListItem)
                .where(UserListItem.id == uuid.UUID(item_id))
                .options(selectinload(UserListItem.media))
            )
            item_obj = item_res.scalars().first()
            if item_obj and item_obj.media:
                if rating is not None:
                    episode_scale = calculate_episode_rating_scale(rating)
                    await update_user_profile(session, user_id, item_obj, item_obj.media, explicit_scale=episode_scale)
                else:
                    completion_boost = 1.5 if is_completed else None
                    await update_user_profile(session, user_id, item_obj, item_obj.media, explicit_scale=completion_boost)
        except Exception as e:
            logger.warning(f"Error in background update_user_profile: {e}")

async def _async_post_process_list_item_update(user_id: str, item_id: str):
    from app.core.database import AsyncSessionLocal
    from app.tracking.models import UserListItem
    from sqlalchemy.orm import selectinload
    
    async with AsyncSessionLocal() as session:
        try:
            from app.recommendation.profile_manager import update_user_profile
            item_res = await session.execute(
                select(UserListItem)
                .where(UserListItem.id == uuid.UUID(item_id))
                .options(
                    selectinload(UserListItem.media),
                    selectinload(UserListItem.episodes)
                )
            )
            item_obj = item_res.scalars().first()
            if item_obj and item_obj.media:
                await update_user_profile(session, user_id, item_obj, item_obj.media)
        except Exception as e:
            logger.warning(f"Error in background update_list_item profile update: {e}")

async def _async_post_process_bulk(
    user_id: str,
    item_id: str,
    is_completed: bool
):
    from app.core.database import AsyncSessionLocal
    from app.tracking.models import UserListItem
    from sqlalchemy.orm import selectinload
    
    async with AsyncSessionLocal() as session:
        try:
            from app.recommendation.profile_manager import update_user_profile
            item_res = await session.execute(
                select(UserListItem)
                .where(UserListItem.id == uuid.UUID(item_id))
                .options(selectinload(UserListItem.media))
            )
            item_obj = item_res.scalars().first()
            if item_obj and item_obj.media:
                completion_boost = 1.5 if is_completed else None
                await update_user_profile(session, user_id, item_obj, item_obj.media, explicit_scale=completion_boost)
        except Exception as e:
            logger.warning(f"Error in background bulk update_user_profile: {e}")

async def add_episode_progress(
    db: AsyncSession, 
    user_id: str, 
    item_id: str, 
    progress: UserEpisodeProgressCreate,
    background_tasks: Optional[BackgroundTasks] = None
):
    try:
        item_uuid = uuid.UUID(item_id)
        user_uuid = uuid.UUID(user_id)
    except ValueError:
        return None

    from sqlalchemy.orm import selectinload
    result = await db.execute(
        select(UserListItem)
        .where(UserListItem.id == item_uuid, UserListItem.user_id == user_uuid)
        .options(selectinload(UserListItem.media))
    )
    item = result.scalars().first()
    if not item:
        return None

    # Validação de data de lançamento do episódio (Fast-path com cache de temporada)
    if item.media and item.media.media_type == "tv":
        from app.details.services import fetch_episode_details
        from app.core.utils import is_date_released
        from fastapi import HTTPException
        try:
            ep_data = await fetch_episode_details(item.media.tmdb_id, progress.season_number, progress.episode_number)
            if not is_date_released(ep_data.air_date):
                raise HTTPException(
                    status_code=400,
                    detail=f"O episódio S{progress.season_number} E{progress.episode_number} ainda não foi lançado (estreia prevista: {ep_data.air_date or 'Indefinida'})."
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.warning(f"Error checking episode release date: {e}")

    # Check if episode already marked
    prog_result = await db.execute(
        select(UserEpisodeProgress)
        .where(
            UserEpisodeProgress.user_list_item_id == item_uuid,
            UserEpisodeProgress.season_number == progress.season_number,
            UserEpisodeProgress.episode_number == progress.episode_number
        )
    )
    existing = prog_result.scalars().first()
    if existing:
        if progress.rating is not None and existing.rating != progress.rating:
            existing.rating = progress.rating
            await db.commit()
            await db.refresh(existing)
            if background_tasks:
                background_tasks.add_task(
                    _async_post_process_episode,
                    user_id=user_id,
                    item_id=str(item.id),
                    media_id=item.media.tmdb_id if item.media else None,
                    rating=progress.rating,
                    is_completed=False,
                    needs_completion_check=False
                )
            else:
                asyncio.create_task(
                    _async_post_process_episode(
                        user_id=user_id,
                        item_id=str(item.id),
                        media_id=item.media.tmdb_id if item.media else None,
                        rating=progress.rating,
                        is_completed=False,
                        needs_completion_check=False
                    )
                )
        return existing

    new_progress = UserEpisodeProgress(
        user_list_item_id=item_uuid,
        season_number=progress.season_number,
        episode_number=progress.episode_number,
        rating=progress.rating
    )
    db.add(new_progress)
    
    # Update last_watched_at
    item.last_watched_at = datetime.now(timezone.utc)
    
    is_completed = False
    needs_completion_check = False
    if item.media and item.media.media_type == "tv":
        cached_tv, _ = details_cache.get_with_status(f"details:tv:{item.media.tmdb_id}")
        if cached_tv and cached_tv.get("status") in ("Ended", "Canceled") and cached_tv.get("number_of_episodes"):
            all_prog = await db.execute(select(UserEpisodeProgress).where(UserEpisodeProgress.user_list_item_id == item_uuid))
            total_marked = len(all_prog.scalars().all()) + 1
            if total_marked >= cached_tv["number_of_episodes"]:
                item.status = "completed"
                is_completed = True
        else:
            needs_completion_check = True

    if not is_completed and item.status == "plan_to_watch":
        item.status = "watching"

    media_ref = item.media
    tmdb_id_val = media_ref.tmdb_id if media_ref else None

    await db.commit()
    await db.refresh(new_progress)
    
    if tmdb_id_val:
        tv_status_cache.delete_prefix(f"tv_status:{tmdb_id_val}")
    
    invalidate_user_stats_cache(user_id)

    # Fast-Path: Processa perfis e recomendações em segundo plano sem travar a resposta HTTP
    if background_tasks:
        background_tasks.add_task(
            _async_post_process_episode,
            user_id=user_id,
            item_id=str(item.id),
            media_id=tmdb_id_val,
            rating=progress.rating,
            is_completed=is_completed,
            needs_completion_check=needs_completion_check
        )
    else:
        asyncio.create_task(
            _async_post_process_episode(
                user_id=user_id,
                item_id=str(item.id),
                media_id=tmdb_id_val,
                rating=progress.rating,
                is_completed=is_completed,
                needs_completion_check=needs_completion_check
            )
        )

    return new_progress

async def update_episode_rating(
    db: AsyncSession,
    user_id: str,
    item_id_or_tmdb_id: str,
    season_number: int,
    episode_number: int,
    rating: Optional[float],
    background_tasks: Optional[BackgroundTasks] = None
):
    try:
        user_uuid = uuid.UUID(user_id)
    except ValueError:
        return None

    from sqlalchemy.orm import selectinload
    from app.media.models import Media

    # Tenta achar por item_id (UUID)
    item = None
    try:
        item_uuid = uuid.UUID(item_id_or_tmdb_id)
        res = await db.execute(
            select(UserListItem)
            .where(UserListItem.id == item_uuid, UserListItem.user_id == user_uuid)
            .options(selectinload(UserListItem.media))
        )
        item = res.scalars().first()
    except ValueError:
        pass

    # Se não achou por UUID, tenta por tmdb_id (int)
    if not item:
        try:
            tmdb_id_int = int(item_id_or_tmdb_id)
            res = await db.execute(
                select(UserListItem)
                .join(UserListItem.media)
                .where(Media.tmdb_id == tmdb_id_int, UserListItem.user_id == user_uuid)
                .options(selectinload(UserListItem.media))
            )
            item = res.scalars().first()
        except ValueError:
            pass

    # Se a série ainda não está na lista do usuário, cria automaticamente como watching
    if not item:
        try:
            tmdb_id_int = int(item_id_or_tmdb_id)
            from app.tracking.schemas import UserListItemCreate
            item = await add_to_list(db, user_id, UserListItemCreate(
                tmdb_id=tmdb_id_int,
                media_type="tv",
                status="watching"
            ))
        except Exception:
            return None

    if not item:
        return None

    # Valida se o episódio já foi lançado caso uma avaliação seja enviada
    if rating is not None and item.media and item.media.media_type == "tv":
        from app.details.services import fetch_episode_details
        from app.core.utils import is_date_released
        from fastapi import HTTPException
        try:
            ep_data = await fetch_episode_details(item.media.tmdb_id, season_number, episode_number)
            if not is_date_released(ep_data.air_date):
                raise HTTPException(
                    status_code=400,
                    detail=f"O episódio S{season_number} E{episode_number} ainda não foi lançado (estreia prevista: {ep_data.air_date or 'Indefinida'})."
                )
        except HTTPException:
            raise
        except Exception as e:
            print(f"Error checking episode release date: {e}")

    # Busca ou cria o registro de progresso do episódio
    prog_result = await db.execute(
        select(UserEpisodeProgress)
        .where(
            UserEpisodeProgress.user_list_item_id == item.id,
            UserEpisodeProgress.season_number == season_number,
            UserEpisodeProgress.episode_number == episode_number
        )
    )
    prog = prog_result.scalars().first()
    if not prog:
        prog = UserEpisodeProgress(
            user_list_item_id=item.id,
            season_number=season_number,
            episode_number=episode_number,
            rating=rating
        )
        db.add(prog)
    else:
        prog.rating = rating

    item.last_watched_at = datetime.now(timezone.utc)
    if item.status == "plan_to_watch":
        item.status = "watching"

    media_ref = item.media
    tmdb_id_val = media_ref.tmdb_id if media_ref else None

    await db.commit()
    await db.refresh(prog)

    if rating is not None and media_ref:
        if background_tasks:
            background_tasks.add_task(
                _async_post_process_episode,
                user_id=user_id,
                item_id=str(item.id),
                media_id=tmdb_id_val,
                rating=rating,
                is_completed=False,
                needs_completion_check=False
            )
        else:
            asyncio.create_task(
                _async_post_process_episode(
                    user_id=user_id,
                    item_id=str(item.id),
                    media_id=tmdb_id_val,
                    rating=rating,
                    is_completed=False,
                    needs_completion_check=False
                )
            )

    if tmdb_id_val:
        tv_status_cache.delete_prefix(f"tv_status:{tmdb_id_val}")

    return prog

async def update_episode_rewatch(
    db: AsyncSession,
    user_id: str,
    item_id_or_tmdb_id: str,
    season_number: int,
    episode_number: int,
    rewatch_count: int,
    background_tasks: Optional[BackgroundTasks] = None
):
    try:
        user_uuid = uuid.UUID(user_id)
    except ValueError:
        return None

    from sqlalchemy.orm import selectinload
    from app.media.models import Media

    item = None
    try:
        item_uuid = uuid.UUID(item_id_or_tmdb_id)
        res = await db.execute(
            select(UserListItem)
            .where(UserListItem.id == item_uuid, UserListItem.user_id == user_uuid)
            .options(selectinload(UserListItem.media))
        )
        item = res.scalars().first()
    except ValueError:
        pass

    if not item:
        try:
            tmdb_id_int = int(item_id_or_tmdb_id)
            res = await db.execute(
                select(UserListItem)
                .join(UserListItem.media)
                .where(Media.tmdb_id == tmdb_id_int, UserListItem.user_id == user_uuid)
                .options(selectinload(UserListItem.media))
            )
            item = res.scalars().first()
        except ValueError:
            pass

    if not item and rewatch_count > 0:
        try:
            tmdb_id_int = int(item_id_or_tmdb_id)
            from app.tracking.schemas import UserListItemCreate
            item = await add_to_list(db, user_id, UserListItemCreate(
                tmdb_id=tmdb_id_int,
                media_type="tv",
                status="watching"
            ))
        except Exception:
            return None

    if not item:
        return None

    target_rewatch = max(0, rewatch_count)

    if target_rewatch > 0 and item.media and item.media.media_type == "tv":
        from app.details.services import fetch_episode_details
        from app.core.utils import is_date_released
        from fastapi import HTTPException
        try:
            ep_data = await fetch_episode_details(item.media.tmdb_id, season_number, episode_number)
            if not is_date_released(ep_data.air_date):
                raise HTTPException(
                    status_code=400,
                    detail=f"O episódio S{season_number} E{episode_number} ainda não foi lançado (estreia prevista: {ep_data.air_date or 'Indefinida'})."
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.warning(f"Error checking episode release date: {e}")

    prog_result = await db.execute(
        select(UserEpisodeProgress)
        .where(
            UserEpisodeProgress.user_list_item_id == item.id,
            UserEpisodeProgress.season_number == season_number,
            UserEpisodeProgress.episode_number == episode_number
        )
    )
    prog = prog_result.scalars().first()
    old_ep_rewatch = (prog.rewatch_count or 0) if prog else 0

    if not prog:
        prog = UserEpisodeProgress(
            user_list_item_id=item.id,
            season_number=season_number,
            episode_number=episode_number,
            rewatch_count=target_rewatch
        )
        db.add(prog)
    else:
        prog.rewatch_count = target_rewatch
        if target_rewatch > old_ep_rewatch:
            prog.watched_at = datetime.now(timezone.utc)

    if target_rewatch > old_ep_rewatch:
        item.last_watched_at = datetime.now(timezone.utc)
    if item.status == "plan_to_watch":
        item.status = "watching"

    # Sincroniza o rewatch_count da obra toda caso todos os episódios estejam reassistidos
    await sync_tv_rewatch_from_episodes(db, item)

    media_ref = item.media
    tmdb_id_val = media_ref.tmdb_id if media_ref else None

    await db.commit()
    await db.refresh(prog)

    if tmdb_id_val:
        tv_status_cache.delete_prefix(f"tv_status:{tmdb_id_val}")
    invalidate_user_stats_cache(user_id)

    if media_ref:
        if background_tasks:
            background_tasks.add_task(_async_post_process_list_item_update, user_id=user_id, item_id=str(item.id))
        else:
            asyncio.create_task(_async_post_process_list_item_update(user_id=user_id, item_id=str(item.id)))

    return prog

async def update_season_rewatch(
    db: AsyncSession,
    user_id: str,
    item_id: str,
    season_number: int,
    rewatch_count: int,
    background_tasks: Optional[BackgroundTasks] = None
):
    try:
        item_uuid = uuid.UUID(item_id)
        user_uuid = uuid.UUID(user_id)
    except ValueError:
        return None

    from sqlalchemy.orm import selectinload
    from fastapi import HTTPException
    result = await db.execute(
        select(UserListItem)
        .where(UserListItem.id == item_uuid, UserListItem.user_id == user_uuid)
        .options(selectinload(UserListItem.media))
    )
    item = result.scalars().first()
    if not item or not item.media or item.media.media_type != "tv":
        return None

    from app.details.services import fetch_tv_details, fetch_season_details
    from app.core.utils import is_date_released

    target_rewatch = max(0, rewatch_count)
    tv_data = None
    try:
        tv_data = await fetch_tv_details(item.media.tmdb_id)
    except Exception:
        pass

    prog_res = await db.execute(
        select(UserEpisodeProgress).where(UserEpisodeProgress.user_list_item_id == item_uuid)
    )
    existing_map = {(p.season_number, p.episode_number): p for p in prog_res.scalars().all()}

    released_ep_nums = []
    try:
        season_details = await fetch_season_details(item.media.tmdb_id, season_number)
        for ep in (season_details.episodes or []):
            if is_date_released(ep.air_date):
                released_ep_nums.append(ep.episode_number)
    except Exception as e:
        logger.warning(f"Failed to fetch season {season_number} details in update_season_rewatch: {e}")
        if tv_data:
            season_meta = next((s for s in tv_data.seasons if s.season_number == season_number), None)
            if season_meta and (not season_meta.air_date or is_date_released(season_meta.air_date)):
                released_ep_nums = list(range(1, (season_meta.episode_count or 0) + 1))

    if not released_ep_nums:
        raise HTTPException(
            status_code=400,
            detail=f"A temporada {season_number} ainda não possui episódios lançados."
        )

    new_progresses = []
    now_utc = datetime.now(timezone.utc)
    for ep_num in released_ep_nums:
        ep_prog = existing_map.get((season_number, ep_num))
        if ep_prog:
            if target_rewatch > (ep_prog.rewatch_count or 0):
                ep_prog.watched_at = now_utc
            ep_prog.rewatch_count = target_rewatch
        elif target_rewatch > 0:
            new_progresses.append(
                UserEpisodeProgress(
                    user_list_item_id=item.id,
                    season_number=season_number,
                    episode_number=ep_num,
                    rewatch_count=target_rewatch
                )
            )

    if new_progresses:
        db.add_all(new_progresses)

    if target_rewatch > 0:
        item.last_watched_at = now_utc
        if item.status == "plan_to_watch":
            item.status = "watching"

    await sync_tv_rewatch_from_episodes(db, item, tv_data=tv_data)

    media_ref = item.media
    tmdb_id_val = media_ref.tmdb_id if media_ref else None

    await db.commit()
    await db.refresh(item)

    if tmdb_id_val:
        tv_status_cache.delete_prefix(f"tv_status:{tmdb_id_val}")
    invalidate_user_stats_cache(user_id)

    if media_ref:
        if background_tasks:
            background_tasks.add_task(_async_post_process_list_item_update, user_id=user_id, item_id=str(item.id))
        else:
            asyncio.create_task(_async_post_process_list_item_update(user_id=user_id, item_id=str(item.id)))

    return {
        "status": "ok",
        "season_rewatch_count": target_rewatch,
        "series_rewatch_count": item.rewatch_count or 0
    }

async def get_episode_progress(db: AsyncSession, user_id: str, item_id: str):
    try:
        item_uuid = uuid.UUID(item_id)
        user_uuid = uuid.UUID(user_id)
    except ValueError:
        return []

    # First, verify if the item belongs to the user
    result = await db.execute(
        select(UserListItem)
        .where(UserListItem.id == item_uuid, UserListItem.user_id == user_uuid)
    )
    item = result.scalars().first()
    if not item:
        return []

    prog_result = await db.execute(
        select(UserEpisodeProgress)
        .where(UserEpisodeProgress.user_list_item_id == item_uuid)
    )
    return prog_result.scalars().all()

async def remove_episode_progress(db: AsyncSession, user_id: str, item_id: str, season_number: int, episode_number: int):
    try:
        item_uuid = uuid.UUID(item_id)
        user_uuid = uuid.UUID(user_id)
    except ValueError:
        return False

    from sqlalchemy.orm import selectinload
    # Verify if the item belongs to the user
    result = await db.execute(
        select(UserListItem)
        .where(UserListItem.id == item_uuid, UserListItem.user_id == user_uuid)
        .options(selectinload(UserListItem.media))
    )
    item = result.scalars().first()
    if not item:
        return False

    tmdb_id_val = item.media.tmdb_id if item.media else None

    prog_result = await db.execute(
        select(UserEpisodeProgress)
        .where(
            UserEpisodeProgress.user_list_item_id == item_uuid,
            UserEpisodeProgress.season_number == season_number,
            UserEpisodeProgress.episode_number == episode_number
        )
    )
    existing = prog_result.scalars().first()
    if existing:
        await db.delete(existing)
        if item.status == "completed":
            item.status = "watching"
        if item.rewatch_count and item.rewatch_count > 0:
            item.rewatch_count = 0
        await db.commit()
        if tmdb_id_val:
            tv_status_cache.delete_prefix(f"tv_status:{tmdb_id_val}")
        invalidate_user_stats_cache(user_id)
        return True
    
    return False

async def mark_season_watched(
    db: AsyncSession,
    user_id: str,
    item_id: str,
    season_number: int,
    background_tasks: Optional[BackgroundTasks] = None
):
    try:
        item_uuid = uuid.UUID(item_id)
        user_uuid = uuid.UUID(user_id)
    except ValueError:
        return None

    from sqlalchemy.orm import selectinload
    from fastapi import HTTPException
    result = await db.execute(
        select(UserListItem)
        .where(UserListItem.id == item_uuid, UserListItem.user_id == user_uuid)
        .options(selectinload(UserListItem.media))
    )
    item = result.scalars().first()
    if not item or not item.media or item.media.media_type != "tv":
        return None

    from app.details.services import fetch_tv_details, fetch_season_details
    from app.core.utils import is_date_released

    tv_data = None
    try:
        tv_data = await fetch_tv_details(item.media.tmdb_id)
    except Exception:
        pass

    prog_res = await db.execute(
        select(UserEpisodeProgress).where(UserEpisodeProgress.user_list_item_id == item_uuid)
    )
    existing_map = {(p.season_number, p.episode_number): p for p in prog_res.scalars().all()}

    new_progresses = []
    released_in_season = 0
    has_unreleased_in_season = False

    try:
        season_details = await fetch_season_details(item.media.tmdb_id, season_number)
        for ep in (season_details.episodes or []):
            if is_date_released(ep.air_date):
                released_in_season += 1
                if (season_number, ep.episode_number) not in existing_map:
                    new_progresses.append(
                        UserEpisodeProgress(
                            user_list_item_id=item.id,
                            season_number=season_number,
                            episode_number=ep.episode_number
                        )
                    )
            else:
                has_unreleased_in_season = True
    except Exception as e:
        logger.warning(f"Failed to fetch season {season_number} details in mark_season_watched: {e}")
        if tv_data:
            season_meta = next((s for s in tv_data.seasons if s.season_number == season_number), None)
            if season_meta and (not season_meta.air_date or is_date_released(season_meta.air_date)):
                released_in_season = season_meta.episode_count
                for ep_num in range(1, season_meta.episode_count + 1):
                    if (season_number, ep_num) not in existing_map:
                        new_progresses.append(
                            UserEpisodeProgress(
                                user_list_item_id=item.id,
                                season_number=season_number,
                                episode_number=ep_num
                            )
                        )

    if released_in_season == 0:
        raise HTTPException(
            status_code=400,
            detail=f"A temporada {season_number} ainda não possui episódios lançados."
        )

    if new_progresses:
        db.add_all(new_progresses)
        item.last_watched_at = datetime.now(timezone.utc)

        total_episodes_now = len(existing_map) + len(new_progresses)
        is_completed = False
        if tv_data:
            is_ended = tv_data.status in ("Ended", "Canceled")
            if is_ended and not has_unreleased_in_season and tv_data.number_of_episodes and total_episodes_now >= tv_data.number_of_episodes:
                item.status = "completed"
                is_completed = True
            elif item.status == "plan_to_watch":
                item.status = "watching"
        elif item.status == "plan_to_watch":
            item.status = "watching"

        media_ref = item.media
        tmdb_id_val = media_ref.tmdb_id if media_ref else None

        await db.commit()
        await db.refresh(item)

        if media_ref:
            if background_tasks:
                background_tasks.add_task(
                    _async_post_process_bulk,
                    user_id=user_id,
                    item_id=str(item.id),
                    is_completed=is_completed
                )
            else:
                asyncio.create_task(
                    _async_post_process_bulk(
                        user_id=user_id,
                        item_id=str(item.id),
                        is_completed=is_completed
                    )
                )

        if tmdb_id_val:
            tv_status_cache.delete_prefix(f"tv_status:{tmdb_id_val}")
        invalidate_user_stats_cache(user_id)

    return True

async def remove_season_progress(
    db: AsyncSession,
    user_id: str,
    item_id: str,
    season_number: int
):
    try:
        item_uuid = uuid.UUID(item_id)
        user_uuid = uuid.UUID(user_id)
    except ValueError:
        return False

    from sqlalchemy.orm import selectinload
    from sqlalchemy import delete
    result = await db.execute(
        select(UserListItem)
        .where(UserListItem.id == item_uuid, UserListItem.user_id == user_uuid)
        .options(selectinload(UserListItem.media))
    )
    item = result.scalars().first()
    if not item:
        return False

    tmdb_id_val = item.media.tmdb_id if item.media else None

    await db.execute(
        delete(UserEpisodeProgress).where(
            UserEpisodeProgress.user_list_item_id == item_uuid,
            UserEpisodeProgress.season_number == season_number
        )
    )
    if item.status == "completed":
        item.status = "watching"
    if item.rewatch_count and item.rewatch_count > 0:
        item.rewatch_count = 0

    await db.commit()
    if tmdb_id_val:
        tv_status_cache.delete_prefix(f"tv_status:{tmdb_id_val}")
    invalidate_user_stats_cache(user_id)
    return True

async def bulk_mark_episodes(
    db: AsyncSession, 
    user_id: str, 
    item_id: str, 
    target: UserEpisodeProgressCreate,
    background_tasks: Optional[BackgroundTasks] = None
):
    try:
        item_uuid = uuid.UUID(item_id)
        user_uuid = uuid.UUID(user_id)
    except ValueError:
        return None

    # Check item belongs to user
    from sqlalchemy.orm import selectinload
    from app.tracking.models import UserListItem
    result = await db.execute(select(UserListItem).where(UserListItem.id == item_uuid, UserListItem.user_id == user_uuid).options(selectinload(UserListItem.media)))
    item = result.scalars().first()
    if not item or item.media.media_type != "tv":
        return None

    from app.details.services import fetch_tv_details, fetch_season_details
    from app.core.utils import is_date_released
    try:
        tv_data = await fetch_tv_details(item.media.tmdb_id)
    except Exception:
        return None

    prog_res = await db.execute(select(UserEpisodeProgress).where(UserEpisodeProgress.user_list_item_id == item_uuid))
    existing_map = {(p.season_number, p.episode_number): p for p in prog_res.scalars().all()}

    new_progresses = []
    has_unreleased = False
    for season in tv_data.seasons:
        if season.season_number > 0 and season.season_number <= target.season_number:
            max_ep = target.episode_number if season.season_number == target.season_number else season.episode_count
            try:
                season_details = await fetch_season_details(item.media.tmdb_id, season.season_number)
                for ep in season_details.episodes:
                    if ep.episode_number <= max_ep:
                        if is_date_released(ep.air_date):
                            if (season.season_number, ep.episode_number) not in existing_map:
                                new_progresses.append(UserEpisodeProgress(
                                    user_list_item_id=item.id,
                                    season_number=season.season_number,
                                    episode_number=ep.episode_number
                                ))
                        else:
                            has_unreleased = True
            except Exception as e:
                logger.warning(f"Failed to fetch season {season.season_number} details in bulk_mark: {e}")
                if not (season.air_date and not is_date_released(season.air_date)):
                    for ep_num in range(1, max_ep + 1):
                        if (season.season_number, ep_num) not in existing_map:
                            new_progresses.append(UserEpisodeProgress(
                                user_list_item_id=item.id,
                                season_number=season.season_number,
                                episode_number=ep_num
                            ))
    
    if new_progresses:
        db.add_all(new_progresses)
        item.last_watched_at = datetime.now(timezone.utc)
        
        is_ended = tv_data.status in ("Ended", "Canceled")
        total_episodes_now = len(existing_map) + len(new_progresses)
        
        is_completed = False
        if is_ended and not has_unreleased and total_episodes_now >= tv_data.number_of_episodes:
            item.status = "completed"
            is_completed = True
        elif item.status == "plan_to_watch":
            item.status = "watching"
            
        media_ref = item.media
        tmdb_id_val = media_ref.tmdb_id if media_ref else None

        await db.commit()
        await db.refresh(item)

        # Trigger Profile Update in background
        if media_ref:
            if background_tasks:
                background_tasks.add_task(
                    _async_post_process_bulk,
                    user_id=user_id,
                    item_id=str(item.id),
                    is_completed=is_completed
                )
            else:
                asyncio.create_task(
                    _async_post_process_bulk(
                        user_id=user_id,
                        item_id=str(item.id),
                        is_completed=is_completed
                    )
                )

        if tmdb_id_val:
            tv_status_cache.delete_prefix(f"tv_status:{tmdb_id_val}")
        invalidate_user_stats_cache(user_id)
    return True


GENRE_NORMALIZATION = {
    "action & adventure": "Ação e Aventura",
    "action": "Ação",
    "adventure": "Aventura",
    "animation": "Animação",
    "comedy": "Comédia",
    "crime": "Crime",
    "documentary": "Documentário",
    "drama": "Drama",
    "family": "Família",
    "fantasy": "Fantasia",
    "history": "História",
    "horror": "Terror",
    "music": "Música",
    "mystery": "Mistério",
    "romance": "Romance",
    "sci-fi & fantasy": "Ficção Científica e Fantasia",
    "science fiction": "Ficção Científica",
    "tv movie": "Cinema TV",
    "thriller": "Suspense",
    "war & politics": "Guerra e Política",
    "war": "Guerra",
    "western": "Faroeste",
    "ação": "Ação",
    "ação e aventura": "Ação e Aventura",
    "aventura": "Aventura",
    "animação": "Animação",
    "comédia": "Comédia",
    "documentário": "Documentário",
    "família": "Família",
    "fantasia": "Fantasia",
    "história": "História",
    "terror": "Terror",
    "música": "Música",
    "mistério": "Mistério",
    "ficção científica": "Ficção Científica",
    "ficção científica e fantasia": "Ficção Científica e Fantasia",
    "suspense": "Suspense",
    "guerra": "Guerra",
    "guerra e política": "Guerra e Política",
    "faroeste": "Faroeste",
}

MONTH_NAMES_PT = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]

def normalize_genre(name: str) -> str:
    clean = name.strip()
    return GENRE_NORMALIZATION.get(clean.lower(), clean)

_revalidating_stats_keys: set = set()

async def recompute_and_cache_stats(user_id: str, period: str, media_type: str):
    """Recomputa as estatísticas do usuário em background e atualiza o cache."""
    cache_key = f"stats:{user_id}:{period}:{media_type}"
    try:
        from app.core.database import AsyncSessionLocal
        async with AsyncSessionLocal() as session:
            await _compute_user_statistics(session, user_id, period=period, media_type=media_type)
    except Exception as e:
        logger.error("Erro ao revalidar estatísticas em background para usuário %s: %s", user_id, e)
    finally:
        _revalidating_stats_keys.discard(cache_key)

async def _compute_user_statistics(db: AsyncSession, user_id: str, period: str = "all", media_type: str = "all"):
    try:
        user_uuid = uuid.UUID(user_id)
    except ValueError:
        return {}

    cache_key = f"stats:{user_id}:{period}:{media_type}"
    now = datetime.now(timezone.utc)
    cutoff = None
    days_span = 365
    if period == "30days":
        cutoff = now - timedelta(days=30)
        days_span = 30
    elif period == "6months":
        cutoff = now - timedelta(days=180)
        days_span = 180
    elif period == "year":
        cutoff = now - timedelta(days=365)
        days_span = 365

    # 1. Busca os títulos do catálogo do usuário com campos necessários
    items_query = (
        select(UserListItem)
        .where(UserListItem.user_id == user_uuid)
        .options(
            load_only(
                UserListItem.id,
                UserListItem.user_id,
                UserListItem.media_id,
                UserListItem.status,
                UserListItem.rating,
                UserListItem.rewatch_count,
                UserListItem.last_watched_at,
                UserListItem.created_at,
                UserListItem.updated_at
            ),
            selectinload(UserListItem.media).options(
                load_only(
                    Media.id,
                    Media.media_type,
                    Media.title,
                    Media.genres,
                    Media.runtime,
                    Media.directors,
                    Media.main_cast,
                    Media.cast_ids,
                    Media.crew_ids
                )
            )
        )
    )
    if media_type != "all":
        items_query = items_query.join(UserListItem.media).where(Media.media_type == media_type)

    result = await db.execute(items_query)
    items = result.scalars().all()

    # 2. Pushdown para SQL: agregação de episódios e timeline diretamente no PostgreSQL
    ep_agg_map = {}
    ep_timeline_map = {}
    earliest_ep_date = now

    if media_type != "movie" and items:
        ep_where = ""
        params = {"user_uuid": user_uuid}
        if cutoff is not None:
            ep_where = "AND COALESCE(uep.watched_at, uli.last_watched_at, uli.updated_at, uli.created_at) >= :cutoff"
            params["cutoff"] = cutoff

        ep_agg_sql = text(f"""
            SELECT 
                uep.user_list_item_id,
                SUM(1 + COALESCE(uep.rewatch_count, 0)) AS total_eps_with_rewatch,
                MIN(COALESCE(uep.watched_at, uli.last_watched_at, uli.updated_at, uli.created_at)) AS earliest_watched_at
            FROM user_episode_progress uep
            JOIN user_list_items uli ON uli.id = uep.user_list_item_id
            WHERE uli.user_id = :user_uuid
              {ep_where}
            GROUP BY uep.user_list_item_id
        """)
        res_ep_agg = await db.execute(ep_agg_sql, params)
        for row in res_ep_agg.fetchall():
            ep_agg_map[row.user_list_item_id] = {
                "total_eps": int(row.total_eps_with_rewatch or 0),
                "earliest": row.earliest_watched_at
            }
            if row.earliest_watched_at:
                dt = row.earliest_watched_at
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                if dt < earliest_ep_date:
                    earliest_ep_date = dt

        num_months = 12 if period in ("year", "all") else 6
        timeline_start = now - timedelta(days=num_months * 31)
        ep_timeline_sql = text("""
            SELECT 
                TO_CHAR(COALESCE(uep.watched_at, uli.last_watched_at, uli.updated_at, uli.created_at), 'YYYY-MM') AS month_key,
                SUM(1 + COALESCE(uep.rewatch_count, 0)) AS ep_count,
                SUM((1 + COALESCE(uep.rewatch_count, 0)) * COALESCE(m.runtime, 0)) AS ep_minutes
            FROM user_episode_progress uep
            JOIN user_list_items uli ON uli.id = uep.user_list_item_id
            JOIN media m ON m.id = uli.media_id
            WHERE uli.user_id = :user_uuid
              AND COALESCE(uep.watched_at, uli.last_watched_at, uli.updated_at, uli.created_at) >= :timeline_start
            GROUP BY TO_CHAR(COALESCE(uep.watched_at, uli.last_watched_at, uli.updated_at, uli.created_at), 'YYYY-MM')
        """)
        res_ep_tl = await db.execute(ep_timeline_sql, {"user_uuid": user_uuid, "timeline_start": timeline_start})
        for row in res_ep_tl.fetchall():
            ep_timeline_map[row.month_key] = {
                "count": int(row.ep_count or 0),
                "minutes": int(row.ep_minutes or 0)
            }

    # 3. Inicializa Timeline de Atividade Mensal
    num_months = 12 if period in ("year", "all") else 6
    activity_months = {}
    for i in range(num_months - 1, -1, -1):
        m_date = now - timedelta(days=i * 30)
        key = f"{m_date.year:04d}-{m_date.month:02d}"
        tl_data = ep_timeline_map.get(key, {"count": 0, "minutes": 0})
        activity_months[key] = {
            "minutes": tl_data["minutes"],
            "count": tl_data["count"],
            "label": f"{MONTH_NAMES_PT[m_date.month - 1]}/{str(m_date.year)[2:]}"
        }

    total_movies_watched = 0
    total_episodes_watched = 0
    total_time_movies_minutes = 0
    total_time_tv_minutes = 0

    rating_distribution = {str(i): 0 for i in range(1, 6)}
    all_ratings = []
    series_counts = {"completed": 0, "watching": 0, "plan_to_watch": 0, "dropped": 0, "paused": 0}
    movies_counts = {"completed": 0, "watching": 0, "plan_to_watch": 0, "dropped": 0, "paused": 0}
    total_counts = {"completed": 0, "watching": 0, "plan_to_watch": 0, "dropped": 0, "paused": 0}

    genres_stats = {}
    cast_stats = {}
    directors_stats = {}
    earliest_date = min(now, earliest_ep_date)

    for item in items:
        m = item.media
        if not m:
            continue

        if media_type != "all" and m.media_type != media_type:
            continue

        runtime = m.runtime or 0

        item_date = item.last_watched_at or item.updated_at or item.created_at
        if item_date:
            if item_date.tzinfo is None:
                item_date = item_date.replace(tzinfo=timezone.utc)
            if item_date < earliest_date:
                earliest_date = item_date

        if m.media_type == "tv":
            if item.status in series_counts:
                series_counts[item.status] += 1
        elif m.media_type == "movie":
            if item.status in movies_counts:
                movies_counts[item.status] += 1

        if item.status in total_counts:
            total_counts[item.status] += 1

        item_time_minutes = 0
        is_watched_in_period = False

        if m.media_type == "movie":
            if item.status in ("completed", "watching"):
                if cutoff is None or (item_date and item_date >= cutoff):
                    is_watched_in_period = True
                    factor = 1 + (item.rewatch_count or 0)
                    total_movies_watched += factor
                    item_time_minutes = runtime * factor
                    total_time_movies_minutes += item_time_minutes

                    if item_date:
                        m_key = f"{item_date.year:04d}-{item_date.month:02d}"
                        if m_key in activity_months:
                            activity_months[m_key]["minutes"] += item_time_minutes
                            activity_months[m_key]["count"] += factor

        elif m.media_type == "tv":
            ep_info = ep_agg_map.get(item.id)
            ep_count_in_period = ep_info["total_eps"] if ep_info else 0
            if ep_count_in_period > 0 or (cutoff is None and item.status in ("completed", "watching")):
                is_watched_in_period = True
                total_episodes_watched += ep_count_in_period
                item_time_minutes = runtime * ep_count_in_period
                total_time_tv_minutes += item_time_minutes
        if cutoff is not None:
            if not is_watched_in_period:
                continue
        else:
            if not is_watched_in_period and not (item.rating and item.rating > 0):
                continue

        if item.rating and item.rating > 0:
            r_key = str(min(5, max(1, int(round(item.rating)))))
            if r_key in rating_distribution:
                rating_distribution[r_key] += 1
            all_ratings.append(item.rating)

        raw_genres = m.genres or []
        normalized_genres = set()
        for g in raw_genres:
            g_raw_name = g.get("name") if isinstance(g, dict) else g
            if g_raw_name:
                normalized_genres.add(normalize_genre(g_raw_name))

        for g_name in normalized_genres:
            g_entry = genres_stats.setdefault(g_name, {"count": 0, "totalTime": 0, "ratings": []})
            g_entry["count"] += 1
            g_entry["totalTime"] += item_time_minutes
            if item.rating:
                g_entry["ratings"].append(item.rating)

        # Mapeamento rápido de person_id e profile_path via cast_ids e crew_ids salvos no Media
        cast_person_map = {}
        cast_profile_map = {}
        if m.cast_ids and isinstance(m.cast_ids, list):
            for c_info in m.cast_ids:
                if isinstance(c_info, dict) and c_info.get("name"):
                    if c_info.get("person_id"):
                        cast_person_map[c_info["name"]] = c_info["person_id"]
                    if c_info.get("profile_path"):
                        cast_profile_map[c_info["name"]] = c_info["profile_path"]

        crew_person_map = {}
        crew_profile_map = {}
        if m.crew_ids and isinstance(m.crew_ids, list):
            for c_info in m.crew_ids:
                if isinstance(c_info, dict) and c_info.get("name"):
                    if c_info.get("person_id"):
                        crew_person_map[c_info["name"]] = c_info["person_id"]
                    if c_info.get("profile_path"):
                        crew_profile_map[c_info["name"]] = c_info["profile_path"]

        raw_cast = m.main_cast or []
        if isinstance(raw_cast, list):
            for actor in raw_cast[:5]:
                a_name = actor.get("name") if isinstance(actor, dict) else str(actor).strip()
                if a_name:
                    c_entry = cast_stats.setdefault(a_name, {"count": 0, "totalTime": 0, "ratings": [], "person_id": None, "profile_path": None})
                    c_entry["count"] += 1
                    c_entry["totalTime"] += item_time_minutes
                    if not c_entry.get("person_id") and a_name in cast_person_map:
                        c_entry["person_id"] = cast_person_map[a_name]
                    if not c_entry.get("profile_path") and a_name in cast_profile_map:
                        c_entry["profile_path"] = cast_profile_map[a_name]
                    if item.rating:
                        c_entry["ratings"].append(item.rating)

        raw_dirs = m.directors or []
        if isinstance(raw_dirs, list):
            for director in raw_dirs:
                d_name = director.get("name") if isinstance(director, dict) else str(director).strip()
                if d_name:
                    d_entry = directors_stats.setdefault(d_name, {"count": 0, "totalTime": 0, "ratings": [], "person_id": None, "profile_path": None})
                    d_entry["count"] += 1
                    d_entry["totalTime"] += item_time_minutes
                    if not d_entry.get("person_id") and d_name in crew_person_map:
                        d_entry["person_id"] = crew_person_map[d_name]
                    if not d_entry.get("profile_path") and d_name in crew_profile_map:
                        d_entry["profile_path"] = crew_profile_map[d_name]
                    if item.rating:
                        d_entry["ratings"].append(item.rating)

    total_time = total_time_movies_minutes + total_time_tv_minutes

    if period == "all":
        delta_days = (now - earliest_date).days
        days_span = max(30, delta_days)

    daily_avg_mins = round(total_time / days_span) if days_span > 0 else 0
    weekly_avg_mins = round((total_time / days_span) * 7) if days_span > 0 else 0

    # Taxas de conclusão dinâmicas
    active_series = series_counts["completed"] + series_counts["watching"] + series_counts["plan_to_watch"]
    series_completion_rate = round((series_counts["completed"] / active_series) * 100, 1) if active_series > 0 else 0.0

    active_movies = movies_counts["completed"] + movies_counts["watching"] + movies_counts["plan_to_watch"]
    movies_completion_rate = round((movies_counts["completed"] / active_movies) * 100, 1) if active_movies > 0 else 0.0

    active_total = total_counts["completed"] + total_counts["watching"] + total_counts["plan_to_watch"]
    overall_completion_rate = round((total_counts["completed"] / active_total) * 100, 1) if active_total > 0 else 0.0

    if media_type == "movie":
        dynamic_completion_rate = movies_completion_rate
        dynamic_completed = movies_counts["completed"]
        dynamic_pending = movies_counts["watching"] + movies_counts["plan_to_watch"]
        dynamic_plan = movies_counts["plan_to_watch"]
    elif media_type == "tv":
        dynamic_completion_rate = series_completion_rate
        dynamic_completed = series_counts["completed"]
        dynamic_pending = series_counts["watching"] + series_counts["plan_to_watch"]
        dynamic_plan = series_counts["plan_to_watch"]
    else:
        dynamic_completion_rate = overall_completion_rate
        dynamic_completed = total_counts["completed"]
        dynamic_pending = total_counts["watching"] + total_counts["plan_to_watch"]
        dynamic_plan = total_counts["plan_to_watch"]

    avg_rating = round(sum(all_ratings) / len(all_ratings), 1) if all_ratings else 0.0

    def format_ranking(stats_dict, limit=15, sort_by_count=False):
        res = []
        if sort_by_count:
            # Ranking inteligente: prioriza presença em obras distintas, desempatando por tempo assistido e nota
            sorted_items = sorted(
                stats_dict.items(),
                key=lambda x: (
                    x[1]["count"],
                    x[1]["totalTime"],
                    sum(x[1]["ratings"]) / len(x[1]["ratings"]) if x[1]["ratings"] else 0
                ),
                reverse=True
            )[:limit]
        else:
            sorted_items = sorted(
                stats_dict.items(),
                key=lambda x: (x[1]["totalTime"], x[1]["count"]),
                reverse=True
            )[:limit]

        for name, data in sorted_items:
            res.append({
                "name": name,
                "person_id": data.get("person_id"),
                "profile_path": data.get("profile_path"),
                "count": data["count"],
                "totalTime": data["totalTime"],
                "avgRating": round(sum(data["ratings"]) / len(data["ratings"]), 1) if data["ratings"] else None,
                "percentage": round((data["totalTime"] / total_time) * 100, 1) if total_time > 0 else 0.0
            })
        return res

    genres_ranking = format_ranking(genres_stats, limit=20)
    cast_ranking = format_ranking(cast_stats, limit=50, sort_by_count=True)
    directors_ranking = format_ranking(directors_stats, limit=50, sort_by_count=True)

    timeline = []
    for k in sorted(activity_months.keys()):
        val = activity_months[k]
        timeline.append({
            "month": val["label"],
            "hours": round(val["minutes"] / 60, 1),
            "count": val["count"]
        })

    output = {
        "totalTime": total_time,
        "totalTimeMovies": total_time_movies_minutes,
        "totalTimeTv": total_time_tv_minutes,
        "totalMovies": total_movies_watched,
        "totalEpisodes": total_episodes_watched,
        "ratingDistribution": [{"rating": k, "count": v} for k, v in sorted(rating_distribution.items(), key=lambda x: int(x[0]))],
        "topGenres": genres_ranking[:10],
        "kpis": {
            "averageRating": avg_rating,
            "totalRated": len(all_ratings),
            "completionRate": dynamic_completion_rate,
            "completedCount": dynamic_completed,
            "pendingCount": dynamic_pending,
            "planToWatchCount": dynamic_plan,
            "seriesCompleted": series_counts["completed"],
            "seriesWatching": series_counts["watching"],
            "seriesPlanToWatch": series_counts["plan_to_watch"],
            "seriesCompletionRate": series_completion_rate,
            "moviesCompleted": movies_counts["completed"],
            "moviesWatching": movies_counts["watching"],
            "moviesPlanToWatch": movies_counts["plan_to_watch"],
            "moviesCompletionRate": movies_completion_rate,
            "dailyAverageMinutes": daily_avg_mins,
            "weeklyAverageMinutes": weekly_avg_mins,
        },
        "rankings": {
            "genres": genres_ranking,
            "cast": cast_ranking,
            "directors": directors_ranking
        },
        "timeline": timeline
    }

    # Nível 3: Persistência em tabela de sumarização pré-calculada para leituras ultra-rápidas
    if period == "all" and media_type == "all":
        try:
            from sqlalchemy.dialects.postgresql import insert
            from app.tracking.models import UserStatsSummary
            stmt = insert(UserStatsSummary).values(
                user_id=user_uuid,
                total_time_movies=output["totalTimeMovies"],
                total_time_tv=output["totalTimeTv"],
                total_movies_watched=output["totalMovies"],
                total_episodes_watched=output["totalEpisodes"],
                average_rating=output["kpis"]["averageRating"],
                total_rated=output["kpis"]["totalRated"],
                rating_distribution=output["ratingDistribution"],
                top_genres=output["topGenres"],
                rankings=output["rankings"],
                timeline=output["timeline"],
                kpis=output["kpis"]
            ).on_conflict_do_update(
                index_elements=[UserStatsSummary.user_id],
                set_={
                    "total_time_movies": output["totalTimeMovies"],
                    "total_time_tv": output["totalTimeTv"],
                    "total_movies_watched": output["totalMovies"],
                    "total_episodes_watched": output["totalEpisodes"],
                    "average_rating": output["kpis"]["averageRating"],
                    "total_rated": output["kpis"]["totalRated"],
                    "rating_distribution": output["ratingDistribution"],
                    "top_genres": output["topGenres"],
                    "rankings": output["rankings"],
                    "timeline": output["timeline"],
                    "kpis": output["kpis"],
                    "updated_at": datetime.now(timezone.utc)
                }
            )
            await db.execute(stmt)
            await db.commit()
        except Exception as e:
            logger.warning("Não foi possível persistir user_stats_summary: %s", e)

    stats_cache.set(cache_key, output)
    return output

async def get_user_statistics(
    db: AsyncSession, 
    user_id: str, 
    period: str = "all", 
    media_type: str = "all",
    background_tasks: Optional[BackgroundTasks] = None
):
    """
    Retorna as estatísticas do usuário com suporte a Stale-While-Revalidate (SWR).
    Se houver cache (mesmo expirado), retorna instantaneamente (<2ms) e agenda
    recomputação assíncrona em background.
    """
    try:
        user_uuid = uuid.UUID(user_id)
    except ValueError:
        return {}

    cache_key = f"stats:{user_id}:{period}:{media_type}"
    cached_data, is_stale = stats_cache.get_with_status(cache_key)

    if cached_data is not None:
        if is_stale and cache_key not in _revalidating_stats_keys:
            _revalidating_stats_keys.add(cache_key)
            if background_tasks is not None:
                background_tasks.add_task(recompute_and_cache_stats, user_id, period, media_type)
            else:
                asyncio.create_task(recompute_and_cache_stats(user_id, period, media_type))
        return cached_data

    # Nível 3: Leitura direta e instantânea (<1ms) de user_stats_summary para o resumo global
    if period == "all" and media_type == "all":
        try:
            from app.tracking.models import UserStatsSummary
            res_summary = await db.execute(
                select(UserStatsSummary).where(UserStatsSummary.user_id == user_uuid)
            )
            summary_row = res_summary.scalar_one_or_none()
            if summary_row:
                output = {
                    "totalTime": summary_row.total_time_movies + summary_row.total_time_tv,
                    "totalTimeMovies": summary_row.total_time_movies,
                    "totalTimeTv": summary_row.total_time_tv,
                    "totalMovies": summary_row.total_movies_watched,
                    "totalEpisodes": summary_row.total_episodes_watched,
                    "ratingDistribution": summary_row.rating_distribution,
                    "topGenres": summary_row.top_genres,
                    "kpis": summary_row.kpis,
                    "rankings": summary_row.rankings,
                    "timeline": summary_row.timeline
                }
                stats_cache.set(cache_key, output)
                if cache_key not in _revalidating_stats_keys:
                    _revalidating_stats_keys.add(cache_key)
                    if background_tasks is not None:
                        background_tasks.add_task(recompute_and_cache_stats, user_id, period, media_type)
                    else:
                        asyncio.create_task(recompute_and_cache_stats(user_id, period, media_type))
                return output
        except Exception as e:
            logger.warning("Falha ao buscar user_stats_summary: %s", e)

    return await _compute_user_statistics(db, user_id, period=period, media_type=media_type)

_revalidating_ranking_keys: set = set()

async def recompute_and_cache_ranking(
    user_id: str,
    media_type: str,
    sort_by: str,
    period: str,
    page: int,
    page_size: int,
    search: Optional[str]
):
    """Recomputa o ranking de mídias em background e atualiza o cache."""
    clean_search = (search or "").strip().lower()
    cache_key = f"stats:{user_id}:rankings:{media_type}:{sort_by}:{period}:{page}:{page_size}:{clean_search}"
    try:
        from app.core.database import AsyncSessionLocal
        async with AsyncSessionLocal() as session:
            await _compute_user_media_ranking(session, user_id, media_type, sort_by, period, page, page_size, search)
    except Exception as e:
        logger.error("Erro ao revalidar ranking em background para usuário %s: %s", user_id, e)
    finally:
        _revalidating_ranking_keys.discard(cache_key)

async def _compute_user_media_ranking(
    db: AsyncSession,
    user_id: str,
    media_type: str = "movie",
    sort_by: str = "time",
    period: str = "all",
    page: int = 1,
    page_size: int = 10,
    search: Optional[str] = None
):
    try:
        user_uuid = uuid.UUID(user_id)
    except ValueError:
        return {
            "items": [],
            "total_items": 0,
            "page": page,
            "page_size": page_size,
            "total_pages": 0,
            "media_type": media_type,
            "sort_by": sort_by,
            "period": period
        }

    clean_search = (search or "").strip().lower()
    page = max(1, int(page))
    page_size = max(1, min(50, int(page_size)))
    offset = (page - 1) * page_size

    cache_key = f"stats:{user_id}:rankings:{media_type}:{sort_by}:{period}:{page}:{page_size}:{clean_search}"

    now = datetime.now(timezone.utc)
    cutoff = None
    if period == "30days":
        cutoff = now - timedelta(days=30)
    elif period == "6months":
        cutoff = now - timedelta(days=180)
    elif period == "year":
        cutoff = now - timedelta(days=365)

    params = {
        "user_uuid": user_uuid,
        "limit": page_size,
        "offset": offset
    }

    if media_type == "movie":
        where_conditions = ["uli.user_id = :user_uuid", "m.media_type = 'movie'"]
        if cutoff is not None:
            where_conditions.append("uli.status IN ('completed', 'watching')")
            where_conditions.append("COALESCE(uli.last_watched_at, uli.updated_at, uli.created_at) >= :cutoff")
            params["cutoff"] = cutoff
        else:
            where_conditions.append("(uli.status IN ('completed', 'watching') OR COALESCE(uli.rewatch_count, 0) > 0 OR uli.rating IS NOT NULL)")

        if clean_search:
            where_conditions.append("m.title ILIKE :search_pattern")
            params["search_pattern"] = f"%{clean_search}%"

        where_clause = " AND ".join(where_conditions)

        if sort_by == "rating":
            order_clause = "CASE WHEN uli.rating IS NOT NULL THEN 1 ELSE 0 END DESC, uli.rating DESC NULLS LAST, (COALESCE(m.runtime, 0) * (1 + COALESCE(uli.rewatch_count, 0))) DESC, uli.id ASC"
        elif sort_by == "rewatch":
            order_clause = "COALESCE(uli.rewatch_count, 0) DESC, (COALESCE(m.runtime, 0) * (1 + COALESCE(uli.rewatch_count, 0))) DESC, uli.rating DESC NULLS LAST, uli.id ASC"
        elif sort_by == "episodes":
            order_clause = "(CASE WHEN uli.status = 'completed' THEN 1 ELSE 0 END) DESC, (COALESCE(m.runtime, 0) * (1 + COALESCE(uli.rewatch_count, 0))) DESC, uli.id ASC"
        elif sort_by == "recent":
            order_clause = "COALESCE(uli.last_watched_at, uli.updated_at, uli.created_at) DESC NULLS LAST, (COALESCE(m.runtime, 0) * (1 + COALESCE(uli.rewatch_count, 0))) DESC, uli.id ASC"
        elif sort_by == "title":
            order_clause = "LOWER(m.title) ASC, uli.id ASC"
        else:  # time
            order_clause = "(COALESCE(m.runtime, 0) * (1 + COALESCE(uli.rewatch_count, 0))) DESC, uli.rating DESC NULLS LAST, uli.id ASC"

        query_sql = text(f"""
            SELECT 
                uli.id AS item_id,
                m.id AS media_id,
                m.tmdb_id,
                m.title,
                m.media_type,
                m.poster_path,
                m.release_date,
                COALESCE(m.runtime, 0) AS runtime,
                m.genres,
                uli.rating,
                COALESCE(uli.rewatch_count, 0) AS rewatch_count,
                (COALESCE(m.runtime, 0) * (1 + COALESCE(uli.rewatch_count, 0))) AS total_time_minutes,
                CASE WHEN uli.status = 'completed' THEN 1 ELSE 0 END AS episodes_watched,
                uli.status,
                COALESCE(uli.last_watched_at, uli.updated_at, uli.created_at) AS last_watched_at,
                COUNT(*) OVER() AS total_items_count
            FROM user_list_items uli
            JOIN media m ON m.id = uli.media_id
            WHERE {where_clause}
            ORDER BY {order_clause}
            LIMIT :limit OFFSET :offset
        """)

    else:  # tv
        ep_cutoff_clause = ""
        if cutoff is not None:
            ep_cutoff_clause = "AND uep.watched_at >= :cutoff"
            params["cutoff"] = cutoff

        tv_filter = "COALESCE(ea.ep_count, 0) > 0" if cutoff is not None else "(COALESCE(ea.ep_count, 0) > 0 OR uli.status IN ('completed', 'watching') OR uli.rating IS NOT NULL)"
        search_filter = ""
        if clean_search:
            search_filter = "AND m.title ILIKE :search_pattern"
            params["search_pattern"] = f"%{clean_search}%"

        if sort_by == "rating":
            order_clause = "CASE WHEN rating IS NOT NULL THEN 1 ELSE 0 END DESC, rating DESC NULLS LAST, total_time_minutes DESC, item_id ASC"
        elif sort_by == "rewatch":
            order_clause = "rewatch_count DESC, total_time_minutes DESC, rating DESC NULLS LAST, item_id ASC"
        elif sort_by == "episodes":
            order_clause = "episodes_watched DESC, total_time_minutes DESC, rating DESC NULLS LAST, item_id ASC"
        elif sort_by == "recent":
            order_clause = "last_watched_at DESC NULLS LAST, total_time_minutes DESC, item_id ASC"
        elif sort_by == "title":
            order_clause = "LOWER(title) ASC, item_id ASC"
        else:  # time
            order_clause = "total_time_minutes DESC, rating DESC NULLS LAST, item_id ASC"

        if cutoff is None:
            # Nível 3: Fast-path O(1) usando contadores desnormalizados de UserListItem sem CTE
            query_sql = text(f"""
                SELECT 
                    uli.id AS item_id,
                    m.id AS media_id,
                    m.tmdb_id,
                    m.title,
                    m.media_type,
                    m.poster_path,
                    m.release_date,
                    COALESCE(m.runtime, 0) AS runtime,
                    m.genres,
                    uli.rating,
                    COALESCE(uli.rewatch_count, 0) AS rewatch_count,
                    uli.total_time_minutes,
                    uli.episodes_watched_count AS episodes_watched,
                    uli.status,
                    COALESCE(uli.last_watched_at, uli.updated_at, uli.created_at) AS last_watched_at,
                    COUNT(*) OVER() AS total_items_count
                FROM user_list_items uli
                JOIN media m ON m.id = uli.media_id
                WHERE uli.user_id = :user_uuid
                  AND m.media_type = 'tv'
                  AND (uli.episodes_watched_count > 0 OR uli.status IN ('completed', 'watching') OR uli.rating IS NOT NULL)
                  {search_filter}
                ORDER BY {order_clause}
                LIMIT :limit OFFSET :offset
            """)
        else:
            query_sql = text(f"""
                WITH ep_agg AS (
                    SELECT 
                        uep.user_list_item_id,
                        SUM(1 + COALESCE(uep.rewatch_count, 0)) AS ep_count,
                        MAX(uep.watched_at) AS max_ep_watched_at,
                        AVG(uep.rating) FILTER (WHERE uep.rating IS NOT NULL AND uep.rating > 0) AS avg_ep_rating
                    FROM user_episode_progress uep
                    JOIN user_list_items uli ON uli.id = uep.user_list_item_id
                    WHERE uli.user_id = :user_uuid
                      {ep_cutoff_clause}
                    GROUP BY uep.user_list_item_id
                ),
                tv_candidates AS (
                    SELECT 
                        uli.id AS item_id,
                        m.id AS media_id,
                        m.tmdb_id,
                        m.title,
                        m.media_type,
                        m.poster_path,
                        m.release_date,
                        COALESCE(m.runtime, 0) AS runtime,
                        m.genres,
                        COALESCE(uli.rating, ROUND(CAST(ea.avg_ep_rating AS numeric), 1)) AS rating,
                        COALESCE(uli.rewatch_count, 0) AS rewatch_count,
                        (COALESCE(m.runtime, 0) * COALESCE(ea.ep_count, 0)) AS total_time_minutes,
                        COALESCE(ea.ep_count, 0) AS episodes_watched,
                        uli.status,
                        GREATEST(
                            COALESCE(ea.max_ep_watched_at, '1970-01-01'::timestamptz),
                            COALESCE(uli.last_watched_at, uli.updated_at, uli.created_at, '1970-01-01'::timestamptz)
                        ) AS last_watched_at
                    FROM user_list_items uli
                    JOIN media m ON m.id = uli.media_id
                    LEFT JOIN ep_agg ea ON ea.user_list_item_id = uli.id
                    WHERE uli.user_id = :user_uuid
                      AND m.media_type = 'tv'
                      AND {tv_filter}
                      {search_filter}
                )
                SELECT 
                    *,
                    COUNT(*) OVER() AS total_items_count
                FROM tv_candidates
                ORDER BY {order_clause}
                LIMIT :limit OFFSET :offset
            """)

    res = await db.execute(query_sql, params)
    rows = res.fetchall()

    total_items = rows[0].total_items_count if rows else 0
    total_pages = max(1, (total_items + page_size - 1) // page_size) if total_items > 0 else 0

    page_items = []
    start = (page - 1) * page_size
    for idx, row in enumerate(rows):
        raw_genres = row.genres or []
        genres_list = []
        for g in raw_genres:
            g_name = g.get("name") if isinstance(g, dict) else str(g)
            if g_name:
                genres_list.append(normalize_genre(g_name))

        page_items.append({
            "id": row.item_id,
            "media_id": row.media_id,
            "tmdb_id": row.tmdb_id,
            "title": row.title,
            "media_type": row.media_type,
            "poster_path": row.poster_path,
            "release_date": row.release_date,
            "runtime": row.runtime,
            "genres": genres_list[:3],
            "rating": float(row.rating) if row.rating is not None else None,
            "rewatch_count": row.rewatch_count,
            "total_time_minutes": row.total_time_minutes,
            "episodes_watched": row.episodes_watched,
            "status": row.status,
            "last_watched_at": row.last_watched_at,
            "rank": start + idx + 1
        })

    output = {
        "items": page_items,
        "total_items": total_items,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
        "media_type": media_type,
        "sort_by": sort_by,
        "period": period
    }

    stats_cache.set(cache_key, output)
    return output


async def get_user_media_ranking(
    db: AsyncSession,
    user_id: str,
    media_type: str = "movie",
    sort_by: str = "time",
    period: str = "all",
    page: int = 1,
    page_size: int = 10,
    search: Optional[str] = None,
    background_tasks: Optional[BackgroundTasks] = None
):
    """
    Retorna o ranking de mídias do usuário com suporte a Stale-While-Revalidate (SWR).
    Se houver cache (mesmo expirado), retorna instantaneamente (<2ms) e agenda
    recomputação assíncrona em background.
    """
    try:
        user_uuid = uuid.UUID(user_id)
    except ValueError:
        return {
            "items": [],
            "total_items": 0,
            "page": page,
            "page_size": page_size,
            "total_pages": 0,
            "media_type": media_type,
            "sort_by": sort_by,
            "period": period
        }

    clean_search = (search or "").strip().lower()
    page = max(1, int(page))
    page_size = max(1, min(50, int(page_size)))

    cache_key = f"stats:{user_id}:rankings:{media_type}:{sort_by}:{period}:{page}:{page_size}:{clean_search}"
    cached_data, is_stale = stats_cache.get_with_status(cache_key)

    if cached_data is not None:
        if is_stale and cache_key not in _revalidating_ranking_keys:
            _revalidating_ranking_keys.add(cache_key)
            if background_tasks is not None:
                background_tasks.add_task(
                    recompute_and_cache_ranking,
                    user_id, media_type, sort_by, period, page, page_size, search
                )
            else:
                asyncio.create_task(
                    recompute_and_cache_ranking(
                        user_id, media_type, sort_by, period, page, page_size, search
                    )
                )
        return cached_data

    return await _compute_user_media_ranking(
        db, user_id, media_type=media_type, sort_by=sort_by, period=period, page=page, page_size=page_size, search=search
    )


_revalidating_people_keys: set = set()

async def recompute_and_cache_people_ranking(
    user_id: str,
    role: str,
    sort_by: str,
    period: str,
    page: int,
    page_size: int,
    search: Optional[str]
):
    """Recomputa o ranking de pessoas (elenco/diretores) em background e atualiza o cache."""
    clean_search = (search or "").strip().lower()
    cache_key = f"stats:{user_id}:people:{role}:{sort_by}:{period}:{page}:{page_size}:{clean_search}"
    try:
        from app.core.database import AsyncSessionLocal
        async with AsyncSessionLocal() as session:
            await _compute_user_people_ranking(session, user_id, role, sort_by, period, page, page_size, search)
    except Exception as e:
        logger.error("Erro ao revalidar people ranking em background para usuário %s: %s", user_id, e)
    finally:
        _revalidating_people_keys.discard(cache_key)

async def _compute_user_people_ranking(
    db: AsyncSession,
    user_id: str,
    role: str = "cast",
    sort_by: str = "count",
    period: str = "all",
    page: int = 1,
    page_size: int = 10,
    search: Optional[str] = None
):
    try:
        user_uuid = uuid.UUID(user_id)
    except ValueError:
        return {
            "items": [],
            "total_items": 0,
            "page": page,
            "page_size": page_size,
            "total_pages": 0,
            "role": role,
            "sort_by": sort_by,
            "period": period,
            "search": search
        }

    clean_search = (search or "").strip().lower()
    page = max(1, int(page))
    page_size = max(1, min(50, int(page_size)))
    cache_key = f"stats:{user_id}:people:{role}:{sort_by}:{period}:{page}:{page_size}:{clean_search}"

    now = datetime.now(timezone.utc)
    cutoff = None
    if period == "30days":
        cutoff = now - timedelta(days=30)
    elif period == "6months":
        cutoff = now - timedelta(days=180)
    elif period == "year":
        cutoff = now - timedelta(days=365)

    params = {"user_uuid": user_uuid}
    cutoff_filter = ""
    if cutoff is not None:
        params["cutoff"] = cutoff
        cutoff_filter = "AND COALESCE(uli.last_watched_at, uli.updated_at, uli.created_at) >= :cutoff"

    query = text(f"""
        SELECT 
            uli.id AS item_id,
            uli.status,
            uli.rating,
            COALESCE(uli.rewatch_count, 0) AS rewatch_count,
            uli.total_time_minutes,
            uli.episodes_watched_count,
            m.title,
            m.media_type,
            COALESCE(m.runtime, 0) AS runtime,
            m.main_cast,
            m.directors,
            m.cast_ids,
            m.crew_ids
        FROM user_list_items uli
        JOIN media m ON m.id = uli.media_id
        WHERE uli.user_id = :user_uuid
          AND (uli.episodes_watched_count > 0 OR uli.status IN ('completed', 'watching') OR uli.rating IS NOT NULL)
          {cutoff_filter}
    """)
    res = await db.execute(query, params)
    rows = res.fetchall()

    people_map = {}
    for r in rows:
        item_time = r.total_time_minutes or 0
        if item_time == 0 and r.media_type == "movie":
            item_time = (r.runtime or 0) * (1 + (r.rewatch_count or 0))

        if role == "directors":
            person_id_map = {c["name"]: c.get("person_id") for c in (r.crew_ids or []) if isinstance(c, dict) and c.get("name")}
            profile_path_map = {c["name"]: c.get("profile_path") for c in (r.crew_ids or []) if isinstance(c, dict) and c.get("name")}
            people_list = r.directors or []
        else:  # cast
            person_id_map = {c["name"]: c.get("person_id") for c in (r.cast_ids or []) if isinstance(c, dict) and c.get("name")}
            profile_path_map = {c["name"]: c.get("profile_path") for c in (r.cast_ids or []) if isinstance(c, dict) and c.get("name")}
            people_list = (r.main_cast or [])[:5]

        for p in people_list:
            name = p.get("name") if isinstance(p, dict) else str(p).strip()
            if not name:
                continue

            if clean_search and clean_search not in name.lower():
                continue

            entry = people_map.setdefault(name, {
                "name": name,
                "person_id": None,
                "profile_path": None,
                "role": role,
                "count": 0,
                "totalTime": 0,
                "ratings": [],
                "sample_titles": []
            })
            entry["count"] += 1
            entry["totalTime"] += item_time
            if not entry["person_id"] and name in person_id_map:
                entry["person_id"] = person_id_map[name]
            if not entry["profile_path"] and name in profile_path_map:
                entry["profile_path"] = profile_path_map[name]
            if r.rating and r.rating > 0:
                entry["ratings"].append(r.rating)
            if len(entry["sample_titles"]) < 3 and r.title not in entry["sample_titles"]:
                entry["sample_titles"].append(r.title)

    all_people = list(people_map.values())
    for p in all_people:
        p["avgRating"] = round(sum(p["ratings"]) / len(p["ratings"]), 1) if p["ratings"] else None

    if sort_by == "time":
        all_people.sort(key=lambda x: (x["totalTime"], x["count"], x["avgRating"] or 0), reverse=True)
    elif sort_by == "rating":
        all_people.sort(key=lambda x: (x["avgRating"] is not None, x["avgRating"] or 0, x["count"], x["totalTime"]), reverse=True)
    elif sort_by == "name":
        all_people.sort(key=lambda x: x["name"].lower())
    else:  # count (default)
        all_people.sort(key=lambda x: (x["count"], x["totalTime"], x["avgRating"] or 0), reverse=True)

    total_items = len(all_people)
    total_pages = max(1, (total_items + page_size - 1) // page_size) if total_items > 0 else 0
    start = (page - 1) * page_size
    sliced = all_people[start : start + page_size]

    page_items = []
    for idx, p in enumerate(sliced):
        page_items.append({
            "rank": start + idx + 1,
            "name": p["name"],
            "person_id": p.get("person_id"),
            "profile_path": p.get("profile_path"),
            "role": p["role"],
            "count": p["count"],
            "totalTime": p["totalTime"],
            "avgRating": p["avgRating"],
            "sample_titles": p["sample_titles"]
        })

    output = {
        "items": page_items,
        "total_items": total_items,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
        "role": role,
        "sort_by": sort_by,
        "period": period,
        "search": search
    }

    stats_cache.set(cache_key, output)
    return output


async def get_user_people_ranking(
    db: AsyncSession,
    user_id: str,
    role: str = "cast",
    sort_by: str = "count",
    period: str = "all",
    page: int = 1,
    page_size: int = 10,
    search: Optional[str] = None,
    background_tasks: Optional[BackgroundTasks] = None
):
    """
    Retorna o ranking paginado de elenco/diretores com suporte a Stale-While-Revalidate (SWR).
    """
    try:
        user_uuid = uuid.UUID(user_id)
    except ValueError:
        return {
            "items": [],
            "total_items": 0,
            "page": page,
            "page_size": page_size,
            "total_pages": 0,
            "role": role,
            "sort_by": sort_by,
            "period": period,
            "search": search
        }

    clean_search = (search or "").strip().lower()
    page = max(1, int(page))
    page_size = max(1, min(50, int(page_size)))

    cache_key = f"stats:{user_id}:people:{role}:{sort_by}:{period}:{page}:{page_size}:{clean_search}"
    cached_data, is_stale = stats_cache.get_with_status(cache_key)

    if cached_data is not None:
        if is_stale and cache_key not in _revalidating_people_keys:
            _revalidating_people_keys.add(cache_key)
            if background_tasks is not None:
                background_tasks.add_task(
                    recompute_and_cache_people_ranking,
                    user_id, role, sort_by, period, page, page_size, search
                )
            else:
                asyncio.create_task(
                    recompute_and_cache_people_ranking(
                        user_id, role, sort_by, period, page, page_size, search
                    )
                )
        return cached_data

    return await _compute_user_people_ranking(
        db, user_id, role=role, sort_by=sort_by, period=period, page=page, page_size=page_size, search=search
    )



