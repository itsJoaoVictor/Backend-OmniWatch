from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload, load_only
from app.tracking.models import UserListItem, UserEpisodeProgress
from app.tracking.schemas import UserListItemCreate, UserListItemUpdate, UserEpisodeProgressCreate
from app.media.services import get_media_by_tmdb_id, create_media
from app.media.schemas import MediaCreate
from app.media.models import Media
from app.core.cache import tv_status_cache, stats_cache
import uuid
from typing import Optional
from datetime import datetime, timezone, timedelta
import asyncio

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
    """Invalida o cache de estatísticas do usuário quando houver modificação nos seus registros."""
    stats_cache.delete_prefix(f"stats:{user_id}")

async def get_user_list(db: AsyncSession, user_id: str):
    target_uuid = user_id if isinstance(user_id, uuid.UUID) else uuid.UUID(str(user_id))
    # Carrega a lista com media em uma única consulta rápida no banco (sub-15ms)
    result = await db.execute(
        select(UserListItem)
        .where(UserListItem.user_id == target_uuid)
        .options(selectinload(UserListItem.media))
    )
    items = result.scalars().all()

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
            runtime=runtime
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
    result = await db.execute(
        select(UserListItem)
        .where(UserListItem.user_id == uuid.UUID(user_id), UserListItem.media_id == media.id)
        .options(selectinload(UserListItem.media))
    )
    existing_item = result.scalars().first()
    if existing_item:
        if existing_item.status != final_status:
            existing_item.status = final_status
            if item.rating is not None:
                existing_item.rating = item.rating
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

    new_item = UserListItem(
        user_id=uuid.UUID(user_id),
        media_id=media.id,
        status=final_status,
        rating=item.rating,
        rewatch_count=item.rewatch_count
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
    
    invalidate_user_stats_cache(user_id)
    return saved_item

async def update_list_item(db: AsyncSession, user_id: str, item_id: str, update_data: UserListItemUpdate):
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

    if update_data.status is not None:
        from app.core.utils import is_date_released
        from fastapi import HTTPException

        rel_date = item.media.release_date if item.media else None
        if not rel_date and item.media:
            from app.details.services import fetch_movie_details, fetch_tv_details
            try:
                if item.media.media_type == "movie":
                    tmdb_data = await fetch_movie_details(item.media.tmdb_id)
                else:
                    tmdb_data = await fetch_tv_details(item.media.tmdb_id)
                rel_date = tmdb_data.release_date if hasattr(tmdb_data, 'release_date') else getattr(tmdb_data, 'first_air_date', None)
                if rel_date:
                    item.media.release_date = rel_date
            except Exception:
                pass

        is_released = is_date_released(rel_date)

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
                from app.core.utils import is_date_released
                from fastapi import HTTPException
                rel_date = item.media.release_date
                if not rel_date:
                    from app.details.services import fetch_movie_details
                    try:
                        tmdb_data = await fetch_movie_details(item.media.tmdb_id)
                        rel_date = tmdb_data.release_date
                        if rel_date:
                            item.media.release_date = rel_date
                    except Exception:
                        pass
                if not is_date_released(rel_date):
                    raise HTTPException(
                        status_code=400,
                        detail="Filmes que ainda não estrearam não podem ser marcados como assistidos."
                    )
                item.status = "completed"
            elif item.media.media_type == "tv":
                from app.details.services import fetch_tv_details, fetch_season_details
                from app.core.utils import is_date_released
                from sqlalchemy import delete
                has_unreleased_episodes = False
                new_progresses = []
                try:
                    tv_data = await fetch_tv_details(item.media.tmdb_id)
                    await db.execute(delete(UserEpisodeProgress).where(UserEpisodeProgress.user_list_item_id == item.id))
                    
                    for season in tv_data.seasons:
                        if season.season_number > 0:
                            try:
                                season_details = await fetch_season_details(item.media.tmdb_id, season.season_number)
                                for ep in season_details.episodes:
                                    if is_date_released(ep.air_date):
                                        new_progresses.append(
                                            UserEpisodeProgress(
                                                user_list_item_id=item.id,
                                                season_number=season.season_number,
                                                episode_number=ep.episode_number
                                            )
                                        )
                                    else:
                                        has_unreleased_episodes = True
                            except Exception as e:
                                print(f"Failed to fetch season {season.season_number} details: {e}")
                                if season.air_date and not is_date_released(season.air_date):
                                    has_unreleased_episodes = True
                                else:
                                    for ep_num in range(1, season.episode_count + 1):
                                        new_progresses.append(
                                            UserEpisodeProgress(
                                                user_list_item_id=item.id,
                                                season_number=season.season_number,
                                                episode_number=ep_num
                                            )
                                        )
                    if new_progresses:
                        db.add_all(new_progresses)
                    
                    is_series_ended = tv_data.status in ("Ended", "Canceled")
                    if is_series_ended and not has_unreleased_episodes:
                        item.status = "completed"
                    else:
                        item.status = "watching"
                except Exception as e:
                    print(f"Failed to auto-mark episodes: {e}")
        elif update_data.status == "watching" and item.media.media_type == "movie":
            from app.core.utils import is_date_released
            from fastapi import HTTPException
            rel_date = item.media.release_date
            if not is_date_released(rel_date):
                raise HTTPException(
                    status_code=400,
                    detail="Filmes que ainda não estrearam não podem ser marcados como em andamento."
                )
            item.status = "watching"
        else:
            item.status = update_data.status

        # Regra de negócio: itens em 'plan_to_watch' ou 'upcoming' não possuem nota
        if item.status in ["plan_to_watch", "upcoming"]:
            item.rating = None
    
    if update_data.rating is not None and item.status not in ["plan_to_watch", "upcoming"]:
        item.rating = update_data.rating
    if update_data.rewatch_count is not None:
        item.rewatch_count = update_data.rewatch_count

    media_ref = item.media
    await db.commit()
    await db.refresh(item)
    
    # Trigger Profile Update (rating or watch status might have changed)
    from app.recommendation.profile_manager import update_user_profile
    if media_ref:
        await update_user_profile(db, user_id, item, media_ref)
    
    reloaded = await db.execute(
        select(UserListItem)
        .where(UserListItem.id == item.id)
        .options(
            selectinload(UserListItem.media),
            selectinload(UserListItem.episodes)
        )
    )
    if item.media and item.media.media_type == "tv":
        tv_status_cache.delete_prefix(f"tv_status:{item.media.tmdb_id}")
        if item.status == "watching":
            is_up_to_date, next_ep = await determine_tv_next_episode_status(item.media.tmdb_id, item.episodes)
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
        # Fase 3: Feedback Implícito - sinal negativo (-0.5) ao remover item da lista
        if item.media:
            from app.recommendation.profile_manager import update_user_profile
            await update_user_profile(db, user_id, item, item.media, explicit_scale=-0.5)

        await db.delete(item)
        await db.commit()
        invalidate_user_stats_cache(user_id)
        return True
    return False

async def add_episode_progress(db: AsyncSession, user_id: str, item_id: str, progress: UserEpisodeProgressCreate):
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

    # Validação de data de lançamento do episódio
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
            print(f"Error checking episode release date: {e}")

    # Check if episode already marked
    prog_result = await db.execute(
        select(UserEpisodeProgress)
        .where(
            UserEpisodeProgress.user_list_item_id == uuid.UUID(item_id),
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
            if item.media:
                from app.recommendation.profile_manager import update_user_profile, calculate_episode_rating_scale
                scale = calculate_episode_rating_scale(progress.rating)
                await update_user_profile(db, user_id, item, item.media, explicit_scale=scale)
        return existing

    new_progress = UserEpisodeProgress(
        user_list_item_id=uuid.UUID(item_id),
        season_number=progress.season_number,
        episode_number=progress.episode_number,
        rating=progress.rating
    )
    db.add(new_progress)
    
    # Update last_watched_at
    item.last_watched_at = datetime.now(timezone.utc)
    
    is_completed = False
    if item.media and item.media.media_type == "tv":
        from app.details.services import fetch_tv_details
        try:
            tv_data = await fetch_tv_details(item.media.tmdb_id)
            if tv_data.status in ("Ended", "Canceled"):
                all_prog = await db.execute(select(UserEpisodeProgress).where(UserEpisodeProgress.user_list_item_id == item_uuid))
                total_marked = len(all_prog.scalars().all()) + 1
                if total_marked >= tv_data.number_of_episodes:
                    item.status = "completed"
                    is_completed = True
        except Exception:
            pass

    if not is_completed and item.status == "plan_to_watch":
        item.status = "watching"

    await db.commit()
    await db.refresh(new_progress)
    
    # Trigger Profile Update
    from app.recommendation.profile_manager import update_user_profile, calculate_episode_rating_scale
    if progress.rating is not None:
        episode_scale = calculate_episode_rating_scale(progress.rating)
        await update_user_profile(db, user_id, item, item.media, explicit_scale=episode_scale)
    else:
        completion_boost = 1.5 if is_completed else None
        await update_user_profile(db, user_id, item, item.media, explicit_scale=completion_boost)
    
    if item.media:
        tv_status_cache.delete_prefix(f"tv_status:{item.media.tmdb_id}")
    
    invalidate_user_stats_cache(user_id)
    return new_progress

async def update_episode_rating(
    db: AsyncSession,
    user_id: str,
    item_id_or_tmdb_id: str,
    season_number: int,
    episode_number: int,
    rating: Optional[float]
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

    await db.commit()
    await db.refresh(prog)

    if rating is not None and item.media:
        from app.recommendation.profile_manager import update_user_profile, calculate_episode_rating_scale
        scale = calculate_episode_rating_scale(rating)
        await update_user_profile(db, user_id, item, item.media, explicit_scale=scale)

    if item.media:
        tv_status_cache.delete_prefix(f"tv_status:{item.media.tmdb_id}")

    return prog

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

    # Verify if the item belongs to the user
    result = await db.execute(
        select(UserListItem)
        .where(UserListItem.id == item_uuid, UserListItem.user_id == user_uuid)
    )
    item = result.scalars().first()
    if not item:
        return False

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
        await db.commit()
        if item.media:
            tv_status_cache.delete_prefix(f"tv_status:{item.media.tmdb_id}")
        return True
    
    return False

async def bulk_mark_episodes(db: AsyncSession, user_id: str, item_id: str, target: UserEpisodeProgressCreate):
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
                print(f"Failed to fetch season {season.season_number} details in bulk_mark: {e}")
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
        await db.commit()
        await db.refresh(item)
        
        # Trigger Profile Update with completion boost
        from app.recommendation.profile_manager import update_user_profile
        completion_boost = 1.5 if is_completed else None
        if media_ref:
            await update_user_profile(db, user_id, item, media_ref, explicit_scale=completion_boost)
            tv_status_cache.delete_prefix(f"tv_status:{media_ref.tmdb_id}")
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

async def get_user_statistics(db: AsyncSession, user_id: str, period: str = "all", media_type: str = "all"):
    try:
        user_uuid = uuid.UUID(user_id)
    except ValueError:
        return {}

    # 1. Cache lookup: <1ms response for repeated queries
    cache_key = f"stats:{user_id}:{period}:{media_type}"
    cached_data, is_stale = stats_cache.get_with_status(cache_key)
    if not is_stale and cached_data is not None:
        return cached_data

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

    # 2. Optimized SQL Query: selective column loading (excludes heavy vector embeddings and keywords)
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
                    Media.main_cast
                )
            )
        )
    )
    if media_type != "all":
        items_query = items_query.join(UserListItem.media).where(Media.media_type == media_type)

    result = await db.execute(items_query)
    items = result.scalars().all()

    # 3. Optimized Episode Query: SQL JOIN instead of WHERE IN, pushdown date cutoff to Postgres
    progress_map = {}
    if media_type != "movie" and items:
        prog_query = (
            select(
                UserEpisodeProgress.user_list_item_id,
                UserEpisodeProgress.season_number,
                UserEpisodeProgress.episode_number,
                UserEpisodeProgress.rating,
                UserEpisodeProgress.watched_at
            )
            .join(UserListItem, UserEpisodeProgress.user_list_item_id == UserListItem.id)
            .where(UserListItem.user_id == user_uuid)
        )
        if cutoff is not None:
            prog_query = prog_query.where(UserEpisodeProgress.watched_at >= cutoff)

        prog_result = await db.execute(prog_query)
        progresses = prog_result.all()
        for p in progresses:
            progress_map.setdefault(p.user_list_item_id, []).append(p)

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

    activity_months = {}
    num_months = 12 if period in ("year", "all") else 6
    for i in range(num_months - 1, -1, -1):
        m_date = now - timedelta(days=i * 30)
        key = f"{m_date.year:04d}-{m_date.month:02d}"
        activity_months[key] = {
            "minutes": 0,
            "count": 0,
            "label": f"{MONTH_NAMES_PT[m_date.month - 1]}/{str(m_date.year)[2:]}"
        }

    earliest_date = now

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
            eps = progress_map.get(item.id, [])
            ep_count_in_period = 0
            for ep in eps:
                ep_date = ep.watched_at or item_date
                if ep_date and ep_date.tzinfo is None:
                    ep_date = ep_date.replace(tzinfo=timezone.utc)
                if ep_date and ep_date < earliest_date:
                    earliest_date = ep_date

                if cutoff is None or (ep_date and ep_date >= cutoff):
                    ep_count_in_period += 1
                    if ep_date:
                        m_key = f"{ep_date.year:04d}-{ep_date.month:02d}"
                        if m_key in activity_months:
                            activity_months[m_key]["minutes"] += runtime
                            activity_months[m_key]["count"] += 1

            if ep_count_in_period > 0 or (cutoff is None and item.status in ("completed", "watching")):
                is_watched_in_period = True
                total_episodes_watched += ep_count_in_period
                item_time_minutes = runtime * ep_count_in_period
                total_time_tv_minutes += item_time_minutes

        if not is_watched_in_period and cutoff is not None:
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

        raw_cast = m.main_cast or []
        if isinstance(raw_cast, list):
            for actor in raw_cast[:5]:
                a_name = actor.get("name") if isinstance(actor, dict) else str(actor).strip()
                if a_name:
                    c_entry = cast_stats.setdefault(a_name, {"count": 0, "totalTime": 0, "ratings": []})
                    c_entry["count"] += 1
                    c_entry["totalTime"] += item_time_minutes
                    if item.rating:
                        c_entry["ratings"].append(item.rating)

        raw_dirs = m.directors or []
        if isinstance(raw_dirs, list):
            for director in raw_dirs:
                d_name = director.get("name") if isinstance(director, dict) else str(director).strip()
                if d_name:
                    d_entry = directors_stats.setdefault(d_name, {"count": 0, "totalTime": 0, "ratings": []})
                    d_entry["count"] += 1
                    d_entry["totalTime"] += item_time_minutes
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

    def format_ranking(stats_dict, limit=15):
        res = []
        sorted_items = sorted(stats_dict.items(), key=lambda x: (x[1]["totalTime"], x[1]["count"]), reverse=True)[:limit]
        for name, data in sorted_items:
            res.append({
                "name": name,
                "count": data["count"],
                "totalTime": data["totalTime"],
                "avgRating": round(sum(data["ratings"]) / len(data["ratings"]), 1) if data["ratings"] else None,
                "percentage": round((data["totalTime"] / total_time) * 100, 1) if total_time > 0 else 0.0
            })
        return res

    genres_ranking = format_ranking(genres_stats, limit=20)
    cast_ranking = format_ranking(cast_stats, limit=15)
    directors_ranking = format_ranking(directors_stats, limit=15)

    timeline = []
    for k in sorted(activity_months.keys()):
        val = activity_months[k]
        timeline.append({
            "month": val["label"],
            "hours": round(val["minutes"] / 60, 1),
            "count": val["count"]
        })

    return {
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

    cache_key = f"stats:{user_id}:rankings:{media_type}:{sort_by}:{period}:{page}:{page_size}:{clean_search}"
    cached_data, is_stale = stats_cache.get_with_status(cache_key)
    if not is_stale and cached_data is not None:
        return cached_data

    now = datetime.now(timezone.utc)
    cutoff = None
    if period == "30days":
        cutoff = now - timedelta(days=30)
    elif period == "6months":
        cutoff = now - timedelta(days=180)
    elif period == "year":
        cutoff = now - timedelta(days=365)

    items_query = (
        select(UserListItem)
        .join(UserListItem.media)
        .where(
            UserListItem.user_id == user_uuid,
            Media.media_type == media_type
        )
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
                    Media.tmdb_id,
                    Media.media_type,
                    Media.title,
                    Media.poster_path,
                    Media.release_date,
                    Media.genres,
                    Media.runtime
                )
            )
        )
    )

    if clean_search:
        items_query = items_query.where(Media.title.ilike(f"%{clean_search}%"))

    result = await db.execute(items_query)
    user_items = result.scalars().all()

    progress_map = {}
    if media_type == "tv" and user_items:
        prog_query = (
            select(
                UserEpisodeProgress.user_list_item_id,
                UserEpisodeProgress.season_number,
                UserEpisodeProgress.episode_number,
                UserEpisodeProgress.rating,
                UserEpisodeProgress.watched_at
            )
            .join(UserListItem, UserEpisodeProgress.user_list_item_id == UserListItem.id)
            .where(UserListItem.user_id == user_uuid)
        )
        if cutoff is not None:
            prog_query = prog_query.where(UserEpisodeProgress.watched_at >= cutoff)

        prog_result = await db.execute(prog_query)
        for p in prog_result.all():
            progress_map.setdefault(p.user_list_item_id, []).append(p)

    candidates = []
    for item in user_items:
        m = item.media
        if not m:
            continue

        runtime = m.runtime or 0
        item_date = item.last_watched_at or item.updated_at or item.created_at
        if item_date and item_date.tzinfo is None:
            item_date = item_date.replace(tzinfo=timezone.utc)

        raw_genres = m.genres or []
        genres_list = []
        for g in raw_genres:
            g_name = g.get("name") if isinstance(g, dict) else str(g)
            if g_name:
                genres_list.append(normalize_genre(g_name))

        if media_type == "movie":
            is_valid = False
            if cutoff is not None:
                if item_date and item_date >= cutoff and item.status in ("completed", "watching"):
                    is_valid = True
            else:
                if item.status in ("completed", "watching") or (item.rewatch_count and item.rewatch_count > 0) or item.rating is not None:
                    is_valid = True

            if not is_valid:
                continue

            factor = 1 + (item.rewatch_count or 0)
            total_time = runtime * factor
            episodes_watched = 1 if item.status == "completed" else 0

            candidates.append({
                "id": item.id,
                "media_id": m.id,
                "tmdb_id": m.tmdb_id,
                "title": m.title,
                "media_type": "movie",
                "poster_path": m.poster_path,
                "release_date": m.release_date,
                "runtime": runtime,
                "genres": genres_list[:3],
                "rating": item.rating,
                "rewatch_count": item.rewatch_count or 0,
                "total_time_minutes": total_time,
                "episodes_watched": episodes_watched,
                "status": item.status,
                "last_watched_at": item_date
            })

        elif media_type == "tv":
            eps = progress_map.get(item.id, [])
            ep_count = len(eps)

            is_valid = False
            if cutoff is not None:
                if ep_count > 0:
                    is_valid = True
            else:
                if ep_count > 0 or item.status in ("completed", "watching") or item.rating is not None:
                    is_valid = True

            if not is_valid:
                continue

            total_time = runtime * ep_count
            last_date = item_date
            for ep in eps:
                ep_dt = ep.watched_at
                if ep_dt:
                    if ep_dt.tzinfo is None:
                        ep_dt = ep_dt.replace(tzinfo=timezone.utc)
                    if not last_date or ep_dt > last_date:
                        last_date = ep_dt

            final_rating = item.rating
            if final_rating is None:
                ep_ratings = [ep.rating for ep in eps if ep.rating is not None and ep.rating > 0]
                if ep_ratings:
                    final_rating = round(sum(ep_ratings) / len(ep_ratings), 1)

            candidates.append({
                "id": item.id,
                "media_id": m.id,
                "tmdb_id": m.tmdb_id,
                "title": m.title,
                "media_type": "tv",
                "poster_path": m.poster_path,
                "release_date": m.release_date,
                "runtime": runtime,
                "genres": genres_list[:3],
                "rating": final_rating,
                "rewatch_count": item.rewatch_count or 0,
                "total_time_minutes": total_time,
                "episodes_watched": ep_count,
                "status": item.status,
                "last_watched_at": last_date
            })

    def sort_key(x):
        r_val = x["rating"] if x["rating"] is not None else -1.0
        t_val = x["total_time_minutes"]
        time_key = x["last_watched_at"].timestamp() if x["last_watched_at"] else 0

        if sort_by == "rating":
            return (x["rating"] is not None, r_val, t_val)
        elif sort_by == "rewatch":
            return (x["rewatch_count"], t_val, r_val)
        elif sort_by == "episodes":
            return (x["episodes_watched"], t_val, r_val)
        elif sort_by == "recent":
            return (time_key, t_val)
        elif sort_by == "title":
            return x["title"].lower()
        else: # "time" (default)
            return (t_val, r_val)

    reverse_sort = (sort_by != "title")
    candidates.sort(key=sort_key, reverse=reverse_sort)

    total_items = len(candidates)
    total_pages = max(1, (total_items + page_size - 1) // page_size) if total_items > 0 else 0
    start = (page - 1) * page_size
    end = start + page_size
    page_items = candidates[start:end]

    for idx, obj in enumerate(page_items):
        obj["rank"] = start + idx + 1

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

