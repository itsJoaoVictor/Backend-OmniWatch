from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload
from app.tracking.models import UserListItem, UserEpisodeProgress
from app.tracking.schemas import UserListItemCreate, UserListItemUpdate, UserEpisodeProgressCreate
from app.media.services import get_media_by_tmdb_id, create_media
from app.media.schemas import MediaCreate
import uuid
from typing import Optional
from datetime import datetime, timezone

async def get_user_list(db: AsyncSession, user_id: str):
    target_uuid = user_id if isinstance(user_id, uuid.UUID) else uuid.UUID(str(user_id))
    result = await db.execute(
        select(UserListItem)
        .where(UserListItem.user_id == target_uuid)
        .options(selectinload(UserListItem.media))
    )
    return result.scalars().all()

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
        
        if not title or not runtime or not genres or not directors or not main_cast:
            from app.details.services import fetch_movie_details, fetch_tv_details
            try:
                if item.media_type == "movie":
                    tmdb_data = await fetch_movie_details(item.tmdb_id)
                    title = title or tmdb_data.title
                    poster_path = poster_path or tmdb_data.poster_path
                    backdrop_path = backdrop_path or tmdb_data.backdrop_path
                    runtime = tmdb_data.runtime or 0
                    original_language = tmdb_data.original_language or None
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
    
    # Reload with media relationship
    result = await db.execute(
        select(UserListItem)
        .where(UserListItem.id == new_item.id)
        .options(selectinload(UserListItem.media))
    )
    saved_item = result.scalars().first()
    
    # Trigger Profile Update
    from app.recommendation.profile_manager import update_user_profile
    await update_user_profile(db, user_id, saved_item, saved_item.media)

    # Dispara o cacheamento local do pôster em segundo plano (LRU ~45KB)
    if saved_item and saved_item.media and saved_item.media.poster_path:
        import asyncio
        from app.images.router import ensure_image_cached
        asyncio.create_task(ensure_image_cached(saved_item.media.poster_path, "w342"))
    
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
    
    if update_data.rating is not None:
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
        .options(selectinload(UserListItem.media))
    )
    item = reloaded.scalars().first() or item
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

    # Trigger Profile Update com peso moderado (0.4x)
    if rating is not None and item.media:
        from app.recommendation.profile_manager import update_user_profile, calculate_episode_rating_scale
        scale = calculate_episode_rating_scale(rating)
        await update_user_profile(db, user_id, item, item.media, explicit_scale=scale)

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
    return True


async def get_user_statistics(db: AsyncSession, user_id: str):
    try:
        user_uuid = uuid.UUID(user_id)
    except ValueError:
        return {}

    # Get all list items with their media
    result = await db.execute(
        select(UserListItem)
        .where(UserListItem.user_id == user_uuid)
        .options(selectinload(UserListItem.media))
    )
    items = result.scalars().all()

    # Get all episode progress for this user
    # Optimize by querying all progress for the user's items
    item_ids = [item.id for item in items]
    progress_map = {}
    if item_ids:
        prog_result = await db.execute(
            select(UserEpisodeProgress)
            .where(UserEpisodeProgress.user_list_item_id.in_(item_ids))
        )
        progresses = prog_result.scalars().all()
        for p in progresses:
            progress_map[p.user_list_item_id] = progress_map.get(p.user_list_item_id, 0) + 1

    total_movies_watched = 0
    total_episodes_watched = 0
    total_time_movies_minutes = 0
    total_time_tv_minutes = 0
    rating_distribution = {str(i): 0 for i in range(1, 11)}
    genres_count = {}

    for item in items:
        # Rating
        if item.rating:
            rating_key = str(int(item.rating))
            if rating_key in rating_distribution:
                rating_distribution[rating_key] += 1
        
        # Genres
        if item.media.genres:
            for genre in item.media.genres:
                # Handle both dict ({"name": "Action"}) and string ("Action") forms
                g_name = genre.get('name') if isinstance(genre, dict) else genre
                if g_name:
                    genres_count[g_name] = genres_count.get(g_name, 0) + 1

        # Watch Time
        runtime = item.media.runtime or 0
        if item.media.media_type == 'movie':
            if item.status in ('completed', 'watching'): # completed movies count
                total_movies_watched += 1 + item.rewatch_count
                total_time_movies_minutes += runtime * (1 + item.rewatch_count)
        elif item.media.media_type == 'tv':
            episodes_watched = progress_map.get(item.id, 0)
            total_episodes_watched += episodes_watched
            total_time_tv_minutes += runtime * episodes_watched

    # Format output
    return {
        'totalTime': total_time_movies_minutes + total_time_tv_minutes,
        'totalTimeMovies': total_time_movies_minutes,
        'totalTimeTv': total_time_tv_minutes,
        'totalMovies': total_movies_watched,
        'totalEpisodes': total_episodes_watched,
        'ratingDistribution': [{'rating': k, 'count': v} for k, v in sorted(rating_distribution.items(), key=lambda x: int(x[0]))],
        'topGenres': [{'name': k, 'count': v} for k, v in sorted(genres_count.items(), key=lambda item: item[1], reverse=True)[:10]]
    }

