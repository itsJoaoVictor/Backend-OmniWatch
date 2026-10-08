import asyncio
import uuid
import logging
import time
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload
from sqlalchemy import delete

from app.core.database import AsyncSessionLocal
from app.media_collections.models import Collection, CollectionItem, UserCollection, UserCollectionSuggestion
from app.media.models import Media
from app.tracking.models import UserListItem
from app.tracking.schemas import UserListItemCreate
from app.tracking.services import add_to_list
from app.notifications.models import Notification
from app.details.services import fetch_collection_details

logger = logging.getLogger(__name__)

async def get_or_create_collection(db: AsyncSession, tmdb_collection_id: int) -> Collection:
    result = await db.execute(
        select(Collection)
        .where(Collection.tmdb_id == tmdb_collection_id)
        .options(selectinload(Collection.items))
    )
    collection = result.scalars().first()

    # Fetch TMDB data
    tmdb_data = await fetch_collection_details(tmdb_collection_id)

    if not collection:
        collection = Collection(
            tmdb_id=tmdb_collection_id,
            name=tmdb_data.name,
            overview=tmdb_data.overview,
            poster_path=tmdb_data.poster_path,
            backdrop_path=tmdb_data.backdrop_path,
            parts_count=len(tmdb_data.parts),
            last_synced_at=datetime.now(timezone.utc)
        )
        db.add(collection)
        await db.commit()
        await db.refresh(collection)
    else:
        collection.name = tmdb_data.name
        collection.overview = tmdb_data.overview
        collection.poster_path = tmdb_data.poster_path
        collection.backdrop_path = tmdb_data.backdrop_path
        collection.parts_count = len(tmdb_data.parts)
        collection.last_synced_at = datetime.now(timezone.utc)
        await db.commit()
        await db.refresh(collection)

    # Sync collection items
    existing_items_res = await db.execute(
        select(CollectionItem).where(CollectionItem.collection_id == collection.id)
    )
    existing_items_map = {item.tmdb_id: item for item in existing_items_res.scalars().all()}

    for part in tmdb_data.parts:
        if part.id not in existing_items_map:
            new_item = CollectionItem(
                collection_id=collection.id,
                tmdb_id=part.id,
                title=part.title,
                release_date=part.release_date,
                poster_path=part.poster_path,
                backdrop_path=part.backdrop_path
            )
            db.add(new_item)

    await db.commit()

    # Refresh with items
    refreshed = await db.execute(
        select(Collection)
        .where(Collection.id == collection.id)
        .options(selectinload(Collection.items))
    )
    return refreshed.scalars().first()


async def follow_collection(db: AsyncSession, user_id: str, tmdb_collection_id: int) -> Dict[str, Any]:
    user_uuid = uuid.UUID(str(user_id))

    collection = await get_or_create_collection(db, tmdb_collection_id)

    # Check if already followed
    res_uc = await db.execute(
        select(UserCollection).where(
            UserCollection.user_id == user_uuid,
            UserCollection.collection_id == collection.id
        )
    )
    existing_follow = res_uc.scalars().first()
    if not existing_follow:
        new_follow = UserCollection(
            user_id=user_uuid,
            collection_id=collection.id,
            auto_add=True
        )
        db.add(new_follow)
        await db.commit()

    # Add movies to user list
    movies_added = 0
    movies_already_in_list = 0

    # Get user list items for these movies
    for item in collection.items:
        # Check if already in user's list
        media_res = await db.execute(select(Media).where(Media.tmdb_id == item.tmdb_id))
        media = media_res.scalars().first()

        # Auto-recover poster_path / backdrop_path if media in DB was missing them
        if media:
            updated = False
            if not media.poster_path and item.poster_path:
                media.poster_path = item.poster_path
                updated = True
            if not media.backdrop_path and item.backdrop_path:
                media.backdrop_path = item.backdrop_path
                updated = True
            if updated:
                await db.commit()
                await db.refresh(media)

        already_in_list = False
        if media:
            uli_res = await db.execute(
                select(UserListItem).where(
                    UserListItem.user_id == user_uuid,
                    UserListItem.media_id == media.id
                )
            )
            if uli_res.scalars().first():
                already_in_list = True

        if already_in_list:
            movies_already_in_list += 1
        else:
            try:
                # Add to list as plan_to_watch with complete poster and backdrop
                create_payload = UserListItemCreate(
                    tmdb_id=item.tmdb_id,
                    media_type="movie",
                    title=item.title,
                    poster_path=item.poster_path,
                    backdrop_path=item.backdrop_path,
                    release_date=item.release_date,
                    status="plan_to_watch"
                )
                saved_item = await add_to_list(db, str(user_id), create_payload)
                if saved_item and saved_item.media_id and not item.media_id:
                    item.media_id = saved_item.media_id
                    await db.commit()
                movies_added += 1
            except Exception as e:
                logger.error(f"Error adding movie {item.tmdb_id} to user list: {e}")

    # Remove qualquer sugestão existente para essa coleção
    await db.execute(
        delete(UserCollectionSuggestion).where(
            UserCollectionSuggestion.user_id == user_uuid,
            UserCollectionSuggestion.collection_tmdb_id == tmdb_collection_id
        )
    )
    await db.commit()

    return {
        "success": True,
        "message": f"Coleção '{collection.name}' adicionada com sucesso!",
        "collection_id": collection.id,
        "tmdb_id": collection.tmdb_id,
        "name": collection.name,
        "movies_added": movies_added,
        "movies_already_in_list": movies_already_in_list
    }


async def unfollow_collection(db: AsyncSession, user_id: str, tmdb_collection_id: int) -> bool:
    user_uuid = uuid.UUID(str(user_id))
    col_res = await db.execute(select(Collection).where(Collection.tmdb_id == tmdb_collection_id))
    collection = col_res.scalars().first()
    if not collection:
        return False

    stmt = delete(UserCollection).where(
        UserCollection.user_id == user_uuid,
        UserCollection.collection_id == collection.id
    )
    await db.execute(stmt)
    await db.commit()
    return True


async def get_collection_status(db: AsyncSession, user_id: str, tmdb_collection_id: int) -> Dict[str, Any]:
    user_uuid = uuid.UUID(str(user_id))
    col_res = await db.execute(
        select(Collection)
        .where(Collection.tmdb_id == tmdb_collection_id)
        .options(selectinload(Collection.items))
    )
    collection = col_res.scalars().first()
    if not collection:
        return {
            "is_following": False,
            "auto_add": True,
            "total_movies": 0,
            "watched_movies": 0,
            "collection_id": None
        }

    uc_res = await db.execute(
        select(UserCollection).where(
            UserCollection.user_id == user_uuid,
            UserCollection.collection_id == collection.id
        )
    )
    user_col = uc_res.scalars().first()

    # Count watched movies
    watched_count = 0
    total_count = len(collection.items)

    for item in collection.items:
        if item.media_id:
            uli_res = await db.execute(
                select(UserListItem).where(
                    UserListItem.user_id == user_uuid,
                    UserListItem.media_id == item.media_id
                )
            )
            uli = uli_res.scalars().first()
            if uli and uli.status == "completed":
                watched_count += 1

    return {
        "is_following": user_col is not None,
        "auto_add": user_col.auto_add if user_col else True,
        "total_movies": total_count,
        "watched_movies": watched_count,
        "collection_id": collection.id
    }


async def get_user_collections(db: AsyncSession, user_id: str) -> List[Dict[str, Any]]:
    user_uuid = uuid.UUID(str(user_id))

    stmt = (
        select(UserCollection)
        .where(UserCollection.user_id == user_uuid)
        .options(
            selectinload(UserCollection.collection).selectinload(Collection.items)
        )
        .order_by(UserCollection.created_at.desc())
    )
    result = await db.execute(stmt)
    user_collections = result.scalars().all()

    # Preload user's list items to quickly determine status and rating
    uli_res = await db.execute(
        select(UserListItem)
        .where(UserListItem.user_id == user_uuid)
        .options(selectinload(UserListItem.media))
    )
    user_items = uli_res.scalars().all()
    user_items_by_tmdb: Dict[int, UserListItem] = {}
    for ui in user_items:
        if ui.media and ui.media.tmdb_id:
            user_items_by_tmdb[ui.media.tmdb_id] = ui

    response_list = []
    for uc in user_collections:
        col = uc.collection
        if not col:
            continue

        items_data = []
        watched_count = 0

        # Sort items by release_date
        sorted_items = sorted(col.items, key=lambda x: x.release_date or "9999-99-99")

        for item in sorted_items:
            ui = user_items_by_tmdb.get(item.tmdb_id)
            status = ui.status if ui else None
            rating = ui.rating if ui else None
            if status == "completed":
                watched_count += 1

            items_data.append({
                "id": item.id,
                "tmdb_id": item.tmdb_id,
                "title": item.title,
                "release_date": item.release_date,
                "poster_path": item.poster_path,
                "backdrop_path": item.backdrop_path,
                "media_id": item.media_id,
                "status": status,
                "rating": rating
            })

        total_movies = len(items_data)
        completion_pct = round((watched_count / total_movies) * 100, 1) if total_movies > 0 else 0.0

        response_list.append({
            "id": col.id,
            "tmdb_id": col.tmdb_id,
            "name": col.name,
            "overview": col.overview,
            "poster_path": col.poster_path,
            "backdrop_path": col.backdrop_path,
            "total_movies": total_movies,
            "watched_movies": watched_count,
            "completion_percentage": completion_pct,
            "items": items_data,
            "created_at": uc.created_at
        })

    return response_list


async def sync_collection_from_tmdb(db: AsyncSession, tmdb_collection_id: int, user_id: Optional[str] = None) -> Dict[str, Any]:
    """Sync a collection from TMDB. Detects new movies, adds to followers lists, reconciles missing items for user, and generates notifications."""
    col_res = await db.execute(
        select(Collection)
        .where(Collection.tmdb_id == tmdb_collection_id)
        .options(selectinload(Collection.items), selectinload(Collection.followers))
    )
    collection = col_res.scalars().first()
    if not collection:
        return {"success": False, "message": "Collection not found"}

    tmdb_data = await fetch_collection_details(tmdb_collection_id)

    # Update metadata
    collection.name = tmdb_data.name
    collection.overview = tmdb_data.overview
    collection.poster_path = tmdb_data.poster_path
    collection.backdrop_path = tmdb_data.backdrop_path
    collection.parts_count = len(tmdb_data.parts)
    collection.last_synced_at = datetime.now(timezone.utc)

    existing_tmdb_ids = {item.tmdb_id for item in collection.items}
    new_parts = [p for p in tmdb_data.parts if p.id not in existing_tmdb_ids]

    new_items_added = 0
    notifications_created = 0

    for part in new_parts:
        new_item = CollectionItem(
            collection_id=collection.id,
            tmdb_id=part.id,
            title=part.title,
            release_date=part.release_date,
            poster_path=part.poster_path,
            backdrop_path=part.backdrop_path
        )
        db.add(new_item)
        await db.flush()
        new_items_added += 1

        # For each follower with auto_add=True, add to their list and notify
        for follower in collection.followers:
            if not follower.auto_add:
                continue

            try:
                # Add to user list with complete poster and backdrop
                create_payload = UserListItemCreate(
                    tmdb_id=part.id,
                    media_type="movie",
                    title=part.title,
                    poster_path=part.poster_path,
                    backdrop_path=part.backdrop_path,
                    release_date=part.release_date,
                    status="plan_to_watch"
                )
                saved_item = await add_to_list(db, str(follower.user_id), create_payload)
                if saved_item and saved_item.media_id and not new_item.media_id:
                    new_item.media_id = saved_item.media_id

                # Create Notification
                notif = Notification(
                    user_id=follower.user_id,
                    title=f"Novo filme na coleção {collection.name}",
                    message=f"'{part.title}' foi adicionado automaticamente à sua lista como 'Quero Ver'.",
                    media_id=saved_item.media_id if saved_item else None
                )
                db.add(notif)
                notifications_created += 1
            except Exception as e:
                logger.error(f"Error auto-adding part {part.id} for user {follower.user_id}: {e}")

    # Reconciliação para o usuário chamador: adiciona filmes existentes da coleção que não estejam na lista
    reconciled_count = 0
    if user_id:
        try:
            target_uuid = uuid.UUID(str(user_id))
            # Recarregar itens da coleção atualizados
            current_items_res = await db.execute(
                select(CollectionItem).where(CollectionItem.collection_id == collection.id)
            )
            all_col_items = current_items_res.scalars().all()

            for item in all_col_items:
                media_res = await db.execute(select(Media).where(Media.tmdb_id == item.tmdb_id))
                media = media_res.scalars().first()

                in_user_list = False
                if media:
                    uli_res = await db.execute(
                        select(UserListItem).where(
                            UserListItem.user_id == target_uuid,
                            UserListItem.media_id == media.id
                        )
                    )
                    if uli_res.scalars().first():
                        in_user_list = True

                if not in_user_list:
                    create_payload = UserListItemCreate(
                        tmdb_id=item.tmdb_id,
                        media_type="movie",
                        title=item.title,
                        poster_path=item.poster_path,
                        backdrop_path=item.backdrop_path,
                        release_date=item.release_date,
                        status="plan_to_watch"
                    )
                    saved_item = await add_to_list(db, str(user_id), create_payload)
                    if saved_item and saved_item.media_id and not item.media_id:
                        item.media_id = saved_item.media_id
                    reconciled_count += 1
        except Exception as e:
            logger.error(f"Error during user reconciliation for collection {collection.id}: {e}")

    await db.commit()

    return {
        "success": True,
        "collection_name": collection.name,
        "new_parts_count": new_items_added,
        "reconciled_count": reconciled_count,
        "notifications_sent": notifications_created
    }


async def sync_all_active_collections():
    """Background task to sync all followed collections."""
    try:
        async with AsyncSessionLocal() as db:
            # Find collections with active followers
            stmt = select(Collection).join(UserCollection).distinct()
            res = await db.execute(stmt)
            active_collections = res.scalars().all()

            logger.info(f"Starting background sync for {len(active_collections)} followed collections...")
            for col in active_collections:
                try:
                    await sync_collection_from_tmdb(db, col.tmdb_id)
                except Exception as ex:
                    logger.error(f"Failed to sync collection {col.tmdb_id} ({col.name}): {ex}")

            logger.info("Background collections sync finished.")
    except Exception as e:
        logger.error(f"Error in sync_all_active_collections loop: {e}")


async def run_collection_sync_loop(interval_hours: int = 24):
    """Periodic loop running every interval_hours."""
    while True:
        try:
            await asyncio.sleep(interval_hours * 3600)
            await sync_all_active_collections()
        except asyncio.CancelledError:
            logger.info("Collection sync loop cancelled.")
            break
        except Exception as e:
            logger.error(f"Unexpected error in run_collection_sync_loop: {e}")
            await asyncio.sleep(60)


# ==============================================================================
# COLEÇÕES: ALERTA E SUGESTÕES EM SEGUNDO PLANO
# ==============================================================================

_user_suggestion_debounce_tasks: Dict[str, asyncio.Task] = {}
_user_last_scanned: Dict[str, float] = {}


async def generate_user_collection_suggestions(db: AsyncSession, user_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Identifica de forma inteligente e em segundo plano franquias/coleções das quais
    o usuário adicionou >= 1 filme e que possuem outros filmes ainda não adicionados
    (total_movies > movies_in_list ou total_movies >= 2), mas que ele ainda NÃO está seguindo ativamente.
    
    Salva/atualiza em 'user_collection_suggestions' para que o endpoint responda
    em tempo recorde (< 5ms) sem nenhuma latência externa.
    """
    user_uuid = uuid.UUID(str(user_id))

    # 1. Obter tmdb_ids das coleções que o usuário já segue
    followed_stmt = (
        select(Collection.tmdb_id)
        .join(UserCollection, UserCollection.collection_id == Collection.id)
        .where(UserCollection.user_id == user_uuid)
    )
    res_followed = await db.execute(followed_stmt)
    followed_col_tmdb_ids = set(res_followed.scalars().all())

    # 2. Obter filmes na lista do usuário
    uli_stmt = (
        select(UserListItem)
        .where(UserListItem.user_id == user_uuid)
        .options(selectinload(UserListItem.media))
    )
    res_uli = await db.execute(uli_stmt)
    user_movies = [ui for ui in res_uli.scalars().all() if ui.media and ui.media.media_type == "movie"]

    if not user_movies:
        await db.execute(
            delete(UserCollectionSuggestion).where(UserCollectionSuggestion.user_id == user_uuid)
        )
        await db.commit()
        return []

    # 3. Enriquecer filmes que ainda não possuem collection_tmdb_id mapeado
    unassigned_movies = [ui for ui in user_movies if ui.media.collection_tmdb_id is None]
    if unassigned_movies:
        # A. Checar primeiro na tabela collection_items existente no banco (0ms de rede)
        unassigned_tmdb_ids = [ui.media.tmdb_id for ui in unassigned_movies]
        ci_res = await db.execute(
            select(CollectionItem)
            .where(CollectionItem.tmdb_id.in_(unassigned_tmdb_ids))
            .options(selectinload(CollectionItem.collection))
        )
        ci_map = {ci.tmdb_id: ci for ci in ci_res.scalars().all()}

        for ui in unassigned_movies:
            if ui.media.tmdb_id in ci_map:
                ci = ci_map[ui.media.tmdb_id]
                ui.media.collection_tmdb_id = ci.collection.tmdb_id
                ui.media.collection_name = ci.collection.name
                db.add(ui.media)

        await db.commit()

        # B. Para os que ainda não foram identificados, buscar no cache/TMDB
        still_unassigned = [ui for ui in unassigned_movies if ui.media.collection_tmdb_id is None]
        if still_unassigned:
            from app.details.services import fetch_movie_details
            for ui in still_unassigned:
                try:
                    tmdb_data = await fetch_movie_details(ui.media.tmdb_id)
                    if tmdb_data and getattr(tmdb_data, "belongs_to_collection", None):
                        ui.media.collection_tmdb_id = tmdb_data.belongs_to_collection.id
                        ui.media.collection_name = tmdb_data.belongs_to_collection.name
                    else:
                        # 0 indica que o filme foi checado e é obra independente (sem coleção)
                        ui.media.collection_tmdb_id = 0
                    db.add(ui.media)
                except Exception as ex:
                    logger.debug(f"Could not fetch movie collection for tmdb_id={ui.media.tmdb_id}: {ex}")

            await db.commit()

    # 4. Agrupar os filmes do usuário por collection_tmdb_id (mapeia tmdb_id -> título para evitar contagem duplicada)
    from collections import defaultdict
    col_to_movies: Dict[int, Dict[int, str]] = defaultdict(dict)
    for ui in user_movies:
        col_id = ui.media.collection_tmdb_id
        if col_id and col_id > 0 and col_id not in followed_col_tmdb_ids:
            col_to_movies[col_id][ui.media.tmdb_id] = ui.media.title

    # 5. Filtrar e gerar sugestões
    candidate_col_tmdb_ids = set()

    for col_tmdb_id, movies_map in col_to_movies.items():
        movie_titles = list(movies_map.values())
        movies_in_list = len(movie_titles)

        # Obter dados da coleção
        col_res = await db.execute(select(Collection).where(Collection.tmdb_id == col_tmdb_id))
        col = col_res.scalars().first()

        name = None
        overview = None
        poster_path = None
        backdrop_path = None
        total_movies = 0

        if col:
            name = col.name
            overview = col.overview
            poster_path = col.poster_path
            backdrop_path = col.backdrop_path
            total_movies = int(col.parts_count or 0)
        else:
            try:
                tmdb_col = await fetch_collection_details(col_tmdb_id)
                name = tmdb_col.name
                overview = tmdb_col.overview
                poster_path = tmdb_col.poster_path
                backdrop_path = tmdb_col.backdrop_path
                total_movies = len(tmdb_col.parts) if tmdb_col.parts else 0
            except Exception as e:
                logger.warning(f"Failed to fetch collection details for {col_tmdb_id}: {e}")
                continue

        # Regra de sugestão (Opção 2 - Sugerir com >= 1 filme):
        # Qualquer filme adicionado (movies_in_list >= 1) que pertença a uma coleção oficial
        # com outros filmes pendentes (total_movies > movies_in_list ou total_movies >= 2 ou total_movies == 0)
        # e que o usuário ainda não siga.
        is_candidate = (movies_in_list >= 1) and (
            total_movies > movies_in_list or total_movies >= 2 or total_movies == 0
        )

        if not is_candidate:
            continue

        candidate_col_tmdb_ids.add(col_tmdb_id)

        # Checar se já existe sugestão registrada para este usuário
        sugg_res = await db.execute(
            select(UserCollectionSuggestion).where(
                UserCollectionSuggestion.user_id == user_uuid,
                UserCollectionSuggestion.collection_tmdb_id == col_tmdb_id
            )
        )
        existing_sugg = sugg_res.scalars().first()

        if existing_sugg:
            existing_sugg.name = name or existing_sugg.name
            existing_sugg.overview = overview or existing_sugg.overview
            existing_sugg.poster_path = poster_path or existing_sugg.poster_path
            existing_sugg.backdrop_path = backdrop_path or existing_sugg.backdrop_path
            existing_sugg.total_movies = total_movies
            existing_sugg.movies_in_list = movies_in_list
            existing_sugg.matched_movie_titles = movie_titles
            db.add(existing_sugg)
        else:
            new_sugg = UserCollectionSuggestion(
                user_id=user_uuid,
                collection_tmdb_id=col_tmdb_id,
                name=name or f"Coleção #{col_tmdb_id}",
                overview=overview,
                poster_path=poster_path,
                backdrop_path=backdrop_path,
                total_movies=total_movies,
                movies_in_list=movies_in_list,
                matched_movie_titles=movie_titles,
                is_dismissed=False
            )
            db.add(new_sugg)

    # 6. Remover sugestões que agora pertencem a coleções seguidas ou que não são mais candidatas
    all_user_suggs_res = await db.execute(
        select(UserCollectionSuggestion).where(UserCollectionSuggestion.user_id == user_uuid)
    )
    all_user_suggs = all_user_suggs_res.scalars().all()
    for sugg in all_user_suggs:
        if sugg.collection_tmdb_id in followed_col_tmdb_ids or sugg.collection_tmdb_id not in candidate_col_tmdb_ids:
            await db.delete(sugg)

    await db.commit()

    return await get_user_collection_suggestions(db, str(user_id), limit=limit)


async def get_user_collection_suggestions(db: AsyncSession, user_id: str, limit: Optional[int] = 5) -> List[Dict[str, Any]]:
    """
    Retorna instantaneamente (< 5ms) todas as sugestões ativas e não dispensadas do usuário,
    com limite configurável de 3 a 5 sugestões (padrão 5) ordenadas por relevância e data.
    """
    user_uuid = uuid.UUID(str(user_id))
    stmt = (
        select(UserCollectionSuggestion)
        .where(
            UserCollectionSuggestion.user_id == user_uuid,
            UserCollectionSuggestion.is_dismissed == False
        )
        .order_by(
            UserCollectionSuggestion.movies_in_list.desc(),
            UserCollectionSuggestion.created_at.desc()
        )
    )
    if limit is not None and limit > 0:
        stmt = stmt.limit(limit)

    res = await db.execute(stmt)
    records = res.scalars().all()

    return [
        {
            "id": r.id,
            "tmdb_id": r.collection_tmdb_id,
            "name": r.name,
            "overview": r.overview,
            "poster_path": r.poster_path,
            "backdrop_path": r.backdrop_path,
            "total_movies": r.total_movies,
            "movies_in_list": r.movies_in_list,
            "matched_movie_titles": r.matched_movie_titles or [],
            "created_at": r.created_at
        }
        for r in records
    ]


async def dismiss_collection_suggestion(db: AsyncSession, user_id: str, collection_tmdb_id: int) -> bool:
    """Dispensar sugestão de coleção para que não reapareça para o usuário."""
    user_uuid = uuid.UUID(str(user_id))
    stmt = (
        select(UserCollectionSuggestion)
        .where(
            UserCollectionSuggestion.user_id == user_uuid,
            UserCollectionSuggestion.collection_tmdb_id == collection_tmdb_id
        )
    )
    res = await db.execute(stmt)
    sugg = res.scalars().first()
    if not sugg:
        return False
    sugg.is_dismissed = True
    await db.commit()
    return True


def schedule_collection_suggestions_scan(user_id: str, debounce_seconds: int = 4) -> None:
    """Agenda a verificação de sugestões em segundo plano com debounce inteligente."""
    user_id_str = str(user_id)
    if user_id_str in _user_suggestion_debounce_tasks:
        task = _user_suggestion_debounce_tasks.pop(user_id_str)
        if not task.done():
            task.cancel()

    async def _runner():
        try:
            if debounce_seconds > 0:
                await asyncio.sleep(debounce_seconds)
            logger.info(f"[COLLECTION-SUGGESTIONS] Executando scan em background para usuário {user_id_str}...")
            async with AsyncSessionLocal() as session:
                await generate_user_collection_suggestions(session, user_id_str)
            _user_last_scanned[user_id_str] = time.time()
            logger.info(f"[COLLECTION-SUGGESTIONS] Scan finalizado para usuário {user_id_str}.")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"[COLLECTION-SUGGESTIONS] Erro no scan para {user_id_str}: {e}")
        finally:
            if _user_suggestion_debounce_tasks.get(user_id_str) is asyncio.current_task():
                _user_suggestion_debounce_tasks.pop(user_id_str, None)

    _user_suggestion_debounce_tasks[user_id_str] = asyncio.create_task(_runner())


async def scan_all_users_collection_suggestions():
    """Varredura periódica de todos os usuários cadastrados."""
    try:
        from app.users.models import User
        async with AsyncSessionLocal() as db:
            users_res = await db.execute(select(User.id))
            user_ids = [str(uid) for uid in users_res.scalars().all()]

        logger.info(f"[COLLECTION-SUGGESTIONS] Iniciando varredura para {len(user_ids)} usuários...")
        for uid in user_ids:
            try:
                async with AsyncSessionLocal() as session:
                    await generate_user_collection_suggestions(session, uid)
            except Exception as ex:
                logger.error(f"[COLLECTION-SUGGESTIONS] Erro ao escanear coleções para {uid}: {ex}")
        logger.info("[COLLECTION-SUGGESTIONS] Varredura periódica concluída com sucesso.")
    except Exception as e:
        logger.error(f"[COLLECTION-SUGGESTIONS] Falha na rotina scan_all_users_collection_suggestions: {e}")


async def run_collection_suggestions_loop(interval_hours: int = 24):
    """Loop diário em background para manter sugestões de coleções atualizadas."""
    while True:
        try:
            await asyncio.sleep(interval_hours * 3600)
            await scan_all_users_collection_suggestions()
        except asyncio.CancelledError:
            logger.info("Loop de sugestões de coleções cancelado.")
            break
        except Exception as e:
            logger.error(f"Erro inesperado no run_collection_suggestions_loop: {e}")
            await asyncio.sleep(60)

