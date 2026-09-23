import asyncio
import uuid
import logging
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload
from sqlalchemy import delete

from app.core.database import AsyncSessionLocal
from app.media_collections.models import Collection, CollectionItem, UserCollection
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


async def sync_collection_from_tmdb(db: AsyncSession, tmdb_collection_id: int) -> Dict[str, Any]:
    """Sync a collection from TMDB. Detects new movies, adds to followers lists, and generates notifications."""
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

    await db.commit()

    return {
        "success": True,
        "collection_name": collection.name,
        "new_parts_count": new_items_added,
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
