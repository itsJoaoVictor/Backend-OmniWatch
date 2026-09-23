import asyncio
from datetime import datetime, timezone, timedelta
import httpx
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import delete
from typing import List

from app.media.models import Media, MediaRelease
from app.tracking.models import UserListItem
from app.core.config import settings
from app.core.database import AsyncSessionLocal

async def sync_single_media_release(db: AsyncSession, media_id: str):
    """Fetches and updates release/episode data for a single media item."""
    stmt_media = select(Media).where(Media.id == media_id)
    result = await db.execute(stmt_media)
    media = result.scalars().first()
    
    if not media:
        return
        
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {settings.TMDB_API_KEY}"
    }

    async with httpx.AsyncClient() as client:
        try:
            if media.media_type == "movie":
                url = f"{settings.TMDB_BASE_URL}/movie/{media.tmdb_id}"
                response = await client.get(url, headers=headers, params={"language": "pt-BR"}, timeout=10.0)
                if response.status_code == 200:
                    data = response.json()
                    rel_date_str = data.get("release_date")
                    if rel_date_str:
                        try:
                            r_date = datetime.strptime(rel_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                            await db.execute(delete(MediaRelease).where(MediaRelease.media_id == media.id))
                            
                            new_release = MediaRelease(
                                media_id=media.id,
                                title=media.title,
                                description=f"Lancamento do filme {media.title}",
                                release_date=r_date,
                            )
                            db.add(new_release)
                        except ValueError:
                            pass
            elif media.media_type == "tv":
                url = f"{settings.TMDB_BASE_URL}/tv/{media.tmdb_id}"
                params = {"language": "pt-BR"}
                response = await client.get(url, headers=headers, params=params, timeout=10.0)
                if response.status_code == 200:
                    data = response.json()
                    next_episode = data.get("next_episode_to_air")
                    
                    await db.execute(delete(MediaRelease).where(MediaRelease.media_id == media.id))

                    if next_episode and next_episode.get("air_date"):
                        r_date = datetime.strptime(next_episode.get("air_date"), "%Y-%m-%d").replace(tzinfo=timezone.utc)
                        ep_title = next_episode.get("name", "Novo Episodio")
                        ep_desc = next_episode.get("overview", "")
                        s_num = next_episode.get("season_number")
                        e_num = next_episode.get("episode_number")
                        
                        new_release = MediaRelease(
                            media_id=media.id,
                            title=f"T{s_num}E{e_num} - {ep_title}",
                            description=ep_desc,
                            release_date=r_date,
                            season_number=s_num,
                            episode_number=e_num
                        )
                        db.add(new_release)
        except Exception as e:
            print(f"Error syncing media {media.id}: {e}")
            
    await db.commit()

async def revert_completed_shows(db: AsyncSession):
    """Auto-revert completed TV shows to watching if a new episode has aired."""
    now_utc = datetime.now(timezone.utc)
    stmt_revert = select(UserListItem).join(MediaRelease, UserListItem.media_id == MediaRelease.media_id).where(
        UserListItem.status == "completed",
        MediaRelease.release_date <= now_utc
    )
    result_revert = await db.execute(stmt_revert)
    items_to_revert = result_revert.scalars().all()
    
    if items_to_revert:
        for item in items_to_revert:
            item.status = "watching"
        await db.commit()

async def fetch_changed_tmdb_ids(media_type: str, start_date: str, end_date: str) -> List[int]:
    """Fetches all changed TMDB IDs for a given media type within a date range."""
    changed_ids = set()
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {settings.TMDB_API_KEY}"
    }
    
    url = f"{settings.TMDB_BASE_URL}/{media_type}/changes"
    
    async with httpx.AsyncClient() as client:
        page = 1
        total_pages = 1
        
        while page <= total_pages:
            params = {
                "start_date": start_date,
                "end_date": end_date,
                "page": page
            }
            try:
                response = await client.get(url, headers=headers, params=params, timeout=10.0)
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])
                    for item in results:
                        if "id" in item:
                            changed_ids.add(item["id"])
                    
                    total_pages = data.get("total_pages", 1)
                    page += 1
                else:
                    break
            except Exception as e:
                print(f"Error fetching {media_type} changes page {page}: {e}")
                break
                
    return list(changed_ids)

async def run_calendar_sync_loop(interval_hours: int = 24):
    """Background task to sync calendar data using the TMDB Changes API."""
    print("Calendar sync loop started.")
    while True:
        try:
            print("Running scheduled calendar sync...")
            now_utc = datetime.now(timezone.utc)
            start_date = (now_utc - timedelta(days=1)).strftime("%Y-%m-%d")
            end_date = now_utc.strftime("%Y-%m-%d")
            
            # 1. Fetch changed TMDB IDs
            changed_movies = await fetch_changed_tmdb_ids("movie", start_date, end_date)
            changed_tv = await fetch_changed_tmdb_ids("tv", start_date, end_date)
            
            all_changed_tmdb_ids = set(changed_movies + changed_tv)
            
            if all_changed_tmdb_ids:
                async with AsyncSessionLocal() as db:
                    # 2. Find tracked media in our DB that overlap with the changed IDs
                    stmt_tracked_ids = select(UserListItem.media_id).where(
                        UserListItem.status.in_(["plan_to_watch", "completed", "watching"])
                    ).distinct()
                    res_tracked = await db.execute(stmt_tracked_ids)
                    tracked_media_ids = res_tracked.scalars().all()
                    
                    if tracked_media_ids:
                        # Chunk the tracked media IDs or just use IN clause if not millions
                        stmt_media_to_update = select(Media.id).where(
                            Media.id.in_(tracked_media_ids),
                            Media.tmdb_id.in_(all_changed_tmdb_ids)
                        )
                        res_media = await db.execute(stmt_media_to_update)
                        media_ids_to_update = res_media.scalars().all()
                        
                        # 3. Update each matching media
                        for m_id in media_ids_to_update:
                            await sync_single_media_release(db, m_id)
                    
                    # 4. Revert completed shows if new episodes aired
                    await revert_completed_shows(db)
                    
            print("Calendar sync complete.")
        except Exception as e:
            print(f"Error in calendar sync loop: {e}")
        
        await asyncio.sleep(interval_hours * 3600)
