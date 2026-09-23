import asyncio
from datetime import datetime, timezone
import httpx
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import delete

from app.media.models import Media, MediaRelease
from app.tracking.models import UserListItem
from app.core.config import settings

async def sync_media_releases(db: AsyncSession):
    stmt = select(UserListItem.media_id).where(
        UserListItem.status.in_(["plan_to_watch", "completed", "watching"])
    ).distinct()
    result = await db.execute(stmt)
    tracked_media_ids = result.scalars().all()

    if not tracked_media_ids:
        return

    stmt_media = select(Media).where(Media.id.in_(tracked_media_ids))
    media_result = await db.execute(stmt_media)
    medias = media_result.scalars().all()

    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {settings.TMDB_API_KEY}"
    }

    async with httpx.AsyncClient() as client:
        for media in medias:
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

    # Auto-revert completed TV shows to watching if a new episode has aired
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
