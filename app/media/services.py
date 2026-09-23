from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.media.models import Media
from app.media.schemas import MediaCreate

async def get_media_by_tmdb_id(db: AsyncSession, tmdb_id: int):
    result = await db.execute(select(Media).where(Media.tmdb_id == tmdb_id))
    return result.scalars().first()

async def create_media(db: AsyncSession, media: MediaCreate):
    db_media = Media(**media.model_dump())
    db.add(db_media)
    await db.commit()
    await db.refresh(db_media)
    return db_media
