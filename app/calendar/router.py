from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import joinedload
from datetime import datetime, date
from typing import List, Optional
import calendar

from app.core.database import get_db
from app.auth.dependencies import get_current_user_id
from app.users.models import User
from app.media.models import MediaRelease, Media
from app.tracking.models import UserListItem
from pydantic import BaseModel
from pydantic import UUID4

router = APIRouter(prefix="/calendar", tags=["calendar"])

class MediaSimple(BaseModel):
    id: UUID4
    tmdb_id: int
    title: str
    poster_path: Optional[str] = None
    media_type: str

class ReleaseEventSchema(BaseModel):
    id: UUID4
    media_id: UUID4
    title: str
    description: Optional[str] = None
    release_date: datetime
    season_number: Optional[int] = None
    episode_number: Optional[int] = None
    media: MediaSimple

    class Config:
        from_attributes = True

@router.get("/my-releases", response_model=List[ReleaseEventSchema])
async def get_my_releases(
    month: Optional[str] = Query(None, description="Format YYYY-MM. If not provided, gets all upcoming releases"),
    limit: Optional[int] = Query(None, ge=1, le=200, description="Max releases to return (default 100 for agenda)"),
    offset: Optional[int] = Query(0, ge=0, description="Offset for pagination"),
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    from datetime import timezone
    
    start_date = None
    end_date = None
    
    if month:
        try:
            year, m = map(int, month.split("-"))
            _, last_day = calendar.monthrange(year, m)
            start_date = datetime(year, m, 1, tzinfo=timezone.utc)
            end_date = datetime(year, m, last_day, 23, 59, 59, tzinfo=timezone.utc)
        except ValueError:
            from fastapi import HTTPException
            raise HTTPException(status_code=400, detail="Invalid month format. Use YYYY-MM.")

    # Consulta única otimizada com JOIN indexado direto entre MediaRelease e UserListItem.
    # Elimina o duplo round-trip e a cláusula IN (...) com centenas de IDs em memória.
    stmt_releases = (
        select(MediaRelease)
        .options(joinedload(MediaRelease.media))
        .join(
            UserListItem,
            UserListItem.media_id == MediaRelease.media_id
        )
        .where(
            UserListItem.user_id == user_id,
            UserListItem.status.in_(["plan_to_watch", "completed", "watching", "upcoming"])
        )
    )
    
    if start_date and end_date:
        stmt_releases = stmt_releases.where(
            MediaRelease.release_date >= start_date,
            MediaRelease.release_date <= end_date
        )
    else:
        # Default: apenas lançamentos futuros
        stmt_releases = stmt_releases.where(
            MediaRelease.release_date >= datetime.now(timezone.utc)
        )
        
    stmt_releases = stmt_releases.order_by(MediaRelease.release_date.asc())

    # Paginação e limite seguro para evitar payloads de múltiplos megabytes
    if offset:
        stmt_releases = stmt_releases.offset(offset)
    if limit is not None:
        stmt_releases = stmt_releases.limit(limit)
    elif not month:
        # Limite padrão seguro para a agenda contínua
        stmt_releases = stmt_releases.limit(100)
        
    result = await db.execute(stmt_releases)
    releases = result.scalars().all()

    return releases
