from fastapi import APIRouter, Depends, Query, Response
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import joinedload, load_only
from datetime import datetime, date, timedelta
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

    class Config:
        from_attributes = True

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
    response: Response,
    month: Optional[str] = Query(None, description="Format YYYY-MM. If not provided, gets all upcoming releases"),
    from_date: Optional[str] = Query(None, description="Format YYYY-MM-DD. Start date for releases (defaults to current date)"),
    limit: Optional[int] = Query(None, ge=1, le=200, description="Max releases to return (default 100 for agenda)"),
    offset: Optional[int] = Query(0, ge=0, description="Offset for pagination"),
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    # Cache HTTP privado no navegador do usuário por 3 minutos, com stale-while-revalidate de 5 minutos
    response.headers["Cache-Control"] = "private, max-age=180, stale-while-revalidate=300"
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
    elif from_date:
        try:
            # Data inicial informada pelo cliente no fuso horário do usuário (00:00:00 UTC)
            start_date = datetime.strptime(from_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            from fastapi import HTTPException
            raise HTTPException(status_code=400, detail="Invalid from_date format. Use YYYY-MM-DD.")
    else:
        # Fallback padrão: início do dia de hoje no fuso horário oficial do servidor (America/Sao_Paulo)
        from app.core.utils import get_server_today
        today_server = get_server_today()
        start_date = datetime(today_server.year, today_server.month, today_server.day, tzinfo=timezone.utc)

    # Consulta única otimizada com JOIN indexado direto entre MediaRelease e UserListItem
    # e projeção restrita aos campos necessários de Media (eliminando colunas pesadas de embedding e JSON).
    stmt_releases = (
        select(MediaRelease)
        .options(
            joinedload(MediaRelease.media).load_only(
                Media.id,
                Media.tmdb_id,
                Media.title,
                Media.poster_path,
                Media.media_type
            )
        )
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
    elif start_date:
        stmt_releases = stmt_releases.where(
            MediaRelease.release_date >= start_date
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
