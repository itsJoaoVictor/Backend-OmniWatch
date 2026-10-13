import uuid
from typing import Optional, Set
from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.core.database import get_db
from app.core.security import decode_token
from app.users.models import User
from app.media.models import Media
from app.tracking.models import UserListItem
from app.recommendation.service import get_active_dismissed_tmdb_ids
from app.person.schemas import PersonDetailsResponse, PersonDiscoverResponse
from app.person.services import fetch_person_details, compute_person_discover_suggestions

router = APIRouter()


def _extract_optional_user_id(request: Request) -> Optional[str]:
    token = request.cookies.get("access_token")
    if not token:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
    if not token:
        return None
    payload = decode_token(token)
    if not payload or "sub" not in payload or payload.get("type") != "access":
        return None
    return payload["sub"]


async def _get_user_watched_or_dismissed_ids(db: AsyncSession, user_id: str) -> Set[int]:
    try:
        target_uuid = user_id if isinstance(user_id, uuid.UUID) else uuid.UUID(str(user_id))
        stmt = (
            select(Media.tmdb_id)
            .join(UserListItem, UserListItem.media_id == Media.id)
            .where(
                UserListItem.user_id == target_uuid,
                UserListItem.status.in_(["completed", "watching", "dropped"]),
            )
        )
        result = await db.execute(stmt)
        watched = set(result.scalars().all())
        dismissed = await get_active_dismissed_tmdb_ids(db, user_id)
        return watched.union(dismissed)
    except Exception:
        return set()


@router.get("/{person_id}", response_model=PersonDetailsResponse)
async def get_person_details(person_id: int):
    """
    Get details for a person (actor, director, etc.), including their credits.
    """
    return await fetch_person_details(person_id)


@router.get("/{person_id}/discover", response_model=PersonDiscoverResponse)
async def get_person_discover_suggestions(
    person_id: int,
    request: Request,
    limit: int = Query(12, ge=1, le=30),
    db: AsyncSession = Depends(get_db),
):
    """
    Retorna sugestões de obras de destaque da pessoa que o usuário ainda não assistiu,
    ordenadas pelo Score de Destaque Personalizado (40% Protagonismo + 35% Qualidade + 25% Afinidade).
    """
    person = await fetch_person_details(person_id)
    user_id = _extract_optional_user_id(request)

    user_vector = None
    taste_clusters = None
    excluded_ids: Set[int] = set()

    if user_id:
        try:
            target_uuid = user_id if isinstance(user_id, uuid.UUID) else uuid.UUID(str(user_id))
            res = await db.execute(select(User).where(User.id == target_uuid))
            user = res.scalars().first()
            if user:
                if user.feature_vector:
                    user_vector = user.feature_vector
                if user.taste_clusters:
                    taste_clusters = user.taste_clusters
            excluded_ids = await _get_user_watched_or_dismissed_ids(db, user_id)
        except Exception:
            pass

    return compute_person_discover_suggestions(
        person=person,
        user_vector=user_vector,
        taste_clusters=taste_clusters,
        exclude_tmdb_ids=excluded_ids,
        limit=limit,
    )


