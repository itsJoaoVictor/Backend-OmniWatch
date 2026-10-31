from fastapi import APIRouter, Depends, Query, status, HTTPException, Response
from sqlalchemy import func
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
import uuid

from app.core.database import get_db
from app.auth.dependencies import get_current_user_id
from app.friends.models import Friendship
from app.friends.schemas import (
    UserSearchResult,
    FriendUserResponse,
    FriendRequestsResponse,
    SendFriendRequest,
    FriendFeedResponse
)
from app.friends.services import (
    search_users,
    send_friend_request,
    get_friends_list,
    get_friend_requests,
    accept_friend_request,
    reject_friend_request,
    cancel_friend_request,
    remove_friend,
    get_friend_feed
)

router = APIRouter(prefix="/friends", tags=["friends"])


def parse_uuid(uid_str: str) -> uuid.UUID:
    try:
        return uuid.UUID(uid_str) if isinstance(uid_str, str) else uid_str
    except ValueError:
        raise HTTPException(status_code=400, detail="ID de usuário inválido.")


@router.get("/pending-count")
async def get_pending_count(
    response: Response,
    current_user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    response.headers["Cache-Control"] = "private, max-age=30, stale-while-revalidate=60"
    u_uuid = parse_uuid(current_user_id)
    res = await db.execute(
        select(func.count(Friendship.id)).where(
            Friendship.addressee_id == u_uuid,
            Friendship.status == "pending"
        )
    )
    count = res.scalar() or 0
    return {"count": count}


@router.get("/feed", response_model=FriendFeedResponse)
async def get_feed_endpoint(
    response: Response,
    page: int = Query(1, ge=1, description="Número da página (legado)"),
    limit: int = Query(20, ge=1, le=50, description="Itens por página"),
    cursor: Optional[str] = Query(None, description="Cursor temporal ISO do último item (before_timestamp)"),
    media_type: Optional[str] = Query(None, description="Filtrar por 'movie' ou 'tv'"),
    friend: Optional[str] = Query(None, description="Filtrar por username ou ID de um amigo"),
    current_user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    response.headers["Cache-Control"] = "private, max-age=20, stale-while-revalidate=40"
    u_uuid = parse_uuid(current_user_id)
    return await get_friend_feed(db, u_uuid, page=page, limit=limit, media_type=media_type, cursor=cursor, friend=friend)

get_feed = get_feed_endpoint


@router.get("", response_model=List[FriendUserResponse])
async def list_friends(
    current_user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    u_uuid = parse_uuid(current_user_id)
    return await get_friends_list(db, u_uuid)

@router.get("/requests", response_model=FriendRequestsResponse)
async def list_requests(
    current_user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    u_uuid = parse_uuid(current_user_id)
    return await get_friend_requests(db, u_uuid)

@router.get("/search", response_model=List[UserSearchResult])
async def search_users_endpoint(
    q: str = Query(..., min_length=1, description="Termo de busca por username ou nome"),
    limit: int = Query(20, ge=1, le=50),
    current_user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    u_uuid = parse_uuid(current_user_id)
    return await search_users(db, u_uuid, q, limit=limit)

@router.post("/requests", status_code=status.HTTP_201_CREATED)
async def send_request(
    body: SendFriendRequest,
    current_user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    u_uuid = parse_uuid(current_user_id)
    friendship = await send_friend_request(
        db,
        current_user_id=u_uuid,
        addressee_id=body.addressee_id,
        username=body.username
    )
    return {"message": "Solicitação enviada com sucesso.", "friendship_id": friendship.id, "status": friendship.status}

@router.post("/requests/{friendship_id}/accept")
async def accept_request(
    friendship_id: uuid.UUID,
    current_user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    u_uuid = parse_uuid(current_user_id)
    friendship = await accept_friend_request(db, friendship_id, u_uuid)
    return {"message": "Solicitação aceita com sucesso.", "friendship_id": friendship.id}

@router.post("/requests/{friendship_id}/reject")
async def reject_request(
    friendship_id: uuid.UUID,
    current_user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    u_uuid = parse_uuid(current_user_id)
    return await reject_friend_request(db, friendship_id, u_uuid)

@router.delete("/requests/{friendship_id}")
async def cancel_request(
    friendship_id: uuid.UUID,
    current_user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    u_uuid = parse_uuid(current_user_id)
    return await cancel_friend_request(db, friendship_id, u_uuid)

@router.delete("/{friendship_id}")
async def delete_friend(
    friendship_id: uuid.UUID,
    current_user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    u_uuid = parse_uuid(current_user_id)
    return await remove_friend(db, friendship_id, u_uuid)
