from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Dict, Any

from app.core.database import get_db
from app.auth.dependencies import get_current_user_id
from app.media_collections.schemas import (
    CollectionFollowResponse,
    CollectionStatusResponse,
    UserCollectionDetailResponse
)
from app.media_collections.services import (
    follow_collection,
    unfollow_collection,
    get_collection_status,
    get_user_collections,
    sync_collection_from_tmdb
)

router = APIRouter(prefix="/collections", tags=["collections"])

@router.get("/following", response_model=List[UserCollectionDetailResponse])
async def list_followed_collections(
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    """List all collections followed by current user with watching progress."""
    return await get_user_collections(db, user_id)

@router.get("/{tmdb_id}/status", response_model=CollectionStatusResponse)
async def check_collection_status(
    tmdb_id: int,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    """Check if the user is following a specific collection."""
    status_data = await get_collection_status(db, user_id, tmdb_id)
    return CollectionStatusResponse(**status_data)

@router.post("/{tmdb_id}/follow", response_model=CollectionFollowResponse)
async def follow_user_collection(
    tmdb_id: int,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    """Follow a collection and bulk-add all unwatched movies to user's list as plan_to_watch."""
    try:
        result = await follow_collection(db, user_id, tmdb_id)
        return CollectionFollowResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao seguir coleção: {str(e)}"
        )

@router.delete("/{tmdb_id}/follow")
async def unfollow_user_collection(
    tmdb_id: int,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    """Unfollow a collection. Does not delete movies already in user list."""
    success = await unfollow_collection(db, user_id, tmdb_id)
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Coleção não encontrada ou não seguida.")
    return {"success": True, "message": "Você deixou de seguir esta coleção com sucesso."}

@router.post("/{tmdb_id}/sync")
async def trigger_collection_sync(
    tmdb_id: int,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    """Trigger an on-demand sync of a collection from TMDB to check for new movies."""
    result = await sync_collection_from_tmdb(db, tmdb_id)
    return result
