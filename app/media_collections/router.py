from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Dict, Any

from app.core.database import get_db
from app.auth.dependencies import get_current_user_id
from app.media_collections.schemas import (
    CollectionFollowResponse,
    CollectionStatusResponse,
    UserCollectionDetailResponse,
    CollectionSuggestionResponse
)
from app.media_collections.services import (
    follow_collection,
    unfollow_collection,
    get_collection_status,
    get_user_collections,
    sync_collection_from_tmdb,
    get_user_collection_suggestions,
    dismiss_collection_suggestion,
    schedule_collection_suggestions_scan
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
    """Trigger an on-demand sync of a collection from TMDB to check for new movies and reconcile missing items."""
    result = await sync_collection_from_tmdb(db, tmdb_id, user_id=user_id)
    return result


@router.get("/suggestions", response_model=List[CollectionSuggestionResponse])
async def list_collection_suggestions(
    limit: int = Query(5, ge=1, le=50, description="Limite de sugestões de coleções a retornar"),
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    """
    Retorna instantaneamente (< 5ms) sugestões de coleções geradas em segundo plano.
    Limite configurável de 3 a 5 (padrão 5).
    Se o usuário ainda não tiver sido escaneado recentemente, agenda um scan em background com cooldown.
    """
    suggestions = await get_user_collection_suggestions(db, user_id, limit=limit)
    import time
    from app.media_collections.services import _user_last_scanned
    now = time.time()
    last_scanned = _user_last_scanned.get(str(user_id), 0)
    if not suggestions and (now - last_scanned > 300):
        schedule_collection_suggestions_scan(user_id, debounce_seconds=2)
    return suggestions


@router.post("/suggestions/{tmdb_id}/dismiss")
async def dismiss_user_collection_suggestion(
    tmdb_id: int,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    """Dispensar sugestão de coleção para o usuário atual."""
    success = await dismiss_collection_suggestion(db, user_id, tmdb_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Sugestão não encontrada ou já dispensada."
        )
    return {"success": True, "message": "Sugestão dispensada com sucesso."}


@router.post("/suggestions/scan")
async def trigger_collection_suggestions_scan(
    user_id: str = Depends(get_current_user_id)
):
    """Dispara explicitamente o cálculo de sugestões em segundo plano."""
    schedule_collection_suggestions_scan(user_id, debounce_seconds=0)
    return {"success": True, "message": "Verificação de coleções agendada em segundo plano."}

