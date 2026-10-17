from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
from uuid import UUID

from app.core.database import get_db
from app.auth.dependencies import get_current_user_id
from app.universes.schemas import (
    UniverseFollowResponse,
    UniverseStatusResponse,
    UserUniverseDetailResponse,
    UniverseSuggestionResponse,
    UniverseFollowRequest,
    ToggleIgnoreItemRequest,
    UpdateUniverseSettingsRequest,
)
from app.universes.services import (
    follow_universe,
    unfollow_universe,
    get_universe_status,
    get_user_universes,
    get_universe_details_public,
    sync_universe_from_sources,
    get_cached_user_universe_suggestions,
    dismiss_universe_suggestion,
    restore_universe_suggestion,
    get_dismissed_universe_suggestions,
    schedule_universe_suggestions_scan,
    toggle_universe_item_ignored,
    update_universe_settings,
)

router = APIRouter(prefix="/universes", tags=["universes"])


@router.get("/following", response_model=List[UserUniverseDetailResponse])
async def list_followed_universes(
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    """Lista todos os universos acompanhados pelo usuário logado com métricas de progresso e notas."""
    return await get_user_universes(db, UUID(str(user_id)))


@router.get("/suggestions", response_model=List[UniverseSuggestionResponse])
async def list_universe_suggestions(
    limit: int = Query(default=20, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    """Retorna as sugestões de universos descobertas em segundo plano com base na lista do usuário."""
    return await get_cached_user_universe_suggestions(db, UUID(str(user_id)), limit=limit)


@router.post("/suggestions/scan")
async def trigger_suggestions_scan(
    user_id: str = Depends(get_current_user_id),
):
    """Dispara uma varredura assíncrona manual de sugestões de universos para o usuário."""
    schedule_universe_suggestions_scan(str(user_id), debounce_seconds=0.0)
    return {"message": "Varredura de sugestões de universos agendada com sucesso em background."}


@router.post("/suggestions/{universe_key}/dismiss")
async def dismiss_suggestion(
    universe_key: str,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    """Descarta uma sugestão de universo para que não seja mais exibida ao usuário."""
    success = await dismiss_universe_suggestion(db, UUID(str(user_id)), universe_key)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Sugestão de universo não encontrada.",
        )
    return {"success": True, "message": "Sugestão dispensada com sucesso."}


@router.get("/suggestions/dismissed", response_model=List[UniverseSuggestionResponse])
async def list_dismissed_suggestions(
    limit: int = Query(default=50, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    """Retorna as sugestões de universos que foram dispensadas pelo usuário."""
    return await get_dismissed_universe_suggestions(db, UUID(str(user_id)), limit=limit)


@router.post("/suggestions/{universe_key}/restore")
async def restore_suggestion(
    universe_key: str,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    """Restaura uma sugestão de universo previamente dispensada."""
    success = await restore_universe_suggestion(db, UUID(str(user_id)), universe_key)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Sugestão de universo não encontrada.",
        )
    return {"success": True, "message": "Sugestão restaurada com sucesso."}


@router.get("/{universe_key}", response_model=UserUniverseDetailResponse)
async def get_universe_details(
    universe_key: str,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    """
    Retorna os detalhes completos de um universo, seu catálogo de mídias ordenado
    cronologicamente e o cruzamento com o perfil do usuário.
    """
    detail = await get_universe_details_public(db, UUID(str(user_id)), universe_key)
    if not detail:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Universo não encontrado ou não compatível com o formato multimídia.",
        )
    return detail


@router.get("/{universe_key}/status", response_model=UniverseStatusResponse)
async def check_universe_status(
    universe_key: str,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    """Verifica se o usuário segue um universo específico."""
    return await get_universe_status(db, UUID(str(user_id)), universe_key)


@router.post("/{universe_key}/follow", response_model=UniverseFollowResponse)
async def follow_user_universe(
    universe_key: str,
    payload: Optional[UniverseFollowRequest] = None,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    """Segue um universo e adiciona os filmes e séries selecionados à lista do usuário."""
    ignored_list = (
        [(x.tmdb_id, x.media_type) for x in payload.ignored_items]
        if payload and payload.ignored_items
        else []
    )
    auto_add = payload.auto_add if payload is not None else True
    result = await follow_universe(
        db,
        UUID(str(user_id)),
        universe_key,
        auto_add=auto_add,
        ignored_items=ignored_list,
    )
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Não foi possível localizar ou montar os dados deste universo.",
        )
    return result


@router.delete("/{universe_key}/follow")
async def unfollow_user_universe(
    universe_key: str,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    """Deixa de seguir um universo (mantendo as obras já adicionadas na lista)."""
    success = await unfollow_universe(db, UUID(str(user_id)), universe_key)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Universo não encontrado ou não seguido.",
        )
    return {"success": True, "message": "Você deixou de seguir este universo com sucesso."}


@router.post("/{universe_key}/items/{tmdb_id}/toggle-ignore")
async def toggle_item_ignored(
    universe_key: str,
    tmdb_id: int,
    payload: ToggleIgnoreItemRequest,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    """Alterna o status de ignorado de uma obra específica no universo."""
    success = await toggle_universe_item_ignored(
        db,
        UUID(str(user_id)),
        universe_key,
        tmdb_id,
        payload.media_type,
        payload.is_ignored,
    )
    return {"success": success, "tmdb_id": tmdb_id, "is_ignored": payload.is_ignored}


@router.patch("/{universe_key}/settings")
async def update_settings(
    universe_key: str,
    payload: UpdateUniverseSettingsRequest,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    """Atualiza configurações do universo como auto_add."""
    success = await update_universe_settings(
        db, UUID(str(user_id)), universe_key, payload.auto_add
    )
    return {"success": success, "auto_add": payload.auto_add}


@router.post("/{universe_key}/sync")
async def trigger_universe_sync(
    universe_key: str,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    """Sincroniza manualmente um universo para buscar novos filmes/séries e reconciliar itens pendentes."""
    universe = await sync_universe_from_sources(db, universe_key, user_id=UUID(str(user_id)))
    if not universe:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Universo não encontrado para sincronização.",
        )
    return {
        "success": True,
        "message": f"Universo '{universe.name}' sincronizado com sucesso!",
        "items_count": universe.items_count,
    }
