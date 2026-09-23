from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List

from app.core.database import get_db
from app.auth.dependencies import get_current_user_id
from app.tracking.schemas import (
    UserListItemCreate, UserListItemUpdate, UserListItemResponse,
    UserEpisodeProgressCreate, UserEpisodeProgressResponse, UserEpisodeRatingUpdate
)
from app.tracking.services import (
    get_user_list, add_to_list, update_list_item, remove_from_list, add_episode_progress
)

router = APIRouter()

@router.get("/my-list", response_model=List[UserListItemResponse])
async def get_my_list(
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    items = await get_user_list(db, user_id)
    return items

from fastapi import BackgroundTasks

@router.post("/my-list", response_model=UserListItemResponse, status_code=status.HTTP_201_CREATED)
async def add_item_to_list(
    item: UserListItemCreate,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    new_item = await add_to_list(db, user_id, item)
    
    # Trigger background sync for this media
    from app.calendar.sync import sync_media_releases
    # AsyncSession doesn't work well across background tasks, so we do a quick hack
    from app.core.database import AsyncSessionLocal

    async def run_sync():
        async with AsyncSessionLocal() as session:
            await sync_media_releases(session)
            
    background_tasks.add_task(run_sync)
    
    return new_item

@router.patch("/my-list/{item_id}", response_model=UserListItemResponse)
async def update_item(
    item_id: str,
    update_data: UserListItemUpdate,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    updated = await update_list_item(db, user_id, item_id, update_data)
    if not updated:
        raise HTTPException(status_code=404, detail="Item não encontrado")
    return updated

@router.delete("/my-list/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_item(
    item_id: str,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    success = await remove_from_list(db, user_id, item_id)
    if not success:
        raise HTTPException(status_code=404, detail="Item não encontrado")
    return None

@router.post("/my-list/{item_id}/progress", response_model=UserEpisodeProgressResponse, status_code=status.HTTP_201_CREATED)
async def mark_episode_watched(
    item_id: str,
    progress: UserEpisodeProgressCreate,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    new_progress = await add_episode_progress(db, user_id, item_id, progress)
    if not new_progress:
        raise HTTPException(status_code=404, detail="Item não encontrado ou você não tem permissão")
    return new_progress

@router.post("/my-list/{item_id}/progress/bulk", status_code=status.HTTP_200_OK)
async def bulk_mark_episodes_watched(
    item_id: str,
    progress: UserEpisodeProgressCreate,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    from app.tracking.services import bulk_mark_episodes
    success = await bulk_mark_episodes(db, user_id, item_id, progress)
    if not success:
        raise HTTPException(status_code=404, detail="Item não encontrado ou falha ao marcar em lote")
    return {"status": "ok"}

@router.get("/my-list/{item_id}/progress", response_model=List[UserEpisodeProgressResponse])
async def get_item_episode_progress(
    item_id: str,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    from app.tracking.services import get_episode_progress
    progress = await get_episode_progress(db, user_id, item_id)
    return progress

@router.delete("/my-list/{item_id}/progress/{season_number}/{episode_number}", status_code=status.HTTP_204_NO_CONTENT)
async def unmark_episode_watched(
    item_id: str,
    season_number: int,
    episode_number: int,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    from app.tracking.services import remove_episode_progress
    success = await remove_episode_progress(db, user_id, item_id, season_number, episode_number)
    if not success:
        raise HTTPException(status_code=404, detail="Progresso não encontrado ou permissão negada")
    return None

@router.patch("/my-list/{item_id}/progress/{season_number}/{episode_number}/rating", response_model=UserEpisodeProgressResponse)
async def rate_episode(
    item_id: str,
    season_number: int,
    episode_number: int,
    rating_data: UserEpisodeRatingUpdate,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    from app.tracking.services import update_episode_rating
    prog = await update_episode_rating(db, user_id, item_id, season_number, episode_number, rating_data.rating)
    if not prog:
        raise HTTPException(status_code=404, detail="Item ou episódio não encontrado")
    return prog

@router.get("/statistics")
async def get_statistics(
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    from app.tracking.services import get_user_statistics
    stats = await get_user_statistics(db, user_id)
    return stats
