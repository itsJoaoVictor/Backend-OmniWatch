import uuid
from typing import List, Optional
from fastapi import APIRouter, Depends, Query, status, Response, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.auth.dependencies import get_current_user_id
from app.custom_lists import services
from app.custom_lists.schemas import (
    CustomListCreate,
    CustomListUpdate,
    CustomListItemCreate,
    CustomListItemsBulkCreate,
    CustomListItemUpdate,
    CustomListSummaryOut,
    CustomListDetailOut,
    CustomListItemOut,
    CustomListItemsPagedOut,
    ReorderItemsRequest,
    MediaListMembership
)

router = APIRouter(prefix="/custom-lists", tags=["custom-lists"])

@router.post("", response_model=CustomListSummaryOut, status_code=status.HTTP_201_CREATED)
async def create_list(
    data: CustomListCreate,
    user_id_str: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    user_id = uuid.UUID(user_id_str)
    return await services.create_custom_list(db, user_id, data)

@router.get("/my", response_model=List[CustomListSummaryOut])
async def get_my_lists(
    request: Request,
    response: Response,
    user_id_str: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    user_id = uuid.UUID(user_id_str)
    
    # Validação rápida de ETag em O(1) via índice
    etag = await services.get_my_custom_lists_etag(db, user_id)
    if_none_match = request.headers.get("if-none-match")

    if if_none_match and if_none_match == etag:
        return Response(
            status_code=status.HTTP_304_NOT_MODIFIED,
            headers={
                "ETag": etag,
                "Cache-Control": "private, max-age=60, stale-while-revalidate=120",
            }
        )

    response.headers["ETag"] = etag
    response.headers["Cache-Control"] = "private, max-age=60, stale-while-revalidate=120"
    return await services.get_my_custom_lists(db, user_id)

@router.get("/membership/{tmdb_id}", response_model=List[MediaListMembership])
async def get_media_membership(
    tmdb_id: int,
    media_type: str = Query(..., description="'movie' or 'tv'"),
    user_id_str: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    user_id = uuid.UUID(user_id_str)
    return await services.get_user_lists_membership_for_media(db, user_id, tmdb_id, media_type)

@router.get("/{list_id}", response_model=CustomListDetailOut)
async def get_list_detail(
    list_id: uuid.UUID,
    user_id_str: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    user_id = uuid.UUID(user_id_str)
    return await services.get_custom_list_detail(db, list_id, user_id)

@router.get("/{list_id}/items", response_model=CustomListItemsPagedOut)
async def get_list_items_paged(
    list_id: uuid.UUID,
    cursor: Optional[int] = Query(None, description="Cursor da última posição (keyset pagination)"),
    limit: int = Query(20, ge=1, le=100, description="Itens por página"),
    user_id_str: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    user_id = uuid.UUID(user_id_str)
    return await services.get_custom_list_items_paged(db, list_id, user_id, cursor=cursor, limit=limit)

@router.put("/{list_id}", response_model=CustomListSummaryOut)
async def update_list(
    list_id: uuid.UUID,
    data: CustomListUpdate,
    user_id_str: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    user_id = uuid.UUID(user_id_str)
    return await services.update_custom_list(db, list_id, user_id, data)

@router.delete("/{list_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_list(
    list_id: uuid.UUID,
    user_id_str: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    user_id = uuid.UUID(user_id_str)
    await services.delete_custom_list(db, list_id, user_id)
    return None

@router.post("/{list_id}/items", response_model=CustomListItemOut, status_code=status.HTTP_201_CREATED)
async def add_item_to_list(
    list_id: uuid.UUID,
    item_in: CustomListItemCreate,
    user_id_str: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    user_id = uuid.UUID(user_id_str)
    return await services.add_item_to_custom_list(db, list_id, user_id, item_in)

@router.post("/{list_id}/items/bulk", response_model=List[CustomListItemOut], status_code=status.HTTP_201_CREATED)
async def add_items_bulk_to_list(
    list_id: uuid.UUID,
    data: CustomListItemsBulkCreate,
    user_id_str: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    user_id = uuid.UUID(user_id_str)
    return await services.add_items_bulk_to_custom_list(db, list_id, user_id, data.items)

@router.delete("/{list_id}/items/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_item_from_list(
    list_id: uuid.UUID,
    item_id: uuid.UUID,
    user_id_str: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    user_id = uuid.UUID(user_id_str)
    await services.remove_item_from_custom_list(db, list_id, user_id, item_id)
    return None

@router.put("/{list_id}/items/reorder", status_code=status.HTTP_200_OK)
async def reorder_items(
    list_id: uuid.UUID,
    data: ReorderItemsRequest,
    user_id_str: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    user_id = uuid.UUID(user_id_str)
    await services.reorder_custom_list_items(db, list_id, user_id, data.item_ids)
    return {"message": "Itens reordenados com sucesso"}

@router.patch("/{list_id}/items/{item_id}", response_model=CustomListItemOut)
async def update_item(
    list_id: uuid.UUID,
    item_id: uuid.UUID,
    data: CustomListItemUpdate,
    user_id_str: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    user_id = uuid.UUID(user_id_str)
    return await services.update_custom_list_item(db, list_id, user_id, item_id, data)
