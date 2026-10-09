import uuid
from typing import Optional, List
from sqlalchemy import select, func, desc, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from fastapi import HTTPException, status

from app.custom_lists.models import CustomList, CustomListItem
from app.custom_lists.schemas import (
    CustomListCreate,
    CustomListUpdate,
    CustomListItemCreate,
    CustomListItemUpdate,
    CustomListSummaryOut,
    CustomListDetailOut,
    CustomListItemOut,
    MediaListMembership
)
from app.media.models import Media
from app.users.models import User

async def create_custom_list(db: AsyncSession, user_id: uuid.UUID, data: CustomListCreate) -> CustomListSummaryOut:
    new_list = CustomList(
        user_id=user_id,
        title=data.title.strip(),
        description=data.description.strip() if data.description else None,
        is_public=False,
        is_ranked=data.is_ranked,
        cover_backdrop_path=data.cover_backdrop_path,
        cover_poster_path=data.cover_poster_path,
        items_count=0
    )
    db.add(new_list)
    await db.commit()
    await db.refresh(new_list)

    # Obter nome do usuário
    user_res = await db.execute(select(User.name).where(User.id == user_id))
    user_name = user_res.scalar_one_or_none()

    return CustomListSummaryOut(
        id=new_list.id,
        user_id=new_list.user_id,
        user_name=user_name,
        title=new_list.title,
        description=new_list.description,
        is_ranked=new_list.is_ranked,
        cover_backdrop_path=new_list.cover_backdrop_path,
        cover_poster_path=new_list.cover_poster_path,
        items_count=0,
        preview_posters=[],
        created_at=new_list.created_at,
        updated_at=new_list.updated_at
    )

async def get_my_custom_lists(db: AsyncSession, user_id: uuid.UUID) -> List[CustomListSummaryOut]:
    query = (
        select(CustomList, User.name.label("user_name"))
        .join(User, CustomList.user_id == User.id)
        .where(CustomList.user_id == user_id)
        .order_by(desc(CustomList.updated_at))
    )
    res = await db.execute(query)
    rows = res.all()

    output = []
    for list_obj, user_name in rows:
        # Obter os primeiros 4 pôsteres para preview
        posters_res = await db.execute(
            select(CustomListItem.poster_path)
            .where(CustomListItem.list_id == list_obj.id, CustomListItem.poster_path.isnot(None))
            .order_by(CustomListItem.position)
            .limit(4)
        )
        preview_posters = [p for p in posters_res.scalars().all() if p]

        output.append(
            CustomListSummaryOut(
                id=list_obj.id,
                user_id=list_obj.user_id,
                user_name=user_name,
                title=list_obj.title,
                description=list_obj.description,
                is_ranked=list_obj.is_ranked,
                cover_backdrop_path=list_obj.cover_backdrop_path,
                cover_poster_path=list_obj.cover_poster_path,
                items_count=list_obj.items_count,
                preview_posters=preview_posters,
                created_at=list_obj.created_at,
                updated_at=list_obj.updated_at
            )
        )
    return output

async def get_custom_list_detail(
    db: AsyncSession,
    list_id: uuid.UUID,
    user_id: uuid.UUID
) -> CustomListDetailOut:
    query = (
        select(CustomList, User.name.label("user_name"))
        .join(User, CustomList.user_id == User.id)
        .where(CustomList.id == list_id)
        .options(selectinload(CustomList.items))
    )
    res = await db.execute(query)
    row = res.first()
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Lista não encontrada")

    list_obj, user_name = row

    if str(user_id) != str(list_obj.user_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Esta lista é privada e pertence a outro usuário")

    # Ordenar itens por position
    sorted_items = sorted(list_obj.items, key=lambda x: x.position)
    
    total_runtime = sum(item.runtime or 0 for item in sorted_items)
    preview_posters = [item.poster_path for item in sorted_items[:4] if item.poster_path]

    items_out = [
        CustomListItemOut(
            id=item.id,
            list_id=item.list_id,
            media_id=item.media_id,
            tmdb_id=item.tmdb_id,
            media_type=item.media_type,
            title=item.title,
            poster_path=item.poster_path,
            backdrop_path=item.backdrop_path,
            release_date=item.release_date,
            runtime=item.runtime or 0,
            position=item.position,
            note=item.note,
            created_at=item.created_at
        )
        for item in sorted_items
    ]

    return CustomListDetailOut(
        id=list_obj.id,
        user_id=list_obj.user_id,
        user_name=user_name,
        title=list_obj.title,
        description=list_obj.description,
        is_ranked=list_obj.is_ranked,
        cover_backdrop_path=list_obj.cover_backdrop_path,
        cover_poster_path=list_obj.cover_poster_path,
        items_count=len(sorted_items),
        preview_posters=preview_posters,
        created_at=list_obj.created_at,
        updated_at=list_obj.updated_at,
        items=items_out,
        total_runtime_minutes=total_runtime,
        is_owner=True
    )

async def update_custom_list(
    db: AsyncSession,
    list_id: uuid.UUID,
    user_id: uuid.UUID,
    data: CustomListUpdate
) -> CustomListSummaryOut:
    res = await db.execute(select(CustomList).where(CustomList.id == list_id))
    list_obj = res.scalar_one_or_none()
    if not list_obj:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Lista não encontrada")
    if str(list_obj.user_id) != str(user_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Permissão negada")

    if data.title is not None:
        list_obj.title = data.title.strip()
    if data.description is not None:
        list_obj.description = data.description.strip() if data.description else None
    if data.is_ranked is not None:
        list_obj.is_ranked = data.is_ranked
    if data.cover_backdrop_path is not None:
        list_obj.cover_backdrop_path = data.cover_backdrop_path
    if data.cover_poster_path is not None:
        list_obj.cover_poster_path = data.cover_poster_path

    await db.commit()
    await db.refresh(list_obj)

    user_res = await db.execute(select(User.name).where(User.id == user_id))
    user_name = user_res.scalar_one_or_none()

    return CustomListSummaryOut(
        id=list_obj.id,
        user_id=list_obj.user_id,
        user_name=user_name,
        title=list_obj.title,
        description=list_obj.description,
        is_ranked=list_obj.is_ranked,
        cover_backdrop_path=list_obj.cover_backdrop_path,
        cover_poster_path=list_obj.cover_poster_path,
        items_count=list_obj.items_count,
        preview_posters=[],
        created_at=list_obj.created_at,
        updated_at=list_obj.updated_at
    )

async def delete_custom_list(db: AsyncSession, list_id: uuid.UUID, user_id: uuid.UUID) -> bool:
    res = await db.execute(select(CustomList).where(CustomList.id == list_id))
    list_obj = res.scalar_one_or_none()
    if not list_obj:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Lista não encontrada")
    if str(list_obj.user_id) != str(user_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Permissão negada")

    await db.delete(list_obj)
    await db.commit()
    return True

async def add_item_to_custom_list(
    db: AsyncSession,
    list_id: uuid.UUID,
    user_id: uuid.UUID,
    item_in: CustomListItemCreate
) -> CustomListItemOut:
    res = await db.execute(select(CustomList).where(CustomList.id == list_id))
    list_obj = res.scalar_one_or_none()
    if not list_obj:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Lista não encontrada")
    if str(list_obj.user_id) != str(user_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Permissão negada")

    # Verifica se já está na lista
    existing = await db.execute(
        select(CustomListItem).where(
            CustomListItem.list_id == list_id,
            CustomListItem.tmdb_id == item_in.tmdb_id,
            CustomListItem.media_type == item_in.media_type
        )
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Este item já está nesta lista")

    # Determinar media_id caso já exista na base de media
    media_res = await db.execute(
        select(Media.id, Media.runtime).where(
            Media.tmdb_id == item_in.tmdb_id,
            Media.media_type == item_in.media_type
        )
    )
    media_row = media_res.first()
    media_id = media_row[0] if media_row else None
    runtime = item_in.runtime or (media_row[1] if media_row and media_row[1] else 0)

    # Determina próxima position
    max_pos_res = await db.execute(
        select(func.coalesce(func.max(CustomListItem.position), -1)).where(
            CustomListItem.list_id == list_id
        )
    )
    next_pos = (max_pos_res.scalar_one() or 0) + 1 if item_in.position is None else item_in.position

    new_item = CustomListItem(
        list_id=list_id,
        media_id=media_id,
        tmdb_id=item_in.tmdb_id,
        media_type=item_in.media_type,
        title=item_in.title,
        poster_path=item_in.poster_path,
        backdrop_path=item_in.backdrop_path,
        release_date=item_in.release_date,
        runtime=runtime,
        position=next_pos,
        note=item_in.note
    )
    db.add(new_item)

    # Atualiza contagem e capa da lista se ainda vazia
    list_obj.items_count = list_obj.items_count + 1
    if not list_obj.cover_backdrop_path and item_in.backdrop_path:
        list_obj.cover_backdrop_path = item_in.backdrop_path
    if not list_obj.cover_poster_path and item_in.poster_path:
        list_obj.cover_poster_path = item_in.poster_path

    await db.commit()
    await db.refresh(new_item)

    return CustomListItemOut(
        id=new_item.id,
        list_id=new_item.list_id,
        media_id=new_item.media_id,
        tmdb_id=new_item.tmdb_id,
        media_type=new_item.media_type,
        title=new_item.title,
        poster_path=new_item.poster_path,
        backdrop_path=new_item.backdrop_path,
        release_date=new_item.release_date,
        runtime=new_item.runtime or 0,
        position=new_item.position,
        note=new_item.note,
        created_at=new_item.created_at
    )

async def remove_item_from_custom_list(
    db: AsyncSession,
    list_id: uuid.UUID,
    user_id: uuid.UUID,
    item_id: uuid.UUID
) -> bool:
    res = await db.execute(select(CustomList).where(CustomList.id == list_id))
    list_obj = res.scalar_one_or_none()
    if not list_obj:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Lista não encontrada")
    if str(list_obj.user_id) != str(user_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Permissão negada")

    item_res = await db.execute(
        select(CustomListItem).where(
            CustomListItem.id == item_id,
            CustomListItem.list_id == list_id
        )
    )
    item_obj = item_res.scalar_one_or_none()
    if not item_obj:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Item não encontrado na lista")

    await db.delete(item_obj)
    list_obj.items_count = max(0, list_obj.items_count - 1)

    # Reindexar posições dos itens restantes
    remaining = await db.execute(
        select(CustomListItem)
        .where(CustomListItem.list_id == list_id)
        .order_by(CustomListItem.position)
    )
    items = remaining.scalars().all()
    for idx, it in enumerate(items):
        it.position = idx

    await db.commit()
    return True

async def reorder_custom_list_items(
    db: AsyncSession,
    list_id: uuid.UUID,
    user_id: uuid.UUID,
    item_ids: List[uuid.UUID]
) -> bool:
    res = await db.execute(select(CustomList).where(CustomList.id == list_id))
    list_obj = res.scalar_one_or_none()
    if not list_obj:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Lista não encontrada")
    if str(list_obj.user_id) != str(user_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Permissão negada")

    items_res = await db.execute(select(CustomListItem).where(CustomListItem.list_id == list_id))
    items_by_id = {item.id: item for item in items_res.scalars().all()}

    for position, item_id in enumerate(item_ids):
        if item_id in items_by_id:
            items_by_id[item_id].position = position

    await db.commit()
    return True

async def update_custom_list_item(
    db: AsyncSession,
    list_id: uuid.UUID,
    user_id: uuid.UUID,
    item_id: uuid.UUID,
    data: CustomListItemUpdate
) -> CustomListItemOut:
    res = await db.execute(select(CustomList).where(CustomList.id == list_id))
    list_obj = res.scalar_one_or_none()
    if not list_obj:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Lista não encontrada")
    if str(list_obj.user_id) != str(user_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Permissão negada")

    item_res = await db.execute(
        select(CustomListItem).where(
            CustomListItem.id == item_id,
            CustomListItem.list_id == list_id
        )
    )
    item_obj = item_res.scalar_one_or_none()
    if not item_obj:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Item não encontrado na lista")

    if data.note is not None:
        item_obj.note = data.note.strip() if data.note else None
    if data.position is not None:
        item_obj.position = data.position

    await db.commit()
    await db.refresh(item_obj)

    return CustomListItemOut(
        id=item_obj.id,
        list_id=item_obj.list_id,
        media_id=item_obj.media_id,
        tmdb_id=item_obj.tmdb_id,
        media_type=item_obj.media_type,
        title=item_obj.title,
        poster_path=item_obj.poster_path,
        backdrop_path=item_obj.backdrop_path,
        release_date=item_obj.release_date,
        runtime=item_obj.runtime or 0,
        position=item_obj.position,
        note=item_obj.note,
        created_at=item_obj.created_at
    )

async def get_user_lists_membership_for_media(
    db: AsyncSession,
    user_id: uuid.UUID,
    tmdb_id: int,
    media_type: str
) -> List[MediaListMembership]:
    # Pega todas as listas do usuário
    user_lists_res = await db.execute(
        select(CustomList.id, CustomList.title)
        .where(CustomList.user_id == user_id)
        .order_by(desc(CustomList.updated_at))
    )
    user_lists = user_lists_res.all()

    if not user_lists:
        return []

    # Pega itens correspondentes
    items_res = await db.execute(
        select(CustomListItem.list_id, CustomListItem.id)
        .where(
            CustomListItem.list_id.in_([l[0] for l in user_lists]),
            CustomListItem.tmdb_id == tmdb_id,
            CustomListItem.media_type == media_type
        )
    )
    matched_items = {row[0]: row[1] for row in items_res.all()}

    output = []
    for l_id, title in user_lists:
        output.append(
            MediaListMembership(
                list_id=l_id,
                title=title,
                contains_media=l_id in matched_items,
                item_id=matched_items.get(l_id)
            )
        )
    return output
