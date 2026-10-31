import uuid
from datetime import datetime, timezone
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
    CustomListItemsPagedOut,
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

async def get_my_custom_lists_etag(db: AsyncSession, user_id: uuid.UUID) -> str:
    res = await db.execute(
        select(func.max(CustomList.updated_at), func.count(CustomList.id))
        .where(CustomList.user_id == user_id)
    )
    row = res.first()
    max_updated, count = row if row else (None, 0)
    ts = int(max_updated.timestamp()) if max_updated else 0
    return f'W/"{str(user_id)[:8]}-{count}-{ts}"'

async def get_my_custom_lists(db: AsyncSession, user_id: uuid.UUID) -> List[CustomListSummaryOut]:
    query = (
        select(CustomList, User.name.label("user_name"))
        .join(User, CustomList.user_id == User.id)
        .where(CustomList.user_id == user_id)
        .order_by(desc(CustomList.updated_at))
    )
    res = await db.execute(query)
    rows = res.all()

    if not rows:
        return []

    # Otimização N+1: Buscar os primeiros 4 pôsteres de todas as listas em lote usando Window Function
    list_ids = [list_obj.id for list_obj, _ in rows]
    subq = (
        select(
            CustomListItem.list_id,
            CustomListItem.poster_path,
            func.row_number().over(
                partition_by=CustomListItem.list_id,
                order_by=CustomListItem.position
            ).label("rn")
        )
        .where(
            CustomListItem.list_id.in_(list_ids),
            CustomListItem.poster_path.isnot(None)
        )
        .subquery()
    )

    posters_query = (
        select(subq.c.list_id, subq.c.poster_path)
        .where(subq.c.rn <= 4)
        .order_by(subq.c.list_id, subq.c.rn)
    )
    posters_res = await db.execute(posters_query)

    posters_map = {}
    for lid, poster in posters_res.all():
        posters_map.setdefault(lid, []).append(poster)

    output = []
    for list_obj, user_name in rows:
        preview_posters = posters_map.get(list_obj.id, [])

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

async def get_custom_list_items_paged(
    db: AsyncSession,
    list_id: uuid.UUID,
    user_id: uuid.UUID,
    cursor: Optional[int] = None,
    limit: int = 20
) -> CustomListItemsPagedOut:
    list_res = await db.execute(
        select(CustomList.user_id, CustomList.items_count).where(CustomList.id == list_id)
    )
    row = list_res.first()
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Lista não encontrada")
    owner_id, total_count = row
    if str(owner_id) != str(user_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Permissão negada")

    stmt = select(CustomListItem).where(CustomListItem.list_id == list_id)
    if cursor is not None:
        stmt = stmt.where(CustomListItem.position > cursor)

    stmt = stmt.order_by(CustomListItem.position.asc()).limit(limit + 1)
    res = await db.execute(stmt)
    rows = res.scalars().all()

    has_more = len(rows) > limit
    page_rows = rows[:limit]
    next_cursor = page_rows[-1].position if (has_more and page_rows) else None

    return CustomListItemsPagedOut(
        items=[
            CustomListItemOut(
                id=it.id,
                list_id=it.list_id,
                media_id=it.media_id,
                tmdb_id=it.tmdb_id,
                media_type=it.media_type,
                title=it.title,
                poster_path=it.poster_path,
                backdrop_path=it.backdrop_path,
                release_date=it.release_date,
                runtime=it.runtime or 0,
                position=it.position,
                note=it.note,
                created_at=it.created_at
            )
            for it in page_rows
        ],
        next_cursor=next_cursor,
        has_more=has_more,
        total_count=total_count
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

async def add_items_bulk_to_custom_list(
    db: AsyncSession,
    list_id: uuid.UUID,
    user_id: uuid.UUID,
    items_in: List[CustomListItemCreate]
) -> List[CustomListItemOut]:
    res = await db.execute(select(CustomList).where(CustomList.id == list_id))
    list_obj = res.scalar_one_or_none()
    if not list_obj:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Lista não encontrada")
    if str(list_obj.user_id) != str(user_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Permissão negada")

    if not items_in:
        return []

    # 1. Deduplicar itens recebidos no próprio payload
    unique_incoming: List[CustomListItemCreate] = []
    seen = set()
    for item in items_in:
        key = (item.tmdb_id, item.media_type)
        if key not in seen:
            seen.add(key)
            unique_incoming.append(item)

    # 2. Consultar itens já existentes na lista para não tentar duplicar
    incoming_tmdb_ids = [it.tmdb_id for it in unique_incoming]
    existing_res = await db.execute(
        select(CustomListItem.tmdb_id, CustomListItem.media_type).where(
            CustomListItem.list_id == list_id,
            CustomListItem.tmdb_id.in_(incoming_tmdb_ids)
        )
    )
    existing_keys = {(r[0], r[1]) for r in existing_res.all()}

    to_insert = [it for it in unique_incoming if (it.tmdb_id, it.media_type) not in existing_keys]
    if not to_insert:
        return []

    # 3. Buscar correspondência na tabela media para reaproveitar media_id e runtime
    media_res = await db.execute(
        select(Media.id, Media.tmdb_id, Media.media_type, Media.runtime).where(
            Media.tmdb_id.in_([it.tmdb_id for it in to_insert])
        )
    )
    media_map = {(row[1], row[2]): (row[0], row[3]) for row in media_res.all()}

    # 4. Determinar próxima position
    max_pos_res = await db.execute(
        select(func.coalesce(func.max(CustomListItem.position), -1)).where(
            CustomListItem.list_id == list_id
        )
    )
    curr_pos = max_pos_res.scalar_one() or -1

    now = datetime.now(timezone.utc)
    new_items: List[CustomListItem] = []
    for it in to_insert:
        curr_pos += 1
        m_info = media_map.get((it.tmdb_id, it.media_type))
        media_id = m_info[0] if m_info else None
        runtime = it.runtime or (m_info[1] if m_info and m_info[1] else 0)

        new_item = CustomListItem(
            id=uuid.uuid4(),
            list_id=list_id,
            media_id=media_id,
            tmdb_id=it.tmdb_id,
            media_type=it.media_type,
            title=it.title,
            poster_path=it.poster_path,
            backdrop_path=it.backdrop_path,
            release_date=it.release_date,
            runtime=runtime,
            position=it.position if it.position is not None else curr_pos,
            note=it.note,
            created_at=now
        )
        new_items.append(new_item)

    db.add_all(new_items)

    # 5. Atualizar contador e capas da lista se vazias
    list_obj.items_count = list_obj.items_count + len(new_items)
    for it in new_items:
        if not list_obj.cover_backdrop_path and it.backdrop_path:
            list_obj.cover_backdrop_path = it.backdrop_path
        if not list_obj.cover_poster_path and it.poster_path:
            list_obj.cover_poster_path = it.poster_path

    await db.commit()

    return [
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
        for item in new_items
    ]

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

    removed_pos = item_obj.position
    await db.delete(item_obj)
    list_obj.items_count = max(0, list_obj.items_count - 1)

    # Reindexar posições dos itens restantes em O(1) via SQL UPDATE direto
    await db.execute(
        update(CustomListItem)
        .where(
            CustomListItem.list_id == list_id,
            CustomListItem.position > removed_pos
        )
        .values(position=CustomListItem.position - 1)
    )

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
