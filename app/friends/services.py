import uuid
from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import or_, and_, func
from fastapi import HTTPException, status

from app.users.models import User
from app.friends.models import Friendship
from app.notifications.models import Notification
from app.media.models import Media
from app.tracking.models import UserListItem, UserEpisodeProgress
from app.friends.schemas import (
    UserSearchResult,
    FriendUserResponse,
    FriendRequestItem,
    FriendRequestsResponse,
    UserSummary,
    FriendFeedItem,
    FriendFeedResponse
)

async def search_users(
    db: AsyncSession,
    current_user_id: uuid.UUID,
    query: str,
    limit: int = 20
) -> List[UserSearchResult]:
    clean = query.strip().lower()
    if not clean:
        return []

    # Busca usuários cujo username ou nome correspondam à busca (excluindo o usuário logado)
    user_query = (
        select(User)
        .where(
            User.id != current_user_id,
            or_(
                func.lower(User.username).like(f"%{clean}%"),
                func.lower(User.name).like(f"%{clean}%")
            )
        )
        .limit(limit)
    )
    res = await db.execute(user_query)
    found_users = res.scalars().all()

    if not found_users:
        return []

    found_ids = [u.id for u in found_users]

    # Busca relacionamentos existentes entre o usuário atual e os usuários encontrados
    rel_query = select(Friendship).where(
        or_(
            and_(Friendship.requester_id == current_user_id, Friendship.addressee_id.in_(found_ids)),
            and_(Friendship.addressee_id == current_user_id, Friendship.requester_id.in_(found_ids))
        )
    )
    rel_res = await db.execute(rel_query)
    friendships = rel_res.scalars().all()

    # Mapeia cada relacionamento por id do outro usuário
    rel_map = {}
    for f in friendships:
        other_id = f.addressee_id if f.requester_id == current_user_id else f.requester_id
        rel_map[other_id] = f

    results: List[UserSearchResult] = []
    for u in found_users:
        f = rel_map.get(u.id)
        rel_status = "none"
        f_id = None

        if f:
            f_id = f.id
            if f.status == "accepted":
                rel_status = "friends"
            elif f.status == "pending":
                if f.requester_id == current_user_id:
                    rel_status = "pending_sent"
                else:
                    rel_status = "pending_received"

        results.append(
            UserSearchResult(
                id=u.id,
                name=u.name,
                username=u.username,
                relationship_status=rel_status,
                friendship_id=f_id
            )
        )

    return results

async def send_friend_request(
    db: AsyncSession,
    current_user_id: uuid.UUID,
    addressee_id: Optional[uuid.UUID] = None,
    username: Optional[str] = None
) -> Friendship:
    # Obtém o usuário de destino
    if username:
        clean_username = username.strip().lower().replace("@", "")
        res = await db.execute(select(User).where(func.lower(User.username) == clean_username))
        target_user = res.scalar_one_or_none()
    elif addressee_id:
        res = await db.execute(select(User).where(User.id == addressee_id))
        target_user = res.scalar_one_or_none()
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Informe o id ou o nome de usuário do destinatário."
        )

    if not target_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Usuário não encontrado."
        )

    if target_user.id == current_user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Você não pode enviar uma solicitação de amizade para si mesmo."
        )

    # Verifica se já existe amizade ou solicitação
    rel_query = select(Friendship).where(
        or_(
            and_(Friendship.requester_id == current_user_id, Friendship.addressee_id == target_user.id),
            and_(Friendship.addressee_id == current_user_id, Friendship.requester_id == target_user.id)
        )
    )
    rel_res = await db.execute(rel_query)
    existing = rel_res.scalar_one_or_none()

    if existing:
        if existing.status == "accepted":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Vocês já são amigos."
            )
        if existing.status == "pending":
            if existing.requester_id == current_user_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Você já enviou uma solicitação de amizade para este usuário."
                )
            else:
                # O outro usuário já havia enviado solicitação: aceita automaticamente!
                existing.status = "accepted"
                # Notifica o outro usuário
                current_u_res = await db.execute(select(User).where(User.id == current_user_id))
                curr_user = current_u_res.scalar_one_or_none()
                curr_name = curr_user.name if curr_user else "Um usuário"
                notif = Notification(
                    user_id=target_user.id,
                    title="Solicitação de amizade aceita",
                    message=f"{curr_name} aceitou sua solicitação de amizade."
                )
                db.add(notif)
                await db.commit()
                await db.refresh(existing)
                return existing

    # Cria nova solicitação
    new_friendship = Friendship(
        requester_id=current_user_id,
        addressee_id=target_user.id,
        status="pending"
    )
    db.add(new_friendship)

    # Cria notificação para o destinatário
    curr_res = await db.execute(select(User).where(User.id == current_user_id))
    curr_user = curr_res.scalar_one_or_none()
    sender_display = f"@{curr_user.username}" if curr_user and curr_user.username else (curr_user.name if curr_user else "Alguém")
    notif = Notification(
        user_id=target_user.id,
        title="Nova solicitação de amizade",
        message=f"{sender_display} enviou uma solicitação de amizade para você."
    )
    db.add(notif)

    await db.commit()
    await db.refresh(new_friendship)
    return new_friendship

async def get_friends_list(
    db: AsyncSession,
    current_user_id: uuid.UUID
) -> List[FriendUserResponse]:
    query = (
        select(Friendship)
        .where(
            Friendship.status == "accepted",
            or_(
                Friendship.requester_id == current_user_id,
                Friendship.addressee_id == current_user_id
            )
        )
        .order_by(Friendship.updated_at.desc())
    )
    res = await db.execute(query)
    friendships = res.scalars().all()

    if not friendships:
        return []

    # Carrega os dados dos amigos
    other_ids = [
        f.addressee_id if f.requester_id == current_user_id else f.requester_id
        for f in friendships
    ]

    users_res = await db.execute(select(User).where(User.id.in_(other_ids)))
    users_dict = {u.id: u for u in users_res.scalars().all()}

    friends: List[FriendUserResponse] = []
    for f in friendships:
        other_id = f.addressee_id if f.requester_id == current_user_id else f.requester_id
        u = users_dict.get(other_id)
        if u:
            friends.append(
                FriendUserResponse(
                    id=u.id,
                    name=u.name,
                    username=u.username,
                    email=u.email,
                    friendship_id=f.id,
                    since=f.updated_at
                )
            )

    return friends

async def get_friend_requests(
    db: AsyncSession,
    current_user_id: uuid.UUID
) -> FriendRequestsResponse:
    # Solicitações recebidas pendentes
    rec_query = (
        select(Friendship)
        .where(
            Friendship.addressee_id == current_user_id,
            Friendship.status == "pending"
        )
        .order_by(Friendship.created_at.desc())
    )
    rec_res = await db.execute(rec_query)
    received_friendships = rec_res.scalars().all()

    # Solicitações enviadas pendentes
    sent_query = (
        select(Friendship)
        .where(
            Friendship.requester_id == current_user_id,
            Friendship.status == "pending"
        )
        .order_by(Friendship.created_at.desc())
    )
    sent_res = await db.execute(sent_query)
    sent_friendships = sent_res.scalars().all()

    # Carrega os dados dos usuários envolvidos
    all_user_ids = set(
        [f.requester_id for f in received_friendships] +
        [f.addressee_id for f in sent_friendships]
    )

    users_dict = {}
    if all_user_ids:
        u_res = await db.execute(select(User).where(User.id.in_(list(all_user_ids))))
        users_dict = {u.id: u for u in u_res.scalars().all()}

    received_items: List[FriendRequestItem] = []
    for f in received_friendships:
        u = users_dict.get(f.requester_id)
        if u:
            received_items.append(
                FriendRequestItem(
                    friendship_id=f.id,
                    user=UserSummary(id=u.id, name=u.name, username=u.username, email=u.email),
                    created_at=f.created_at
                )
            )

    sent_items: List[FriendRequestItem] = []
    for f in sent_friendships:
        u = users_dict.get(f.addressee_id)
        if u:
            sent_items.append(
                FriendRequestItem(
                    friendship_id=f.id,
                    user=UserSummary(id=u.id, name=u.name, username=u.username, email=u.email),
                    created_at=f.created_at
                )
            )

    return FriendRequestsResponse(received=received_items, sent=sent_items)

async def accept_friend_request(
    db: AsyncSession,
    friendship_id: uuid.UUID,
    current_user_id: uuid.UUID
) -> Friendship:
    res = await db.execute(
        select(Friendship).where(
            Friendship.id == friendship_id,
            Friendship.addressee_id == current_user_id,
            Friendship.status == "pending"
        )
    )
    friendship = res.scalar_one_or_none()
    if not friendship:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Solicitação de amizade não encontrada ou já respondida."
        )

    friendship.status = "accepted"

    # Notifica quem enviou a solicitação
    curr_res = await db.execute(select(User).where(User.id == current_user_id))
    curr_user = curr_res.scalar_one_or_none()
    display = f"@{curr_user.username}" if curr_user and curr_user.username else (curr_user.name if curr_user else "Alguém")
    notif = Notification(
        user_id=friendship.requester_id,
        title="Solicitação de amizade aceita",
        message=f"{display} aceitou sua solicitação de amizade."
    )
    db.add(notif)

    await db.commit()
    await db.refresh(friendship)
    return friendship

async def reject_friend_request(
    db: AsyncSession,
    friendship_id: uuid.UUID,
    current_user_id: uuid.UUID
) -> dict:
    res = await db.execute(
        select(Friendship).where(
            Friendship.id == friendship_id,
            Friendship.addressee_id == current_user_id,
            Friendship.status == "pending"
        )
    )
    friendship = res.scalar_one_or_none()
    if not friendship:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Solicitação de amizade não encontrada."
        )

    await db.delete(friendship)
    await db.commit()
    return {"message": "Solicitação de amizade recusada."}

async def cancel_friend_request(
    db: AsyncSession,
    friendship_id: uuid.UUID,
    current_user_id: uuid.UUID
) -> dict:
    res = await db.execute(
        select(Friendship).where(
            Friendship.id == friendship_id,
            Friendship.requester_id == current_user_id,
            Friendship.status == "pending"
        )
    )
    friendship = res.scalar_one_or_none()
    if not friendship:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Solicitação de amizade não encontrada."
        )

    await db.delete(friendship)
    await db.commit()
    return {"message": "Solicitação de amizade cancelada."}

async def remove_friend(
    db: AsyncSession,
    friendship_id: uuid.UUID,
    current_user_id: uuid.UUID
) -> dict:
    # Pode receber friendship_id diretamente ou tentar localizar
    res = await db.execute(
        select(Friendship).where(
            Friendship.id == friendship_id,
            Friendship.status == "accepted",
            or_(
                Friendship.requester_id == current_user_id,
                Friendship.addressee_id == current_user_id
            )
        )
    )
    friendship = res.scalar_one_or_none()

    if not friendship:
        # Tenta verificar se o id passado era o id do amigo (User.id)
        alt_res = await db.execute(
            select(Friendship).where(
                Friendship.status == "accepted",
                or_(
                    and_(Friendship.requester_id == current_user_id, Friendship.addressee_id == friendship_id),
                    and_(Friendship.addressee_id == current_user_id, Friendship.requester_id == friendship_id)
                )
            )
        )
        friendship = alt_res.scalar_one_or_none()

    if not friendship:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Amizade não encontrada."
        )

    await db.delete(friendship)
    await db.commit()
    return {"message": "Amizade desfeita com sucesso."}

async def get_friend_feed(
    db: AsyncSession,
    current_user_id: uuid.UUID,
    page: int = 1,
    limit: int = 20
) -> FriendFeedResponse:
    # 1. Obter IDs de todos os amigos com status 'accepted'
    friendships_res = await db.execute(
        select(Friendship).where(
            Friendship.status == "accepted",
            or_(
                Friendship.requester_id == current_user_id,
                Friendship.addressee_id == current_user_id
            )
        )
    )
    friendships = friendships_res.scalars().all()

    if not friendships:
        return FriendFeedResponse(
            items=[],
            total=0,
            page=page,
            limit=limit,
            has_more=False
        )

    friend_ids = [
        f.addressee_id if f.requester_id == current_user_id else f.requester_id
        for f in friendships
    ]

    # 2. Buscar episódios recentes assistidos pelos amigos
    ep_query = (
        select(
            UserEpisodeProgress,
            UserListItem,
            Media,
            User
        )
        .join(UserListItem, UserEpisodeProgress.user_list_item_id == UserListItem.id)
        .join(Media, UserListItem.media_id == Media.id)
        .join(User, UserListItem.user_id == User.id)
        .where(UserListItem.user_id.in_(friend_ids))
        .order_by(UserEpisodeProgress.watched_at.desc())
        .limit(300)
    )
    ep_res = await db.execute(ep_query)
    ep_records = ep_res.all()

    # Agrupar episódios assistidos da mesma série na mesma data pelo mesmo amigo
    # chave: (user_id, media_id, season_number, date_str)
    ep_groups = {}
    for ep, uli, media, user in ep_records:
        if not ep.watched_at:
            continue
        date_str = ep.watched_at.strftime("%Y-%m-%d")
        key = (user.id, media.id, ep.season_number, date_str)
        if key not in ep_groups:
            ep_groups[key] = {
                "user": user,
                "media": media,
                "uli": uli,
                "season_number": ep.season_number,
                "episodes": [],
                "ratings": []
            }
        ep_groups[key]["episodes"].append((ep.episode_number, ep.watched_at))
        if ep.rating is not None:
            ep_groups[key]["ratings"].append(ep.rating)

    feed_items = []

    for key, group in ep_groups.items():
        user = group["user"]
        media = group["media"]
        uli = group["uli"]
        season_num = group["season_number"]
        ep_data = group["episodes"]
        ep_numbers = sorted(list(set(e[0] for e in ep_data)))
        latest_watched_at = max(e[1] for e in ep_data)

        # Determina o rating: rating dos episódios ou da série na lista
        user_rating = group["ratings"][0] if group["ratings"] else uli.rating

        if len(ep_numbers) == 1:
            episodes_label = f"T{season_num} E{ep_numbers[0]}"
            action_type = "watched_episode"
        else:
            min_ep = min(ep_numbers)
            max_ep = max(ep_numbers)
            if ep_numbers == list(range(min_ep, max_ep + 1)):
                episodes_label = f"T{season_num}: Eps. {min_ep} a {max_ep}"
            else:
                episodes_label = f"T{season_num}: Eps. {', '.join(str(x) for x in ep_numbers)}"
            action_type = "binge_watched"

        item_id = f"ep_{user.id}_{media.id}_{season_num}_{min(ep_numbers)}_{max(ep_numbers)}_{int(latest_watched_at.timestamp())}"

        feed_items.append(
            FriendFeedItem(
                id=item_id,
                user_id=user.id,
                user_name=user.name,
                username=user.username,
                media_id=media.id,
                tmdb_id=media.tmdb_id,
                media_type=media.media_type,
                title=media.title,
                poster_path=media.poster_path,
                backdrop_path=media.backdrop_path,
                rating=user_rating,
                season_number=season_num,
                episode_numbers=ep_numbers,
                episodes_label=episodes_label,
                watched_at=latest_watched_at,
                action_type=action_type
            )
        )

    # 3. Buscar filmes recentes assistidos pelos amigos
    movie_query = (
        select(
            UserListItem,
            Media,
            User
        )
        .join(Media, UserListItem.media_id == Media.id)
        .join(User, UserListItem.user_id == User.id)
        .where(
            UserListItem.user_id.in_(friend_ids),
            Media.media_type == "movie",
            or_(
                UserListItem.status == "completed",
                UserListItem.last_watched_at.is_not(None)
            )
        )
        .order_by(
            func.coalesce(UserListItem.last_watched_at, UserListItem.updated_at).desc()
        )
        .limit(100)
    )
    movie_res = await db.execute(movie_query)
    movie_records = movie_res.all()

    for uli, media, user in movie_records:
        watched_at = uli.last_watched_at or uli.updated_at or uli.created_at
        if not watched_at:
            continue

        item_id = f"mov_{user.id}_{media.id}_{int(watched_at.timestamp())}"
        feed_items.append(
            FriendFeedItem(
                id=item_id,
                user_id=user.id,
                user_name=user.name,
                username=user.username,
                media_id=media.id,
                tmdb_id=media.tmdb_id,
                media_type="movie",
                title=media.title,
                poster_path=media.poster_path,
                backdrop_path=media.backdrop_path,
                rating=uli.rating,
                season_number=None,
                episode_numbers=[],
                episodes_label=None,
                watched_at=watched_at,
                action_type="watched_movie"
            )
        )

    # 4. Ordenar todos os itens por watched_at decrescente
    feed_items.sort(key=lambda x: x.watched_at, reverse=True)

    # 5. Aplicar paginação
    total = len(feed_items)
    start = (page - 1) * limit
    end = start + limit
    paginated_items = feed_items[start:end]
    has_more = end < total

    return FriendFeedResponse(
        items=paginated_items,
        total=total,
        page=page,
        limit=limit,
        has_more=has_more
    )
