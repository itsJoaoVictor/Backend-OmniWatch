import uuid
from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import load_only
from sqlalchemy import or_, and_, func
from fastapi import HTTPException, status

from app.users.models import User
from app.friends.models import Friendship, FriendRecommendation
from app.notifications.models import Notification
from app.notifications.push_service import trigger_web_push_background
from app.media.models import Media
from app.tracking.models import UserListItem, UserEpisodeProgress
from app.friends.schemas import (
    UserSearchResult,
    FriendUserResponse,
    FriendRequestItem,
    FriendRequestsResponse,
    UserSummary,
    FriendFeedItem,
    FriendFeedResponse,
    FriendEligibilityItem,
    SendRecommendationRequest,
    FriendRecommendationResponse
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
        .options(load_only(User.id, User.name, User.username))
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

    users_res = await db.execute(
        select(User)
        .options(load_only(User.id, User.name, User.username, User.email))
        .where(User.id.in_(other_ids))
    )
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
    # Consulta unificada para solicitações pendentes (recebidas e enviadas)
    query = (
        select(Friendship)
        .where(
            Friendship.status == "pending",
            or_(
                Friendship.addressee_id == current_user_id,
                Friendship.requester_id == current_user_id
            )
        )
        .order_by(Friendship.created_at.desc())
    )
    res = await db.execute(query)
    all_pending = res.scalars().all()

    received_friendships = [f for f in all_pending if f.addressee_id == current_user_id]
    sent_friendships = [f for f in all_pending if f.requester_id == current_user_id]

    # Carrega os dados dos usuários envolvidos com load_only
    all_user_ids = set(
        [f.requester_id for f in received_friendships] +
        [f.addressee_id for f in sent_friendships]
    )

    users_dict = {}
    if all_user_ids:
        u_res = await db.execute(
            select(User)
            .options(load_only(User.id, User.name, User.username, User.email))
            .where(User.id.in_(list(all_user_ids)))
        )
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
    limit: int = 20,
    media_type: Optional[str] = None,
    cursor: Optional[str] = None,
    friend: Optional[str] = None
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
            has_more=False,
            next_cursor=None
        )

    friend_ids = [
        f.addressee_id if f.requester_id == current_user_id else f.requester_id
        for f in friendships
    ]

    # Filtrar por amigo específico se fornecido (por username ou UUID)
    if friend:
        clean_friend = friend.strip().replace("@", "")
        target_friend_id = None
        try:
            parsed_uuid = uuid.UUID(clean_friend)
            if parsed_uuid in friend_ids:
                target_friend_id = parsed_uuid
        except ValueError:
            user_res = await db.execute(
                select(User.id).where(func.lower(User.username) == clean_friend.lower())
            )
            found_id = user_res.scalar_one_or_none()
            if found_id and found_id in friend_ids:
                target_friend_id = found_id

        if not target_friend_id:
            # Amigo não existe ou não faz parte do círculo de amizades aceitas
            return FriendFeedResponse(
                items=[],
                total=0,
                page=page,
                limit=limit,
                has_more=False,
                next_cursor=None
            )
        
        friend_ids = [target_friend_id]

    # Normalizar media_type se fornecido
    filter_type = media_type.lower().strip() if media_type else None
    if filter_type not in ("movie", "tv"):
        filter_type = None

    # Parse do cursor temporal (before_timestamp) se fornecido
    cursor_dt = None
    if cursor:
        try:
            clean_cursor = cursor.strip().replace("Z", "+00:00")
            cursor_dt = datetime.fromisoformat(clean_cursor)
        except Exception:
            cursor_dt = None

    feed_items = []

    # Se houver cursor, usamos limites fixos e enxutos O(limit), sem crescer com a profundidade da página
    if cursor_dt:
        ep_limit = max(40, limit * 3)
        movie_limit = max(20, limit * 2)
    else:
        # Modo legado (sem cursor): escala dinâmica proporcional
        needed_count = page * limit
        ep_limit = min(300, max(60, needed_count * 4))
        movie_limit = min(100, max(30, needed_count * 2))

    # 2. Buscar episódios recentes assistidos pelos amigos (ignorado se filtro for apenas filme)
    if filter_type != "movie":
        ep_conditions = [
            UserListItem.user_id.in_(friend_ids),
            UserEpisodeProgress.watched_at.is_not(None)
        ]
        if cursor_dt:
            ep_conditions.append(UserEpisodeProgress.watched_at < cursor_dt)

        ep_query = (
            select(
                UserEpisodeProgress.season_number,
                UserEpisodeProgress.episode_number,
                UserEpisodeProgress.rating.label("ep_rating"),
                UserEpisodeProgress.watched_at,
                UserListItem.rating.label("uli_rating"),
                Media.id.label("media_id"),
                Media.tmdb_id,
                Media.media_type,
                Media.title,
                Media.poster_path,
                Media.backdrop_path,
                User.id.label("user_id"),
                User.name.label("user_name"),
                User.username,
            )
            .join(UserListItem, UserEpisodeProgress.user_list_item_id == UserListItem.id)
            .join(Media, UserListItem.media_id == Media.id)
            .join(User, UserListItem.user_id == User.id)
            .where(*ep_conditions)
            .order_by(UserEpisodeProgress.watched_at.desc())
            .limit(ep_limit)
        )
        ep_res = await db.execute(ep_query)
        ep_records = ep_res.all()

        # Agrupar episódios assistidos da mesma série na mesma data pelo mesmo amigo
        # chave: (user_id, media_id, season_number, date_str)
        ep_groups = {}
        for row in ep_records:
            if not row.watched_at:
                continue
            date_str = row.watched_at.strftime("%Y-%m-%d")
            key = (row.user_id, row.media_id, row.season_number, date_str)
            if key not in ep_groups:
                ep_groups[key] = {
                    "user_id": row.user_id,
                    "user_name": row.user_name,
                    "username": row.username,
                    "media_id": row.media_id,
                    "tmdb_id": row.tmdb_id,
                    "media_type": row.media_type,
                    "title": row.title,
                    "poster_path": row.poster_path,
                    "backdrop_path": row.backdrop_path,
                    "uli_rating": row.uli_rating,
                    "season_number": row.season_number,
                    "episodes": [],
                    "ratings": []
                }
            ep_groups[key]["episodes"].append((row.episode_number, row.watched_at))
            if row.ep_rating is not None:
                ep_groups[key]["ratings"].append(row.ep_rating)

        for key, group in ep_groups.items():
            user_id = group["user_id"]
            user_name = group["user_name"]
            username = group["username"]
            media_id = group["media_id"]
            tmdb_id = group["tmdb_id"]
            media_type_val = group["media_type"]
            title = group["title"]
            poster_path = group["poster_path"]
            backdrop_path = group["backdrop_path"]
            season_num = group["season_number"]
            ep_data = group["episodes"]
            ep_numbers = sorted(list(set(e[0] for e in ep_data)))
            latest_watched_at = max(e[1] for e in ep_data)

            # Determina o rating: rating dos episódios ou da série na lista
            user_rating = group["ratings"][0] if group["ratings"] else group["uli_rating"]

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

            item_id = f"ep_{user_id}_{media_id}_{season_num}_{min(ep_numbers)}_{max(ep_numbers)}_{int(latest_watched_at.timestamp())}"

            feed_items.append(
                FriendFeedItem(
                    id=item_id,
                    user_id=user_id,
                    user_name=user_name,
                    username=username,
                    media_id=media_id,
                    tmdb_id=tmdb_id,
                    media_type=media_type_val,
                    title=title,
                    poster_path=poster_path,
                    backdrop_path=backdrop_path,
                    rating=user_rating,
                    season_number=season_num,
                    episode_numbers=ep_numbers,
                    episodes_label=episodes_label,
                    watched_at=latest_watched_at,
                    action_type=action_type
                )
            )

    # 3. Buscar filmes recentes assistidos pelos amigos (ignorado se filtro for apenas série)
    if filter_type != "tv":
        movie_query = (
            select(
                UserListItem.rating.label("uli_rating"),
                UserListItem.last_watched_at,
                UserListItem.updated_at,
                UserListItem.created_at,
                Media.id.label("media_id"),
                Media.tmdb_id,
                Media.title,
                Media.poster_path,
                Media.backdrop_path,
                User.id.label("user_id"),
                User.name.label("user_name"),
                User.username,
            )
            .join(Media, UserListItem.media_id == Media.id)
            .join(User, UserListItem.user_id == User.id)
        )

        movie_conditions = [
            UserListItem.user_id.in_(friend_ids),
            Media.media_type == "movie",
            or_(
                UserListItem.status == "completed",
                UserListItem.last_watched_at.is_not(None)
            )
        ]
        if cursor_dt:
            movie_conditions.append(
                or_(
                    UserListItem.last_watched_at < cursor_dt,
                    and_(UserListItem.last_watched_at.is_(None), UserListItem.updated_at < cursor_dt)
                )
            )

        movie_query = (
            movie_query
            .where(*movie_conditions)
            .order_by(
                UserListItem.last_watched_at.desc().nulls_last(),
                UserListItem.updated_at.desc().nulls_last()
            )
            .limit(movie_limit)
        )
        movie_res = await db.execute(movie_query)
        movie_records = movie_res.all()

        for row in movie_records:
            watched_at = row.last_watched_at or row.updated_at or row.created_at
            if not watched_at:
                continue

            item_id = f"mov_{row.user_id}_{row.media_id}_{int(watched_at.timestamp())}"
            feed_items.append(
                FriendFeedItem(
                    id=item_id,
                    user_id=row.user_id,
                    user_name=row.user_name,
                    username=row.username,
                    media_id=row.media_id,
                    tmdb_id=row.tmdb_id,
                    media_type="movie",
                    title=row.title,
                    poster_path=row.poster_path,
                    backdrop_path=row.backdrop_path,
                    rating=row.uli_rating,
                    season_number=None,
                    episode_numbers=[],
                    episodes_label=None,
                    watched_at=watched_at,
                    action_type="watched_movie"
                )
            )

    # 4. Ordenar todos os itens por watched_at decrescente
    feed_items.sort(key=lambda x: x.watched_at, reverse=True)

    # 5. Aplicar paginação (por Cursor ou Offset legado)
    if cursor_dt:
        paginated_items = feed_items[:limit]
        has_more = len(feed_items) > limit
        total = len(paginated_items)
    else:
        total = len(feed_items)
        start = (page - 1) * limit
        end = start + limit
        paginated_items = feed_items[start:end]
        has_more = end < total

    next_cursor = None
    if paginated_items and has_more:
        last_watched = paginated_items[-1].watched_at
        if hasattr(last_watched, "isoformat"):
            next_cursor = last_watched.isoformat()
        else:
            next_cursor = str(last_watched)

    return FriendFeedResponse(
        items=paginated_items,
        total=total,
        page=page,
        limit=limit,
        has_more=has_more,
        next_cursor=next_cursor
    )


async def get_recommendation_eligibility(
    db: AsyncSession,
    current_user_id: uuid.UUID,
    tmdb_id: int,
    media_type: str
) -> List[FriendEligibilityItem]:
    # 1. Obter amigos aceitos
    friends = await get_friends_list(db, current_user_id)
    if not friends:
        return []

    friend_ids = [f.id for f in friends]

    # 2. Verificar se a mídia existe no banco local
    media_res = await db.execute(
        select(Media).where(Media.tmdb_id == tmdb_id, Media.media_type == media_type)
    )
    media = media_res.scalar_one_or_none()

    # 3. Mapear status de lista dos amigos
    list_status_by_user = {}
    if media:
        list_res = await db.execute(
            select(UserListItem).where(
                UserListItem.user_id.in_(friend_ids),
                UserListItem.media_id == media.id
            )
        )
        for item in list_res.scalars().all():
            list_status_by_user[item.user_id] = item.status

    # 4. Verificar se já existe recomendação pendente enviada por este usuário
    pending_recs_res = await db.execute(
        select(FriendRecommendation.recipient_id).where(
            FriendRecommendation.sender_id == current_user_id,
            FriendRecommendation.recipient_id.in_(friend_ids),
            FriendRecommendation.tmdb_id == tmdb_id,
            FriendRecommendation.media_type == media_type,
            FriendRecommendation.status == "pending"
        )
    )
    pending_recipient_ids = set(pending_recs_res.scalars().all())

    # 5. Montar lista de elegibilidade
    results: List[FriendEligibilityItem] = []
    for f in friends:
        ineligible_reason = None
        if f.id in pending_recipient_ids:
            ineligible_reason = "PENDING_RECOMMENDATION"
        elif f.id in list_status_by_user:
            st = list_status_by_user[f.id]
            if st in ("completed", "watched"):
                ineligible_reason = "ALREADY_WATCHED"
            else:
                ineligible_reason = "ALREADY_IN_LIST"

        results.append(
            FriendEligibilityItem(
                id=f.id,
                name=f.name,
                username=f.username,
                email=f.email,
                is_eligible=(ineligible_reason is None),
                ineligible_reason=ineligible_reason
            )
        )

    return results


async def send_friend_recommendations(
    db: AsyncSession,
    current_user_id: uuid.UUID,
    payload: SendRecommendationRequest
) -> List[FriendRecommendationResponse]:
    if not payload.recipient_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Selecione ao menos um amigo para enviar a indicação."
        )

    curr_res = await db.execute(select(User).where(User.id == current_user_id))
    current_user = curr_res.scalar_one_or_none()
    if not current_user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Usuário não encontrado.")

    eligibility_list = await get_recommendation_eligibility(
        db, current_user_id, payload.tmdb_id, payload.media_type
    )
    eligibility_map = {item.id: item for item in eligibility_list}

    recipients_res = await db.execute(
        select(User).where(User.id.in_(payload.recipient_ids))
    )
    recipients_dict = {u.id: u for u in recipients_res.scalars().all()}

    media_res = await db.execute(
        select(Media).where(Media.tmdb_id == payload.tmdb_id, Media.media_type == payload.media_type)
    )
    media = media_res.scalar_one_or_none()

    clean_message = payload.message.strip() if payload.message else None
    if clean_message and len(clean_message) > 280:
        clean_message = clean_message[:280]

    sender_display = f"@{current_user.username}" if current_user.username else current_user.name
    created_recs = []

    for recipient_id in payload.recipient_ids:
        eligibility = eligibility_map.get(recipient_id)
        if not eligibility:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"O usuário {recipient_id} não é seu amigo."
            )
        if not eligibility.is_eligible:
            reason_map = {
                "ALREADY_WATCHED": "já assistiu",
                "ALREADY_IN_LIST": "já tem na lista",
                "PENDING_RECOMMENDATION": "já possui uma indicação pendente"
            }
            msg = reason_map.get(eligibility.ineligible_reason, "não está elegível")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"{eligibility.name} {msg} para este título."
            )

        rec = FriendRecommendation(
            sender_id=current_user_id,
            recipient_id=recipient_id,
            media_type=payload.media_type,
            tmdb_id=payload.tmdb_id,
            media_title=payload.media_title,
            media_poster_path=payload.media_poster_path,
            media_release_date=payload.media_release_date,
            media_vote_average=payload.media_vote_average,
            message=clean_message,
            status="pending"
        )
        db.add(rec)
        created_recs.append((rec, recipient_id))

        notif_msg = f"{sender_display} indicou '{payload.media_title}' para você."
        if clean_message:
            snippet = clean_message if len(clean_message) <= 60 else f"{clean_message[:57]}..."
            notif_msg += f' "{snippet}"'

        notif = Notification(
            user_id=recipient_id,
            title="Indicação de Amigo",
            message=notif_msg,
            media_id=media.id if media else None
        )
        db.add(notif)

        poster_icon = f"https://image.tmdb.org/t/p/w185{media.poster_path}" if (media and media.poster_path) else None
        trigger_web_push_background(
            user_id=recipient_id,
            title=f"🎬 Indicação de {sender_display}",
            body=notif_msg,
            url="/my-list?tab=recommendations",
            icon=poster_icon
        )

    await db.commit()

    response_items = []
    sender_summary = UserSummary(
        id=current_user.id,
        name=current_user.name,
        username=current_user.username,
        email=current_user.email
    )
    for rec, recipient_id in created_recs:
        await db.refresh(rec)
        rec_u = recipients_dict.get(recipient_id)
        rec_summary = UserSummary(
            id=rec_u.id,
            name=rec_u.name,
            username=rec_u.username,
            email=rec_u.email
        ) if rec_u else UserSummary(id=recipient_id, name="Amigo")

        response_items.append(
            FriendRecommendationResponse(
                id=rec.id,
                sender=sender_summary,
                recipient=rec_summary,
                media_type=rec.media_type,
                tmdb_id=rec.tmdb_id,
                media_title=rec.media_title,
                media_poster_path=rec.media_poster_path,
                media_release_date=rec.media_release_date,
                media_vote_average=rec.media_vote_average,
                message=rec.message,
                status=rec.status,
                created_at=rec.created_at,
                updated_at=rec.updated_at,
                in_user_list=False,
                user_list_status=None
            )
        )

    return response_items


async def get_received_recommendations(
    db: AsyncSession,
    current_user_id: uuid.UUID,
    status_filter: str = "pending",
    limit: int = 50,
    offset: int = 0
) -> List[FriendRecommendationResponse]:
    stmt = (
        select(FriendRecommendation)
        .where(FriendRecommendation.recipient_id == current_user_id)
        .order_by(FriendRecommendation.created_at.desc())
    )
    if status_filter == "pending":
        stmt = stmt.where(FriendRecommendation.status == "pending")
    elif status_filter == "history":
        stmt = stmt.where(FriendRecommendation.status.in_(["accepted", "dismissed"]))

    stmt = stmt.limit(limit).offset(offset)
    res = await db.execute(stmt)
    recs = res.scalars().all()

    if not recs:
        return []

    sender_ids = list({r.sender_id for r in recs})
    users_res = await db.execute(select(User).where(User.id.in_(sender_ids + [current_user_id])))
    users_dict = {u.id: u for u in users_res.scalars().all()}

    tmdb_keys = {(r.tmdb_id, r.media_type) for r in recs}
    media_filters = [and_(Media.tmdb_id == tid, Media.media_type == mtype) for tid, mtype in tmdb_keys]
    media_res = await db.execute(select(Media).where(or_(*media_filters)))
    medias = media_res.scalars().all()
    media_by_key = {(m.tmdb_id, m.media_type): m for m in medias}

    media_ids = [m.id for m in medias]
    user_list_map = {}
    if media_ids:
        list_res = await db.execute(
            select(UserListItem).where(
                UserListItem.user_id == current_user_id,
                UserListItem.media_id.in_(media_ids)
            )
        )
        for it in list_res.scalars().all():
            user_list_map[it.media_id] = it.status

    current_user = users_dict.get(current_user_id)
    recipient_summary = UserSummary(
        id=current_user.id,
        name=current_user.name,
        username=current_user.username,
        email=current_user.email
    ) if current_user else UserSummary(id=current_user_id, name="Você")

    results: List[FriendRecommendationResponse] = []
    for r in recs:
        snd = users_dict.get(r.sender_id)
        sender_summary = UserSummary(
            id=snd.id,
            name=snd.name,
            username=snd.username,
            email=snd.email
        ) if snd else UserSummary(id=r.sender_id, name="Amigo")

        m = media_by_key.get((r.tmdb_id, r.media_type))
        user_list_status = user_list_map.get(m.id) if m else None

        results.append(
            FriendRecommendationResponse(
                id=r.id,
                sender=sender_summary,
                recipient=recipient_summary,
                media_type=r.media_type,
                tmdb_id=r.tmdb_id,
                media_title=r.media_title,
                media_poster_path=r.media_poster_path,
                media_release_date=r.media_release_date,
                media_vote_average=r.media_vote_average,
                message=r.message,
                status=r.status,
                created_at=r.created_at,
                updated_at=r.updated_at,
                in_user_list=user_list_status is not None,
                user_list_status=user_list_status
            )
        )

    return results


async def update_recommendation_status(
    db: AsyncSession,
    current_user_id: uuid.UUID,
    recommendation_id: uuid.UUID,
    new_status: str
) -> FriendRecommendationResponse:
    if new_status not in ("accepted", "dismissed"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Status inválido.")

    stmt = select(FriendRecommendation).where(
        FriendRecommendation.id == recommendation_id,
        FriendRecommendation.recipient_id == current_user_id
    )
    res = await db.execute(stmt)
    rec = res.scalar_one_or_none()
    if not rec:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Indicação não encontrada.")

    rec.status = new_status
    await db.commit()
    await db.refresh(rec)

    users_res = await db.execute(select(User).where(User.id.in_([rec.sender_id, rec.recipient_id])))
    users_dict = {u.id: u for u in users_res.scalars().all()}

    snd = users_dict.get(rec.sender_id)
    rcp = users_dict.get(rec.recipient_id)

    return FriendRecommendationResponse(
        id=rec.id,
        sender=UserSummary(id=snd.id, name=snd.name, username=snd.username, email=snd.email) if snd else UserSummary(id=rec.sender_id, name="Amigo"),
        recipient=UserSummary(id=rcp.id, name=rcp.name, username=rcp.username, email=rcp.email) if rcp else UserSummary(id=rec.recipient_id, name="Você"),
        media_type=rec.media_type,
        tmdb_id=rec.tmdb_id,
        media_title=rec.media_title,
        media_poster_path=rec.media_poster_path,
        media_release_date=rec.media_release_date,
        media_vote_average=rec.media_vote_average,
        message=rec.message,
        status=rec.status,
        created_at=rec.created_at,
        updated_at=rec.updated_at,
        in_user_list=False,
        user_list_status=None
    )

