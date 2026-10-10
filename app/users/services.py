from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func
from fastapi import HTTPException, status
from app.users.models import User
from app.users.schemas import UserCreate, RESERVED_USERNAMES
from app.core.security import get_password_hash
import uuid

async def create_user(db: AsyncSession, user_data: UserCreate) -> User:
    # Check email duplicate
    result = await db.execute(select(User).where(User.email == user_data.email))
    existing_user = result.scalar_one_or_none()
    
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered"
        )

    # Check username duplicate if provided
    if user_data.username:
        uname_clean = user_data.username.lower().strip()
        result_uname = await db.execute(select(User).where(func.lower(User.username) == uname_clean))
        if result_uname.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Nome de usuário já está em uso"
            )

    # Hash password
    hashed_pwd = get_password_hash(user_data.password)

    # Create user
    new_user = User(
        name=user_data.name,
        email=user_data.email,
        username=user_data.username.lower().strip() if user_data.username else None,
        password_hash=hashed_pwd,
        role="user"
    )

    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)

    return new_user

async def check_username_availability(
    db: AsyncSession,
    username: str,
    exclude_user_id: uuid.UUID | None = None
) -> tuple[bool, str]:
    clean = username.strip().lower()
    if len(clean) < 3 or len(clean) > 30:
        return False, "O nome de usuário deve ter entre 3 e 30 caracteres."
    
    import re
    if not re.match(r"^[a-z0-9._-]+$", clean):
        return False, "Apenas letras minúsculas, números, '.', '-' e '_' são permitidos."
        
    if clean in RESERVED_USERNAMES:
        return False, "Este nome de usuário é reservado."

    query = select(User).where(func.lower(User.username) == clean)
    if exclude_user_id:
        query = query.where(User.id != exclude_user_id)
        
    res = await db.execute(query)
    existing = res.scalar_one_or_none()
    if existing:
        return False, "Nome de usuário já está em uso."
        
    return True, "Nome de usuário disponível."

async def update_user_username(
    db: AsyncSession,
    user_id: uuid.UUID,
    new_username: str
) -> User:
    clean = new_username.strip().lower()
    
    # Busca usuário
    res = await db.execute(select(User).where(User.id == user_id))
    user = res.scalar_one_or_none()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Usuário não encontrado"
        )
        
    if user.username and user.username.lower() == clean:
        return user
        
    available, msg = await check_username_availability(db, clean, exclude_user_id=user_id)
    if not available:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=msg
        )
        
    user.username = clean
    await db.commit()
    await db.refresh(user)
    return user

async def get_public_user_profile(
    db: AsyncSession,
    identifier: str,
    current_user_id: uuid.UUID
):
    from sqlalchemy import or_, and_
    from app.friends.models import Friendship
    from app.tracking.models import UserListItem
    from app.media.models import Media
    from app.custom_lists.models import CustomList
    from app.users.schemas import (
        PublicUserProfileResponse,
        PublicUserProfileStats,
        PublicUserProfileList,
        PublicUserProfileFavorite,
        PublicUserTrackedItem,
    )

    clean_id = identifier.strip().lstrip("@")
    target_user = None

    # Tenta resolver por UUID se for formato válido
    try:
        target_uuid = uuid.UUID(clean_id)
        res = await db.execute(select(User).where(User.id == target_uuid))
        target_user = res.scalar_one_or_none()
    except (ValueError, AttributeError):
        pass

    # Se não encontrado por UUID, busca por username case-insensitive
    if not target_user:
        res = await db.execute(select(User).where(func.lower(User.username) == clean_id.lower()))
        target_user = res.scalar_one_or_none()

    if not target_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Usuário não encontrado"
        )

    # 1. Determina status de relacionamento
    rel_status = "none"
    f_id = None

    if target_user.id == current_user_id:
        rel_status = "self"
    else:
        rel_query = select(Friendship).where(
            or_(
                and_(Friendship.requester_id == current_user_id, Friendship.addressee_id == target_user.id),
                and_(Friendship.addressee_id == current_user_id, Friendship.requester_id == target_user.id)
            )
        )
        rel_res = await db.execute(rel_query)
        friendship = rel_res.scalar_one_or_none()

        if friendship:
            f_id = friendship.id
            if friendship.status == "accepted":
                rel_status = "friends"
            elif friendship.status == "pending":
                if friendship.requester_id == current_user_id:
                    rel_status = "pending_sent"
                else:
                    rel_status = "pending_received"

    # 2. Estatísticas de consumo
    try:
        from app.tracking.services import get_user_statistics
        stats_data = await get_user_statistics(db, str(target_user.id))
    except Exception:
        stats_data = {}

    total_movies = stats_data.get("totalMovies", 0) if isinstance(stats_data, dict) else 0
    total_episodes = stats_data.get("totalEpisodes", 0) if isinstance(stats_data, dict) else 0
    total_time_minutes = stats_data.get("totalTime", 0) if isinstance(stats_data, dict) else 0
    total_time_hours = round(total_time_minutes / 60, 1)
    kpis = stats_data.get("kpis", {}) if isinstance(stats_data, dict) else {}
    average_rating = kpis.get("averageRating", 0.0)
    completed_count = kpis.get("completedCount", 0)

    stats_obj = PublicUserProfileStats(
        total_movies=total_movies,
        total_episodes=total_episodes,
        total_time_minutes=total_time_minutes,
        total_time_hours=total_time_hours,
        average_rating=average_rating,
        completed_count=completed_count
    )

    # 3. Listas customizadas públicas
    lists_res = await db.execute(
        select(CustomList)
        .where(
            CustomList.user_id == target_user.id,
            CustomList.is_public == True
        )
        .order_by(CustomList.created_at.desc())
        .limit(20)
    )
    custom_lists_rows = lists_res.scalars().all()
    public_lists = [
        PublicUserProfileList.model_validate(cl)
        for cl in custom_lists_rows
    ]

    # 4. Obras favoritas
    fav_res = await db.execute(
        select(UserListItem, Media)
        .join(Media, UserListItem.media_id == Media.id)
        .where(
            UserListItem.user_id == target_user.id,
            UserListItem.is_favorite == True
        )
        .order_by(UserListItem.updated_at.desc())
        .limit(30)
    )
    fav_rows = fav_res.all()
    favorite_media = [
        PublicUserProfileFavorite(
            media_id=media.id,
            tmdb_id=media.tmdb_id,
            media_type=media.media_type,
            title=media.title,
            poster_path=media.poster_path,
            rating=item.rating
        )
        for item, media in fav_rows
    ]

    # 5. Obras acompanhadas (My List)
    try:
        from app.tracking.services import get_user_list
        user_list_items = await get_user_list(db, str(target_user.id))
    except Exception:
        user_list_items = []

    tracked_media = [
        PublicUserTrackedItem(
            id=item.id,
            media_id=item.media_id,
            tmdb_id=item.media.tmdb_id,
            media_type=item.media.media_type,
            title=item.media.title,
            poster_path=item.media.poster_path,
            backdrop_path=item.media.backdrop_path,
            status=item.status,
            rating=item.rating,
            is_favorite=item.is_favorite,
            last_watched_at=item.last_watched_at
        )
        for item in user_list_items
        if item.media is not None
    ]

    return PublicUserProfileResponse(
        id=target_user.id,
        name=target_user.name,
        username=target_user.username,
        created_at=target_user.created_at,
        relationship_status=rel_status,
        friendship_id=f_id,
        stats=stats_obj,
        public_lists=public_lists,
        favorite_media=favorite_media,
        tracked_media=tracked_media
    )


