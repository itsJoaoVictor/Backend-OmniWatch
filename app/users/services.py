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

