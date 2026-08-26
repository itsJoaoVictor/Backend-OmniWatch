from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from fastapi import HTTPException, status
from app.users.models import User
from app.users.schemas import UserCreate
from app.core.security import get_password_hash

async def create_user(db: AsyncSession, user_data: UserCreate) -> User:
    # Check email duplicate
    result = await db.execute(select(User).where(User.email == user_data.email))
    existing_user = result.scalar_one_or_none()
    
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered"
        )

    # Hash password
    hashed_pwd = get_password_hash(user_data.password)

    # Create user
    new_user = User(
        name=user_data.name,
        email=user_data.email,
        password_hash=hashed_pwd,
        role="user"
    )

    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)

    return new_user
