from fastapi import APIRouter, Depends, status, Request, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
import logging
import uuid

from app.users.schemas import (
    UserCreate,
    UserResponse,
    UsernameUpdate,
    UsernameCheckResponse,
    PublicUserProfileResponse,
)
from app.users.services import (
    create_user,
    update_user_username,
    check_username_availability,
    get_public_user_profile,
)
from app.users.models import User
from app.auth.dependencies import get_current_user_id
from app.core.database import get_db
from slowapi import Limiter
from slowapi.util import get_remote_address

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)
logger = logging.getLogger("omniwatch.audit")
logger.setLevel(logging.INFO)

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit("5/minute")
async def register_user(request: Request, user: UserCreate, db: AsyncSession = Depends(get_db)):
    client_ip = request.client.host if request.client else "Unknown IP"
    try:
        new_user = await create_user(db, user)
        logger.info(f"AUDIT SUCCESS: Cadastro realizado com sucesso | Email: {user.email} | IP: {client_ip}")
        return new_user
    except Exception as e:
        logger.error(f"AUDIT FAILURE: Falha ao cadastrar usuário | Email: {user.email} | IP: {client_ip} | Motivo: {str(e)}")
        raise

@router.get("/me", response_model=UserResponse)
async def get_current_user_profile(
    current_user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    try:
        u_uuid = uuid.UUID(current_user_id) if isinstance(current_user_id, str) else current_user_id
    except ValueError:
        raise HTTPException(status_code=400, detail="ID de usuário inválido")

    res = await db.execute(select(User).where(User.id == u_uuid))
    user = res.scalars().first()
    if not user:
        raise HTTPException(status_code=404, detail="Usuário não encontrado")
    return user

@router.get("/check-username", response_model=UsernameCheckResponse)
@limiter.limit("60/minute")
async def check_username(
    request: Request,
    username: str = Query(..., min_length=1, max_length=50),
    db: AsyncSession = Depends(get_db)
):
    available, msg = await check_username_availability(db, username)
    return UsernameCheckResponse(available=available, message=msg)

@router.patch("/me/username", response_model=UserResponse)
@limiter.limit("15/minute")
async def update_username(
    request: Request,
    body: UsernameUpdate,
    current_user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    try:
        u_uuid = uuid.UUID(current_user_id) if isinstance(current_user_id, str) else current_user_id
    except ValueError:
        raise HTTPException(status_code=400, detail="ID de usuário inválido")

    updated_user = await update_user_username(db, u_uuid, body.username)
    return updated_user

@router.put("/me/username", response_model=UserResponse)
@limiter.limit("15/minute")
async def update_username_put(
    request: Request,
    body: UsernameUpdate,
    current_user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    return await update_username(request, body, current_user_id, db)

@router.get("/profile/{identifier}", response_model=PublicUserProfileResponse)
@limiter.limit("60/minute")
async def get_user_public_profile(
    request: Request,
    identifier: str,
    current_user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    try:
        u_uuid = uuid.UUID(current_user_id) if isinstance(current_user_id, str) else current_user_id
    except ValueError:
        raise HTTPException(status_code=400, detail="ID de usuário inválido")

    return await get_public_user_profile(db, identifier=identifier, current_user_id=u_uuid)


