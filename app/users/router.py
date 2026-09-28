from fastapi import APIRouter, Depends, status, Request
from sqlalchemy.ext.asyncio import AsyncSession
import logging

from app.users.schemas import UserCreate, UserResponse
from app.users.services import create_user
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

from app.auth.dependencies import get_current_user_id
from fastapi import HTTPException
import uuid
from sqlalchemy.future import select
from app.users.models import User

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

