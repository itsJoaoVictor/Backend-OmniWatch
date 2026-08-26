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
