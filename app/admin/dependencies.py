from fastapi import Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
import uuid

from app.auth.dependencies import get_current_user_id
from app.core.database import get_db
from app.users.models import User

# Apenas este e-mail tem permissão de Administrador para ver o status do servidor
ADMIN_EMAILS = {"jvgf98@gmail.com"}

async def get_current_admin_user(
    current_user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
) -> User:
    try:
        u_uuid = uuid.UUID(current_user_id) if isinstance(current_user_id, str) else current_user_id
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Identificador de usuário inválido"
        )

    result = await db.execute(select(User).filter(User.id == u_uuid))
    user = result.scalars().first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Usuário não encontrado"
        )

    user_email = (user.email or "").strip().lower()
    if user_email not in ADMIN_EMAILS:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Acesso restrito: somente o administrador (jvgf98@gmail.com) pode visualizar o painel do servidor."
        )

    return user
