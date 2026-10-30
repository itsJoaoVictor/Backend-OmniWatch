from fastapi import APIRouter, Depends, HTTPException, status, Response, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from datetime import datetime, timedelta, timezone
from app.core.database import get_db
from app.users.models import User, RefreshToken
from app.core.security import verify_password, get_password_hash, create_access_token, create_refresh_token, decode_token, hash_token
from app.auth.schemas import LoginRequest, TokenResponse, ForgotPasswordRequest
from app.core.rate_limit import limiter
from app.core.config import settings
import uuid
import asyncio
from sqlalchemy import func

router = APIRouter(prefix="/auth", tags=["auth"])

ACCESS_TOKEN_COOKIE_MAX_AGE = 7 * 24 * 3600   # 7 dias (604800s)
REFRESH_TOKEN_COOKIE_MAX_AGE = 365 * 24 * 3600 # 1 ano (31536000s)
REFRESH_GRACE_PERIOD_SECONDS = 60              # 60s de tolerância para concorrência de rede/tabs

@router.post("/login", response_model=TokenResponse)
@limiter.limit("10/minute")
async def login(request: Request, response: Response, login_data: LoginRequest, db: AsyncSession = Depends(get_db)):
    identifier = login_data.email.lower().strip()
    
    result = await db.execute(
        select(User).where(
            (func.lower(User.email) == identifier) | 
            (func.lower(User.username) == identifier)
        )
    )
    user = result.scalars().first()
    
    if not user:
        await asyncio.sleep(1)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Credenciais inválidas")
        
    if not verify_password(login_data.password, user.password_hash):
        await asyncio.sleep(1)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Credenciais inválidas")
    
    access_token = create_access_token(data={"sub": str(user.id), "role": user.role})
    refresh_token = create_refresh_token(data={"sub": str(user.id)})
    
    now = datetime.now(timezone.utc)
    rt_db = RefreshToken(
        user_id=user.id,
        token_hash=hash_token(refresh_token),
        expires_at=now + timedelta(days=365)
    )
    db.add(rt_db)
    await db.commit()
    
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        secure=not settings.DEBUG,
        samesite="lax" if settings.DEBUG else "none",
        max_age=ACCESS_TOKEN_COOKIE_MAX_AGE,
        path="/"
    )
    response.set_cookie(
        key="refresh_token",
        value=refresh_token,
        httponly=True,
        secure=not settings.DEBUG,
        samesite="lax" if settings.DEBUG else "none",
        max_age=REFRESH_TOKEN_COOKIE_MAX_AGE,
        path="/"
    )
    
    return TokenResponse(
        message="Login realizado com sucesso",
        user={
            "id": str(user.id),
            "name": user.name,
            "email": user.email,
            "username": user.username,
            "role": user.role,
            "last_seen_release_id": user.last_seen_release_id or 0,
        }
    )

@router.post("/refresh")
async def refresh_token(request: Request, response: Response, db: AsyncSession = Depends(get_db)):
    old_refresh_token = request.cookies.get("refresh_token")
    if not old_refresh_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh token ausente")
        
    payload = decode_token(old_refresh_token)
    if not payload or payload.get("type") != "refresh":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token inválido")
        
    result = await db.execute(select(RefreshToken).where(RefreshToken.token_hash == hash_token(old_refresh_token)))
    rt_db = result.scalars().first()
    
    if not rt_db:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token não encontrado")
        
    now = datetime.now(timezone.utc)
    
    # Grace Period para requisições concorrentes / retries de rede
    if rt_db.revoked:
        is_within_grace_period = False
        if rt_db.revoked_at:
            revoked_time = rt_db.revoked_at if rt_db.revoked_at.tzinfo else rt_db.revoked_at.replace(tzinfo=timezone.utc)
            if (now - revoked_time).total_seconds() <= REFRESH_GRACE_PERIOD_SECONDS:
                is_within_grace_period = True
                
        if not is_within_grace_period:
            # Sessão expirada/revogada além do período de tolerância:
            # NUNCA revoga as sessões de outros dispositivos! Rejeita apenas esta tentativa.
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Sessão expirada")
        
    expires_at = rt_db.expires_at if rt_db.expires_at.tzinfo else rt_db.expires_at.replace(tzinfo=timezone.utc)
    if expires_at < now:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expirado")
        
    user_res = await db.execute(select(User).where(User.id == rt_db.user_id))
    user = user_res.scalars().first()
    
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Usuário não encontrado")
        
    # Marca o token antigo como revogado com a data/hora para controle de grace period
    rt_db.revoked = True
    rt_db.revoked_at = now
    
    new_access_token = create_access_token(data={"sub": str(user.id), "role": user.role})
    new_refresh_token = create_refresh_token(data={"sub": str(user.id)})
    
    new_rt_db = RefreshToken(
        user_id=user.id,
        token_hash=hash_token(new_refresh_token),
        expires_at=now + timedelta(days=365)
    )
    db.add(new_rt_db)
    await db.commit()
    
    response.set_cookie(
        key="access_token",
        value=new_access_token,
        httponly=True,
        secure=not settings.DEBUG,
        samesite="lax" if settings.DEBUG else "none",
        max_age=ACCESS_TOKEN_COOKIE_MAX_AGE,
        path="/"
    )
    response.set_cookie(
        key="refresh_token",
        value=new_refresh_token,
        httponly=True,
        secure=not settings.DEBUG,
        samesite="lax" if settings.DEBUG else "none",
        max_age=REFRESH_TOKEN_COOKIE_MAX_AGE,
        path="/"
    )
    
    return {"message": "Sessão renovada"}

@router.post("/logout")
async def logout(request: Request, response: Response, db: AsyncSession = Depends(get_db)):
    old_refresh_token = request.cookies.get("refresh_token")
    if old_refresh_token:
        result = await db.execute(select(RefreshToken).where(RefreshToken.token_hash == hash_token(old_refresh_token)))
        rt_db = result.scalars().first()
        if rt_db and not rt_db.revoked:
            rt_db.revoked = True
            rt_db.revoked_at = datetime.now(timezone.utc)
            await db.commit()
            
    # Limpa cookies em path="/" e também no path legado "/api/auth/refresh" para garantir remoção
    response.set_cookie(
        key="access_token", value="", max_age=0, path="/",
        secure=not settings.DEBUG, samesite="lax" if settings.DEBUG else "none"
    )
    response.set_cookie(
        key="refresh_token", value="", max_age=0, path="/",
        secure=not settings.DEBUG, samesite="lax" if settings.DEBUG else "none"
    )
    response.set_cookie(
        key="refresh_token", value="", max_age=0, path="/api/auth/refresh",
        secure=not settings.DEBUG, samesite="lax" if settings.DEBUG else "none"
    )
    
    return {"message": "Logout realizado com sucesso"}

@router.post("/forgot-password")
@limiter.limit("5/minute")
async def forgot_password(request: Request, payload: ForgotPasswordRequest, db: AsyncSession = Depends(get_db)):
    email = payload.email.lower().strip()
    username = payload.username.lower().strip()

    result = await db.execute(
        select(User).where(
            (func.lower(User.email) == email) &
            (func.lower(User.username) == username)
        )
    )
    user = result.scalars().first()

    if not user:
        await asyncio.sleep(1)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Não encontramos uma conta com a combinação de e-mail e nome de usuário informados."
        )

    user.password_hash = get_password_hash(payload.new_password)
    user.failed_login_attempts = 0
    user.locked_until = None

    # Na troca de senha explícita, revoga todas as sessões anteriores por segurança
    await db.execute(
        RefreshToken.__table__.update()
        .where(RefreshToken.user_id == user.id)
        .values(revoked=True, revoked_at=datetime.now(timezone.utc))
    )
    await db.commit()

    return {"message": "Senha redefinida com sucesso."}
