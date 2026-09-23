from fastapi import APIRouter, Depends, HTTPException, status, Response, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from datetime import datetime, timedelta, timezone
from app.core.database import get_db
from app.users.models import User, RefreshToken
from app.core.security import verify_password, create_access_token, create_refresh_token, decode_token, hash_token
from app.auth.schemas import LoginRequest, TokenResponse
from app.core.rate_limit import limiter
import uuid
import asyncio

router = APIRouter(prefix="/auth", tags=["auth"])

@router.post("/login", response_model=TokenResponse)
@limiter.limit("10/minute")
async def login(request: Request, response: Response, login_data: LoginRequest, db: AsyncSession = Depends(get_db)):
    email = login_data.email.lower().strip()
    
    result = await db.execute(select(User).where(User.email == email))
    user = result.scalars().first()
    
    if not user:
        await asyncio.sleep(1)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Credenciais inválidas")
        
    if not verify_password(login_data.password, user.password_hash):
        await asyncio.sleep(1)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Credenciais inválidas")
    
    access_token = create_access_token(data={"sub": str(user.id), "role": user.role})
    refresh_token = create_refresh_token(data={"sub": str(user.id)})
    
    rt_db = RefreshToken(
        user_id=user.id,
        token_hash=hash_token(refresh_token),
        expires_at=datetime.now(timezone.utc) + timedelta(days=7)
    )
    db.add(rt_db)
    await db.commit()
    
    from app.core.config import settings
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        secure=not settings.DEBUG,
        samesite="lax" if settings.DEBUG else "none",
        max_age=900,
        path="/"
    )
    response.set_cookie(
        key="refresh_token",
        value=refresh_token,
        httponly=True,
        secure=not settings.DEBUG,
        samesite="lax" if settings.DEBUG else "none",
        max_age=604800,
        path="/api/auth/refresh"
    )
    
    return TokenResponse(
        message="Login realizado com sucesso",
        user={"id": str(user.id), "name": user.name, "email": user.email, "role": user.role}
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
        
    if rt_db.revoked:
        await db.execute(RefreshToken.__table__.update().where(RefreshToken.user_id == rt_db.user_id).values(revoked=True))
        await db.commit()
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Sessão comprometida")
        
    if rt_db.expires_at.replace(tzinfo=timezone.utc) < datetime.now(timezone.utc):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expirado")
        
    user_res = await db.execute(select(User).where(User.id == rt_db.user_id))
    user = user_res.scalars().first()
    
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Usuário não encontrado")
        
    rt_db.revoked = True
    
    new_access_token = create_access_token(data={"sub": str(user.id), "role": user.role})
    new_refresh_token = create_refresh_token(data={"sub": str(user.id)})
    
    new_rt_db = RefreshToken(
        user_id=user.id,
        token_hash=hash_token(new_refresh_token),
        expires_at=datetime.now(timezone.utc) + timedelta(days=7)
    )
    db.add(new_rt_db)
    await db.commit()
    
    from app.core.config import settings
    response.set_cookie(
        key="access_token",
        value=new_access_token,
        httponly=True,
        secure=not settings.DEBUG,
        samesite="lax" if settings.DEBUG else "none",
        max_age=900,
        path="/"
    )
    response.set_cookie(
        key="refresh_token",
        value=new_refresh_token,
        httponly=True,
        secure=not settings.DEBUG,
        samesite="lax" if settings.DEBUG else "none",
        max_age=604800,
        path="/api/auth/refresh"
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
            await db.commit()
            
    from app.core.config import settings
    response.set_cookie(
        key="access_token", value="", max_age=0, path="/",
        secure=not settings.DEBUG, samesite="lax" if settings.DEBUG else "none"
    )
    response.set_cookie(
        key="refresh_token", value="", max_age=0, path="/api/auth/refresh",
        secure=not settings.DEBUG, samesite="lax" if settings.DEBUG else "none"
    )
    
    return {"message": "Logout realizado com sucesso"}
