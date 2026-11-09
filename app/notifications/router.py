from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import delete, update
from typing import List, Optional
from pydantic import BaseModel, UUID4
from datetime import datetime
import uuid

from app.core.database import get_db
from app.auth.dependencies import get_current_user_id
from app.users.models import User
from app.notifications.models import Notification, PushSubscription
from app.notifications.push_service import (
    VAPID_PUBLIC_KEY,
    PYWEBPUSH_AVAILABLE,
    send_web_push_notification,
    trigger_web_push_background,
)

router = APIRouter(prefix="/notifications", tags=["notifications"])

class PushSubscriptionKeys(BaseModel):
    p256dh: str
    auth: str

class PushSubscriptionCreate(BaseModel):
    endpoint: str
    keys: PushSubscriptionKeys
    user_agent: Optional[str] = None

class NotificationSchema(BaseModel):
    id: UUID4
    title: str
    message: str
    is_read: bool
    media_id: Optional[UUID4] = None
    created_at: datetime

    class Config:
        from_attributes = True

@router.get("", response_model=List[NotificationSchema], include_in_schema=False)
@router.get("/", response_model=List[NotificationSchema])
async def get_notifications(
    unread_only: bool = False,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    stmt = select(Notification).where(Notification.user_id == user_id).order_by(Notification.created_at.desc())
    if unread_only:
        stmt = stmt.where(Notification.is_read == False)
    
    result = await db.execute(stmt)
    notifications = result.scalars().all()
    return notifications

@router.post("/read-all")
async def mark_all_as_read(
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    stmt = (
        update(Notification)
        .where(Notification.user_id == user_id, Notification.is_read == False)
        .values(is_read=True)
    )
    await db.execute(stmt)
    await db.commit()
    return {"success": True}

@router.delete("/clear-all")
@router.delete("", include_in_schema=False)
@router.delete("/")
async def clear_all_notifications(
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    stmt = delete(Notification).where(Notification.user_id == user_id)
    await db.execute(stmt)
    await db.commit()
    return {"success": True, "message": "All notifications cleared"}

@router.post("/{notification_id}/read")
async def mark_as_read(
    notification_id: UUID4,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    stmt = select(Notification).where(Notification.id == notification_id, Notification.user_id == user_id)
    result = await db.execute(stmt)
    notification = result.scalars().first()
    
    if not notification:
        raise HTTPException(status_code=404, detail="Not found")
        
    notification.is_read = True
    await db.commit()
    return {"success": True}

@router.delete("/{notification_id}")
async def delete_notification(
    notification_id: UUID4,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    stmt = select(Notification).where(Notification.id == notification_id, Notification.user_id == user_id)
    result = await db.execute(stmt)
    notification = result.scalars().first()
    
    if not notification:
        raise HTTPException(status_code=404, detail="Not found")
        
    await db.delete(notification)
    await db.commit()
    return {"success": True}


@router.get("/push-key")
async def get_push_public_key():
    """
    Retorna a chave pública VAPID para registro do Service Worker no frontend.
    """
    return {"public_key": VAPID_PUBLIC_KEY}


@router.post("/push-subscription")
async def save_push_subscription(
    payload: PushSubscriptionCreate,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    """
    Cadastra ou atualiza o endpoint de Web Push de um aparelho do usuário.
    """
    u_uuid = uuid.UUID(user_id) if isinstance(user_id, str) else user_id
    
    # Verifica se o endpoint já existe no banco
    stmt = select(PushSubscription).where(PushSubscription.endpoint == payload.endpoint)
    res = await db.execute(stmt)
    existing = res.scalar_one_or_none()

    if existing:
        existing.user_id = u_uuid
        existing.p256dh = payload.keys.p256dh
        existing.auth = payload.keys.auth
        if payload.user_agent:
            existing.user_agent = payload.user_agent
    else:
        new_sub = PushSubscription(
            user_id=u_uuid,
            endpoint=payload.endpoint,
            p256dh=payload.keys.p256dh,
            auth=payload.keys.auth,
            user_agent=payload.user_agent
        )
        db.add(new_sub)

    await db.commit()
    return {"success": True, "message": "Inscrição Web Push salva com sucesso."}


@router.delete("/push-subscription")
async def remove_push_subscription(
    endpoint: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    """
    Remove a inscrição de Web Push (ex: logout ou desativação de notificações no aparelho).
    """
    u_uuid = uuid.UUID(user_id) if isinstance(user_id, str) else user_id
    stmt = delete(PushSubscription).where(PushSubscription.user_id == u_uuid)
    if endpoint:
        stmt = stmt.where(PushSubscription.endpoint == endpoint)
    
    await db.execute(stmt)
    await db.commit()
    return {"success": True, "message": "Inscrição removida."}


@router.post("/push-test")
async def test_push_notification(
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    """
    Envia uma notificação push de teste para todos os aparelhos registrados do usuário logado.
    """
    u_uuid = uuid.UUID(user_id) if isinstance(user_id, str) else user_id

    if not PYWEBPUSH_AVAILABLE:
        return {
            "success": False,
            "message": "O servidor ainda não possui a biblioteca de Web Push instalada (pywebpush no Python do servidor)."
        }

    stmt = select(PushSubscription).where(PushSubscription.user_id == u_uuid)
    res = await db.execute(stmt)
    subscriptions = res.scalars().all()
    if not subscriptions:
        return {
            "success": False,
            "message": "Nenhum aparelho ativo registrado no banco. Toque em 'Desativar' e depois 'Ativar' no card acima primeiro."
        }

    count, last_error = await send_web_push_notification(
        db,
        user_id=u_uuid,
        title="OmniWatch 🎬",
        body="As notificações push no seu celular estão ativadas e funcionando perfeitamente!",
        url="/my-list?tab=recommendations"
    )
    if count == 0:
        msg = f"Falha ao despachar notificação: {last_error}" if last_error else "Falha ao despachar notificação para o serviço do Google/Apple. Verifique se o aparelho ainda está conectado."
        return {
            "success": False,
            "message": msg
        }
    return {
        "success": True,
        "message": f"Notificação de teste enviada para {count} aparelho(s)!"
    }


