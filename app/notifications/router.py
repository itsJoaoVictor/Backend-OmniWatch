from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import List, Optional
from pydantic import BaseModel, UUID4
from datetime import datetime

from app.core.database import get_db
from app.auth.dependencies import get_current_user_id
from app.users.models import User
from app.notifications.models import Notification

router = APIRouter(prefix="/notifications", tags=["notifications"])

class NotificationSchema(BaseModel):
    id: UUID4
    title: str
    message: str
    is_read: bool
    media_id: Optional[UUID4] = None
    created_at: datetime

    class Config:
        from_attributes = True

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
