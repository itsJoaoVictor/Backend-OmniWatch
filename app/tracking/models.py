import uuid
from sqlalchemy import Column, String, DateTime, Integer, Float, ForeignKey, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from app.core.database import Base

class UserListItem(Base):
    __tablename__ = "user_list_items"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    media_id = Column(UUID(as_uuid=True), ForeignKey("media.id", ondelete="CASCADE"), nullable=False, index=True)
    status = Column(String, nullable=False, default="plan_to_watch", index=True)
    rating = Column(Float, nullable=True)
    rewatch_count = Column(Integer, default=0, nullable=False)
    last_watched_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    user = relationship("User")
    media = relationship("Media")

class UserEpisodeProgress(Base):
    __tablename__ = "user_episode_progress"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    user_list_item_id = Column(UUID(as_uuid=True), ForeignKey("user_list_items.id", ondelete="CASCADE"), nullable=False, index=True)
    season_number = Column(Integer, nullable=False)
    episode_number = Column(Integer, nullable=False)
    rating = Column(Float, nullable=True)
    watched_at = Column(DateTime(timezone=True), server_default=func.now())

    user_list_item = relationship("UserListItem")
