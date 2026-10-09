import uuid
from sqlalchemy import Column, String, DateTime, Integer, Boolean, ForeignKey, UniqueConstraint, func, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from app.core.database import Base
from app.users.models import User
from app.media.models import Media

class CustomList(Base):
    __tablename__ = "custom_lists"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    is_public = Column(Boolean, default=True, nullable=False, index=True)
    is_ranked = Column(Boolean, default=False, nullable=False)
    cover_backdrop_path = Column(String(500), nullable=True)
    cover_poster_path = Column(String(500), nullable=True)
    items_count = Column(Integer, default=0, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    user = relationship("User")
    items = relationship(
        "CustomListItem",
        back_populates="custom_list",
        cascade="all, delete-orphan",
        order_by="CustomListItem.position"
    )

class CustomListItem(Base):
    __tablename__ = "custom_list_items"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    list_id = Column(UUID(as_uuid=True), ForeignKey("custom_lists.id", ondelete="CASCADE"), nullable=False, index=True)
    media_id = Column(UUID(as_uuid=True), ForeignKey("media.id", ondelete="SET NULL"), nullable=True, index=True)
    tmdb_id = Column(Integer, nullable=False, index=True)
    media_type = Column(String(20), nullable=False)  # 'movie' | 'tv'
    title = Column(String(255), nullable=False)
    poster_path = Column(String(500), nullable=True)
    backdrop_path = Column(String(500), nullable=True)
    release_date = Column(String(50), nullable=True)
    runtime = Column(Integer, nullable=True, default=0)
    position = Column(Integer, nullable=False, default=0)
    note = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    custom_list = relationship("CustomList", back_populates="items")
    media = relationship("Media")

    __table_args__ = (
        UniqueConstraint("list_id", "tmdb_id", "media_type", name="uq_custom_list_item"),
    )
