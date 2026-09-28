import uuid
from sqlalchemy import Column, String, Integer, DateTime, Boolean, ForeignKey, func, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID, JSONB
from app.core.database import Base

class UserRecommendation(Base):
    __tablename__ = "user_recommendations"

    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
        index=True
    )
    explore_items = Column(JSONB, nullable=False, default=list)
    upcoming_items = Column(JSONB, nullable=False, default=dict)
    personas_items = Column(JSONB, nullable=False, default=list)
    is_stale = Column(Boolean, nullable=False, default=True)
    last_generated_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

class UserDismissedRecommendation(Base):
    """
    Registra obras que o usuário dispensou ('Não tenho interesse').
    Guarda o embedding para penalidade semântica contextual e uma data de expiração (Snooze).
    """
    __tablename__ = "user_dismissed_recommendations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    tmdb_id = Column(Integer, nullable=False, index=True)
    media_type = Column(String, nullable=False)  # 'movie' ou 'tv'
    title = Column(String, nullable=False, default="")
    poster_path = Column(String, nullable=True)
    embedding = Column(JSONB, nullable=True)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        UniqueConstraint("user_id", "tmdb_id", "media_type", name="uq_user_dismissed_media"),
    )
