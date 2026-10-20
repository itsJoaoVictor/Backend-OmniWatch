import uuid
from sqlalchemy import Column, String, DateTime, Integer, Float, Boolean, ForeignKey, func, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from app.core.database import Base

class UserListItem(Base):
    __tablename__ = "user_list_items"
    __allow_unmapped__ = True

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    media_id = Column(UUID(as_uuid=True), ForeignKey("media.id", ondelete="CASCADE"), nullable=False, index=True)
    status = Column(String, nullable=False, default="plan_to_watch", index=True)
    rating = Column(Float, nullable=True)
    rewatch_count = Column(Integer, default=0, nullable=False)
    is_favorite = Column(Boolean, default=False, nullable=False, index=True)
    episodes_watched_count = Column(Integer, default=0, nullable=False, server_default="0")
    total_time_minutes = Column(Integer, default=0, nullable=False, server_default="0")
    last_watched_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    user = relationship("User")
    media = relationship("Media")
    episodes = relationship("UserEpisodeProgress", back_populates="user_list_item", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_user_list_items_user_watched", "user_id", "last_watched_at"),
    )

    # Atributos voláteis calculados em tempo de execução
    is_up_to_date: bool = None
    next_episode: dict = None

class UserEpisodeProgress(Base):
    __tablename__ = "user_episode_progress"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    user_list_item_id = Column(UUID(as_uuid=True), ForeignKey("user_list_items.id", ondelete="CASCADE"), nullable=False, index=True)
    season_number = Column(Integer, nullable=False)
    episode_number = Column(Integer, nullable=False)
    rating = Column(Float, nullable=True)
    rewatch_count = Column(Integer, default=0, nullable=False)
    watched_at = Column(DateTime(timezone=True), server_default=func.now())

    user_list_item = relationship("UserListItem", back_populates="episodes")

    __table_args__ = (
        Index("idx_user_episodes_watched_at", "watched_at"),
        Index("idx_user_episodes_item_watched", "user_list_item_id", "watched_at"),
    )

class UserStatsSummary(Base):
    __tablename__ = "user_stats_summary"

    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    total_time_movies = Column(Integer, default=0, nullable=False, server_default="0")
    total_time_tv = Column(Integer, default=0, nullable=False, server_default="0")
    total_movies_watched = Column(Integer, default=0, nullable=False, server_default="0")
    total_episodes_watched = Column(Integer, default=0, nullable=False, server_default="0")
    average_rating = Column(Float, nullable=True)
    total_rated = Column(Integer, default=0, nullable=False, server_default="0")
    rating_distribution = Column(JSONB, nullable=False, server_default="{}")
    top_genres = Column(JSONB, nullable=False, server_default="[]")
    rankings = Column(JSONB, nullable=False, server_default="{}")
    timeline = Column(JSONB, nullable=False, server_default="[]")
    kpis = Column(JSONB, nullable=False, server_default="{}")
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    user = relationship("User")

