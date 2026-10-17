import uuid
from sqlalchemy import Column, String, DateTime, Integer, Boolean, ForeignKey, UniqueConstraint, func, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from app.core.database import Base
from app.users.models import User
from app.media.models import Media


class Universe(Base):
    __tablename__ = "universes"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    universe_key = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    overview = Column(String, nullable=True)
    poster_path = Column(String, nullable=True)
    backdrop_path = Column(String, nullable=True)
    items_count = Column(Integer, default=0, nullable=False)
    movies_count = Column(Integer, default=0, nullable=False)
    tv_count = Column(Integer, default=0, nullable=False)
    collections_count = Column(Integer, default=0, nullable=False)
    last_synced_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    items = relationship("UniverseItem", back_populates="universe", cascade="all, delete-orphan")
    followers = relationship("UserUniverse", back_populates="universe", cascade="all, delete-orphan")


class UniverseItem(Base):
    __tablename__ = "universe_items"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    universe_id = Column(UUID(as_uuid=True), ForeignKey("universes.id", ondelete="CASCADE"), nullable=False, index=True)
    media_id = Column(UUID(as_uuid=True), ForeignKey("media.id", ondelete="SET NULL"), nullable=True, index=True)
    tmdb_id = Column(Integer, nullable=False, index=True)
    media_type = Column(String, nullable=False, default="movie", index=True)
    title = Column(String, nullable=False)
    release_date = Column(String, nullable=True)
    poster_path = Column(String, nullable=True)
    backdrop_path = Column(String, nullable=True)
    collection_name = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    universe = relationship("Universe", back_populates="items")
    media = relationship("Media")

    __table_args__ = (
        UniqueConstraint("universe_id", "tmdb_id", "media_type", name="uq_universe_item_universe_tmdb_type"),
    )


class UserUniverse(Base):
    __tablename__ = "user_universes"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    universe_id = Column(UUID(as_uuid=True), ForeignKey("universes.id", ondelete="CASCADE"), nullable=False, index=True)
    auto_add = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User")
    universe = relationship("Universe", back_populates="followers")

    __table_args__ = (
        UniqueConstraint("user_id", "universe_id", name="uq_user_universe"),
    )


class UserUniverseSuggestion(Base):
    __tablename__ = "user_universe_suggestions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    universe_key = Column(String, nullable=False, index=True)
    name = Column(String, nullable=False)
    overview = Column(String, nullable=True)
    poster_path = Column(String, nullable=True)
    backdrop_path = Column(String, nullable=True)
    total_items = Column(Integer, default=0, nullable=False)
    movies_count = Column(Integer, default=0, nullable=False)
    tv_count = Column(Integer, default=0, nullable=False)
    collections_count = Column(Integer, default=0, nullable=False)
    items_in_list = Column(Integer, default=0, nullable=False)
    matched_titles = Column(JSON, default=list, nullable=False)
    discovered_members = Column(JSON, default=list, nullable=False)
    is_dismissed = Column(Boolean, default=False, nullable=False, index=True)
    dismissed_items_count = Column(Integer, default=0, nullable=True)
    dismissed_media_types = Column(JSON, default=list, nullable=True)
    dismissed_collection_ids = Column(JSON, default=list, nullable=True)
    is_reactivated = Column(Boolean, default=False, nullable=False, index=True)
    reactivation_reason = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    user = relationship("User")

    __table_args__ = (
        UniqueConstraint("user_id", "universe_key", name="uq_user_universe_suggestion"),
    )


class UserUniverseIgnoredItem(Base):
    __tablename__ = "user_universe_ignored_items"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    universe_id = Column(UUID(as_uuid=True), ForeignKey("universes.id", ondelete="CASCADE"), nullable=False, index=True)
    tmdb_id = Column(Integer, nullable=False, index=True)
    media_type = Column(String, nullable=False, default="movie", index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User")
    universe = relationship("Universe")

    __table_args__ = (
        UniqueConstraint("user_id", "universe_id", "tmdb_id", "media_type", name="uq_user_universe_ignored_item"),
    )

