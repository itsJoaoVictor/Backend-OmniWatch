import uuid
from sqlalchemy import Column, String, DateTime, Integer, JSON, func, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from app.core.database import Base

class Media(Base):
    __tablename__ = "media"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    tmdb_id = Column(Integer, unique=True, index=True, nullable=False)
    media_type = Column(String, nullable=False, index=True)
    title = Column(String, nullable=False)
    poster_path = Column(String, nullable=True)
    backdrop_path = Column(String, nullable=True)
    original_language = Column(String, nullable=True)
    # Lista de {"id": int, "name": str} conforme padrão TMDB
    genres = Column(JSON, default=list)
    directors = Column(JSON, default=list)
    main_cast = Column(JSON, default=list)
    keywords = Column(JSON, default=list)
    # Fase 4: person_id do TMDB para atores e diretores (lista de {name, person_id})
    cast_ids = Column(JSON, nullable=True)
    crew_ids = Column(JSON, nullable=True)
    # Fase 6: Embedding semântico denso (lista de floats)
    embedding = Column(JSON, nullable=True)
    release_date = Column(String, nullable=True)
    runtime = Column(Integer, nullable=True, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    releases = relationship("MediaRelease", back_populates="media", cascade="all, delete-orphan")

class MediaRelease(Base):
    __tablename__ = "media_releases"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    media_id = Column(UUID(as_uuid=True), ForeignKey("media.id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String, nullable=False)
    description = Column(String, nullable=True)
    release_date = Column(DateTime(timezone=True), nullable=False, index=True)
    season_number = Column(Integer, nullable=True)
    episode_number = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    media = relationship("Media", back_populates="releases")
