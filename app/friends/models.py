import uuid
from sqlalchemy import Column, String, DateTime, Integer, Float, ForeignKey, UniqueConstraint, Index, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from app.core.database import Base

class Friendship(Base):
    __tablename__ = "friendships"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    requester_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    addressee_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    status = Column(String(20), server_default="pending", default="pending", nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    requester = relationship("User", foreign_keys=[requester_id])
    addressee = relationship("User", foreign_keys=[addressee_id])

    __table_args__ = (
        UniqueConstraint("requester_id", "addressee_id", name="uq_friendship_requester_addressee"),
        Index("ix_friendships_addressee_status", "addressee_id", "status"),
        Index("ix_friendships_requester_status", "requester_id", "status"),
    )

class FriendRecommendation(Base):
    __tablename__ = "friend_recommendations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    sender_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    recipient_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    media_type = Column(String(10), nullable=False)  # 'movie' ou 'tv'
    tmdb_id = Column(Integer, nullable=False, index=True)
    media_title = Column(String(255), nullable=False)
    media_poster_path = Column(String(255), nullable=True)
    media_release_date = Column(String(30), nullable=True)
    media_vote_average = Column(Float, nullable=True)
    message = Column(String(280), nullable=True)
    status = Column(String(20), server_default="pending", default="pending", nullable=False, index=True)  # pending, accepted, dismissed
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    sender = relationship("User", foreign_keys=[sender_id])
    recipient = relationship("User", foreign_keys=[recipient_id])

    __table_args__ = (
        Index("ix_friend_recs_recipient_status", "recipient_id", "status"),
        Index("ix_friend_recs_sender_media", "sender_id", "media_type", "tmdb_id"),
    )

