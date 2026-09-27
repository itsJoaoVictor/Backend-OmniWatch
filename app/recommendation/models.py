from sqlalchemy import Column, DateTime, Boolean, ForeignKey, func
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
