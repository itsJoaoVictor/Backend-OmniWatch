"""add_friend_recommendations_table

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-11-08 21:12:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = 'b2c3d4e5f6a7'
down_revision: Union[str, Sequence[str], None] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    tables = insp.get_table_names()

    if 'friend_recommendations' not in tables:
        op.create_table(
            'friend_recommendations',
            sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
            sa.Column('sender_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
            sa.Column('recipient_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
            sa.Column('media_type', sa.String(length=10), nullable=False),
            sa.Column('tmdb_id', sa.Integer(), nullable=False),
            sa.Column('media_title', sa.String(length=255), nullable=False),
            sa.Column('media_poster_path', sa.String(length=255), nullable=True),
            sa.Column('media_release_date', sa.String(length=30), nullable=True),
            sa.Column('media_vote_average', sa.Float(), nullable=True),
            sa.Column('message', sa.String(length=280), nullable=True),
            sa.Column('status', sa.String(length=20), server_default='pending', nullable=False),
            sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
        )
        op.create_index('ix_friend_recommendations_id', 'friend_recommendations', ['id'])
        op.create_index('ix_friend_recommendations_sender_id', 'friend_recommendations', ['sender_id'])
        op.create_index('ix_friend_recommendations_recipient_id', 'friend_recommendations', ['recipient_id'])
        op.create_index('ix_friend_recommendations_tmdb_id', 'friend_recommendations', ['tmdb_id'])
        op.create_index('ix_friend_recommendations_status', 'friend_recommendations', ['status'])
        op.create_index('ix_friend_recommendations_created_at', 'friend_recommendations', ['created_at'])
        op.create_index('ix_friend_recs_recipient_status', 'friend_recommendations', ['recipient_id', 'status'])
        op.create_index('ix_friend_recs_sender_media', 'friend_recommendations', ['sender_id', 'media_type', 'tmdb_id'])


def downgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    tables = insp.get_table_names()

    if 'friend_recommendations' in tables:
        op.drop_table('friend_recommendations')
