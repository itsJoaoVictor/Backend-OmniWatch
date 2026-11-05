"""add_etapas_4_5_6_recommendation_fields

Revision ID: a1b2c3d4e5f6
Revises: f3a4b5c6d7e8
Create Date: 2026-11-05 01:50:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, Sequence[str], None] = 'f3a4b5c6d7e8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)

    # 1. user_recommendations
    ur_cols = [c['name'] for c in insp.get_columns('user_recommendations')]
    if 'recent_vibe_items' not in ur_cols:
        op.add_column(
            'user_recommendations',
            sa.Column('recent_vibe_items', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'[]'::jsonb"), nullable=False)
        )
    if 'impressions' not in ur_cols:
        op.add_column(
            'user_recommendations',
            sa.Column('impressions', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False)
        )

    # 2. media
    media_cols = [c['name'] for c in insp.get_columns('media')]
    if 'popularity' not in media_cols:
        op.add_column('media', sa.Column('popularity', sa.Float(), server_default='0.0', nullable=True))
    if 'vote_average' not in media_cols:
        op.add_column('media', sa.Column('vote_average', sa.Float(), server_default='0.0', nullable=True))
    if 'vote_count' not in media_cols:
        op.add_column('media', sa.Column('vote_count', sa.Integer(), server_default='0', nullable=True))
    if 'belongs_to_collection' not in media_cols:
        op.add_column('media', sa.Column('belongs_to_collection', sa.JSON(), nullable=True))
    if 'watch_providers_br' not in media_cols:
        op.add_column('media', sa.Column('watch_providers_br', sa.JSON(), server_default=sa.text("'[]'::json"), nullable=True))

    # 3. recommendation_interaction_events
    tables = insp.get_table_names()
    if 'recommendation_interaction_events' not in tables:
        op.create_table(
            'recommendation_interaction_events',
            sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
            sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
            sa.Column('media_type', sa.String(length=20), nullable=False, server_default='movie'),
            sa.Column('tmdb_id', sa.Integer(), nullable=False),
            sa.Column('event_type', sa.String(length=50), nullable=False),
            sa.Column('carousel_type', sa.String(length=50), nullable=True, server_default='explore'),
            sa.Column('position', sa.Integer(), nullable=True, server_default='0'),
            sa.Column('score_presented', sa.Float(), nullable=True),
            sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False)
        )
        op.create_index('idx_rec_events_user_id', 'recommendation_interaction_events', ['user_id'])
        op.create_index('idx_rec_events_tmdb_id', 'recommendation_interaction_events', ['tmdb_id'])
        op.create_index('idx_rec_events_event_type', 'recommendation_interaction_events', ['event_type'])
        op.create_index('idx_rec_events_created_at', 'recommendation_interaction_events', ['created_at'])


def downgrade() -> None:
    op.drop_table('recommendation_interaction_events')
    op.drop_column('media', 'watch_providers_br')
    op.drop_column('media', 'belongs_to_collection')
    op.drop_column('media', 'vote_count')
    op.drop_column('media', 'vote_average')
    op.drop_column('media', 'popularity')
    op.drop_column('user_recommendations', 'impressions')
    op.drop_column('user_recommendations', 'recent_vibe_items')
