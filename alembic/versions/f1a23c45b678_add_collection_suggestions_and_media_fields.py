"""add_collection_suggestions_and_media_fields

Revision ID: f1a23c45b678
Revises: e8c23d45ef12
Create Date: 2026-10-08 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'f1a23c45b678'
down_revision: Union[str, Sequence[str], None] = 'e8c23d45ef12'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Adicionar colunas de coleção na tabela media
    op.add_column('media', sa.Column('collection_tmdb_id', sa.Integer(), nullable=True))
    op.add_column('media', sa.Column('collection_name', sa.String(), nullable=True))
    op.create_index(op.f('ix_media_collection_tmdb_id'), 'media', ['collection_tmdb_id'], unique=False)

    # 2. Criar tabela de sugestões de coleções para o usuário
    op.create_table(
        'user_collection_suggestions',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('user_id', sa.UUID(), nullable=False),
        sa.Column('collection_tmdb_id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('overview', sa.String(), nullable=True),
        sa.Column('poster_path', sa.String(), nullable=True),
        sa.Column('backdrop_path', sa.String(), nullable=True),
        sa.Column('total_movies', sa.Integer(), server_default='0', nullable=False),
        sa.Column('movies_in_list', sa.Integer(), server_default='0', nullable=False),
        sa.Column('matched_movie_titles', postgresql.JSONB(astext_type=sa.Text()), server_default='[]', nullable=False),
        sa.Column('is_dismissed', sa.Boolean(), server_default=sa.text('false'), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('user_id', 'collection_tmdb_id', name='uq_user_collection_suggestion')
    )
    op.create_index(op.f('ix_user_collection_suggestions_id'), 'user_collection_suggestions', ['id'], unique=False)
    op.create_index(op.f('ix_user_collection_suggestions_user_id'), 'user_collection_suggestions', ['user_id'], unique=False)
    op.create_index(op.f('ix_user_collection_suggestions_collection_tmdb_id'), 'user_collection_suggestions', ['collection_tmdb_id'], unique=False)
    op.create_index(op.f('ix_user_collection_suggestions_is_dismissed'), 'user_collection_suggestions', ['is_dismissed'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_user_collection_suggestions_is_dismissed'), table_name='user_collection_suggestions')
    op.drop_index(op.f('ix_user_collection_suggestions_collection_tmdb_id'), table_name='user_collection_suggestions')
    op.drop_index(op.f('ix_user_collection_suggestions_user_id'), table_name='user_collection_suggestions')
    op.drop_index(op.f('ix_user_collection_suggestions_id'), table_name='user_collection_suggestions')
    op.drop_table('user_collection_suggestions')

    op.drop_index(op.f('ix_media_collection_tmdb_id'), table_name='media')
    op.drop_column('media', 'collection_name')
    op.drop_column('media', 'collection_tmdb_id')
