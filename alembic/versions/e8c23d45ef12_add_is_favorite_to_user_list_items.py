"""add_is_favorite_to_user_list_items

Revision ID: e8c23d45ef12
Revises: b9bd8a7b8889
Create Date: 2026-10-03 18:55:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e8c23d45ef12'
down_revision: Union[str, Sequence[str], None] = 'b9bd8a7b8889'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Adiciona coluna is_favorite se não existir
    op.add_column('user_list_items', sa.Column('is_favorite', sa.Boolean(), server_default=sa.text('false'), nullable=False))
    op.create_index(op.f('ix_user_list_items_is_favorite'), 'user_list_items', ['is_favorite'], unique=False)
    op.create_index('ix_user_list_items_user_id_is_favorite', 'user_list_items', ['user_id', 'is_favorite'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_user_list_items_user_id_is_favorite', table_name='user_list_items')
    op.drop_index(op.f('ix_user_list_items_is_favorite'), table_name='user_list_items')
    op.drop_column('user_list_items', 'is_favorite')
