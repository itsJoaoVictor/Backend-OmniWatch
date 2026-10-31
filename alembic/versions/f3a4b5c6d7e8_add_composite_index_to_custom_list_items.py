"""add_composite_index_to_custom_list_items

Revision ID: f3a4b5c6d7e8
Revises: e1f2a3b4c5d6
Create Date: 2026-10-31 12:15:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = 'f3a4b5c6d7e8'
down_revision: Union[str, Sequence[str], None] = 'e1f2a3b4c5d6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index(
        'idx_custom_list_items_list_pos',
        'custom_list_items',
        ['list_id', 'position'],
        unique=False
    )


def downgrade() -> None:
    op.drop_index('idx_custom_list_items_list_pos', table_name='custom_list_items')
