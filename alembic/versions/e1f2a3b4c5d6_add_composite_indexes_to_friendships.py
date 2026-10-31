"""add_composite_indexes_to_friendships

Revision ID: e1f2a3b4c5d6
Revises: d9e2f345678c
Create Date: 2026-10-30 23:37:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = 'e1f2a3b4c5d6'
down_revision: Union[str, Sequence[str], None] = 'd9e2f345678c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index(
        'ix_friendships_addressee_status',
        'friendships',
        ['addressee_id', 'status'],
        unique=False
    )
    op.create_index(
        'ix_friendships_requester_status',
        'friendships',
        ['requester_id', 'status'],
        unique=False
    )


def downgrade() -> None:
    op.drop_index('ix_friendships_requester_status', table_name='friendships')
    op.drop_index('ix_friendships_addressee_status', table_name='friendships')
