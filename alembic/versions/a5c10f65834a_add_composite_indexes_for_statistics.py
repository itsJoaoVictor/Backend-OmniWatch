"""add_composite_indexes_for_statistics

Revision ID: a5c10f65834a
Revises: 954095a4e1d7
Create Date: 2026-10-17 17:06:55.535012

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a5c10f65834a'
down_revision: Union[str, Sequence[str], None] = '954095a4e1d7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index(
        'ix_user_episode_progress_item_watched',
        'user_episode_progress',
        ['user_list_item_id', 'watched_at'],
        unique=False
    )
    op.create_index(
        'ix_user_list_items_user_status_media',
        'user_list_items',
        ['user_id', 'status', 'media_id'],
        unique=False
    )
    op.create_index(
        'ix_user_list_items_user_last_watched',
        'user_list_items',
        ['user_id', 'last_watched_at'],
        unique=False
    )


def downgrade() -> None:
    op.drop_index('ix_user_list_items_user_last_watched', table_name='user_list_items')
    op.drop_index('ix_user_list_items_user_status_media', table_name='user_list_items')
    op.drop_index('ix_user_episode_progress_item_watched', table_name='user_episode_progress')
