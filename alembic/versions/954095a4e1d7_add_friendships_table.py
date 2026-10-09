"""add_friendships_table

Revision ID: 954095a4e1d7
Revises: f207e415d476
Create Date: 2026-10-09 01:17:08.305021

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '954095a4e1d7'
down_revision: Union[str, Sequence[str], None] = 'f207e415d476'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'friendships',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('requester_id', sa.UUID(), nullable=False),
        sa.Column('addressee_id', sa.UUID(), nullable=False),
        sa.Column('status', sa.String(length=20), server_default='pending', nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['addressee_id'], ['users.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['requester_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('requester_id', 'addressee_id', name='uq_friendship_requester_addressee')
    )
    op.create_index(op.f('ix_friendships_id'), 'friendships', ['id'], unique=False)
    op.create_index(op.f('ix_friendships_requester_id'), 'friendships', ['requester_id'], unique=False)
    op.create_index(op.f('ix_friendships_addressee_id'), 'friendships', ['addressee_id'], unique=False)
    op.create_index(op.f('ix_friendships_status'), 'friendships', ['status'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_friendships_status'), table_name='friendships')
    op.drop_index(op.f('ix_friendships_addressee_id'), table_name='friendships')
    op.drop_index(op.f('ix_friendships_requester_id'), table_name='friendships')
    op.drop_index(op.f('ix_friendships_id'), table_name='friendships')
    op.drop_table('friendships')
