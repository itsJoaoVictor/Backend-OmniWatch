"""add_push_subscriptions_table

Revision ID: c3d4e5f6a7b8
Revises: b2c3d4e5f6a7
Create Date: 2026-11-08 21:46:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = 'c3d4e5f6a7b8'
down_revision: Union[str, Sequence[str], None] = 'b2c3d4e5f6a7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    tables = insp.get_table_names()

    if 'push_subscriptions' not in tables:
        op.create_table(
            'push_subscriptions',
            sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
            sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
            sa.Column('endpoint', sa.String(), nullable=False),
            sa.Column('p256dh', sa.String(), nullable=False),
            sa.Column('auth', sa.String(), nullable=False),
            sa.Column('user_agent', sa.String(), nullable=True),
            sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        )
        op.create_index('ix_push_subscriptions_id', 'push_subscriptions', ['id'])
        op.create_index('ix_push_subscriptions_user_id', 'push_subscriptions', ['user_id'])
        op.create_index('ix_push_subscriptions_endpoint', 'push_subscriptions', ['endpoint'], unique=True)
        op.create_index('ix_push_subscriptions_created_at', 'push_subscriptions', ['created_at'])


def downgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    tables = insp.get_table_names()

    if 'push_subscriptions' in tables:
        op.drop_table('push_subscriptions')
