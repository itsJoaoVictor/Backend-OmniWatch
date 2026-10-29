"""optimize_episode_triggers_o1_delta

Revision ID: d9e2f345678c
Revises: c8d1e234567b
Create Date: 2026-10-28 21:45:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = 'd9e2f345678c'
down_revision: Union[str, Sequence[str], None] = 'c8d1e234567b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Otimiza update_user_list_item_counters para O(1) incremental sem scan em user_episode_progress
    op.execute("""
        CREATE OR REPLACE FUNCTION update_user_list_item_counters()
        RETURNS TRIGGER AS $$
        DECLARE
            target_item_id UUID;
            v_delta_eps INT := 0;
        BEGIN
            IF (TG_OP = 'INSERT') THEN
                target_item_id := NEW.user_list_item_id;
                v_delta_eps := 1 + COALESCE(NEW.rewatch_count, 0);
            ELSIF (TG_OP = 'DELETE') THEN
                target_item_id := OLD.user_list_item_id;
                v_delta_eps := -(1 + COALESCE(OLD.rewatch_count, 0));
            ELSIF (TG_OP = 'UPDATE') THEN
                target_item_id := NEW.user_list_item_id;
                v_delta_eps := (1 + COALESCE(NEW.rewatch_count, 0)) - (1 + COALESCE(OLD.rewatch_count, 0));
            END IF;

            IF v_delta_eps <> 0 AND target_item_id IS NOT NULL THEN
                UPDATE user_list_items uli
                SET 
                    episodes_watched_count = GREATEST(0, uli.episodes_watched_count + v_delta_eps),
                    total_time_minutes = GREATEST(0, uli.total_time_minutes + (v_delta_eps * COALESCE(m.runtime, 0)))
                FROM media m
                WHERE uli.id = target_item_id AND m.id = uli.media_id;
            END IF;

            RETURN NULL;
        END;
        $$ LANGUAGE plpgsql;
    """)

    # 2. Otimiza update_user_monthly_activity para O(1) incremental por chave primária (user_id, year_month)
    op.execute("""
        CREATE OR REPLACE FUNCTION update_user_monthly_activity()
        RETURNS TRIGGER AS $$
        DECLARE
            v_user_id UUID;
            v_runtime INT := 0;
            v_watched_at TIMESTAMPTZ;
            v_year_month VARCHAR(7);
            v_delta_eps INT := 0;
        BEGIN
            IF (TG_OP = 'INSERT') THEN
                SELECT uli.user_id, COALESCE(m.runtime, 0), COALESCE(NEW.watched_at, uli.last_watched_at, uli.updated_at, uli.created_at, now())
                INTO v_user_id, v_runtime, v_watched_at
                FROM user_list_items uli
                JOIN media m ON m.id = uli.media_id
                WHERE uli.id = NEW.user_list_item_id;
                v_delta_eps := 1 + COALESCE(NEW.rewatch_count, 0);
            ELSIF (TG_OP = 'DELETE') THEN
                SELECT uli.user_id, COALESCE(m.runtime, 0), COALESCE(OLD.watched_at, uli.last_watched_at, uli.updated_at, uli.created_at, now())
                INTO v_user_id, v_runtime, v_watched_at
                FROM user_list_items uli
                JOIN media m ON m.id = uli.media_id
                WHERE uli.id = OLD.user_list_item_id;
                v_delta_eps := -(1 + COALESCE(OLD.rewatch_count, 0));
            ELSIF (TG_OP = 'UPDATE') THEN
                SELECT uli.user_id, COALESCE(m.runtime, 0), COALESCE(NEW.watched_at, uli.last_watched_at, uli.updated_at, uli.created_at, now())
                INTO v_user_id, v_runtime, v_watched_at
                FROM user_list_items uli
                JOIN media m ON m.id = uli.media_id
                WHERE uli.id = NEW.user_list_item_id;
                v_delta_eps := (1 + COALESCE(NEW.rewatch_count, 0)) - (1 + COALESCE(OLD.rewatch_count, 0));
            END IF;

            IF v_user_id IS NOT NULL AND v_delta_eps <> 0 THEN
                v_year_month := TO_CHAR(v_watched_at, 'YYYY-MM');

                INSERT INTO user_monthly_activity (user_id, year_month, episodes_watched, episodes_time_minutes, updated_at)
                VALUES (
                    v_user_id,
                    v_year_month,
                    GREATEST(0, v_delta_eps),
                    GREATEST(0, v_delta_eps * v_runtime),
                    now()
                )
                ON CONFLICT (user_id, year_month) DO UPDATE
                SET 
                    episodes_watched = GREATEST(0, user_monthly_activity.episodes_watched + v_delta_eps),
                    episodes_time_minutes = GREATEST(0, user_monthly_activity.episodes_time_minutes + (v_delta_eps * v_runtime)),
                    updated_at = now();
            END IF;

            RETURN NULL;
        END;
        $$ LANGUAGE plpgsql;
    """)


def downgrade() -> None:
    op.execute("""
        CREATE OR REPLACE FUNCTION update_user_list_item_counters()
        RETURNS TRIGGER AS $$
        DECLARE
            target_item_id UUID;
            v_runtime INT;
        BEGIN
            IF (TG_OP = 'DELETE') THEN
                target_item_id := OLD.user_list_item_id;
            ELSE
                target_item_id := NEW.user_list_item_id;
            END IF;

            SELECT COALESCE(m.runtime, 0) INTO v_runtime
            FROM user_list_items uli
            JOIN media m ON m.id = uli.media_id
            WHERE uli.id = target_item_id;

            UPDATE user_list_items
            SET 
                episodes_watched_count = COALESCE((
                    SELECT SUM(1 + COALESCE(rewatch_count, 0))
                    FROM user_episode_progress
                    WHERE user_list_item_id = target_item_id
                ), 0),
                total_time_minutes = COALESCE((
                    SELECT SUM(1 + COALESCE(rewatch_count, 0)) * COALESCE(v_runtime, 0)
                    FROM user_episode_progress
                    WHERE user_list_item_id = target_item_id
                ), 0)
            WHERE id = target_item_id;

            RETURN NULL;
        END;
        $$ LANGUAGE plpgsql;
    """)
    op.execute("""
        CREATE OR REPLACE FUNCTION update_user_monthly_activity()
        RETURNS TRIGGER AS $$
        DECLARE
            v_user_id UUID;
            v_watched_at TIMESTAMPTZ;
            v_year_month VARCHAR(7);
        BEGIN
            IF (TG_OP = 'DELETE') THEN
                SELECT uli.user_id, COALESCE(OLD.watched_at, now())
                INTO v_user_id, v_watched_at
                FROM user_list_items uli
                WHERE uli.id = OLD.user_list_item_id;
            ELSE
                SELECT uli.user_id, COALESCE(NEW.watched_at, now())
                INTO v_user_id, v_watched_at
                FROM user_list_items uli
                WHERE uli.id = NEW.user_list_item_id;
            END IF;

            IF v_user_id IS NOT NULL THEN
                v_year_month := TO_CHAR(v_watched_at, 'YYYY-MM');

                INSERT INTO user_monthly_activity (user_id, year_month, episodes_watched, episodes_time_minutes, updated_at)
                SELECT 
                    v_user_id,
                    v_year_month,
                    COALESCE(SUM(1 + COALESCE(uep.rewatch_count, 0)), 0),
                    COALESCE(SUM((1 + COALESCE(uep.rewatch_count, 0)) * COALESCE(m.runtime, 0)), 0),
                    now()
                FROM user_episode_progress uep
                JOIN user_list_items uli ON uli.id = uep.user_list_item_id
                JOIN media m ON m.id = uli.media_id
                WHERE uli.user_id = v_user_id
                  AND TO_CHAR(COALESCE(uep.watched_at, uli.last_watched_at, uli.updated_at, uli.created_at), 'YYYY-MM') = v_year_month
                ON CONFLICT (user_id, year_month) DO UPDATE
                SET 
                    episodes_watched = EXCLUDED.episodes_watched,
                    episodes_time_minutes = EXCLUDED.episodes_time_minutes,
                    updated_at = now();
            END IF;

            RETURN NULL;
        END;
        $$ LANGUAGE plpgsql;
    """)
