"""add_user_monthly_activity

Revision ID: c8d1e234567b
Revises: b7c9e012345a
Create Date: 2026-10-25 23:45:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = 'c8d1e234567b'
down_revision: Union[str, Sequence[str], None] = 'b7c9e012345a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    # 1. Tabela materializada de atividade mensal por usuário
    op.execute("""
        CREATE TABLE IF NOT EXISTS user_monthly_activity (
            user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            year_month VARCHAR(7) NOT NULL,
            episodes_watched INTEGER NOT NULL DEFAULT 0,
            episodes_time_minutes INTEGER NOT NULL DEFAULT 0,
            movies_watched INTEGER NOT NULL DEFAULT 0,
            movies_time_minutes INTEGER NOT NULL DEFAULT 0,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (user_id, year_month)
        );
    """)

    # 2. Índice composto para leituras de intervalo temporal instantâneas
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_user_monthly_activity_user_date 
        ON user_monthly_activity (user_id, year_month DESC);
    """)

    # 3. Trigger atômica para manutenção incremental das contagens mensais
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

    op.execute("DROP TRIGGER IF EXISTS trg_user_monthly_activity ON user_episode_progress;")
    op.execute("""
        CREATE TRIGGER trg_user_monthly_activity
        AFTER INSERT OR UPDATE OR DELETE ON user_episode_progress
        FOR EACH ROW EXECUTE FUNCTION update_user_monthly_activity();
    """)

    # 4. Backfill inicial dos dados históricos
    op.execute("""
        INSERT INTO user_monthly_activity (user_id, year_month, episodes_watched, episodes_time_minutes, updated_at)
        SELECT 
            uli.user_id,
            TO_CHAR(COALESCE(uep.watched_at, uli.last_watched_at, uli.updated_at, uli.created_at), 'YYYY-MM') AS year_month,
            SUM(1 + COALESCE(uep.rewatch_count, 0)) AS episodes_watched,
            SUM((1 + COALESCE(uep.rewatch_count, 0)) * COALESCE(m.runtime, 0)) AS episodes_time_minutes,
            now()
        FROM user_episode_progress uep
        JOIN user_list_items uli ON uli.id = uep.user_list_item_id
        JOIN media m ON m.id = uli.media_id
        GROUP BY uli.user_id, TO_CHAR(COALESCE(uep.watched_at, uli.last_watched_at, uli.updated_at, uli.created_at), 'YYYY-MM')
        ON CONFLICT (user_id, year_month) DO UPDATE
        SET 
            episodes_watched = EXCLUDED.episodes_watched,
            episodes_time_minutes = EXCLUDED.episodes_time_minutes,
            updated_at = now();
    """)

def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS trg_user_monthly_activity ON user_episode_progress;")
    op.execute("DROP FUNCTION IF EXISTS update_user_monthly_activity();")
    op.execute("DROP TABLE IF EXISTS user_monthly_activity;")
