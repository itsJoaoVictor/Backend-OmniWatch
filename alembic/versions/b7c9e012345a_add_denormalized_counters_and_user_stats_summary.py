"""add_denormalized_counters_and_user_stats_summary

Revision ID: b7c9e012345a
Revises: a5c10f65834a
Create Date: 2026-10-17 17:55:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'b7c9e012345a'
down_revision: Union[str, Sequence[str], None] = 'a5c10f65834a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Colunas desnormalizadas em user_list_items
    op.execute("""
        ALTER TABLE user_list_items 
        ADD COLUMN IF NOT EXISTS episodes_watched_count INTEGER NOT NULL DEFAULT 0;
    """)
    op.execute("""
        ALTER TABLE user_list_items 
        ADD COLUMN IF NOT EXISTS total_time_minutes INTEGER NOT NULL DEFAULT 0;
    """)

    # 2. Índices para ranking instantâneo
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_user_list_items_user_time 
        ON user_list_items (user_id, total_time_minutes DESC);
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_user_list_items_user_episodes 
        ON user_list_items (user_id, episodes_watched_count DESC);
    """)

    # 3. Tabela de sumarização global de usuário
    op.execute("""
        CREATE TABLE IF NOT EXISTS user_stats_summary (
            user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
            total_time_movies INTEGER NOT NULL DEFAULT 0,
            total_time_tv INTEGER NOT NULL DEFAULT 0,
            total_movies_watched INTEGER NOT NULL DEFAULT 0,
            total_episodes_watched INTEGER NOT NULL DEFAULT 0,
            average_rating NUMERIC(3, 2),
            total_rated INTEGER NOT NULL DEFAULT 0,
            rating_distribution JSONB NOT NULL DEFAULT '{}'::jsonb,
            top_genres JSONB NOT NULL DEFAULT '[]'::jsonb,
            rankings JSONB NOT NULL DEFAULT '{}'::jsonb,
            timeline JSONB NOT NULL DEFAULT '[]'::jsonb,
            kpis JSONB NOT NULL DEFAULT '{}'::jsonb,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );
    """)

    # 4. Triggers para manutenção contínua e transacional dos contadores
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

    op.execute("DROP TRIGGER IF EXISTS trg_user_episode_progress_counters ON user_episode_progress;")
    op.execute("""
        CREATE TRIGGER trg_user_episode_progress_counters
        AFTER INSERT OR UPDATE OR DELETE ON user_episode_progress
        FOR EACH ROW EXECUTE FUNCTION update_user_list_item_counters();
    """)

    op.execute("""
        CREATE OR REPLACE FUNCTION update_user_movie_item_counters()
        RETURNS TRIGGER AS $$
        DECLARE
            v_runtime INT;
            v_media_type TEXT;
        BEGIN
            SELECT m.media_type, COALESCE(m.runtime, 0) INTO v_media_type, v_runtime
            FROM media m
            WHERE m.id = NEW.media_id;

            IF v_media_type = 'movie' THEN
                IF NEW.status IN ('completed', 'watching') THEN
                    NEW.episodes_watched_count := 1;
                    NEW.total_time_minutes := (1 + COALESCE(NEW.rewatch_count, 0)) * v_runtime;
                ELSE
                    NEW.episodes_watched_count := 0;
                    NEW.total_time_minutes := 0;
                END IF;
            END IF;

            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)

    op.execute("DROP TRIGGER IF EXISTS trg_user_movie_item_counters ON user_list_items;")
    op.execute("""
        CREATE TRIGGER trg_user_movie_item_counters
        BEFORE INSERT OR UPDATE OF status, rewatch_count ON user_list_items
        FOR EACH ROW EXECUTE FUNCTION update_user_movie_item_counters();
    """)

    # 5. Backfill de séries existentes
    op.execute("""
        WITH ep_counts AS (
            SELECT user_list_item_id, SUM(1 + COALESCE(rewatch_count, 0)) AS ep_count
            FROM user_episode_progress
            GROUP BY user_list_item_id
        )
        UPDATE user_list_items uli
        SET 
            episodes_watched_count = COALESCE(ec.ep_count, 0),
            total_time_minutes = COALESCE(ec.ep_count, 0) * COALESCE(m.runtime, 0)
        FROM media m, ep_counts ec
        WHERE m.id = uli.media_id 
          AND m.media_type = 'tv' 
          AND ec.user_list_item_id = uli.id;
    """)

    # 6. Backfill de filmes existentes
    op.execute("""
        UPDATE user_list_items uli
        SET 
            episodes_watched_count = CASE WHEN uli.status IN ('completed', 'watching') THEN 1 ELSE 0 END,
            total_time_minutes = CASE WHEN uli.status IN ('completed', 'watching') THEN (1 + COALESCE(uli.rewatch_count, 0)) * COALESCE(m.runtime, 0) ELSE 0 END
        FROM media m
        WHERE m.id = uli.media_id AND m.media_type = 'movie';
    """)


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS trg_user_movie_item_counters ON user_list_items;")
    op.execute("DROP FUNCTION IF EXISTS update_user_movie_item_counters();")
    op.execute("DROP TRIGGER IF EXISTS trg_user_episode_progress_counters ON user_episode_progress;")
    op.execute("DROP FUNCTION IF EXISTS update_user_list_item_counters();")
    op.execute("DROP TABLE IF EXISTS user_stats_summary;")
    op.execute("DROP INDEX IF EXISTS ix_user_list_items_user_episodes;")
    op.execute("DROP INDEX IF EXISTS ix_user_list_items_user_time;")
    op.execute("ALTER TABLE user_list_items DROP COLUMN IF EXISTS total_time_minutes;")
    op.execute("ALTER TABLE user_list_items DROP COLUMN IF EXISTS episodes_watched_count;")
