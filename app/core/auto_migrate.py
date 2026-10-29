import os
import asyncio
import logging
from alembic.config import Config
from alembic import command
from sqlalchemy import text
from app.core.database import AsyncSessionLocal

logger = logging.getLogger(__name__)

def _run_alembic_upgrade():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        alembic_ini_path = os.path.abspath(os.path.join(current_dir, "..", "..", "alembic.ini"))
        if os.path.exists(alembic_ini_path):
            alembic_cfg = Config(alembic_ini_path)
            command.upgrade(alembic_cfg, "head")
            logger.info("✅ [AutoMigrate] Alembic upgrade head executado com sucesso.")
        else:
            logger.warning(f"⚠️ [AutoMigrate] Arquivo alembic.ini não encontrado em {alembic_ini_path}")
    except Exception as e:
        logger.warning(f"⚠️ [AutoMigrate] Aviso durante execução do Alembic: {e}")

async def run_auto_migrations():
    """
    Executa migrações automaticamente na inicialização da aplicação,
    dispensando a necessidade de comandos manuais no servidor de produção.
    """
    logger.info("🔄 [AutoMigrate] Verificando e aplicando migrações automaticamente...")
    
    # 1. Executa upgrade do Alembic de forma assíncrona/em thread
    try:
        await asyncio.to_thread(_run_alembic_upgrade)
    except Exception as e:
        logger.warning(f"⚠️ [AutoMigrate] Erro na thread do Alembic: {e}")

    # 2. Garantia idempotente direta no PostgreSQL para 'is_favorite'
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(text("""
                ALTER TABLE user_list_items 
                ADD COLUMN IF NOT EXISTS is_favorite BOOLEAN NOT NULL DEFAULT FALSE;
            """))
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS ix_user_list_items_is_favorite 
                ON user_list_items (is_favorite);
            """))
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS ix_user_list_items_user_id_is_favorite 
                ON user_list_items (user_id, is_favorite);
            """))
            await session.commit()
            logger.info("✅ [AutoMigrate] Coluna 'is_favorite' e índices validados no banco de dados.")

            # 3. Garantia idempotente para colunas de coleção em media
            await session.execute(text("""
                ALTER TABLE media 
                ADD COLUMN IF NOT EXISTS collection_tmdb_id INTEGER;
            """))
            await session.execute(text("""
                ALTER TABLE media 
                ADD COLUMN IF NOT EXISTS collection_name VARCHAR;
            """))
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS ix_media_collection_tmdb_id 
                ON media (collection_tmdb_id);
            """))

            # 4. Garantia idempotente para user_collection_suggestions
            await session.execute(text("""
                CREATE TABLE IF NOT EXISTS user_collection_suggestions (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    collection_tmdb_id INTEGER NOT NULL,
                    name VARCHAR NOT NULL,
                    overview TEXT,
                    poster_path VARCHAR,
                    backdrop_path VARCHAR,
                    total_movies INTEGER NOT NULL DEFAULT 0,
                    movies_in_list INTEGER NOT NULL DEFAULT 0,
                    matched_movie_titles JSONB NOT NULL DEFAULT '[]'::jsonb,
                    is_dismissed BOOLEAN NOT NULL DEFAULT FALSE,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    CONSTRAINT uq_user_collection_suggestion UNIQUE (user_id, collection_tmdb_id)
                );
            """))
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS ix_user_collection_suggestions_user_id 
                ON user_collection_suggestions (user_id);
            """))
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS ix_user_collection_suggestions_collection_tmdb_id 
                ON user_collection_suggestions (collection_tmdb_id);
            """))
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS ix_user_collection_suggestions_is_dismissed 
                ON user_collection_suggestions (is_dismissed);
            """))
            # 5. Garantia idempotente para custom_lists e custom_list_items
            await session.execute(text("""
                CREATE TABLE IF NOT EXISTS custom_lists (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    title VARCHAR(255) NOT NULL,
                    description TEXT,
                    is_public BOOLEAN NOT NULL DEFAULT TRUE,
                    is_ranked BOOLEAN NOT NULL DEFAULT FALSE,
                    cover_backdrop_path VARCHAR(500),
                    cover_poster_path VARCHAR(500),
                    items_count INTEGER NOT NULL DEFAULT 0,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
            """))
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS ix_custom_lists_user_id ON custom_lists (user_id);
            """))
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS ix_custom_lists_is_public ON custom_lists (is_public);
            """))

            await session.execute(text("""
                CREATE TABLE IF NOT EXISTS custom_list_items (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    list_id UUID NOT NULL REFERENCES custom_lists(id) ON DELETE CASCADE,
                    media_id UUID REFERENCES media(id) ON DELETE SET NULL,
                    tmdb_id INTEGER NOT NULL,
                    media_type VARCHAR(20) NOT NULL,
                    title VARCHAR(255) NOT NULL,
                    poster_path VARCHAR(500),
                    backdrop_path VARCHAR(500),
                    release_date VARCHAR(50),
                    runtime INTEGER DEFAULT 0,
                    position INTEGER NOT NULL DEFAULT 0,
                    note TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    CONSTRAINT uq_custom_list_item UNIQUE (list_id, tmdb_id, media_type)
                );
            """))
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS ix_custom_list_items_list_id ON custom_list_items (list_id);
            """))
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS ix_custom_list_items_tmdb_id ON custom_list_items (tmdb_id);
            """))

            await session.commit()
            logger.info("✅ [AutoMigrate] Tabelas 'custom_lists' e 'custom_list_items' validadas no banco de dados.")

            # 6. Garantia idempotente para friendships
            await session.execute(text("""
                CREATE TABLE IF NOT EXISTS friendships (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    requester_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    addressee_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    status VARCHAR(20) NOT NULL DEFAULT 'pending',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    CONSTRAINT uq_friendship_requester_addressee UNIQUE (requester_id, addressee_id)
                );
            """))
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS ix_friendships_requester_id ON friendships (requester_id);
            """))
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS ix_friendships_addressee_id ON friendships (addressee_id);
            """))
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS ix_friendships_status ON friendships (status);
            """))

            # 7. Garantia idempotente para controle de novidades vistas pelo usuário (Changelog / What's New)
            await session.execute(text("""
                ALTER TABLE users 
                ADD COLUMN IF NOT EXISTS last_seen_release_id INTEGER NOT NULL DEFAULT 0;
            """))

            # 8. Garantia idempotente para rewatch_count em user_episode_progress
            await session.execute(text("""
                ALTER TABLE user_episode_progress 
                ADD COLUMN IF NOT EXISTS rewatch_count INTEGER NOT NULL DEFAULT 0;
            """))
            await session.commit()
            logger.info("✅ [AutoMigrate] Tabela 'friendships', 'last_seen_release_id', 'rewatch_count' e índices validados no banco de dados.")

            # 9. Garantia idempotente para Universos (universes, universe_items, user_universes, user_universe_suggestions e colunas em media)
            await session.execute(text("""
                ALTER TABLE media 
                ADD COLUMN IF NOT EXISTS universe_key VARCHAR;
            """))
            await session.execute(text("""
                ALTER TABLE media 
                ADD COLUMN IF NOT EXISTS universe_name VARCHAR;
            """))
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS ix_media_universe_key 
                ON media (universe_key);
            """))

            await session.execute(text("""
                CREATE TABLE IF NOT EXISTS universes (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    universe_key VARCHAR NOT NULL UNIQUE,
                    name VARCHAR NOT NULL,
                    overview TEXT,
                    poster_path VARCHAR,
                    backdrop_path VARCHAR,
                    items_count INTEGER NOT NULL DEFAULT 0,
                    movies_count INTEGER NOT NULL DEFAULT 0,
                    tv_count INTEGER NOT NULL DEFAULT 0,
                    collections_count INTEGER NOT NULL DEFAULT 0,
                    last_synced_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
            """))
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS ix_universes_universe_key ON universes (universe_key);
            """))

            await session.execute(text("""
                CREATE TABLE IF NOT EXISTS universe_items (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    universe_id UUID NOT NULL REFERENCES universes(id) ON DELETE CASCADE,
                    media_id UUID REFERENCES media(id) ON DELETE SET NULL,
                    tmdb_id INTEGER NOT NULL,
                    media_type VARCHAR(20) NOT NULL DEFAULT 'movie',
                    title VARCHAR NOT NULL,
                    release_date VARCHAR,
                    poster_path VARCHAR,
                    backdrop_path VARCHAR,
                    collection_name VARCHAR,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    CONSTRAINT uq_universe_item_universe_tmdb_type UNIQUE (universe_id, tmdb_id, media_type)
                );
            """))
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS ix_universe_items_universe_id ON universe_items (universe_id);
            """))
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS ix_universe_items_tmdb_id ON universe_items (tmdb_id);
            """))

            await session.execute(text("""
                CREATE TABLE IF NOT EXISTS user_universes (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    universe_id UUID NOT NULL REFERENCES universes(id) ON DELETE CASCADE,
                    auto_add BOOLEAN NOT NULL DEFAULT TRUE,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    CONSTRAINT uq_user_universe UNIQUE (user_id, universe_id)
                );
            """))
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS ix_user_universes_user_id ON user_universes (user_id);
            """))
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS ix_user_universes_universe_id ON user_universes (universe_id);
            """))

            await session.execute(text("""
                CREATE TABLE IF NOT EXISTS user_universe_suggestions (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    universe_key VARCHAR NOT NULL,
                    name VARCHAR NOT NULL,
                    overview TEXT,
                    poster_path VARCHAR,
                    backdrop_path VARCHAR,
                    total_items INTEGER NOT NULL DEFAULT 0,
                    movies_count INTEGER NOT NULL DEFAULT 0,
                    tv_count INTEGER NOT NULL DEFAULT 0,
                    collections_count INTEGER NOT NULL DEFAULT 0,
                    items_in_list INTEGER NOT NULL DEFAULT 0,
                    matched_titles JSONB NOT NULL DEFAULT '[]'::jsonb,
                    discovered_members JSONB NOT NULL DEFAULT '[]'::jsonb,
                    is_dismissed BOOLEAN NOT NULL DEFAULT FALSE,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    CONSTRAINT uq_user_universe_suggestion UNIQUE (user_id, universe_key)
                );
            """))
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS ix_user_universe_suggestions_user_id ON user_universe_suggestions (user_id);
            """))
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS ix_user_universe_suggestions_universe_key ON user_universe_suggestions (universe_key);
            """))
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS ix_user_universe_suggestions_is_dismissed ON user_universe_suggestions (is_dismissed);
            """))
            await session.execute(text("""
                ALTER TABLE user_universe_suggestions
                ADD COLUMN IF NOT EXISTS dismissed_items_count INTEGER DEFAULT 0;
            """))
            await session.execute(text("""
                ALTER TABLE user_universe_suggestions
                ADD COLUMN IF NOT EXISTS dismissed_media_types JSONB DEFAULT '[]'::jsonb;
            """))
            await session.execute(text("""
                ALTER TABLE user_universe_suggestions
                ADD COLUMN IF NOT EXISTS dismissed_collection_ids JSONB DEFAULT '[]'::jsonb;
            """))
            await session.execute(text("""
                ALTER TABLE user_universe_suggestions
                ADD COLUMN IF NOT EXISTS is_reactivated BOOLEAN DEFAULT FALSE;
            """))
            await session.execute(text("""
                ALTER TABLE user_universe_suggestions
                ADD COLUMN IF NOT EXISTS reactivation_reason VARCHAR;
            """))
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS ix_user_universe_suggestions_is_reactivated ON user_universe_suggestions (is_reactivated);
            """))
            await session.execute(text("""
                CREATE TABLE IF NOT EXISTS user_universe_ignored_items (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    universe_id UUID NOT NULL REFERENCES universes(id) ON DELETE CASCADE,
                    tmdb_id INTEGER NOT NULL,
                    media_type VARCHAR NOT NULL DEFAULT 'movie',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    CONSTRAINT uq_user_universe_ignored_item UNIQUE (user_id, universe_id, tmdb_id, media_type)
                );
            """))
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS ix_user_universe_ignored_items_user_id ON user_universe_ignored_items (user_id);
            """))
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS ix_user_universe_ignored_items_universe_id ON user_universe_ignored_items (universe_id);
            """))
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS ix_user_universe_ignored_items_tmdb_id ON user_universe_ignored_items (tmdb_id);
            """))
            await session.commit()
            logger.info("✅ [AutoMigrate] Tabelas de Universos e colunas validadas no banco de dados.")

            # 10. Garantia idempotente para índices compostos de estatísticas e rankings
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS ix_user_episode_progress_item_watched 
                ON user_episode_progress (user_list_item_id, watched_at);
            """))
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS ix_user_list_items_user_status_media 
                ON user_list_items (user_id, status, media_id);
            """))
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS ix_user_list_items_user_last_watched 
                ON user_list_items (user_id, last_watched_at);
            """))
            await session.commit()
            logger.info("✅ [AutoMigrate] Índices compostos de estatísticas e rankings validados no banco de dados.")

            # 11. Garantia idempotente para Nível 3: Contadores desnormalizados e user_stats_summary
            await session.execute(text("""
                ALTER TABLE user_list_items 
                ADD COLUMN IF NOT EXISTS episodes_watched_count INTEGER NOT NULL DEFAULT 0;
            """))
            await session.execute(text("""
                ALTER TABLE user_list_items 
                ADD COLUMN IF NOT EXISTS total_time_minutes INTEGER NOT NULL DEFAULT 0;
            """))
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS ix_user_list_items_user_time 
                ON user_list_items (user_id, total_time_minutes DESC);
            """))
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS ix_user_list_items_user_episodes 
                ON user_list_items (user_id, episodes_watched_count DESC);
            """))
            await session.execute(text("""
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
            """))

            # Triggers de atualização automática O(1) incremental (sem scan em user_episode_progress)
            await session.execute(text("""
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
            """))
            await session.execute(text("DROP TRIGGER IF EXISTS trg_user_episode_progress_counters ON user_episode_progress;"))
            await session.execute(text("""
                CREATE TRIGGER trg_user_episode_progress_counters
                AFTER INSERT OR UPDATE OR DELETE ON user_episode_progress
                FOR EACH ROW EXECUTE FUNCTION update_user_list_item_counters();
            """))

            await session.execute(text("""
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
            """))
            await session.execute(text("DROP TRIGGER IF EXISTS trg_user_movie_item_counters ON user_list_items;"))
            await session.execute(text("""
                CREATE TRIGGER trg_user_movie_item_counters
                BEFORE INSERT OR UPDATE OF status, rewatch_count ON user_list_items
                FOR EACH ROW EXECUTE FUNCTION update_user_movie_item_counters();
            """))

            # Backfill idempotente
            await session.execute(text("""
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
                  AND ec.user_list_item_id = uli.id
                  AND (uli.episodes_watched_count = 0 OR uli.total_time_minutes = 0);
            """))

            await session.execute(text("""
                UPDATE user_list_items uli
                SET 
                    episodes_watched_count = CASE WHEN uli.status IN ('completed', 'watching') THEN 1 ELSE 0 END,
                    total_time_minutes = CASE WHEN uli.status IN ('completed', 'watching') THEN (1 + COALESCE(uli.rewatch_count, 0)) * COALESCE(m.runtime, 0) ELSE 0 END
                FROM media m
                WHERE m.id = uli.media_id 
                  AND m.media_type = 'movie'
                  AND uli.status IN ('completed', 'watching')
                  AND (uli.episodes_watched_count = 0 OR uli.total_time_minutes = 0);
            """))
            await session.commit()
            logger.info("✅ [AutoMigrate] Nível 3 (Contadores desnormalizados, triggers e user_stats_summary) validados.")

            # 4. Tabela materializada de atividade mensal (user_monthly_activity)
            await session.execute(text("""
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
            """))
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS ix_user_monthly_activity_user_date 
                ON user_monthly_activity (user_id, year_month DESC);
            """))
            await session.execute(text("""
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
            """))
            await session.execute(text("DROP TRIGGER IF EXISTS trg_user_monthly_activity ON user_episode_progress;"))
            await session.execute(text("""
                CREATE TRIGGER trg_user_monthly_activity
                AFTER INSERT OR UPDATE OR DELETE ON user_episode_progress
                FOR EACH ROW EXECUTE FUNCTION update_user_monthly_activity();
            """))

            # Backfill idempotente
            await session.execute(text("""
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
            """))
            await session.commit()
            logger.info("✅ [AutoMigrate] Tabela materializada user_monthly_activity validada.")
    except Exception as e:
        logger.error(f"❌ [AutoMigrate] Falha ao verificar DDL idempotente: {e}")


