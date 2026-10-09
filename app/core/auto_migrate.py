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
    except Exception as e:
        logger.error(f"❌ [AutoMigrate] Falha ao verificar DDL idempotente: {e}")

