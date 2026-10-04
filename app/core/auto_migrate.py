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
    except Exception as e:
        logger.error(f"❌ [AutoMigrate] Falha ao verificar DDL idempotente: {e}")
