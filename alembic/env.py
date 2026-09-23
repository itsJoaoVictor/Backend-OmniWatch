import asyncio
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.core.database import Base, DATABASE_URL
from app.users import models as users_models
from app.media import models as media_models
from app.tracking import models as tracking_models
from app.notifications import models as notifications_models
from app.media_collections import models as media_collections_models

target_metadata = Base.metadata
config.set_main_option("sqlalchemy.url", DATABASE_URL)

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    
    # --- AUTO CREATE DB START ---
    from app.core.database import DB_USER, encoded_password, DB_HOST, DB_PORT, DB_NAME, DB_DRIVER
    
    sys_url = f"postgresql://{DB_USER}:{encoded_password}@{DB_HOST}:{DB_PORT}/postgres"
    
    try:
        import re
        if DB_DRIVER == "asyncpg":
            import asyncpg
            conn = await asyncpg.connect(sys_url)
            exists = await conn.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", DB_NAME)
            if not exists:
                if not re.match(r"^[a-zA-Z0-9_]+$", DB_NAME):
                    raise ValueError(f"Invalid database name: {DB_NAME}")
                await conn.execute(f'CREATE DATABASE "{DB_NAME}"')
                print(f"Banco de dados '{DB_NAME}' criado com sucesso!")
            await conn.close()
        elif DB_DRIVER in ("psycopg", "psycopg3"):
            import psycopg
            conn = await psycopg.AsyncConnection.connect(sys_url, autocommit=True)
            async with conn.cursor() as cur:
                await cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DB_NAME,))
                exists = await cur.fetchone()
                if not exists:
                    if not re.match(r"^[a-zA-Z0-9_]+$", DB_NAME):
                        raise ValueError(f"Invalid database name: {DB_NAME}")
                    await cur.execute(f'CREATE DATABASE "{DB_NAME}"')
                    print(f"Banco de dados '{DB_NAME}' criado com sucesso!")
            await conn.close()
    except Exception as e:
        print(f"Aviso: Não foi possível checar/criar o banco de dados automaticamente. Detalhe: {e}")
    # --- AUTO CREATE DB END ---

    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
