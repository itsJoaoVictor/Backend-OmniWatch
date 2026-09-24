import asyncio, asyncpg, os, uuid
from dotenv import load_dotenv

load_dotenv()

async def check():
    conn = await asyncpg.connect(
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT")),
        database=os.getenv("DB_NAME")
    )
    r = await conn.fetchrow("""
        SELECT uli.id, uli.status, m.title, m.tmdb_id, m.media_type, m.release_date
        FROM user_list_items uli
        JOIN media m ON uli.media_id = m.id
        WHERE uli.id = $1
    """, uuid.UUID("5297a3f4-8164-4bda-9565-3856b2fc3068"))
    print("Found item:", dict(r) if r else "Not found")
    await conn.close()

asyncio.run(check())
