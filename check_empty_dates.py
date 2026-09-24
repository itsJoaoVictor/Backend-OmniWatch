import asyncio, asyncpg, os
from dotenv import load_dotenv

load_dotenv()

async def c():
    conn = await asyncpg.connect(
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT")),
        database=os.getenv("DB_NAME")
    )
    rows = await conn.fetch("SELECT id, title, tmdb_id, release_date FROM media WHERE media_type = 'movie' AND (release_date IS NULL OR release_date = '');")
    print(f"Total movies with empty or null release_date: {len(rows)}")
    for r in rows:
        print(f"  - {r['title']} (TMDB {r['tmdb_id']}): release_date='{r['release_date']}'")
    await conn.close()

asyncio.run(c())
