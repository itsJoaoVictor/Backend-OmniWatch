import asyncio, os, httpx, asyncpg
from dotenv import load_dotenv

load_dotenv()

async def populate():
    conn = await asyncpg.connect(
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT")),
        database=os.getenv("DB_NAME")
    )
    rows = await conn.fetch("SELECT id, title, tmdb_id, media_type FROM media WHERE media_type = 'movie' AND (release_date IS NULL OR release_date = '');")
    headers = {"Authorization": f"Bearer {os.getenv('TMDB_API_KEY')}"}
    async with httpx.AsyncClient() as client:
        for r in rows:
            res = await client.get(f"https://api.themoviedb.org/3/movie/{r['tmdb_id']}?language=pt-BR", headers=headers)
            if res.status_code == 200:
                data = res.json()
                rel = data.get("release_date")
                status = data.get("status")
                print(f"Movie '{r['title']}' (TMDB {r['tmdb_id']}): status='{status}', release_date='{rel}'")
                if rel:
                    await conn.execute("UPDATE media SET release_date = $1 WHERE id = $2;", rel, r["id"])
                    print(f"  -> Updated release_date to '{rel}'")
                elif status:
                    # Se não tem data, garante que fica vazio mas sabemos que não lançou
                    pass
    await conn.close()

asyncio.run(populate())
