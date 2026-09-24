import asyncio, os, httpx
from dotenv import load_dotenv

load_dotenv()

async def t():
    async with httpx.AsyncClient() as client:
        res = await client.get('https://api.themoviedb.org/3/movie/1320786?language=pt-BR', headers={'Authorization': f'Bearer {os.getenv("TMDB_API_KEY")}'})
        data = res.json()
        print('Terrifier 4 title:', data.get('title'))
        print('Terrifier 4 status:', data.get('status'))
        print('Terrifier 4 release_date:', repr(data.get('release_date')))

asyncio.run(t())
