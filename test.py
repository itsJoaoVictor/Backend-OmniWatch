import asyncio
from app.trending.services import get_trending_week
from app.core.config import settings
import httpx

print(f"API KEY: {settings.TMDB_API_KEY[:10]}...")

async def test():
    try:
        data = await get_trending_week()
        print("SUCCESS:", data.keys())
    except Exception as e:
        print(f"EXCEPTION: {repr(e)}")

asyncio.run(test())
