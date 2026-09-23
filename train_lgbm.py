import asyncio
from app.core.database import AsyncSessionLocal
from app.recommendation.ranker import train_ranker_model

async def main():
    async with AsyncSessionLocal() as db:
        await train_ranker_model(db)

if __name__ == "__main__":
    asyncio.run(main())
