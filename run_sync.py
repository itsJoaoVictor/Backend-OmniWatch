import asyncio
from app.core.database import AsyncSessionLocal
from app.users.models import User
from app.media.models import Media, MediaRelease
from app.tracking.models import UserListItem
from app.calendar.sync import sync_media_releases

async def main():
    async with AsyncSessionLocal() as db:
        await sync_media_releases(db)
        print("Sync complete!")

if __name__ == '__main__':
    asyncio.run(main())
