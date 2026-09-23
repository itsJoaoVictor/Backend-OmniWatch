import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.core.database import AsyncSessionLocal
from app.media.models import Media
from app.details.services import fetch_movie_details, fetch_tv_details

async def repopulate():
    async with AsyncSessionLocal() as db:
        # Find media where main_cast or directors is empty
        # We'll just fetch all and check in python to be safe with JSON arrays
        result = await db.execute(select(Media))
        all_media = result.scalars().all()
        
        updated_count = 0
        for m in all_media:
            needs_update = False
            
            # Convert JSON lists if needed, some might be empty lists []
            genres = m.genres or []
            directors = m.directors or []
            main_cast = m.main_cast or []
            
            if not directors or not main_cast or not genres:
                print(f"Updating '{m.title}' (TMDB ID: {m.tmdb_id}, Type: {m.media_type})...")
                try:
                    if m.media_type == "movie":
                        tmdb_data = await fetch_movie_details(m.tmdb_id)
                    else:
                        tmdb_data = await fetch_tv_details(m.tmdb_id)
                        
                    if not genres and tmdb_data.genres:
                        m.genres = [g.name for g in tmdb_data.genres]
                        needs_update = True
                        
                    if not directors and tmdb_data.credits and tmdb_data.credits.crew:
                        m.directors = [c.name for c in tmdb_data.credits.crew]
                        needs_update = True
                        
                    if not main_cast and tmdb_data.credits and tmdb_data.credits.cast:
                        m.main_cast = [c.name for c in tmdb_data.credits.cast]
                        needs_update = True
                        
                    if needs_update:
                        updated_count += 1
                        
                except Exception as e:
                    print(f"Failed to fetch details for '{m.title}': {e}")
                    
            if needs_update:
                await db.commit()
                print(f"-> Updated '{m.title}' successfully.")
                
        print(f"\nDone! Repopulated {updated_count} media items.")

if __name__ == "__main__":
    asyncio.run(repopulate())
