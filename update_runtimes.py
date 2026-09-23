import asyncio
import sys
import os
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.core.database import engine
from app.media.models import Media
from app.details.services import fetch_movie_details, fetch_tv_details

async_session = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

async def main():
    async with async_session() as db:
        result = await db.execute(select(Media))
        medias = result.scalars().all()
        
        print(f'Verificando {len(medias)} itens para atualizar runtime e generos...')
        
        for media in medias:
            updated = False
            try:
                if media.media_type == 'movie':
                    if media.runtime == None or media.runtime == 0 or not media.genres:
                        data = await fetch_movie_details(media.tmdb_id)
                        if media.runtime == None or media.runtime == 0:
                            media.runtime = data.runtime or 0
                            updated = True
                        if not media.genres and data.genres:
                            media.genres = [g.name for g in data.genres]
                            updated = True
                else:
                    if media.runtime == None or media.runtime == 0 or not media.genres:
                        data = await fetch_tv_details(media.tmdb_id)
                        if media.runtime == None or media.runtime == 0:
                            if data.episode_run_time and len(data.episode_run_time) > 0:
                                media.runtime = data.episode_run_time[0]
                            else:
                                media.runtime = 45 # Default
                            updated = True
                        if not media.genres and data.genres:
                            media.genres = [g.name for g in data.genres]
                            updated = True
                
                if updated:
                    print(f'Atualizado {media.title} - runtime: {media.runtime}, genres: {media.genres}')
            except Exception as e:
                print(f'Erro ao atualizar {media.title}: {e}')
                
            await asyncio.sleep(0.1) # Evitar rate limit
            
        await db.commit()
        print('Atualização concluída.')

if __name__ == '__main__':
    asyncio.run(main())
