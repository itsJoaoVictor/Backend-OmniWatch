import asyncio
import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

from app.core.database import AsyncSessionLocal
from sqlalchemy.future import select
from app.users.models import User
from app.media.models import Media
from app.tracking.models import UserListItem
from app.details.services import fetch_movie_details, fetch_tv_details
from app.tracking.services import invalidate_user_stats_cache

async def enrich():
    print("Iniciando enriquecimento de profile_path para elenco e diretores...")
    t0 = time.perf_counter()

    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Media))
        media_list = result.scalars().all()
        total = len(media_list)
        print(f"Total de {total} obras no banco para processar.")

        sem = asyncio.Semaphore(10)
        updated_count = 0
        error_count = 0

        async def process_media(m):
            nonlocal updated_count, error_count
            async with sem:
                try:
                    if m.media_type == "movie":
                        det = await fetch_movie_details(m.tmdb_id)
                    else:
                        det = await fetch_tv_details(m.tmdb_id)

                    m.cast_ids = det.cast_ids or []
                    m.crew_ids = det.crew_ids or []
                    updated_count += 1
                except Exception as e:
                    error_count += 1
                    print(f"  Erro ao buscar [{m.media_type}] {m.title} ({m.tmdb_id}): {e}")

        # Executa em lotes de 30 para comitar no banco progressivamente
        batch_size = 30
        for i in range(0, total, batch_size):
            batch = media_list[i : i + batch_size]
            await asyncio.gather(*(process_media(m) for m in batch))
            await db.commit()
            print(f"  Progresso: {min(i + batch_size, total)}/{total} obras processadas...")

        print(f"Concluído em {time.perf_counter() - t0:.1f}s! Atualizadas: {updated_count}, Erros: {error_count}")

if __name__ == "__main__":
    asyncio.run(enrich())
