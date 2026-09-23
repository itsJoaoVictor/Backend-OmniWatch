import sys
import os
sys.path.insert(0, os.path.abspath("."))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import asyncio
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from app.core.database import AsyncSessionLocal

import app.users.models
import app.media.models
import app.tracking.models
import app.media_collections.models

from app.tracking.models import UserListItem
from app.images.router import ensure_image_cached, CACHE_DIR

async def warmup_cache():
    print("Iniciando pré-aquecimento de cache dos pôsteres da Minha Lista...")
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(UserListItem).options(selectinload(UserListItem.media))
        )
        items = result.scalars().all()
        print(f"Total de itens na Minha Lista encontrados: {len(items)}")

        cached_count = 0
        skipped_count = 0
        failed_count = 0

        for idx, item in enumerate(items, 1):
            media = item.media
            if not media or not media.poster_path:
                skipped_count += 1
                continue

            print(f"[{idx}/{len(items)}] Cacheando poster de: {media.title} ({media.poster_path})...")
            cached_path = await ensure_image_cached(media.poster_path, "w342")
            if cached_path:
                cached_count += 1
            else:
                failed_count += 1

        # Estatísticas finais da pasta de cache
        total_files = 0
        total_bytes = 0
        if os.path.exists(CACHE_DIR):
            for f in os.scandir(CACHE_DIR):
                if f.is_file():
                    total_files += 1
                    total_bytes += f.stat().st_size

        mb_size = total_bytes / (1024 * 1024)
        print("\n--- RESUMO DO WARM-UP ---")
        print(f"Pôsteres processados com sucesso: {cached_count}")
        print(f"Itens sem pôster/ignorados: {skipped_count}")
        print(f"Falhas no download: {failed_count}")
        print(f"Total de arquivos na pasta de cache: {total_files}")
        print(f"Espaço total ocupado em disco: {mb_size:.2f} MB (Limite LRU: 500 MB)")
        print("Pré-aquecimento concluído!")

if __name__ == "__main__":
    asyncio.run(warmup_cache())
