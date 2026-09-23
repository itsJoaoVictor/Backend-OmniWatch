import os
import re
import time
import httpx
from typing import Optional
from fastapi import APIRouter, HTTPException, Query, Response
from fastapi.responses import FileResponse

router = APIRouter()

CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".cache", "posters"))
os.makedirs(CACHE_DIR, exist_ok=True)

# Quota de Cache em Disco: Limite de 500 MB com redução para 400 MB quando ultrapassado (LRU)
MAX_CACHE_BYTES = 500 * 1024 * 1024
TARGET_CACHE_BYTES = 400 * 1024 * 1024

ALLOWED_SIZES = {"w92", "w154", "w185", "w342", "w500", "w780", "w1280", "original"}
SAFE_FILENAME_REGEX = re.compile(r"^[a-zA-Z0-9_\-\.]+\.(jpg|jpeg|png|webp|svg)$", re.IGNORECASE)

TMDB_BASE_URL = "https://image.tmdb.org/t/p"

def cleanup_cache_if_needed() -> None:
    """
    Política LRU (Least Recently Used):
    Verifica o tamanho total da pasta de cache. Se ultrapassar MAX_CACHE_BYTES (500 MB),
    remove os arquivos acessados há mais tempo até atingir TARGET_CACHE_BYTES (400 MB).
    """
    try:
        entries = []
        total_size = 0
        with os.scandir(CACHE_DIR) as it:
            for entry in it:
                if entry.is_file():
                    stat = entry.stat()
                    entries.append((stat.st_atime, stat.st_size, entry.path))
                    total_size += stat.st_size

        if total_size <= MAX_CACHE_BYTES:
            return

        # Ordena do menor atime (mais antigo) para o maior
        entries.sort(key=lambda x: x[0])

        current_size = total_size
        removed_count = 0
        for atime, size, path in entries:
            if current_size <= TARGET_CACHE_BYTES:
                break
            try:
                os.remove(path)
                current_size -= size
                removed_count += 1
            except OSError:
                pass

        print(f"[ImageCache LRU] Limpeza concluída: {removed_count} arquivos removidos. Tamanho atual: {current_size / (1024*1024):.2f} MB")
    except Exception as e:
        print(f"[ImageCache LRU] Erro ao executar limpeza de cache: {e}")

async def ensure_image_cached(path: Optional[str], size: str = "w342") -> Optional[str]:
    """
    Garante que uma imagem do TMDB esteja salva no disco local do servidor.
    Retorna o caminho absoluto do arquivo em cache ou None em caso de falha.
    """
    if not path or not isinstance(path, str) or not path.strip():
        return None

    clean_size = size if size in ALLOWED_SIZES else "w342"
    clean_path = path.strip().lstrip("/")

    if not SAFE_FILENAME_REGEX.match(clean_path):
        return None

    cache_file_path = os.path.join(CACHE_DIR, f"{clean_size}_{clean_path}")

    # Se já existir em disco, apenas atualiza o tempo de acesso (LRU)
    if os.path.exists(cache_file_path):
        try:
            os.utime(cache_file_path, None)
        except OSError:
            pass
        return cache_file_path

    # Baixa do TMDB de forma assíncrona
    tmdb_url = f"{TMDB_BASE_URL}/{clean_size}/{clean_path}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) OmniWatch/1.0"
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(tmdb_url, headers=headers)
            if resp.status_code == 200:
                with open(cache_file_path, "wb") as f:
                    f.write(resp.content)

                # Verifica se ultrapassou a cota de 500 MB
                cleanup_cache_if_needed()
                return cache_file_path
            else:
                return None
    except Exception as e:
        print(f"[ImageCache] Falha ao pré-carregar {clean_path}: {e}")
        return None

@router.get("/proxy")
async def proxy_image(
    path: str = Query(..., description="Caminho relativo da imagem do TMDB, ex: /686F0CEPmI4ZXjFbWtIHQOBwnfI.jpg"),
    size: str = Query("w342", description="Tamanho TMDB desejado, ex: w342, w500, original"),
):
    clean_size = size if size in ALLOWED_SIZES else "w342"
    clean_path = path.strip().lstrip("/")

    if not SAFE_FILENAME_REGEX.match(clean_path):
        raise HTTPException(status_code=400, detail="Nome de arquivo de imagem inválido.")

    cache_file_path = os.path.join(CACHE_DIR, f"{clean_size}_{clean_path}")
    is_hit = os.path.exists(cache_file_path)

    cached_path = await ensure_image_cached(path, clean_size)
    if not cached_path or not os.path.exists(cached_path):
        raise HTTPException(status_code=404, detail="Imagem não encontrada no TMDB.")

    ext = clean_path.split(".")[-1].lower()
    media_type = f"image/{ext}" if ext != "jpg" else "image/jpeg"

    return FileResponse(
        cached_path,
        media_type=media_type,
        headers={
            "Cache-Control": "public, max-age=31536000, immutable",
            "X-Proxy-Cache": "HIT" if is_hit else "MISS"
        }
    )
