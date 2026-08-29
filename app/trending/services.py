import httpx
from fastapi import HTTPException, status
from app.core.config import settings
from app.core.cache import trending_cache

async def get_trending_week(language: str = "pt-BR", page: int = 1) -> dict:
    cache_key = f"trending:week:{language}:{page}"
    
    # Busca do cache sabendo se está fresco ou desatualizado (stale)
    cached_data, is_stale = trending_cache.get_with_status(cache_key)
    
    # Se o dado existe e está FRESCO, retornamos imediatamente (Hit do Cache)
    if cached_data is not None and not is_stale:
        return cached_data
        
    url = f"{settings.TMDB_BASE_URL}/trending/all/week"
    params = {
        "language": language,
        "page": page
    }
    headers = {
        "Authorization": f"Bearer {settings.TMDB_API_KEY}",
        "accept": "application/json"
    }
    
    async with httpx.AsyncClient() as client:
        try:
            # Tenta buscar os dados novos (para popular um cache vazio ou renovar um stale)
            response = await client.get(url, params=params, headers=headers, timeout=10.0)
            response.raise_for_status()
            data = response.json()
            
            # Atualiza o cache com o novo dado fresquinho
            trending_cache.set(cache_key, data)
            return data
        except httpx.HTTPError as e:
            # MAGIA ACONTECE AQUI (Stale-If-Error):
            # Se a API do TMDB quebrou, mas temos um dado antigo salvo no cache, salvamos o dia!
            if cached_data is not None:
                print(f"⚠️ [RESILIÊNCIA] TMDB falhou. Retornando cache antigo (Stale-If-Error) para {cache_key}. Erro: {e}")
                return cached_data
            
            # Se não temos cache e o TMDB quebrou, aí sim o usuário vê o erro.
            print(f"❌ TMDB HTTP Error: {e}")
            if hasattr(e, "response") and e.response is not None:
                print(f"TMDB Response: {e.response.text}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Failed to fetch trending data from upstream provider and no fallback cache is available."
            )
