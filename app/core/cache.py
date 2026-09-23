import time
from typing import Any, Dict, Tuple
from app.core.config import settings

class SimpleTTLCache:
    def __init__(self, ttl_seconds: int = 86400): # 24 horas por padrão
        self.ttl = 0 if settings.DEBUG else ttl_seconds
        self._cache: Dict[str, dict] = {}
        
    def get_with_status(self, key: str) -> Tuple[Any | None, bool]:
        """
        Retorna (dado, is_stale).
        is_stale é True se o dado não existe ou passou do tempo de TTL.
        """
        if key in self._cache:
            entry = self._cache[key]
            is_stale = (time.time() - entry["timestamp"]) >= self.ttl
            return entry["data"], is_stale
        return None, True

    def set(self, key: str, value: Any) -> None:
        self._cache[key] = {
            "data": value,
            "timestamp": time.time()
        }

    def invalidate_prefix(self, prefix: str) -> None:
        """
        Marca as chaves que começam com o prefixo como expiradas (stale).
        Mantém os dados para o SWR (Stale-While-Revalidate) funcionar.
        """
        for k in self._cache:
            if k.startswith(prefix):
                self._cache[k]["timestamp"] = 0

# Global cache instance for trending items
trending_cache = SimpleTTLCache(ttl_seconds=3600) # 1 hour
search_cache = SimpleTTLCache(ttl_seconds=1800) # 30 minutes
details_cache = SimpleTTLCache(ttl_seconds=3600) # 1 hour