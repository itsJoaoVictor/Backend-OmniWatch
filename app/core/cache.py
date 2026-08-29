import time
from typing import Any, Dict, Tuple

class SimpleTTLCache:
    def __init__(self, ttl_seconds: int = 86400): # 24 horas por padrão
        self.ttl = ttl_seconds
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

# Global cache instance for trending items
trending_cache = SimpleTTLCache(ttl_seconds=86400) # 24 hours
