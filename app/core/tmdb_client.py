import asyncio
import logging
import random
from typing import Any, Dict, Optional, Tuple

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

# Concorrência máxima simultânea contra api.themoviedb.org por event loop
# (evita rajadas 429 quando múltiplas rotinas e funis rodam em paralelo).
TMDB_MAX_CONCURRENCY = 14
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

_clients_by_loop: Dict[int, httpx.AsyncClient] = {}
_semaphores_by_loop: Dict[int, asyncio.Semaphore] = {}


def _current_loop_id() -> int:
    try:
        return id(asyncio.get_running_loop())
    except RuntimeError:
        return 0


def get_tmdb_client() -> httpx.AsyncClient:
    """Retorna um AsyncClient reutilizável vinculado ao event loop atual."""
    loop_id = _current_loop_id()
    client = _clients_by_loop.get(loop_id)
    if client is None or client.is_closed:
        # Limpa clientes de loops antigos fechados
        stale = [k for k, c in _clients_by_loop.items() if c.is_closed]
        for k in stale:
            _clients_by_loop.pop(k, None)
            _semaphores_by_loop.pop(k, None)
        client = httpx.AsyncClient(
            timeout=12.0,
            limits=httpx.Limits(max_keepalive_connections=25, max_connections=60),
        )
        _clients_by_loop[loop_id] = client
    return client


def _get_tmdb_semaphore() -> asyncio.Semaphore:
    loop_id = _current_loop_id()
    sem = _semaphores_by_loop.get(loop_id)
    if sem is None:
        sem = asyncio.Semaphore(TMDB_MAX_CONCURRENCY)
        _semaphores_by_loop[loop_id] = sem
    return sem


def build_tmdb_auth(params: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """
    Normaliza credenciais TMDB suportando tanto Read Access Token v4 (JWT Bearer)
    quanto chave v3 hexadecimal de 32 caracteres.
    """
    clean_params: Dict[str, Any] = dict(params) if params else {}
    headers: Dict[str, str] = {"accept": "application/json"}

    raw_key = (settings.TMDB_API_KEY or "").strip()
    if raw_key.lower().startswith("bearer "):
        raw_key = raw_key[7:].strip()

    # Remove api_key acidentalmente passado se for um JWT Bearer
    if "api_key" in clean_params and ("." in str(clean_params["api_key"]) or len(str(clean_params["api_key"])) > 40):
        clean_params.pop("api_key", None)

    if raw_key:
        if "." in raw_key or len(raw_key) > 40:
            headers["Authorization"] = f"Bearer {raw_key}"
        else:
            clean_params.setdefault("api_key", raw_key)

    return headers, clean_params


def _compute_backoff_delay(attempt: int, response: Optional[httpx.Response] = None, base_delay: float = 0.8) -> float:
    if response is not None and response.status_code == 429:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                parsed = float(retry_after)
                if 0 < parsed <= 20.0:
                    return parsed + random.uniform(0.05, 0.35)
            except ValueError:
                pass
    return min(12.0, base_delay * (2 ** attempt)) + random.uniform(0.05, 0.4)


async def tmdb_request(
    endpoint_or_url: str,
    params: Optional[Dict[str, Any]] = None,
    timeout: float = 12.0,
    max_attempts: int = 4,
) -> httpx.Response:
    """
    Executa GET contra o TMDB com semáforo global de concorrência e retry automático
    com exponential backoff + jitter contra HTTP 429, 5xx e timeouts/erros de rede.
    """
    if endpoint_or_url.startswith("http://") or endpoint_or_url.startswith("https://"):
        url = endpoint_or_url
    else:
        path = endpoint_or_url if endpoint_or_url.startswith("/") else f"/{endpoint_or_url}"
        url = f"{settings.TMDB_BASE_URL}{path}"

    headers, final_params = build_tmdb_auth(params)
    sem = _get_tmdb_semaphore()
    last_exc: Optional[Exception] = None

    for attempt in range(max_attempts):
        try:
            async with sem:
                client = get_tmdb_client()
                response = await client.get(url, headers=headers, params=final_params, timeout=timeout)

            if response.status_code in RETRYABLE_STATUS_CODES and attempt < max_attempts - 1:
                delay = _compute_backoff_delay(attempt, response=response)
                logger.warning(
                    f"[TMDB-RETRY] HTTP {response.status_code} em {endpoint_or_url} "
                    f"(tentativa {attempt + 1}/{max_attempts}) — aguardando {delay:.2f}s"
                )
                await asyncio.sleep(delay)
                continue

            return response

        except (httpx.TimeoutException, httpx.TransportError, httpx.RequestError) as exc:
            last_exc = exc
            if attempt < max_attempts - 1:
                delay = _compute_backoff_delay(attempt)
                logger.warning(
                    f"[TMDB-RETRY] {type(exc).__name__} em {endpoint_or_url} "
                    f"(tentativa {attempt + 1}/{max_attempts}) — aguardando {delay:.2f}s"
                )
                await asyncio.sleep(delay)
                continue
            logger.error(f"[TMDB-ERROR] Falha de transporte definitiva em {endpoint_or_url}: {type(exc).__name__}: {exc}")
            raise

    if last_exc:
        raise last_exc
    raise httpx.ConnectError(f"Falha ao conectar ao TMDB em {endpoint_or_url}")


async def tmdb_get_json(
    endpoint_or_url: str,
    params: Optional[Dict[str, Any]] = None,
    timeout: float = 12.0,
    max_attempts: int = 4,
) -> Optional[Dict[str, Any]]:
    """
    Atalho resiliente para rotinas internas/background: retorna o JSON em caso de HTTP 200
    ou None em caso de erro não-recuperável (404, etc.) sem derrubar o caller.
    """
    if not settings.TMDB_API_KEY:
        return None
    try:
        resp = await tmdb_request(
            endpoint_or_url=endpoint_or_url,
            params=params,
            timeout=timeout,
            max_attempts=max_attempts,
        )
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code != 404:
            logger.warning(f"[TMDB] Status inesperado {resp.status_code} em {endpoint_or_url}")
    except Exception as exc:
        logger.warning(f"[TMDB] Erro final em {endpoint_or_url}: {type(exc).__name__}: {exc}")
    return None
