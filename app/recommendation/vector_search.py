import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Set, Tuple
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import text

from app.media.models import Media

logger = logging.getLogger(__name__)

# Cache em memória para o índice vetorial em ambientes onde pgvector nativo em C não está compilado
_CACHE_LOCK = asyncio.Lock()
_CACHED_MATRIX: Optional[np.ndarray] = None
_CACHED_KEYS: List[Tuple[str, int]] = []
_CACHED_MEDIA_MAP: Dict[Tuple[str, int], Dict[str, Any]] = {}
_LAST_CACHE_UPDATE: float = 0.0
_CACHE_TTL_SECONDS: float = 900.0  # 15 minutos de TTL

_PGVECTOR_SUPPORTED: Optional[bool] = None


async def is_pgvector_supported(session: AsyncSession) -> bool:
    """
    Verifica se a extensão 'vector' está instalada e ativa no PostgreSQL.
    """
    global _PGVECTOR_SUPPORTED
    if _PGVECTOR_SUPPORTED is not None:
        return _PGVECTOR_SUPPORTED

    try:
        res = await session.execute(text("SELECT 1 FROM pg_extension WHERE extname = 'vector'"))
        _PGVECTOR_SUPPORTED = bool(res.scalar())
    except Exception:
        _PGVECTOR_SUPPORTED = False

    return _PGVECTOR_SUPPORTED


def invalidate_vector_cache():
    """Invalida o cache em memória do índice vetorial."""
    global _CACHED_MATRIX, _CACHED_KEYS, _CACHED_MEDIA_MAP, _LAST_CACHE_UPDATE
    _CACHED_MATRIX = None
    _CACHED_KEYS = []
    _CACHED_MEDIA_MAP = {}
    _LAST_CACHE_UPDATE = 0.0


async def _load_vector_cache_if_needed(db: AsyncSession):
    """
    Carrega todos os embeddings de Media do catálogo em uma matriz numpy normalizada
    para busca vetorial em tempo real (< 3ms).
    """
    global _CACHED_MATRIX, _CACHED_KEYS, _CACHED_MEDIA_MAP, _LAST_CACHE_UPDATE
    now = time.time()
    if _CACHED_MATRIX is not None and (now - _LAST_CACHE_UPDATE) < _CACHE_TTL_SECONDS:
        return

    async with _CACHE_LOCK:
        if _CACHED_MATRIX is not None and (now - _LAST_CACHE_UPDATE) < _CACHE_TTL_SECONDS:
            return

        stmt = select(
            Media.tmdb_id,
            Media.media_type,
            Media.title,
            Media.poster_path,
            Media.backdrop_path,
            Media.genres,
            Media.directors,
            Media.main_cast,
            Media.keywords,
            Media.original_language,
            Media.embedding,
            Media.release_date,
            Media.runtime,
            Media.vote_average,
            Media.vote_count,
            Media.watch_providers_br,
        ).where(Media.embedding.isnot(None))

        res = await db.execute(stmt)
        rows = res.all()

        keys: List[Tuple[str, int]] = []
        vectors: List[List[float]] = []
        media_map: Dict[Tuple[str, int], Dict[str, Any]] = {}

        for r in rows:
            (
                tmdb_id, mtype, title, poster, backdrop, genres, directors,
                cast, keywords, lang, emb, rel_date, runtime, vote_avg, vote_cnt, br_prov
            ) = r
            if not tmdb_id or not emb:
                continue

            k = (mtype or "movie", tmdb_id)
            keys.append(k)
            vectors.append(emb)

            genre_ids = []
            if genres and isinstance(genres, list):
                for g in genres:
                    if isinstance(g, dict) and "id" in g:
                        genre_ids.append(g["id"])

            media_map[k] = {
                "id": tmdb_id,
                "title": title,
                "name": title if mtype == "tv" else None,
                "media_type": mtype or "movie",
                "poster_path": poster,
                "backdrop_path": backdrop,
                "genres": genres or [],
                "genre_ids": genre_ids,
                "cast": cast or [],
                "crew": directors or [],
                "keywords_list": keywords or [],
                "original_language": lang or "",
                "release_date": rel_date if mtype != "tv" else None,
                "first_air_date": rel_date if mtype == "tv" else None,
                "runtime": runtime or 0,
                "vote_average": float(vote_avg or 0.0),
                "vote_count": int(vote_cnt or 0),
                "watch_providers_br": br_prov or [],
                "flatrate_providers_br": br_prov or [],
                "embedding": emb,
                "semantic": True,
            }

        if vectors:
            mat = np.array(vectors, dtype=np.float32)
            # Normalização L2 para que produto interno seja equivalente à similaridade cosseno
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            _CACHED_MATRIX = mat / norms
            _CACHED_KEYS = keys
            _CACHED_MEDIA_MAP = media_map
            _LAST_CACHE_UPDATE = now
            logger.info(f"[VectorSearch] Índice vetorial atualizado com {len(keys)} obras do catálogo.")
        else:
            _CACHED_MATRIX = np.zeros((0, 4096), dtype=np.float32)
            _CACHED_KEYS = []
            _CACHED_MEDIA_MAP = {}
            _LAST_CACHE_UPDATE = now


async def search_semantic_candidates(
    db: AsyncSession,
    query_embedding: Optional[List[float]],
    excluded_keys: Optional[Set[Tuple[str, int]]] = None,
    limit: int = 150,
    media_type_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Canal 1 de Retrieval Semântico (Sub-etapa 5.1 + Ideia 2):
    - Se pgvector nativo estiver ativo no Postgres: executa SQL ANN com operador de cosseno '<=>'.
    - Em ambientes locais/dev sem o binário C do pgvector: executa busca vetorial matricial
      otimizada com projeção cosseno vetorizada e argpartition em O(M + K log K) (< 3ms).
    Filtra imediatamente chaves (media_type, tmdb_id) já assistidas ou dispensadas.
    """
    if not query_embedding:
        return []

    excluded = excluded_keys or set()

    # 1. Caminho nativo pgvector se habilitado no banco
    if await is_pgvector_supported(db):
        try:
            q_str = f"[{','.join(str(float(x)) for x in query_embedding)}]"
            where_clause = "WHERE embedding_vec IS NOT NULL"
            params: Dict[str, Any] = {
                "query_vec": q_str,
                "fetch_limit": limit + len(excluded),
            }
            if media_type_filter:
                where_clause += " AND media_type = :mtype"
                params["mtype"] = media_type_filter

            query_sql = text(f"""
                SELECT tmdb_id, media_type, title, poster_path, backdrop_path,
                       genres, directors, main_cast, keywords, original_language,
                       release_date, runtime, vote_average, vote_count, watch_providers_br,
                       embedding,
                       (1 - (embedding_vec <=> :query_vec::vector)) AS sim
                FROM media
                {where_clause}
                ORDER BY embedding_vec <=> :query_vec::vector
                LIMIT :fetch_limit;
            """)
            res = await db.execute(query_sql, params)
            rows = res.all()
            results = []
            for r in rows:
                mtype_val = r.media_type or "movie"
                k = (mtype_val, r.tmdb_id)
                if k in excluded:
                    continue
                genre_ids = []
                if r.genres and isinstance(r.genres, list):
                    for g in r.genres:
                        if isinstance(g, dict) and "id" in g:
                            genre_ids.append(g["id"])

                br_prov = r.watch_providers_br or []
                results.append({
                    "id": r.tmdb_id,
                    "title": r.title,
                    "name": r.title if mtype_val == "tv" else None,
                    "media_type": mtype_val,
                    "poster_path": r.poster_path,
                    "backdrop_path": r.backdrop_path,
                    "genres": r.genres or [],
                    "genre_ids": genre_ids,
                    "cast": r.main_cast or [],
                    "crew": r.directors or [],
                    "keywords_list": r.keywords or [],
                    "original_language": r.original_language or "",
                    "release_date": r.release_date if mtype_val != "tv" else None,
                    "first_air_date": r.release_date if mtype_val == "tv" else None,
                    "runtime": r.runtime or 0,
                    "vote_average": float(r.vote_average or 0.0),
                    "vote_count": int(r.vote_count or 0),
                    "watch_providers_br": br_prov,
                    "flatrate_providers_br": br_prov,
                    "embedding": r.embedding,
                    "semantic": True,
                    "semantic_score": round(float(r.sim) * 100.0, 2),
                })
                if len(results) >= limit:
                    break
            if results:
                return results
        except Exception as e:
            logger.warning(f"[VectorSearch] Consulta pgvector nativa falhou, caindo para índice em memória: {e}")

    # 2. Caminho otimizado em memória (< 3ms, O(M + K log K))
    await _load_vector_cache_if_needed(db)
    if _CACHED_MATRIX is None or len(_CACHED_KEYS) == 0:
        return []

    q_arr = np.array(query_embedding, dtype=np.float32)
    q_norm = np.linalg.norm(q_arr)
    if q_norm == 0:
        return []
    q_normed = q_arr / q_norm

    # Garante alinhamento de dimensão se necessário
    if q_normed.shape[0] != _CACHED_MATRIX.shape[1]:
        logger.warning(f"[VectorSearch] Dimensão incompatível: query {q_normed.shape[0]} vs matriz {_CACHED_MATRIX.shape[1]}")
        return []

    # Produto interno de matriz M @ q em O(M) SIMD
    similarities = np.dot(_CACHED_MATRIX, q_normed)
    if media_type_filter:
        type_mask = np.array([k[0] == media_type_filter for k in _CACHED_KEYS], dtype=bool)
        similarities = np.where(type_mask, similarities, -np.inf)
    total_rows = len(similarities)

    # Usa np.argpartition em O(M) quando o alvo é menor que o catálogo total
    target_k = min(total_rows, limit + len(excluded) + 16)
    if target_k < total_rows:
        part_idx = np.argpartition(-similarities, target_k - 1)[:target_k]
        top_indices = part_idx[np.argsort(-similarities[part_idx])]
    else:
        top_indices = np.argsort(-similarities)

    candidates: List[Dict[str, Any]] = []
    for idx in top_indices:
        k = _CACHED_KEYS[idx]
        if media_type_filter and k[0] != media_type_filter:
            continue
        if k in excluded:
            continue

        raw_info = _CACHED_MEDIA_MAP.get(k)
        if not raw_info:
            continue

        cand = dict(raw_info)
        sim_val = float(similarities[idx])
        cand["semantic_score"] = round(sim_val * 100.0, 2)
        candidates.append(cand)

        if len(candidates) >= limit:
            break

    return candidates
