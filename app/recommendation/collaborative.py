from __future__ import annotations
import asyncio
import logging
import time
import uuid
from typing import List, Dict, Any, Set, Optional, Tuple, Union
import numpy as np
import scipy.sparse as sp

try:
    import implicit
except ImportError:
    implicit = None

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload

from app.users.models import User
from app.tracking.models import UserListItem
from app.media.models import Media

logger = logging.getLogger(__name__)

# Cache do modelo ALS em memória
_ALS_LOCK = asyncio.Lock()
_ALS_MODEL: Optional[implicit.als.AlternatingLeastSquares] = None
_USER_ITEM_MATRIX: Optional[sp.csr_matrix] = None
_USER_MAP: Dict[uuid.UUID, int] = {}
_REVERSE_USER_MAP: Dict[int, uuid.UUID] = {}
_ITEM_MAP: Dict[Tuple[str, int], int] = {}
_REVERSE_ITEM_MAP: Dict[int, Tuple[str, int]] = {}
_LAST_ALS_FIT_TIME: float = 0.0
_ALS_TTL_SECONDS: float = 900.0  # 15 minutos


def invalidate_als_cache():
    """Invalida o cache do modelo ALS para retreino na próxima chamada."""
    global _ALS_MODEL, _USER_ITEM_MATRIX, _USER_MAP, _REVERSE_USER_MAP, _ITEM_MAP, _REVERSE_ITEM_MAP, _LAST_ALS_FIT_TIME
    _ALS_MODEL = None
    _USER_ITEM_MATRIX = None
    _USER_MAP = {}
    _REVERSE_USER_MAP = {}
    _ITEM_MAP = {}
    _REVERSE_ITEM_MAP = {}
    _LAST_ALS_FIT_TIME = 0.0


def get_collaborative_rrf_weight() -> float:
    """
    Peso adaptativo do canal colaborativo no RRF conforme a maturidade da base de usuários.
    Em bases pequenas (< 10 usuários ativos), usa peso moderado (0.50) para não sobrepor o conteúdo;
    escala suavemente até 1.20 conforme a base atinge 30+ usuários ativos.
    """
    n_users = len(_USER_MAP)
    if n_users < 10:
        return 0.50
    if n_users >= 30:
        return 1.20
    return round(0.50 + 0.70 * ((n_users - 10) / 20.0), 2)


async def _get_or_train_als_model(db: AsyncSession) -> Tuple[
    Optional[implicit.als.AlternatingLeastSquares],
    Optional[sp.csr_matrix],
    Dict[uuid.UUID, int],
    Dict[int, Tuple[str, int]]
]:
    """
    Treina ou recupera o modelo ALS de fatoração matricial implícita (Sub-etapa 5.2 + Ideia 9)
    ponderado pela confiança dos sinais implícitos/explícitos dos usuários:
      c_{ui} = 1 + alpha * r_{ui}
    """
    global _ALS_MODEL, _USER_ITEM_MATRIX, _USER_MAP, _REVERSE_USER_MAP, _ITEM_MAP, _REVERSE_ITEM_MAP, _LAST_ALS_FIT_TIME

    if implicit is None:
        return None, None, {}, {}

    now = time.time()
    if _ALS_MODEL is not None and (now - _LAST_ALS_FIT_TIME) < _ALS_TTL_SECONDS:
        return _ALS_MODEL, _USER_ITEM_MATRIX, _USER_MAP, _REVERSE_ITEM_MAP

    async with _ALS_LOCK:
        if _ALS_MODEL is not None and (now - _LAST_ALS_FIT_TIME) < _ALS_TTL_SECONDS:
            return _ALS_MODEL, _USER_ITEM_MATRIX, _USER_MAP, _REVERSE_ITEM_MAP

        # Carrega todas as interações de listas no banco
        stmt = (
            select(
                UserListItem.user_id,
                Media.media_type,
                Media.tmdb_id,
                UserListItem.rating,
                UserListItem.status,
                UserListItem.is_favorite,
                UserListItem.rewatch_count
            )
            .join(Media, UserListItem.media_id == Media.id)
        )
        res = await db.execute(stmt)
        rows = res.all()

        if not rows:
            return None, None, {}, {}

        # Filtra apenas sinais reais de consumo positivo (exclui plan_to_watch/upcoming sem nota e obras rejeitadas < 3.0★)
        positive_rows = []
        for r in rows:
            uid, mtype, tmdb_id, rating, status, is_fav, rewatch_cnt = r
            if not uid or not tmdb_id:
                continue
            if rating is not None and float(rating) < 3.0 and not is_fav:
                continue
            if rating is None and status not in ("completed", "watching") and not is_fav and not (rewatch_cnt and rewatch_cnt > 0):
                continue
            positive_rows.append(r)

        if not positive_rows:
            return None, None, {}, {}

        user_ids = sorted(list(set(r[0] for r in positive_rows if r[0])))
        item_keys = sorted(list(set((r[1] or "movie", r[2]) for r in positive_rows if r[2])))

        if not user_ids or not item_keys:
            return None, None, {}, {}

        user_map = {uid: i for i, uid in enumerate(user_ids)}
        rev_user_map = {i: uid for uid, i in user_map.items()}
        item_map = {k: j for j, k in enumerate(item_keys)}
        rev_item_map = {j: k for k, j in item_map.items()}

        rows_idx = []
        cols_idx = []
        data = []

        alpha_weight = 40.0
        for uid, mtype, tmdb_id, rating, status, is_fav, rewatch_cnt in positive_rows:
            u_i = user_map[uid]
            i_j = item_map[(mtype or "movie", tmdb_id)]

            # Cálculo de afinidade r_{ui} e confiança c_{ui} = 1 + alpha * r_{ui}
            r_ui = 0.5  # Base de interação positiva
            if rating:
                r_ui = max(0.2, float(rating) / 2.5)
            elif status == "completed":
                r_ui = 1.2
            elif status == "watching":
                r_ui = 1.0

            if is_fav:
                r_ui += 1.5
            if rewatch_cnt:
                r_ui += min(2.0, (rewatch_cnt or 0) * 0.5)

            conf = 1.0 + (alpha_weight * r_ui)

            rows_idx.append(u_i)
            cols_idx.append(i_j)
            data.append(conf)

        user_item_matrix = sp.csr_matrix((data, (rows_idx, cols_idx)), shape=(len(user_ids), len(item_keys)))

        try:
            # Treina ALS com 32 fatores latentes e regularização L2
            model = implicit.als.AlternatingLeastSquares(
                factors=32,
                regularization=0.05,
                iterations=15,
                random_state=42
            )
            model.fit(user_item_matrix, show_progress=False)

            _ALS_MODEL = model
            _USER_ITEM_MATRIX = user_item_matrix
            _USER_MAP = user_map
            _REVERSE_USER_MAP = rev_user_map
            _ITEM_MAP = item_map
            _REVERSE_ITEM_MAP = rev_item_map
            _LAST_ALS_FIT_TIME = now
            logger.info(f"[ALS] Modelo treinado com sucesso: {len(user_ids)} usuários, {len(item_keys)} obras.")
            return _ALS_MODEL, _USER_ITEM_MATRIX, _USER_MAP, _REVERSE_ITEM_MAP
        except Exception as e:
            logger.error(f"[ALS] Falha ao treinar modelo ALS: {e}")
            return None, None, {}, {}


async def find_similar_users(
    db: AsyncSession,
    target_user_id: str,
    n_neighbors: int = 5,
    min_similarity: float = 0.20
) -> List[Dict[str, Any]]:
    """
    Encontra os usuários mais parecidos utilizando as representações latentes do modelo ALS
    (ou fallback de sobreposição esparsa).
    """
    try:
        target_uuid = uuid.UUID(str(target_user_id))
    except (ValueError, TypeError):
        return []

    model, user_item_matrix, user_map, _ = await _get_or_train_als_model(db)
    if model is None or target_uuid not in user_map:
        return []

    target_idx = user_map[target_uuid]
    user_factors = model.user_factors
    target_factor = user_factors[target_idx]

    # Similaridade cosseno entre os fatores latentes de todos os usuários
    norms = np.linalg.norm(user_factors, axis=1)
    target_norm = np.linalg.norm(target_factor)
    if target_norm == 0 or np.any(norms == 0):
        norms[norms == 0] = 1.0

    dot_prods = np.dot(user_factors, target_factor)
    sims = dot_prods / (norms * (target_norm or 1.0))

    ranked_indices = np.argsort(-sims)
    similar_users = []

    for idx in ranked_indices:
        if idx == target_idx:
            continue
        sim_val = float(sims[idx])
        if sim_val < min_similarity:
            continue
        peer_uuid = _REVERSE_USER_MAP.get(idx)
        if peer_uuid:
            similar_users.append({
                "user_id": str(peer_uuid),
                "similarity": round(sim_val, 3),
                "shared_interests": []
            })
            if len(similar_users) >= n_neighbors:
                break

    return similar_users


async def fetch_collaborative_candidates(
    db: AsyncSession,
    target_user_id: str,
    seen_media_keys: Optional[Union[Set[Tuple[str, int]], Set[int]]] = None,
    limit: int = 60,
    min_similarity: float = 0.15,
    seen_tmdb_ids: Optional[Set[int]] = None,
    media_type_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Canal 3 de Recuperação Colaborativa Implícita (Sub-etapa 5.2 + Ideia 9):
    Gera recomendações a partir da fatoração matricial ALS (implicit), capturando
    correlações latentes ("quem amou X também amou Y") e descobrindo 'hidden gems'.
    """
    effective_seen_keys = seen_media_keys if seen_media_keys is not None else (seen_tmdb_ids or set())

    try:
        target_uuid = uuid.UUID(str(target_user_id))
    except (ValueError, TypeError):
        return []

    model, user_item_matrix, user_map, rev_item_map = await _get_or_train_als_model(db)
    if model is None or not rev_item_map:
        return []

    recommended_keys_with_score: List[Tuple[Tuple[str, int], float]] = []

    if target_uuid in user_map and user_item_matrix is not None:
        target_idx = user_map[target_uuid]
        try:
            fetch_n = limit * 4 if media_type_filter else limit * 2
            ids, scores = model.recommend(
                target_idx,
                user_item_matrix[target_idx],
                N=fetch_n,
                filter_already_liked_items=True
            )
            for item_idx, score in zip(ids, scores):
                m_key = rev_item_map.get(int(item_idx))
                if m_key and m_key not in effective_seen_keys:
                    if media_type_filter and m_key[0] != media_type_filter:
                        continue
                    recommended_keys_with_score.append((m_key, float(score)))
        except Exception as e:
            logger.warning(f"[ALS] Erro no recommend direto: {e}")

    # Fallback para cold-start (usuário novo sem interações prévias no treino):
    # Usa similaridade Item-Item ALS a partir das obras positivas da lista do usuário
    if len(recommended_keys_with_score) < limit:
        try:
            user_items_stmt = select(UserListItem.media_id, UserListItem.rating).where(
                UserListItem.user_id == target_uuid
            )
            ui_res = await db.execute(user_items_stmt)
            liked_media_ids = [r[0] for r in ui_res.all() if (r[1] is None or r[1] >= 3.0)]

            if liked_media_ids:
                media_stmt = select(Media.media_type, Media.tmdb_id).where(Media.id.in_(liked_media_ids))
                m_res = await db.execute(media_stmt)
                user_seed_keys = [
                    (r[0] or "movie", r[1]) for r in m_res.all()
                    if r[1] and (not media_type_filter or (r[0] or "movie") == media_type_filter)
                ]

                # Mapeia para índices do modelo
                seed_indices = [_ITEM_MAP[k] for k in user_seed_keys if k in _ITEM_MAP]
                if seed_indices:
                    sim_ids, sim_scores = model.similar_items(seed_indices[:4], N=limit * 2)
                    if sim_ids.ndim == 2:
                        for row_ids, row_scs in zip(sim_ids, sim_scores):
                            for s_id, s_sc in zip(row_ids, row_scs):
                                m_k = rev_item_map.get(int(s_id))
                                if m_k and m_k not in effective_seen_keys:
                                    if media_type_filter and m_k[0] != media_type_filter:
                                        continue
                                    recommended_keys_with_score.append((m_k, float(s_sc)))
                    else:
                        for s_id, s_sc in zip(sim_ids, sim_scores):
                            m_k = rev_item_map.get(int(s_id))
                            if m_k and m_k not in effective_seen_keys:
                                if media_type_filter and m_k[0] != media_type_filter:
                                    continue
                                recommended_keys_with_score.append((m_k, float(s_sc)))
        except Exception as e:
            logger.warning(f"[ALS] Erro no item-to-item fallback: {e}")

    if not recommended_keys_with_score:
        return []

    # Desduplica mantendo o maior score colaborativo
    best_collab: Dict[Tuple[str, int], float] = {}
    for k, sc in recommended_keys_with_score:
        if k not in best_collab or sc > best_collab[k]:
            best_collab[k] = sc

    sorted_keys = sorted(best_collab.keys(), key=lambda k: best_collab[k], reverse=True)[:limit]
    if not sorted_keys:
        return []

    # Carrega metadados completos de Media no banco
    tmdb_ids = [k[1] for k in sorted_keys]
    media_records = await db.execute(
        select(Media).where(Media.tmdb_id.in_(tmdb_ids))
    )
    media_by_key = {(m.media_type or "movie", m.tmdb_id): m for m in media_records.scalars().all()}

    candidates: List[Dict[str, Any]] = []
    for k in sorted_keys:
        m = media_by_key.get(k)
        if not m:
            continue

        genre_ids = []
        if m.genres and isinstance(m.genres, list):
            for g in m.genres:
                if isinstance(g, dict) and "id" in g:
                    genre_ids.append(g["id"])

        c = {
            "id": m.tmdb_id,
            "title": m.title,
            "media_type": m.media_type or "movie",
            "poster_path": m.poster_path,
            "backdrop_path": m.backdrop_path,
            "genres": m.genres or [],
            "genre_ids": genre_ids,
            "cast": m.main_cast or [],
            "crew": m.directors or [],
            "keywords_list": m.keywords or [],
            "original_language": m.original_language or "",
            "release_date": m.release_date or "",
            "vote_average": float(m.vote_average or 0.0),
            "vote_count": int(m.vote_count or 0),
            "embedding": m.embedding,
            "collaborative": True,
            "collaborative_score": round(float(best_collab[k]), 4),
            "match_tags": ["✨ Descoberta Colaborativa"],
        }
        candidates.append(c)

    return candidates
