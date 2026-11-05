import logging
from typing import Dict, Optional, List, Any
from datetime import datetime, timezone
import math
import numpy as np

logger = logging.getLogger(__name__)
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload
from app.core.config import settings
from app.users.models import User
from app.tracking.models import UserListItem
from app.media.models import Media
from app.recommendation.models import UserDismissedRecommendation
from app.recommendation.vector_builder import build_sparse_vector, merge_vectors, sublinear_scale


def calculate_time_decay(action_date: Optional[datetime], half_life_days: float = 365.0) -> float:
    """
    Decaimento temporal exponencial com meia-vida padrão de 365 dias.
    lambda = ln(2) / half_life_days.
    """
    if not action_date:
        return 1.0
    now = datetime.now(timezone.utc)
    if action_date.tzinfo is None:
        action_date = action_date.replace(tzinfo=timezone.utc)
    delta_days = max(0.0, (now - action_date).total_seconds() / 86400.0)
    decay_lambda = math.log(2) / half_life_days
    return float(math.exp(-decay_lambda * delta_days))


def calculate_rating_scale(
    rating: Optional[float],
    status: Optional[str] = None,
    rewatch_count: int = 0
) -> float:
    """
    Escala de nota explícita ou status implícito:
    - Nota explícita (0 a 5 estrelas / 0 a 10 TMDB):
      5.0 -> +2.0, 4.0 -> +1.5, 3.0 -> +1.0, 2.5 -> +0.5, 2.0 -> -0.5, <= 1.5 -> -1.5 (feedback negativo)
    - Status implícito:
      completed -> +1.2, watching -> +1.0, plan_to_watch -> +0.5, dropped -> -1.0
    """
    if rating is not None:
        if rating >= 4.5:
            return 2.0
        elif rating >= 3.5:
            return 1.5
        elif rating >= 3.0:
            return 1.0
        elif rating >= 2.5:
            return 0.5
        elif rating >= 2.0:
            return -0.5
        else:
            return -1.5

    if status == "completed":
        return 1.2
    elif status == "watching":
        return 1.0
    elif status == "plan_to_watch":
        return 0.5
    elif status == "dropped":
        return -1.0

    return 1.0


def calculate_rewatch_modifier(rewatch_count: int = 0, ep_rewatches: int = 0) -> float:
    """
    Bônus logarítmico suave para rewatches múltiplos (sinal forte de apreço).
    """
    total_rewatches = max(0, rewatch_count or 0) + max(0, ep_rewatches or 0)
    if total_rewatches <= 0:
        return 1.0
    return float(min(2.5, 1.0 + 0.3 * math.log1p(total_rewatches)))


async def rebuild_user_profile(db: AsyncSession, user_id: str) -> Optional[User]:
    """
    Reconstrói o perfil do usuário (vetor esparso e embedding denso) de forma 100% idempotente,
    eliminando contagem dupla em edições e deriva vetorial (Anomalias 2.1.5 e 2.1.9 + Ideias 3 e 4).

    Utiliza:
    - UserListItem + UserDismissedRecommendation
    - Taxonomia Canônica de Gêneros Filme ↔ Série
    - Escala Sublinear: sign(S_f) * ln(1 + |S_f|)
    - Running Rocchio exato: soma positiva ponderada - (gamma = 0.25) * soma negativa
    - Top 800 chaves mais relevantes
    """
    user_result = await db.execute(select(User).where(User.id == user_id))
    user = user_result.scalars().first()
    if not user:
        return None

    # 1. Carrega todas as obras na lista do usuário
    stmt = (
        select(UserListItem)
        .options(selectinload(UserListItem.media), selectinload(UserListItem.episodes))
        .where(UserListItem.user_id == user_id)
    )
    res = await db.execute(stmt)
    list_items = res.scalars().all()

    # 2. Carrega recomendações dispensadas ativas (Snooze não expirado)
    now = datetime.now(timezone.utc)
    dismissed_res = await db.execute(
        select(UserDismissedRecommendation).where(
            UserDismissedRecommendation.user_id == user_id,
            UserDismissedRecommendation.expires_at > now
        )
    )
    dismissed_items = dismissed_res.scalars().all()

    if not list_items and not dismissed_items:
        user.feature_vector = {}
        user.embedding = None
        await db.commit()
        return user

    raw_scores: Dict[str, float] = {}
    pos_emb_sum: Optional[np.ndarray] = None
    pos_weight_total: float = 0.0
    neg_emb_sum: Optional[np.ndarray] = None
    neg_weight_total: float = 0.0

    for item in list_items:
        media = item.media
        if not media:
            continue

        action_date = item.last_watched_at or item.updated_at or item.created_at
        time_decay = calculate_time_decay(action_date)
        rating_scale = calculate_rating_scale(item.rating, item.status, rewatch_count=item.rewatch_count)
        fav_mod = 2.0 if item.is_favorite else 1.0

        episodes_list = getattr(item, "episodes", []) or []
        ep_rewatches = sum((getattr(ep, "rewatch_count", 0) or 0) for ep in episodes_list)
        rewatch_mod = calculate_rewatch_modifier(item.rewatch_count, ep_rewatches)

        # Modificador proporcional de episódios para Séries
        episode_mod = 1.0
        if getattr(media, "media_type", None) == "tv":
            if item.status == "completed":
                episode_mod = 1.2
            else:
                ep_count = getattr(item, "episodes_watched_count", 0) or len(episodes_list)
                episode_mod = 1.0 + min(0.2, 0.2 * (ep_count / 10.0))

        item_scale = time_decay * rating_scale * fav_mod * rewatch_mod * episode_mod

        # Acumulação Esparsa
        m_vec = build_sparse_vector(media)
        for k, v in m_vec.items():
            raw_scores[k] = raw_scores.get(k, 0.0) + (v * item_scale)

        # Acumulação Densa via Rocchio
        if media.embedding:
            m_emb = np.array(media.embedding, dtype=np.float32)
            if m_emb.shape == (settings.EMBEDDING_DIM,):
                norm_m = np.linalg.norm(m_emb)
                if norm_m > 0:
                    m_emb = m_emb / norm_m
                    if item_scale > 0:
                        if pos_emb_sum is None:
                            pos_emb_sum = item_scale * m_emb
                        else:
                            pos_emb_sum += item_scale * m_emb
                        pos_weight_total += item_scale
                    elif item_scale < 0:
                        neg_weight = abs(item_scale)
                        if neg_emb_sum is None:
                            neg_emb_sum = neg_weight * m_emb
                        else:
                            neg_emb_sum += neg_weight * m_emb
                        neg_weight_total += neg_weight

    # Penalidade suave de obras dispensadas no vetor semântico
    for d in dismissed_items:
        if d.embedding:
            d_emb = np.array(d.embedding, dtype=np.float32)
            if d_emb.shape == (settings.EMBEDDING_DIM,):
                norm_d = np.linalg.norm(d_emb)
                if norm_d > 0:
                    d_emb = d_emb / norm_d
                    d_weight = 0.5
                    if neg_emb_sum is None:
                        neg_emb_sum = d_weight * d_emb
                    else:
                        neg_emb_sum += d_weight * d_emb
                    neg_weight_total += d_weight

    # Escala Sublinear e teto em Top 800 features
    scaled_vector: Dict[str, float] = {}
    for k, raw_v in raw_scores.items():
        s_v = sublinear_scale(raw_v)
        if abs(s_v) > 0.001:
            scaled_vector[k] = s_v

    max_abs = max((abs(v) for v in scaled_vector.values()), default=1.0)
    if max_abs > 100.0:
        scaled_vector = {k: (v / max_abs) * 100.0 for k, v in scaled_vector.items()}

    if len(scaled_vector) > 800:
        sorted_feats = sorted(scaled_vector.items(), key=lambda x: abs(x[1]), reverse=True)
        scaled_vector = dict(sorted_feats[:800])

    user.feature_vector = scaled_vector

    # Centroide Denso Exato via Rocchio (gamma = 0.25 para termos negativos)
    if pos_emb_sum is not None and pos_weight_total > 0:
        u_dense = pos_emb_sum / pos_weight_total
        if neg_emb_sum is not None and neg_weight_total > 0:
            gamma = 0.25
            u_dense = u_dense - (gamma / neg_weight_total) * neg_emb_sum
        u_norm = np.linalg.norm(u_dense)
        if u_norm > 0:
            user.embedding = (u_dense / u_norm).tolist()
        else:
            user.embedding = None
    else:
        user.embedding = None

    await db.commit()
    return user


async def update_user_profile(
    db: AsyncSession, 
    user_id: str, 
    list_item: Optional[UserListItem] = None, 
    media: Optional[Media] = None,
    explicit_scale: Optional[float] = None,
    media_vector: Optional[Dict[str, float]] = None,
    update_clusters: bool = False
):
    """
    Atualiza o perfil do usuário garantindo 100% de idempotência através do rebuild_user_profile.
    Gera embedding da obra se ausente, recalcula o vetor de forma vetorizada instantânea e
    agenda atualização assíncrona de recomendações.
    """
    if media and not getattr(media, "embedding", None):
        try:
            from app.recommendation.embeddings import build_text_for_embedding, generate_embedding
            text = build_text_for_embedding(media)
            media_emb = await generate_embedding(text)
            if media_emb:
                media.embedding = media_emb
                await db.commit()
        except Exception as e:
            logger.warning(f"Erro ao gerar embedding de media {getattr(media, 'id', None)}: {e}")

    user = await rebuild_user_profile(db, user_id)
    if not user:
        return

    # V2: Atualização assíncrona/segura dos clusters de gosto (Personas) apenas quando solicitado
    if update_clusters:
        try:
            from app.recommendation.cluster_manager import compute_user_taste_clusters
            await compute_user_taste_clusters(db, user.id)
        except Exception:
            pass
        
    # Marca como stale, ejeta a mídia imediatamente das recomendações e aciona debounce inteligente
    try:
        from app.recommendation.service import (
            mark_user_recommendations_stale,
            eject_media_from_user_recommendations,
            notify_user_media_action,
            schedule_recommendation_computation,
        )
        if media and getattr(media, "tmdb_id", None):
            await eject_media_from_user_recommendations(
                db, user.id, media.tmdb_id, media_type=getattr(media, "media_type", None)
            )
            notify_user_media_action(user.id, media.tmdb_id, debounce_seconds=35)
        else:
            await mark_user_recommendations_stale(db, user.id)
            schedule_recommendation_computation(user.id, force=False, cooldown_seconds=1200)
    except Exception as e:
        logger.warning(f"Erro ao agendar atualização de recomendações para {user.id}: {e}")


