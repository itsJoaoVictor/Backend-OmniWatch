from typing import Dict, Optional
from datetime import datetime, timezone
import math
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.users.models import User
from app.tracking.models import UserListItem
from app.media.models import Media
from app.recommendation.vector_builder import build_sparse_vector, merge_vectors

def calculate_time_decay(action_date: Optional[datetime]) -> float:
    """
    Calculates an exponential decay factor based on how old the action is.
    Recent ratings/actions have a factor close to 1.0. Older ones decay but never reach 0.
    """
    if not action_date:
        return 1.0
        
    now = datetime.now(timezone.utc)
    delta_days = (now - action_date).days
    
    if delta_days < 0:
        delta_days = 0
        
    # Half-life of 180 days
    half_life = 180.0
    decay_factor = math.pow(0.5, delta_days / half_life)
    
    # Floor at 0.1 so we never completely forget old preferences
    return max(0.1, decay_factor)

def calculate_rating_scale(rating: Optional[float], status: Optional[str] = None) -> float:
    """
    Fase 3 - Feedback Implícito & Explícito (Escala 0 a 5 estrelas):
    Diferenciação de pesos por nota:
    - 4.5 a 5.0 -> +1.5 (Amou / High praise)
    - 0.5 a 2.0 -> -1.0 (Odiou / Negative feedback)
    - 2.5 a 4.0 -> Proporcional de 0.0 a 1.0 ((rating - 2.5) / 1.5)
      * 2.5 -> 0.0
      * 3.0 -> 0.33
      * 3.5 -> 0.67
      * 4.0 -> 1.0
    
    Se não houver nota explícita, infere do status de acompanhamento:
    - "completed"     -> +1.5 (Assistiu até o fim / Boost de conclusão)
    - "watching"      -> +1.0 (Engajamento ativo)
    - "plan_to_watch" -> +0.8 (Interesse ao adicionar à Watchlist)
    - "dropped"       -> -0.5 (Abandonou)
    - default         -> +0.5
    """
    if rating is not None:
        if rating >= 4.5:
            return 1.5
        elif rating <= 2.0:
            return -1.0
        else:
            # Linear mapping from 2.5 (0.0) to 4.0 (1.0)
            return (rating - 2.5) / 1.5

    if status == "completed":
        return 1.5
    elif status == "watching":
        return 1.0
    elif status == "plan_to_watch":
        return 0.8
    elif status == "dropped":
        return -0.5

    return 0.5

def calculate_episode_rating_scale(rating: float) -> float:
    """
    Calcula a escala de afinidade de um episódio avaliado individualmente.
    Aplica um fator moderado de 0.4x em relação a uma obra completa.
    Ex: nota 5.0 gera 0.4 * 1.5 = +0.6 de reforço.
    Ex: nota 1.0 gera 0.4 * (-1.0) = -0.4 de penalidade.
    """
    base_scale = calculate_rating_scale(rating)
    return 0.4 * base_scale

async def update_user_profile(
    db: AsyncSession, 
    user_id: str, 
    list_item: Optional[UserListItem] = None, 
    media: Optional[Media] = None,
    explicit_scale: Optional[float] = None,
    media_vector: Optional[Dict[str, float]] = None
):
    """
    Incrementally updates the user's feature vector in O(1) time.
    Supports both explicit updates (ratings/status) and implicit signals:
    - explicit_scale = -0.5 when removed from list
    - explicit_scale = +1.5 when series is completed
    - explicit_scale = +0.2 when details page is visited
    """
    user_result = await db.execute(select(User).where(User.id == user_id))
    user = user_result.scalars().first()
    
    if not user:
        return
        
    current_vector = user.feature_vector or {}
    if not isinstance(current_vector, dict):
        current_vector = {}
        
    # Determine final scale factor
    if explicit_scale is not None:
        final_scale = explicit_scale
    else:
        action_date = (list_item.updated_at or list_item.created_at) if list_item else None
        time_decay = calculate_time_decay(action_date)
        rating_scale = calculate_rating_scale(
            list_item.rating if list_item else None,
            list_item.status if list_item else None
        )
        
        # TV Show episode modifier (if media is TV)
        episode_modifier = 1.0
        if media and media.media_type == "tv":
            episode_modifier = 1.2

        final_scale = time_decay * rating_scale * episode_modifier
    
    # Build media vector if not provided
    if media_vector is None:
        if media is None:
            return
        media_vector = build_sparse_vector(media)
        
    if not media_vector:
        return
    
    # Merge vectors with the calculated scale
    new_vector = merge_vectors(current_vector, media_vector, scale=final_scale)
    
    # Normalize vector to prevent values from exploding to infinity over years
    max_val = max((abs(v) for v in new_vector.values()), default=1.0)
    if max_val > 100.0:
        new_vector = {k: v / max_val * 100.0 for k, v in new_vector.items()}
        
    # Keep only the most relevant features to avoid giant JSONs (top 300)
    if len(new_vector) > 300:
        sorted_items = sorted(new_vector.items(), key=lambda x: abs(x[1]), reverse=True)
        new_vector = dict(sorted_items[:300])

    user.feature_vector = new_vector

    # Fase 6: Atualização incremental do embedding semântico do usuário
    if media:
        media_emb = media.embedding
        if not media_emb:
            try:
                from app.recommendation.embeddings import build_text_for_embedding, generate_embedding
                text = build_text_for_embedding(media)
                media_emb = await generate_embedding(text)
                if media_emb:
                    media.embedding = media_emb
            except Exception:
                media_emb = None

        if media_emb:
            from app.recommendation.embeddings import update_user_embedding_vector
            user.embedding = update_user_embedding_vector(user.embedding, media_emb, final_scale)

    await db.commit()

    # V2: Atualização assíncrona/segura dos clusters de gosto (Personas)
    try:
        from app.recommendation.cluster_manager import compute_user_taste_clusters
        await compute_user_taste_clusters(db, user.id)
    except Exception:
        pass
        
    # Invalida o cache para esse usuário, ativando a revalidação em background na próxima visita
    try:
        from app.recommendation.router import recs_cache, personas_cache
        recs_cache.invalidate_prefix(f"explore:{user_id}")
        personas_cache.invalidate_prefix(f"personas:{user_id}")
    except Exception as e:
        pass
