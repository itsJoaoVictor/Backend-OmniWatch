import logging
logger = logging.getLogger(__name__)
from fastapi import APIRouter, Depends, HTTPException, Path, Query, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Any, Dict, Optional
from app.core.database import get_db
from app.core.cache import SimpleTTLCache
from app.auth.dependencies import get_current_user_id
from app.users.models import User
from app.recommendation.orchestrator import get_personalized_recommendations
from app.recommendation.matcher import calculate_match_score
from app.recommendation.vector_builder import WEIGHTS
from app.details.services import fetch_movie_details, fetch_tv_details

router = APIRouter()

visit_cache = SimpleTTLCache(ttl_seconds=3600)
recs_cache = SimpleTTLCache(ttl_seconds=86400) # 24 horas TTL para recomendações (ou 604800 para 1 semana)
personas_cache = SimpleTTLCache(ttl_seconds=86400) # 24 horas TTL para personas

@router.get("/explore", response_model=List[Any])
async def get_explore_recommendations(
    background_tasks: BackgroundTasks,
    limit: int = Query(50, ge=1, le=100),
    persona_id: Optional[str] = Query(None, description="Filtra recomendações para uma Persona/Cluster específico"),
    current_user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    from sqlalchemy.future import select
    from app.users.models import User
    
    user_result = await db.execute(select(User.updated_at).where(User.id == current_user_id))
    updated_at = user_result.scalars().first()
    ts = updated_at.timestamp() if updated_at else 0
    
    cache_key = f"explore:{current_user_id}:{limit}:{persona_id}:v{ts}"
    cached_data, is_stale = recs_cache.get_with_status(cache_key)

    async def fetch_and_cache():
        from app.core.database import AsyncSessionLocal
        try:
            async with AsyncSessionLocal() as bg_db:
                new_recs = await get_personalized_recommendations(
                    bg_db, current_user_id, top_k=limit, persona_id=persona_id
                )
                recs_cache.set(cache_key, new_recs)
        except Exception as e:
            print(f"Erro no SWR de explore: {e}")

    if cached_data is not None:
        if is_stale:
            recs_cache.set(cache_key, cached_data) # Impede dezenas de tasks de SWR concorrentes
            background_tasks.add_task(fetch_and_cache)
        return cached_data

    try:
        recommendations = await get_personalized_recommendations(
            db, current_user_id, top_k=limit, persona_id=persona_id
        )
        recs_cache.set(cache_key, recommendations)
        return recommendations
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail='Ocorreu um erro interno ao processar a solicitação.')

@router.get("/personas", response_model=List[Any])
async def get_user_personas(
    background_tasks: BackgroundTasks,
    top_k: int = Query(15, ge=1, le=30),
    current_user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    from sqlalchemy.future import select
    from app.users.models import User
    
    user_result = await db.execute(select(User.updated_at).where(User.id == current_user_id))
    updated_at = user_result.scalars().first()
    ts = updated_at.timestamp() if updated_at else 0
    
    cache_key = f"personas:{current_user_id}:{top_k}:v{ts}"
    cached_data, is_stale = personas_cache.get_with_status(cache_key)

    async def fetch_and_cache_personas():
        from app.core.database import AsyncSessionLocal
        from app.recommendation.orchestrator import get_persona_recommendations
        try:
            async with AsyncSessionLocal() as bg_db:
                new_personas = await get_persona_recommendations(bg_db, current_user_id, top_k_per_persona=top_k)
                personas_cache.set(cache_key, new_personas)
        except Exception as e:
            print(f"Erro no SWR de personas: {e}")

    if cached_data is not None:
        if is_stale:
            personas_cache.set(cache_key, cached_data)
            background_tasks.add_task(fetch_and_cache_personas)
        return cached_data

    try:
        from app.recommendation.orchestrator import get_persona_recommendations
        personas = await get_persona_recommendations(db, current_user_id, top_k_per_persona=top_k)
        personas_cache.set(cache_key, personas)
        return personas
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail='Ocorreu um erro interno ao processar a solicitação.')

@router.get("/score/{media_type}/{tmdb_id}")
async def get_item_score(
    media_type: str = Path(...),
    tmdb_id: int = Path(...),
    current_user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """
    Calculates the match score for a specific item on the details page.
    Includes genres, cast, crew, keywords (Fase 1) and original_language (Fase 2).
    """
    from sqlalchemy.future import select
    user_result = await db.execute(select(User).where(User.id == current_user_id))
    user = user_result.scalars().first()
    
    if not user or not user.feature_vector:
        return {"match_score": 0, "match_tags": []}
        
    try:
        if media_type == "movie":
            tmdb_data = await fetch_movie_details(tmdb_id)
        else:
            tmdb_data = await fetch_tv_details(tmdb_id)
            
        candidate_vector = {}
        candidate_vector[f"type_{media_type}"] = WEIGHTS["media_type"]
        
        if tmdb_data.genres:
            for g in tmdb_data.genres:
                candidate_vector[f"genre_{g.name.lower()}"] = WEIGHTS["genre"]
                
        if tmdb_data.credits:
            if tmdb_data.credits.crew:
                for c in tmdb_data.credits.crew:
                    if c.job in ("Director", "Creator"):
                        candidate_vector[f"director_{c.name.lower()}"] = WEIGHTS["director"]
            if tmdb_data.credits.cast:
                for c in tmdb_data.credits.cast[:10]:
                    candidate_vector[f"cast_{c.name.lower()}"] = WEIGHTS["cast"]
                    
        # Fase 1: Keywords
        if hasattr(tmdb_data, "keywords") and tmdb_data.keywords:
            for kw in tmdb_data.keywords:
                candidate_vector[f"keyword_{kw.lower()}"] = WEIGHTS["keyword"]

        # Fase 2: Original Language
        if tmdb_data.original_language and tmdb_data.original_language.lower() != "en":
            candidate_vector[f"lang_{tmdb_data.original_language.lower()}"] = WEIGHTS["language"]
                    
        # Check if we have local embedding
        from app.media.models import Media as MediaModel
        local_media = await db.execute(select(MediaModel).where(MediaModel.tmdb_id == tmdb_id, MediaModel.media_type == media_type))
        local_media_obj = local_media.scalars().first()
        cand_emb = local_media_obj.embedding if local_media_obj else None

        active_personas = [p for p in user.taste_clusters if p.get("is_active", True)] if user.taste_clusters else []
        
        from app.recommendation.embeddings import calculate_cosine_similarity
        
        if active_personas:
            best_final_score = -1.0
            best_tags = []
            
            for p in active_personas:
                p_vec = p.get("feature_vector") or user.feature_vector
                p_emb = p.get("centroid_embedding") or user.embedding
                
                aff_score, tags = calculate_match_score(p_vec, candidate_vector)
                
                if p_emb and cand_emb:
                    sem_sim = calculate_cosine_similarity(p_emb, cand_emb)
                    sem_score = sem_sim * 100.0
                    f_score = round(0.65 * sem_score + 0.35 * aff_score, 1)
                    if sem_sim >= 0.60 and "✨ Sintonia Semântica" not in tags:
                        tags.append("✨ Sintonia Semântica")
                else:
                    f_score = aff_score
                    
                if f_score > best_final_score:
                    best_final_score = f_score
                    best_tags = tags
                    
            score = best_final_score
            top_tags = best_tags
        else:
            score, top_tags = calculate_match_score(user.feature_vector, candidate_vector)
            if user.embedding and cand_emb:
                sem_sim = calculate_cosine_similarity(user.embedding, cand_emb)
                sem_score = sem_sim * 100.0
                score = round(0.65 * sem_score + 0.35 * score, 1)
                if sem_sim >= 0.60 and "✨ Sintonia Semântica" not in top_tags:
                    top_tags.append("✨ Sintonia Semântica")
        
        return {
            "match_score": score,
            "match_tags": top_tags
        }
    except Exception as e:
        # Ignore errors if TMDB fetch fails
        return {"match_score": 0, "match_tags": []}

@router.post("/visit/{media_type}/{tmdb_id}")
async def record_visit(
    media_type: str = Path(...),
    tmdb_id: int = Path(...),
    current_user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """
    Fase 3 - Feedback Implícito:
    Registra visita à página de detalhes com peso leve (+0.2) no vetor do usuário.
    Usa cache de 1 hora por usuário/mídia para evitar spam em F5.
    """
    cache_key = f"visit:{current_user_id}:{media_type}:{tmdb_id}"
    data, is_stale = visit_cache.get_with_status(cache_key)
    if not is_stale and data:
        return {"status": "ok", "recorded": False, "reason": "already_recorded_recently"}

    try:
        from app.media.services import get_media_by_tmdb_id
        from app.recommendation.profile_manager import update_user_profile

        media = await get_media_by_tmdb_id(db, tmdb_id)
        if media:
            await update_user_profile(db, current_user_id, None, media, explicit_scale=0.2)
            visit_cache.set(cache_key, True)
            return {"status": "ok", "recorded": True, "scale": 0.2}

        # Se não existe no banco, busca TMDB para extrair vetor
        if media_type == "movie":
            tmdb_data = await fetch_movie_details(tmdb_id)
        else:
            tmdb_data = await fetch_tv_details(tmdb_id)

        vector = {}
        vector[f"type_{media_type}"] = WEIGHTS["media_type"]
        if tmdb_data.genres:
            for g in tmdb_data.genres:
                vector[f"genre_{g.name.lower()}"] = WEIGHTS["genre"]
        if tmdb_data.credits:
            if tmdb_data.credits.crew:
                for c in tmdb_data.credits.crew:
                    if c.job in ("Director", "Creator"):
                        vector[f"director_{c.name.lower()}"] = WEIGHTS["director"]
            if tmdb_data.credits.cast:
                for c in tmdb_data.credits.cast[:10]:
                    vector[f"cast_{c.name.lower()}"] = WEIGHTS["cast"]
        if hasattr(tmdb_data, "keywords") and tmdb_data.keywords:
            for kw in tmdb_data.keywords:
                vector[f"keyword_{kw.lower()}"] = WEIGHTS["keyword"]
        if tmdb_data.original_language and tmdb_data.original_language.lower() != "en":
            vector[f"lang_{tmdb_data.original_language.lower()}"] = WEIGHTS["language"]

        await update_user_profile(
            db,
            current_user_id,
            None,
            media=None,
            explicit_scale=0.2,
            media_vector=vector
        )
        visit_cache.set(cache_key, True)
        return {"status": "ok", "recorded": True, "scale": 0.2}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

@router.get("/similar-users")
async def get_similar_users(
    current_user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """
    Fase 5 - Collaborative Filtering (Debug/Metrics):
    Retorna os usuários com perfil de gosto mais similar ao usuário autenticado,
    calculado via NearestNeighbors (distância cosseno) sobre os feature vectors esparsos.
    """
    from app.recommendation.collaborative import find_similar_users
    try:
        similar_users = await find_similar_users(
            db,
            current_user_id,
            n_neighbors=5,
            min_similarity=0.10
        )
        return {
            "user_id": current_user_id,
            "similar_users_count": len(similar_users),
            "similar_users": similar_users
        }
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail='Ocorreu um erro interno ao processar a solicitação.')

