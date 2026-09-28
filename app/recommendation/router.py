import logging
logger = logging.getLogger(__name__)
from fastapi import APIRouter, Depends, HTTPException, Path, Query, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Any, Dict, Optional
from app.core.database import get_db
from app.core.cache import SimpleTTLCache
from app.auth.dependencies import get_current_user_id
from app.users.models import User
from app.recommendation.orchestrator import get_personalized_recommendations, get_upcoming_recommendations
from app.recommendation.matcher import calculate_match_score
from app.recommendation.vector_builder import WEIGHTS
from app.details.services import fetch_movie_details, fetch_tv_details

router = APIRouter()

from datetime import datetime, timezone
from pydantic import BaseModel
from app.recommendation.service import (
    get_user_recommendations_record,
    compute_and_save_user_recommendations,
    dismiss_recommendation,
    undismiss_recommendation,
    get_user_dismissed_records,
    get_active_dismissed_tmdb_ids,
    MAX_STALE_SECONDS,
)

class DismissRecommendationRequest(BaseModel):
    tmdb_id: int
    media_type: str
    title: Optional[str] = ""
    poster_path: Optional[str] = None
    days_snooze: Optional[int] = 180

class UndismissRecommendationRequest(BaseModel):
    tmdb_id: int
    media_type: Optional[str] = None

async def _get_user_seen_tmdb_ids(db: AsyncSession, user_id: str) -> set:
    try:
        from app.tracking.services import get_user_list
        user_list = await get_user_list(db, user_id)
        seen = {item.media.tmdb_id for item in user_list if item.media}
        dismissed = await get_active_dismissed_tmdb_ids(db, user_id)
        return seen.union(dismissed)
    except Exception:
        return set()

@router.get("/explore", response_model=List[Any])
async def get_explore_recommendations(
    background_tasks: BackgroundTasks,
    limit: int = Query(50, ge=1, le=100),
    persona_id: Optional[str] = Query(None, description="Filtra recomendações para uma Persona/Cluster específico"),
    current_user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    try:
        record = await get_user_recommendations_record(db, current_user_id)
        seen_ids = await _get_user_seen_tmdb_ids(db, current_user_id)

        # Cold start: se o registro não existe ou ainda não tem itens
        if not record or not record.explore_items:
            res = await compute_and_save_user_recommendations(current_user_id, force=True)
            items = (res.get("explore_items") if res else [])
        else:
            items = record.explore_items
            now = datetime.now(timezone.utc)
            is_expired = (
                record.last_generated_at is None
                or (now - record.last_generated_at).total_seconds() > MAX_STALE_SECONDS
            )
            if record.is_stale or is_expired:
                background_tasks.add_task(compute_and_save_user_recommendations, current_user_id)

        # Filtragem dinâmica defensiva: garante que nenhuma obra na lista do usuário seja exibida
        if seen_ids and items:
            items = [it for it in items if it.get("id") not in seen_ids and it.get("tmdb_id") not in seen_ids]

        if persona_id:
            personas = record.personas_items if record else []
            target = next((p for p in personas if p.get("id") == persona_id), None)
            if target and target.get("recommendations"):
                p_recs = target["recommendations"]
                if seen_ids:
                    p_recs = [it for it in p_recs if it.get("id") not in seen_ids and it.get("tmdb_id") not in seen_ids]
                return p_recs[:limit]
            raw = await get_personalized_recommendations(db, current_user_id, top_k=limit, persona_id=persona_id)
            if seen_ids:
                raw = [it for it in raw if it.get("id") not in seen_ids and it.get("tmdb_id") not in seen_ids]
            return raw[:limit]

        return items[:limit]
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
    try:
        record = await get_user_recommendations_record(db, current_user_id)
        seen_ids = await _get_user_seen_tmdb_ids(db, current_user_id)

        if not record or not record.personas_items:
            res = await compute_and_save_user_recommendations(current_user_id, force=True)
            personas = (res.get("personas_items", []) if res else [])
        else:
            personas = record.personas_items
            now = datetime.now(timezone.utc)
            is_expired = (
                record.last_generated_at is None
                or (now - record.last_generated_at).total_seconds() > MAX_STALE_SECONDS
            )
            if record.is_stale or is_expired:
                background_tasks.add_task(compute_and_save_user_recommendations, current_user_id)

        result = []
        for p in personas:
            p_copy = dict(p)
            recs = p_copy.get("recommendations", [])
            if isinstance(recs, list):
                if seen_ids:
                    recs = [it for it in recs if it.get("id") not in seen_ids and it.get("tmdb_id") not in seen_ids]
                p_copy["recommendations"] = recs[:top_k]
            result.append(p_copy)
        return result
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail='Ocorreu um erro interno ao processar a solicitação.')

@router.get("/upcoming", response_model=Dict[str, Any])
async def get_upcoming_recommendations_endpoint(
    background_tasks: BackgroundTasks,
    limit: int = Query(20, ge=1, le=50),
    current_user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """
    Retorna os próximos lançamentos (Filmes e Séries Inéditas) ranqueados
    pelo perfil de afinidade (Match Score) do usuário.
    """
    try:
        record = await get_user_recommendations_record(db, current_user_id)
        seen_ids = await _get_user_seen_tmdb_ids(db, current_user_id)

        if not record or not record.upcoming_items:
            res = await compute_and_save_user_recommendations(current_user_id, force=True)
            upcoming = (res.get("upcoming_items", {}) if res else {"movies": [], "tv": []})
        else:
            upcoming = record.upcoming_items
            now = datetime.now(timezone.utc)
            is_expired = (
                record.last_generated_at is None
                or (now - record.last_generated_at).total_seconds() > MAX_STALE_SECONDS
            )
            if record.is_stale or is_expired:
                background_tasks.add_task(compute_and_save_user_recommendations, current_user_id)

        movies = upcoming.get("movies") or []
        tv = upcoming.get("tv") or []
        if seen_ids:
            movies = [it for it in movies if it.get("id") not in seen_ids and it.get("tmdb_id") not in seen_ids]
            tv = [it for it in tv if it.get("id") not in seen_ids and it.get("tmdb_id") not in seen_ids]

        return {
            "movies": movies[:limit],
            "tv": tv[:limit]
        }
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail='Ocorreu um erro interno ao processar a solicitação.')

@router.get("/feed", response_model=Dict[str, Any])
async def get_explore_feed(
    background_tasks: BackgroundTasks,
    current_user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """
    Retorna todo o payload de Explore (explore, upcoming e personas)
    em uma única requisição ultrarrápida direta do banco de dados (< 20ms).
    """
    try:
        record = await get_user_recommendations_record(db, current_user_id)
        seen_ids = await _get_user_seen_tmdb_ids(db, current_user_id)

        if not record or (not record.explore_items and not record.personas_items):
            res = await compute_and_save_user_recommendations(current_user_id, force=True)
            explore = res.get("explore_items", []) if res else []
            upcoming = res.get("upcoming_items", {}) if res else {"movies": [], "tv": []}
            personas = res.get("personas_items", []) if res else []
            is_stale = False
            last_gen = datetime.now(timezone.utc).isoformat()
        else:
            explore = record.explore_items
            upcoming = record.upcoming_items
            personas = record.personas_items
            is_stale = record.is_stale
            last_gen = record.last_generated_at.isoformat() if record.last_generated_at else None

            now = datetime.now(timezone.utc)
            is_expired = (
                record.last_generated_at is None
                or (now - record.last_generated_at).total_seconds() > MAX_STALE_SECONDS
            )
            if is_stale or is_expired:
                background_tasks.add_task(compute_and_save_user_recommendations, current_user_id)

        if seen_ids:
            explore = [it for it in explore if it.get("id") not in seen_ids and it.get("tmdb_id") not in seen_ids]
            new_personas = []
            for p in personas:
                p_c = dict(p)
                recs = p_c.get("recommendations", [])
                p_c["recommendations"] = [it for it in recs if it.get("id") not in seen_ids and it.get("tmdb_id") not in seen_ids]
                new_personas.append(p_c)
            personas = new_personas
            up_movies = [it for it in (upcoming.get("movies") or []) if it.get("id") not in seen_ids and it.get("tmdb_id") not in seen_ids]
            up_tv = [it for it in (upcoming.get("tv") or []) if it.get("id") not in seen_ids and it.get("tmdb_id") not in seen_ids]
            upcoming = {"movies": up_movies, "tv": up_tv}

        return {
            "explore": explore,
            "upcoming": upcoming,
            "personas": personas,
            "is_stale": is_stale,
            "last_generated_at": last_gen
        }
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail='Ocorreu um erro interno ao processar a solicitação.')

@router.post("/recompute")
async def force_recompute_recommendations(
    background_tasks: BackgroundTasks,
    current_user_id: str = Depends(get_current_user_id),
):
    """Dispara um recálculo forçado de recomendações em segundo plano."""
    background_tasks.add_task(compute_and_save_user_recommendations, current_user_id, force=True)
    return {"message": "Recálculo forçado de recomendações iniciado em segundo plano."}


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
        best_persona = None
        
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
                    best_persona = p
                    
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
        
        # Enriquecer tags com explicabilidade humana (Porque curtiu X, Talentos, Persona, Microtemas)
        user_list = []
        try:
            from app.user_list.service import get_user_list
            user_list = await get_user_list(db, user.id)
        except Exception:
            pass

        candidate_obj = {
            "id": tmdb_id,
            "title": getattr(tmdb_data, "title", None) or getattr(tmdb_data, "name", "Mídia"),
            "name": getattr(tmdb_data, "name", None),
            "media_type": media_type,
            "genres": [g.name for g in tmdb_data.genres] if tmdb_data.genres else [],
            "keywords": tmdb_data.keywords if hasattr(tmdb_data, "keywords") and tmdb_data.keywords else [],
            "credits": tmdb_data.credits.dict() if tmdb_data.credits and hasattr(tmdb_data.credits, "dict") else None,
            "original_language": getattr(tmdb_data, "original_language", None),
        }

        from app.recommendation.matcher import build_rich_explanation_tags
        rich_tags = build_rich_explanation_tags(
            candidate=candidate_obj,
            user_vector=user.feature_vector,
            user_items=user_list,
            cand_emb=cand_emb,
            target_persona=best_persona,
            fallback_tags=top_tags
        )
        
        return {
            "match_score": score,
            "match_tags": rich_tags
        }
    except Exception as e:
        # Ignore errors if TMDB fetch fails
        return {"match_score": 0, "match_tags": []}

@router.post("/visit/{media_type}/{tmdb_id}")
async def record_visit(
    media_type: str = Path(...),
    tmdb_id: int = Path(...)
):
    """
    Feedback implícito desativado por design:
    Visitas a páginas de detalhes não alteram mais o perfil do usuário
    nem invalidam o cache de recomendações. Mantido como no-op para compatibilidade.
    """
    return {"status": "ok", "recorded": False}

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

@router.post("/dismiss")
async def dismiss_recommendation_endpoint(
    payload: DismissRecommendationRequest,
    current_user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """
    Registra que o usuário não tem interesse na obra.
    Ejeta a obra do carrossel imediatamente e salva a preferência com expiração (snooze de 6-12 meses).
    """
    try:
        item = await dismiss_recommendation(
            db=db,
            user_id=current_user_id,
            tmdb_id=payload.tmdb_id,
            media_type=payload.media_type,
            title=payload.title or "",
            poster_path=payload.poster_path,
            days_snooze=payload.days_snooze or 180
        )
        return {
            "status": "success",
            "message": "Recomendação dispensada com sucesso.",
            "tmdb_id": item.tmdb_id,
            "expires_at": item.expires_at.isoformat()
        }
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail="Erro ao dispensar recomendação.")

@router.post("/undismiss")
async def undismiss_recommendation_endpoint(
    payload: UndismissRecommendationRequest,
    current_user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """
    Restaura uma recomendação previamente dispensada.
    """
    try:
        success = await undismiss_recommendation(
            db=db,
            user_id=current_user_id,
            tmdb_id=payload.tmdb_id,
            media_type=payload.media_type
        )
        return {
            "status": "success" if success else "not_found",
            "restored": success,
            "tmdb_id": payload.tmdb_id
        }
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail="Erro ao restaurar recomendação.")

@router.get("/dismissed")
async def list_dismissed_recommendations_endpoint(
    limit: int = Query(50, ge=1, le=100),
    current_user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """
    Retorna a lista de obras dispensadas ativas do usuário para a tela de configurações/perfil.
    """
    try:
        items = await get_user_dismissed_records(db, current_user_id, limit=limit)
        return [
            {
                "id": str(it.id),
                "tmdb_id": it.tmdb_id,
                "media_type": it.media_type,
                "title": it.title,
                "poster_path": it.poster_path,
                "expires_at": it.expires_at.isoformat(),
                "created_at": it.created_at.isoformat()
            }
            for it in items
        ]
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail="Erro ao listar recomendações dispensadas.")

