import asyncio
import logging
logger = logging.getLogger(__name__)
from fastapi import APIRouter, Depends, HTTPException, Path, Query, BackgroundTasks, Response, Request
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Any, Dict, Optional
from app.core.database import get_db
from app.core.cache import SimpleTTLCache, recommendations_seen_cache
from app.auth.dependencies import get_current_user_id
from app.users.models import User
from app.recommendation.orchestrator import (
    get_personalized_recommendations,
    get_upcoming_recommendations,
    build_candidate_vector,
)
from app.recommendation.matcher import calculate_match_score, compute_omnimatch_score
from app.recommendation.vector_builder import WEIGHTS, normalize_genre_to_canonical
from app.details.services import fetch_movie_details, fetch_tv_details

router = APIRouter()

from datetime import datetime, timezone
from pydantic import BaseModel
from app.recommendation.service import (
    get_user_recommendations_record,
    compute_and_save_user_recommendations,
    schedule_recommendation_computation,
    dismiss_recommendation,
    undismiss_recommendation,
    get_user_dismissed_records,
    get_active_dismissed_tmdb_ids,
    MAX_STALE_SECONDS,
    sanitize_recommendation_item,
    sanitize_recommendation_list,
    check_is_bloated,
    cleanup_bloated_user_record,
    generate_recommendations_etag,
    check_etag_match,
)

RECS_CACHE_HEADER = "private, no-cache, must-revalidate, max-age=0"

class DismissRecommendationRequest(BaseModel):
    tmdb_id: int
    media_type: str
    title: Optional[str] = ""
    poster_path: Optional[str] = None
    days_snooze: Optional[int] = 180

class UndismissRecommendationRequest(BaseModel):
    tmdb_id: int
    media_type: Optional[str] = None

class TelemetryEventItem(BaseModel):
    tmdb_id: int
    media_type: str = "movie"
    event_type: str  # 'impression', 'click', 'add_to_list', 'completed_high', 'dismiss'
    carousel_type: Optional[str] = "explore"
    position: Optional[int] = 0
    score_presented: Optional[float] = None

class TelemetryBatchRequest(BaseModel):
    events: List[TelemetryEventItem]

_seen_ids_locks: Dict[str, asyncio.Lock] = {}

async def _get_user_seen_tmdb_ids(db: AsyncSession, user_id: str) -> set:
    user_key = str(user_id)
    cached_val, is_stale = recommendations_seen_cache.get_with_status(user_key)
    if not is_stale and cached_val is not None:
        return cached_val

    lock = _seen_ids_locks.setdefault(user_key, asyncio.Lock())
    async with lock:
        # Double-check após adquirir o lock para evitar concorrência/thundering herd simultâneo
        cached_val, is_stale = recommendations_seen_cache.get_with_status(user_key)
        if not is_stale and cached_val is not None:
            return cached_val

        try:
            import uuid
            target_uuid = user_id if isinstance(user_id, uuid.UUID) else uuid.UUID(user_key)
            from app.media.models import Media
            from app.tracking.models import UserListItem
            from sqlalchemy.future import select
            
            stmt = (
                select(Media.tmdb_id)
                .join(UserListItem, UserListItem.media_id == Media.id)
                .where(UserListItem.user_id == target_uuid)
            )
            result = await db.execute(stmt)
            seen = set(result.scalars().all())
            dismissed = await get_active_dismissed_tmdb_ids(db, user_id)
            all_seen = seen.union(dismissed)
            recommendations_seen_cache.set(user_key, all_seen)
            return all_seen
        except Exception as e:
            logger.warning(f"Erro ao obter seen_tmdb_ids para {user_id}: {e}")
            return set()

@router.get("/explore", response_model=List[Any])
async def get_explore_recommendations(
    request: Request,
    response: Response,
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
            record = await get_user_recommendations_record(db, current_user_id)
        else:
            items = record.explore_items
            if check_is_bloated(items):
                background_tasks.add_task(cleanup_bloated_user_record, current_user_id)

            now = datetime.now(timezone.utc)
            is_expired = (
                record.last_generated_at is None
                or (now - record.last_generated_at).total_seconds() > MAX_STALE_SECONDS
            )
            if record.is_stale or is_expired:
                schedule_recommendation_computation(current_user_id)

        etag = generate_recommendations_etag(
            current_user_id, record, seen_ids, section="explore", extra=str(persona_id or "")
        )
        response.headers["ETag"] = etag
        response.headers["Cache-Control"] = RECS_CACHE_HEADER

        if check_etag_match(request.headers.get("if-none-match"), etag):
            return Response(status_code=304, headers={"ETag": etag, "Cache-Control": RECS_CACHE_HEADER})

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
                return sanitize_recommendation_list(p_recs[:limit])
            raw = await get_personalized_recommendations(db, current_user_id, top_k=limit, persona_id=persona_id)
            if seen_ids:
                raw = [it for it in raw if it.get("id") not in seen_ids and it.get("tmdb_id") not in seen_ids]
            return sanitize_recommendation_list(raw[:limit])

        return sanitize_recommendation_list(items[:limit])
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail='Ocorreu um erro interno ao processar a solicitação.')

@router.get("/explore/recent-vibe", response_model=List[Any])
async def get_recent_vibe_endpoint(
    request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    limit: int = Query(20, ge=1, le=50),
    current_user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """
    Retorna o carrossel 'Na Sua Vibe Recente' gerado pelo perfil Rocchio de memória curta (21 dias).
    """
    try:
        record = await get_user_recommendations_record(db, current_user_id)
        seen_ids = await _get_user_seen_tmdb_ids(db, current_user_id)

        if not record or not getattr(record, "recent_vibe_items", None):
            res = await compute_and_save_user_recommendations(current_user_id, force=True)
            items = (res.get("recent_vibe_items") if res else [])
            record = await get_user_recommendations_record(db, current_user_id)
        else:
            items = record.recent_vibe_items or []
            now = datetime.now(timezone.utc)
            is_expired = (
                record.last_generated_at is None
                or (now - record.last_generated_at).total_seconds() > MAX_STALE_SECONDS
            )
            if record.is_stale or is_expired:
                schedule_recommendation_computation(current_user_id)

        etag = generate_recommendations_etag(
            current_user_id, record, seen_ids, section="recent-vibe", extra=str(limit)
        )
        response.headers["ETag"] = etag
        response.headers["Cache-Control"] = RECS_CACHE_HEADER

        if check_etag_match(request.headers.get("if-none-match"), etag):
            return Response(status_code=304, headers={"ETag": etag, "Cache-Control": RECS_CACHE_HEADER})

        if seen_ids and items:
            items = [it for it in items if it.get("id") not in seen_ids and it.get("tmdb_id") not in seen_ids]

        return sanitize_recommendation_list(items[:limit])
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail='Ocorreu um erro interno ao processar a solicitação.')

@router.get("/personas", response_model=List[Any])
async def get_user_personas(
    request: Request,
    response: Response,
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
            record = await get_user_recommendations_record(db, current_user_id)
        else:
            personas = record.personas_items
            if personas and any(check_is_bloated(p.get("recommendations")) for p in personas if isinstance(p, dict)):
                background_tasks.add_task(cleanup_bloated_user_record, current_user_id)

            now = datetime.now(timezone.utc)
            is_expired = (
                record.last_generated_at is None
                or (now - record.last_generated_at).total_seconds() > MAX_STALE_SECONDS
            )
            if record.is_stale or is_expired:
                schedule_recommendation_computation(current_user_id)

        etag = generate_recommendations_etag(
            current_user_id, record, seen_ids, section="personas", extra=str(top_k)
        )
        response.headers["ETag"] = etag
        response.headers["Cache-Control"] = RECS_CACHE_HEADER

        if check_etag_match(request.headers.get("if-none-match"), etag):
            return Response(status_code=304, headers={"ETag": etag, "Cache-Control": RECS_CACHE_HEADER})

        result = []
        for p in personas:
            p_copy = dict(p)
            recs = p_copy.get("recommendations", [])
            if isinstance(recs, list):
                if seen_ids:
                    recs = [it for it in recs if it.get("id") not in seen_ids and it.get("tmdb_id") not in seen_ids]
                p_copy["recommendations"] = sanitize_recommendation_list(recs[:top_k])
            result.append(p_copy)

        return result
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail='Ocorreu um erro interno ao processar a solicitação.')

@router.get("/upcoming", response_model=Dict[str, Any])
async def get_upcoming_recommendations_endpoint(
    request: Request,
    response: Response,
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
            record = await get_user_recommendations_record(db, current_user_id)
        else:
            upcoming = record.upcoming_items
            now = datetime.now(timezone.utc)
            is_expired = (
                record.last_generated_at is None
                or (now - record.last_generated_at).total_seconds() > MAX_STALE_SECONDS
            )
            if record.is_stale or is_expired:
                schedule_recommendation_computation(current_user_id)

        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        etag = generate_recommendations_etag(
            current_user_id, record, seen_ids, section="upcoming", extra=f"{limit}:{today_str}"
        )
        response.headers["ETag"] = etag
        response.headers["Cache-Control"] = RECS_CACHE_HEADER

        if check_etag_match(request.headers.get("if-none-match"), etag):
            return Response(status_code=304, headers={"ETag": etag, "Cache-Control": RECS_CACHE_HEADER})

        movies = upcoming.get("movies") or []
        tv = upcoming.get("tv") or upcoming.get("series") or []
        if seen_ids:
            movies = [it for it in movies if it.get("id") not in seen_ids and it.get("tmdb_id") not in seen_ids]
            tv = [it for it in tv if it.get("id") not in seen_ids and it.get("tmdb_id") not in seen_ids]

        movies = [it for it in movies if str(it.get("release_date") or "").strip() >= today_str]
        tv = [it for it in tv if str(it.get("first_air_date") or "").strip() >= today_str]

        clean_movies = sanitize_recommendation_list(movies[:limit])
        clean_tv = sanitize_recommendation_list(tv[:limit])

        return {
            "movies": clean_movies,
            "tv": clean_tv,
            "series": clean_tv
        }
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail='Ocorreu um erro interno ao processar a solicitação.')

@router.get("/feed", response_model=Dict[str, Any])
async def get_explore_feed(
    request: Request,
    response: Response,
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
            recent_vibe = res.get("recent_vibe_items", []) if res else []
            upcoming = res.get("upcoming_items", {}) if res else {"movies": [], "tv": []}
            personas = res.get("personas_items", []) if res else []
            is_stale = False
            last_gen = datetime.now(timezone.utc).isoformat()
            record = await get_user_recommendations_record(db, current_user_id)
        else:
            explore = record.explore_items
            recent_vibe = getattr(record, "recent_vibe_items", []) or []
            upcoming = record.upcoming_items
            personas = record.personas_items
            is_stale = record.is_stale
            last_gen = record.last_generated_at.isoformat() if record.last_generated_at else None

            if check_is_bloated(explore):
                background_tasks.add_task(cleanup_bloated_user_record, current_user_id)

            now = datetime.now(timezone.utc)
            is_expired = (
                record.last_generated_at is None
                or (now - record.last_generated_at).total_seconds() > MAX_STALE_SECONDS
            )
            if is_stale or is_expired:
                schedule_recommendation_computation(current_user_id)

        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        etag = generate_recommendations_etag(
            current_user_id, record, seen_ids, section="feed", extra=today_str
        )
        response.headers["ETag"] = etag
        response.headers["Cache-Control"] = RECS_CACHE_HEADER

        if check_etag_match(request.headers.get("if-none-match"), etag):
            return Response(status_code=304, headers={"ETag": etag, "Cache-Control": RECS_CACHE_HEADER})

        if seen_ids:
            explore = [it for it in explore if it.get("id") not in seen_ids and it.get("tmdb_id") not in seen_ids]
            recent_vibe = [it for it in recent_vibe if it.get("id") not in seen_ids and it.get("tmdb_id") not in seen_ids]
            new_personas = []
            for p in personas:
                p_c = dict(p)
                recs = p_c.get("recommendations", [])
                p_c["recommendations"] = [it for it in recs if it.get("id") not in seen_ids and it.get("tmdb_id") not in seen_ids]
                new_personas.append(p_c)
            personas = new_personas

        up_movies = [it for it in (upcoming.get("movies") or []) if it.get("id") not in seen_ids and it.get("tmdb_id") not in seen_ids and str(it.get("release_date") or "").strip() >= today_str]
        up_tv = [it for it in (upcoming.get("tv") or upcoming.get("series") or []) if it.get("id") not in seen_ids and it.get("tmdb_id") not in seen_ids and str(it.get("first_air_date") or "").strip() >= today_str]
        
        clean_explore = sanitize_recommendation_list(explore)
        clean_recent_vibe = sanitize_recommendation_list(recent_vibe)
        for p in personas:
            if isinstance(p, dict) and p.get("recommendations"):
                sanitize_recommendation_list(p["recommendations"])
        clean_up_movies = sanitize_recommendation_list(up_movies)
        clean_up_tv = sanitize_recommendation_list(up_tv)

        upcoming = {"movies": clean_up_movies, "tv": clean_up_tv, "series": clean_up_tv}

        return {
            "explore": clean_explore,
            "recent_vibe": clean_recent_vibe,
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
            
        candidate_obj = {
            "id": tmdb_id,
            "title": getattr(tmdb_data, "title", None) or getattr(tmdb_data, "name", "Mídia"),
            "name": getattr(tmdb_data, "name", None),
            "media_type": media_type,
            "genres": [g.name if hasattr(g, "name") else g for g in tmdb_data.genres] if tmdb_data.genres else [],
            "keywords": tmdb_data.keywords if hasattr(tmdb_data, "keywords") and tmdb_data.keywords else [],
            "keywords_list": tmdb_data.keywords if hasattr(tmdb_data, "keywords") and tmdb_data.keywords else [],
            "credits": tmdb_data.credits.dict() if tmdb_data.credits and hasattr(tmdb_data.credits, "dict") else None,
            "cast": [c.name for c in tmdb_data.credits.cast[:10]] if tmdb_data.credits and tmdb_data.credits.cast else [],
            "crew": [c.name for c in tmdb_data.credits.crew if getattr(c, "job", "") in ("Director", "Creator")] if tmdb_data.credits and tmdb_data.credits.crew else [],
            "original_language": getattr(tmdb_data, "original_language", "") or "",
        }
        candidate_vector = build_candidate_vector(candidate_obj)
                    
        # Check if we have local embedding
        from app.media.models import Media as MediaModel
        local_media = await db.execute(select(MediaModel).where(MediaModel.tmdb_id == tmdb_id, MediaModel.media_type == media_type))
        local_media_obj = local_media.scalars().first()
        cand_emb = local_media_obj.embedding if local_media_obj else None

        active_personas = [p for p in user.taste_clusters if p.get("is_active", True)] if (user.taste_clusters and len(user.taste_clusters) > 1) else []

        user_list = []
        try:
            from app.tracking.services import get_user_list
            user_list = await get_user_list(db, user.id)
        except Exception:
            pass

        omni = compute_omnimatch_score(
            user_vector=user.feature_vector,
            candidate_vector=candidate_vector,
            user_embedding=user.embedding,
            candidate_embedding=cand_emb,
            active_personas=active_personas,
            vote_average=float(getattr(tmdb_data, "vote_average", 0.0) or 0.0),
            vote_count=int(getattr(tmdb_data, "vote_count", 0) or 0),
            user_item_count=len(user_list),
            candidate=candidate_obj,
            user_items=user_list,
        )

        final_score = omni["match_score"]
        try:
            from app.recommendation.ranker import LightGBMRanker
            lgbm_ranker = LightGBMRanker()
            if lgbm_ranker.is_ready():
                final_score = lgbm_ranker.predict_score(
                    user=user,
                    candidate=candidate_obj,
                    cand_emb=cand_emb,
                    base_omnimatch=final_score,
                    user_items=user_list,
                    active_personas=active_personas,
                )
        except Exception:
            pass

        return {
            "match_score": final_score,
            "match_tags": omni["match_tags"],
            "is_preliminary": omni.get("is_preliminary", False),
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

@router.post("/events")
async def record_recommendation_events(
    payload: TelemetryBatchRequest,
    current_user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """
    Coleta eventos de telemetria leve (impressions, clicks, add_to_list, dismiss)
    para alimentar o treino do ranker LambdaMART sem viés de posição (Sub-etapa 6.1 + Ideia 10).
    """
    try:
        import uuid
        from app.recommendation.models import RecommendationInteractionEvent

        u_uuid = uuid.UUID(current_user_id) if isinstance(current_user_id, str) else current_user_id
        events_to_add = []
        for ev in payload.events:
            events_to_add.append(
                RecommendationInteractionEvent(
                    user_id=u_uuid,
                    tmdb_id=ev.tmdb_id,
                    media_type=ev.media_type or "movie",
                    event_type=(ev.event_type or "impression").strip().lower(),
                    carousel_type=ev.carousel_type or "explore",
                    position=ev.position if ev.position is not None else 0,
                    score_presented=ev.score_presented,
                )
            )

        if events_to_add:
            db.add_all(events_to_add)
            await db.commit()

        return {"status": "success", "recorded": len(events_to_add)}
    except Exception as e:
        logger.exception(f"Erro ao registrar eventos de telemetria: {e}")
        raise HTTPException(status_code=500, detail="Erro interno ao registrar eventos de telemetria.")

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

        # Telemetria: registra evento de dismiss (Sub-etapa 6.1)
        try:
            import uuid
            from app.recommendation.models import RecommendationInteractionEvent
            u_uuid = uuid.UUID(current_user_id) if isinstance(current_user_id, str) else current_user_id
            db.add(
                RecommendationInteractionEvent(
                    user_id=u_uuid,
                    tmdb_id=payload.tmdb_id,
                    media_type=payload.media_type or "movie",
                    event_type="dismiss",
                    carousel_type="explore",
                    position=0,
                )
            )
            await db.commit()
        except Exception as te:
            logger.warning(f"Erro ao registrar telemetria de dismiss: {te}")

        _seen_ids_cache.delete_prefix(str(current_user_id))

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
        _seen_ids_cache.delete_prefix(str(current_user_id))
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

@router.get("/together")
async def get_together_recommendations_endpoint(
    friend_id: str = Query(..., description="ID do amigo com quem fazer o match"),
    unseen_mode: str = Query("one", description="'one' (inédito para pelo menos 1) ou 'both' (inédito para ambos)"),
    media_type: str = Query("all", description="'all', 'movie' ou 'tv'"),
    genre_id: Optional[int] = Query(None, description="ID do gênero TMDB para filtrar"),
    min_vote_average: Optional[float] = Query(None, ge=0.0, le=10.0, description="Nota mínima TMDB"),
    limit: int = Query(30, ge=1, le=100),
    current_user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """
    Gera recomendações conjuntas ('Assistir Juntos') combinando os gostos
    do usuário autenticado e do amigo especificado.
    """
    try:
        import uuid
        u_uuid = uuid.UUID(current_user_id) if isinstance(current_user_id, str) else current_user_id
        f_uuid = uuid.UUID(friend_id) if isinstance(friend_id, str) else friend_id

        from app.recommendation.together_service import get_together_recommendations
        return await get_together_recommendations(
            db=db,
            current_user_id=u_uuid,
            friend_id=f_uuid,
            unseen_mode=unseen_mode,
            media_type=media_type,
            genre_id=genre_id,
            min_vote_average=min_vote_average,
            limit=limit,
        )
    except HTTPException:
        raise
    except ValueError:
        raise HTTPException(status_code=400, detail="ID de amigo inválido.")
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail="Erro interno ao gerar recomendações em dupla.")

