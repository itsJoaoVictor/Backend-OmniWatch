import asyncio
import logging
from typing import List, Dict, Any, Optional
import uuid

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import or_, and_
from fastapi import HTTPException, status

from app.users.models import User
from app.friends.models import Friendship
from app.tracking.models import UserListItem
from app.media.models import Media
from app.recommendation.matcher import calculate_match_score
from app.recommendation.orchestrator import (
    fetch_discover_candidates,
    enrich_candidates_with_details,
    build_candidate_vector,
)
from app.recommendation.embeddings import calculate_cosine_similarity
from app.recommendation.service import get_active_dismissed_tmdb_ids

logger = logging.getLogger(__name__)

async def get_together_recommendations(
    db: AsyncSession,
    current_user_id: uuid.UUID,
    friend_id: uuid.UUID,
    unseen_mode: str = "one",  # "one" | "both"
    media_type: str = "all",   # "all" | "movie" | "tv"
    genre_id: Optional[int] = None,
    min_vote_average: Optional[float] = None,
    limit: int = 30,
) -> Dict[str, Any]:
    """
    Gera recomendações conjuntas ('Assistir Juntos') para o usuário atual e um amigo.
    Utiliza Média Harmônica dos Match Scores e respeita as regras de ineditismo.
    """
    # 1. Validação de Amizade
    friendship_query = select(Friendship).where(
        Friendship.status == "accepted",
        or_(
            and_(Friendship.requester_id == current_user_id, Friendship.addressee_id == friend_id),
            and_(Friendship.addressee_id == current_user_id, Friendship.requester_id == friend_id),
        )
    )
    f_res = await db.execute(friendship_query)
    friendship = f_res.scalar_one_or_none()
    if not friendship:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="A amizade com este usuário não está ativa ou não foi encontrada."
        )

    # 2. Carrega usuários
    users_res = await db.execute(select(User).where(User.id.in_([current_user_id, friend_id])))
    users_map = {u.id: u for u in users_res.scalars().all()}
    user_a = users_map.get(current_user_id)
    user_b = users_map.get(friend_id)

    if not user_a or not user_b:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Usuário ou amigo não encontrado.")

    vector_a = user_a.feature_vector or {}
    vector_b = user_b.feature_vector or {}
    emb_a = user_a.embedding
    emb_b = user_b.embedding

    # 3. Busca lista de vistos/em andamento e recomendações ocultadas de ambos os usuários
    list_items_res = await db.execute(
        select(UserListItem.user_id, Media.tmdb_id, UserListItem.status)
        .join(Media, UserListItem.media_id == Media.id)
        .where(UserListItem.user_id.in_([current_user_id, friend_id]))
    )
    
    seen_a: set = set()
    seen_b: set = set()
    for row in list_items_res.all():
        u_id, tmdb_id, item_status = row[0], row[1], row[2]
        if item_status in ("completed", "watching"):
            if u_id == current_user_id:
                seen_a.add(tmdb_id)
            elif u_id == friend_id:
                seen_b.add(tmdb_id)

    # RN: Recomendações Ocultadas / Dispensadas por A ou por B
    dismissed_a, dismissed_b = await asyncio.gather(
        get_active_dismissed_tmdb_ids(db, str(current_user_id)),
        get_active_dismissed_tmdb_ids(db, str(friend_id)),
    )
    dismissed_both = dismissed_a.union(dismissed_b)

    # 4. Busca candidatos (Discover para A e para B simultaneamente)
    cand_tasks = []
    if vector_a:
        cand_tasks.append(fetch_discover_candidates(vector_a, db=db, user_id=str(current_user_id)))
    else:
        cand_tasks.append(fetch_discover_candidates({}, db=db, user_id=str(current_user_id)))

    if vector_b:
        cand_tasks.append(fetch_discover_candidates(vector_b, db=db, user_id=str(friend_id)))
    else:
        cand_tasks.append(fetch_discover_candidates({}, db=db, user_id=str(friend_id)))

    cand_results = await asyncio.gather(*cand_tasks, return_exceptions=True)

    all_raw_candidates: List[Dict[str, Any]] = []
    seen_cand_ids: set = set()
    for res in cand_results:
        if isinstance(res, list):
            for c in res:
                cid = c.get("id")
                if cid and cid not in seen_cand_ids:
                    seen_cand_ids.add(cid)
                    all_raw_candidates.append(c)

    # 5. Aplica filtros básicos antes do enriquecimento (tipo de mídia, gênero, nota mínima, ineditismo e ocultadas)
    filtered_candidates: List[Dict[str, Any]] = []
    for c in all_raw_candidates:
        tmdb_id = c.get("id")
        c_type = c.get("media_type")

        # RN: Se for recomendação ocultada por QUALQUER um dos dois usuários, descarta imediatamente
        if tmdb_id in dismissed_both:
            continue

        # Filtro de tipo de mídia
        if media_type != "all" and c_type != media_type:
            continue

        # Filtro de gênero (se especificado)
        if genre_id is not None:
            genre_ids = c.get("genre_ids") or []
            if genre_id not in genre_ids:
                continue

        # Filtro de nota mínima TMDB
        if min_vote_average is not None:
            vote_avg = c.get("vote_average", 0.0)
            if vote_avg < min_vote_average:
                continue

        # Filtro de ineditismo
        watched_by_a = tmdb_id in seen_a
        watched_by_b = tmdb_id in seen_b

        if unseen_mode == "both":
            # Inédito para ambos: nenhum dos dois pode ter assistido
            if watched_by_a or watched_by_b:
                continue
        else:
            # Inédito para pelo menos 1 (padrão): exclui apenas se AMBOS assistiram
            if watched_by_a and watched_by_b:
                continue

        c["_watched_by_user"] = watched_by_a
        c["_watched_by_friend"] = watched_by_b
        filtered_candidates.append(c)

    if not filtered_candidates:
        return {
            "friend": {
                "id": str(user_b.id),
                "name": user_b.name,
                "username": user_b.username,
            },
            "items": [],
            "total": 0,
        }

    # 6. Enriquece candidatos com detalhes
    # Criamos um vetor sintetizado para enriquecer com prioridade candidatos promissores
    synth_vector = {}
    for k, v in vector_a.items():
        synth_vector[k] = synth_vector.get(k, 0.0) + v * 0.5
    for k, v in vector_b.items():
        synth_vector[k] = synth_vector.get(k, 0.0) + v * 0.5

    try:
        enriched = await enrich_candidates_with_details(filtered_candidates, user_vector=synth_vector)
    except Exception as e:
        logger.warning(f"Erro ao enriquecer candidatos: {e}")
        enriched = filtered_candidates

    # Busca embeddings locais se existirem
    cand_tmdb_ids = [c["id"] for c in enriched if "id" in c]
    if cand_tmdb_ids:
        try:
            media_records = await db.execute(
                select(Media.tmdb_id, Media.embedding).where(Media.tmdb_id.in_(cand_tmdb_ids))
            )
            emb_map = {row[0]: row[1] for row in media_records.all() if row[1]}
            for c in enriched:
                if c.get("id") in emb_map and not c.get("embedding"):
                    c["embedding"] = emb_map[c["id"]]
        except Exception:
            pass

    # 7. Calcula Match Duplo Harmônico e tags explicativas
    scored_items: List[Dict[str, Any]] = []

    for c in enriched:
        cand_vec = build_candidate_vector(c)
        cand_emb = c.get("embedding")

        # Score Usuário A
        aff_a, tags_a = calculate_match_score(vector_a, cand_vec)
        if emb_a and cand_emb:
            sim_a = calculate_cosine_similarity(emb_a, cand_emb)
            score_a = round(0.65 * (sim_a * 100.0) + 0.35 * aff_a, 1)
        else:
            score_a = aff_a

        # Score Usuário B
        aff_b, tags_b = calculate_match_score(vector_b, cand_vec)
        if emb_b and cand_emb:
            sim_b = calculate_cosine_similarity(emb_b, cand_emb)
            score_b = round(0.65 * (sim_b * 100.0) + 0.35 * aff_b, 1)
        else:
            score_b = aff_b

        # Média Harmônica com prevenção contra zeros
        if score_a <= 5.0 or score_b <= 5.0:
            # Se um dos dois tem afinidade muito baixa/nula, derruba o match conjunto
            combined_score = round(min(score_a, score_b) * 0.5, 1)
        else:
            harmonic = (2.0 * score_a * score_b) / (score_a + score_b)
            # Penalidade sutil para descompasso grande (> 30 pontos de diferença)
            diff = abs(score_a - score_b)
            diff_penalty = (diff - 30.0) * 0.2 if diff > 30.0 else 0.0
            combined_score = round(max(0.0, min(100.0, harmonic - diff_penalty)), 1)

        # Motivos compartilhados
        shared_tags = []
        set_tags_a = set(tags_a)
        set_tags_b = set(tags_b)
        overlap = set_tags_a.intersection(set_tags_b)

        for tag in overlap:
            clean = tag.replace("🎬 ", "").replace("🌟 ", "").replace("🚀 ", "")
            shared_tags.append(f"✨ Ambos curtem {clean}")

        if not shared_tags:
            if tags_a and tags_b:
                shared_tags.append(f"Você: {tags_a[0]}")
                shared_tags.append(f"{user_b.name.split()[0]}: {tags_b[0]}")
            elif tags_a:
                shared_tags.append(tags_a[0])
            elif tags_b:
                shared_tags.append(tags_b[0])

        item_data = {
            "id": c.get("id"),
            "title": c.get("title") or c.get("name") or "Sem título",
            "name": c.get("name"),
            "poster_path": c.get("poster_path"),
            "backdrop_path": c.get("backdrop_path"),
            "media_type": c.get("media_type", "movie"),
            "vote_average": c.get("vote_average", 0.0),
            "overview": c.get("overview", ""),
            "release_date": c.get("release_date") or c.get("first_air_date"),
            "match_score": combined_score,
            "user_score": score_a,
            "friend_score": score_b,
            "match_tags": shared_tags[:3],
            "watched_by_user": c.get("_watched_by_user", False),
            "watched_by_friend": c.get("_watched_by_friend", False),
        }
        scored_items.append(item_data)

    # 8. Ordena por Match Duplo decrescente
    scored_items.sort(key=lambda x: x["match_score"], reverse=True)

    return {
        "friend": {
            "id": str(user_b.id),
            "name": user_b.name,
            "username": user_b.username,
        },
        "items": scored_items[:limit],
        "total": len(scored_items),
    }
