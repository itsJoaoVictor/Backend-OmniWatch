import asyncio
import logging
from typing import List, Dict, Any, Optional, Set, Tuple
import uuid

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import or_, and_
from fastapi import HTTPException, status

from app.users.models import User
from app.friends.models import Friendship
from app.tracking.models import UserListItem
from app.media.models import Media
from app.recommendation.matcher import calculate_match_score, compute_omnimatch_score
from app.recommendation.orchestrator import (
    fetch_discover_candidates,
    enrich_candidates_with_details,
    build_candidate_vector,
    apply_fast_dpp_reranking,
)
from app.recommendation.embeddings import calculate_cosine_similarity
from app.recommendation.service import get_active_dismissed_media_keys

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

    # 3. Busca lista de vistos/em andamento e recomendações ocultadas de ambos os usuários usando chave canônica (media_type, tmdb_id)
    list_items_res = await db.execute(
        select(UserListItem.user_id, Media.media_type, Media.tmdb_id, UserListItem.status)
        .join(Media, UserListItem.media_id == Media.id)
        .where(UserListItem.user_id.in_([current_user_id, friend_id]))
    )
    
    seen_a: Set[Tuple[str, int]] = set()
    seen_b: Set[Tuple[str, int]] = set()
    watchlist_a: Set[Tuple[str, int]] = set()
    watchlist_b: Set[Tuple[str, int]] = set()

    for row in list_items_res.all():
        u_id, m_type, tmdb_id, item_status = row[0], row[1], row[2], row[3]
        item_key = (m_type or "movie", tmdb_id)
        if item_status in ("completed", "watching"):
            if u_id == current_user_id:
                seen_a.add(item_key)
            elif u_id == friend_id:
                seen_b.add(item_key)
        elif item_status == "plan_to_watch":
            if u_id == current_user_id:
                watchlist_a.add(item_key)
            elif u_id == friend_id:
                watchlist_b.add(item_key)

    in_both_watchlists = watchlist_a.intersection(watchlist_b)

    # RN: Recomendações Ocultadas / Dispensadas por A ou por B (chave composta)
    dismissed_a, dismissed_b = await asyncio.gather(
        get_active_dismissed_media_keys(db, str(current_user_id)),
        get_active_dismissed_media_keys(db, str(friend_id)),
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
    seen_cand_keys: Set[Tuple[str, int]] = set()
    for res in cand_results:
        if isinstance(res, list):
            for c in res:
                cid = c.get("id")
                mtype = c.get("media_type") or "movie"
                ckey = (mtype, cid)
                if cid and ckey not in seen_cand_keys:
                    seen_cand_keys.add(ckey)
                    all_raw_candidates.append(c)

    # 4.1 Injeta mídias que estão no 'Quero Ver' de ambos os usuários para garantir que participem do ranking
    if in_both_watchlists:
        try:
            for m_type, tmdb_id in in_both_watchlists:
                ckey = (m_type, tmdb_id)
                if ckey in seen_cand_keys or ckey in dismissed_both:
                    continue
                shared_media_stmt = select(Media).where(
                    Media.media_type == m_type,
                    Media.tmdb_id == tmdb_id
                )
                shared_media_res = await db.execute(shared_media_stmt)
                sm = shared_media_res.scalars().first()
                if sm:
                    seen_cand_keys.add(ckey)
                    genre_ids_list = []
                    if sm.genres and isinstance(sm.genres, list):
                        for g in sm.genres:
                            if isinstance(g, dict) and "id" in g:
                                genre_ids_list.append(g["id"])
                    all_raw_candidates.append({
                        "id": sm.tmdb_id,
                        "title": sm.title,
                        "name": sm.title if sm.media_type == "tv" else None,
                        "media_type": sm.media_type,
                        "poster_path": sm.poster_path,
                        "backdrop_path": sm.backdrop_path,
                        "genre_ids": genre_ids_list,
                        "vote_average": 7.0,
                        "release_date": sm.release_date,
                        "overview": "",
                    })
        except Exception as e:
            logger.warning(f"Erro ao injetar obras do Quero Ver compartilhado: {e}")

    # 5. Aplica filtros básicos antes do enriquecimento (tipo de mídia, gênero, nota mínima, ineditismo e ocultadas)
    filtered_candidates: List[Dict[str, Any]] = []
    for c in all_raw_candidates:
        tmdb_id = c.get("id")
        c_type = c.get("media_type") or "movie"
        cand_key = (c_type, tmdb_id)

        # RN: Se for recomendação ocultada por QUALQUER um dos dois usuários, descarta imediatamente
        if cand_key in dismissed_both:
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

        # Filtro de ineditismo usando chave canônica
        watched_by_a = cand_key in seen_a
        watched_by_b = cand_key in seen_b

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
        enriched = await enrich_candidates_with_details(filtered_candidates, user_vector=synth_vector, db=db)
    except Exception as e:
        logger.warning(f"Erro ao enriquecer candidatos: {e}")
        enriched = filtered_candidates

    # Busca embeddings locais usando chave composta (media_type, tmdb_id)
    cand_tmdb_ids = [c["id"] for c in enriched if "id" in c]
    if cand_tmdb_ids:
        try:
            media_records = await db.execute(
                select(Media.media_type, Media.tmdb_id, Media.embedding).where(Media.tmdb_id.in_(cand_tmdb_ids))
            )
            emb_map = {(row[0], row[1]): row[2] for row in media_records.all() if row[2]}
            for c in enriched:
                m_key = (c.get("media_type", "movie"), c.get("id"))
                if m_key in emb_map and not c.get("embedding"):
                    c["embedding"] = emb_map[m_key]
        except Exception:
            pass

    # 7. Calcula Match Duplo Harmônico e tags explicativas com compute_omnimatch_score
    scored_items: List[Dict[str, Any]] = []
    user_a_count = len(seen_a) + len(watchlist_a)
    user_b_count = len(seen_b) + len(watchlist_b)

    for c in enriched:
        cand_vec = build_candidate_vector(c)
        cand_emb = c.get("embedding")
        vote_avg = float(c.get("vote_average") or 0.0)
        vote_cnt = int(c.get("vote_count") or 0)

        # Score Usuário A
        res_a = compute_omnimatch_score(
            user_vector=vector_a,
            candidate_vector=cand_vec,
            user_embedding=emb_a,
            candidate_embedding=cand_emb,
            active_personas=user_a.taste_clusters if (user_a and user_a.taste_clusters and len(user_a.taste_clusters) > 1) else None,
            vote_average=vote_avg,
            vote_count=vote_cnt,
            user_item_count=user_a_count,
            candidate=c,
        )
        score_a = res_a["match_score"]
        tags_a = res_a["match_tags"]

        # Score Usuário B
        res_b = compute_omnimatch_score(
            user_vector=vector_b,
            candidate_vector=cand_vec,
            user_embedding=emb_b,
            candidate_embedding=cand_emb,
            active_personas=user_b.taste_clusters if (user_b and user_b.taste_clusters and len(user_b.taste_clusters) > 1) else None,
            vote_average=vote_avg,
            vote_count=vote_cnt,
            user_item_count=user_b_count,
            candidate=c,
        )
        score_b = res_b["match_score"]
        tags_b = res_b["match_tags"]

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

        # Cruzamento Matricial de Personas User A x User B (Sub-etapa 4.2 + Ideia 10)
        p_id_a = res_a.get("best_persona_id")
        p_id_b = res_b.get("best_persona_id")
        clusters_a = user_a.taste_clusters or []
        clusters_b = user_b.taste_clusters or []

        p_obj_a = next((p for p in clusters_a if p.get("id") == p_id_a), None) if p_id_a else None
        p_obj_b = next((p for p in clusters_b if p.get("id") == p_id_b), None) if p_id_b else None

        if p_obj_a and p_obj_b:
            name_a = (user_a.name or "Você").split()[0]
            name_b = (user_b.name or "Amigo").split()[0]
            title_a = p_obj_a.get("name") or p_obj_a.get("title") or "Gosto A"
            title_b = p_obj_b.get("name") or p_obj_b.get("title") or "Gosto B"
            if title_a.lower() == title_b.lower():
                shared_tags.insert(0, f"✨ Sintonia mútua: {title_a}")
            else:
                shared_tags.insert(0, f"✨ Sintonia: {title_a} ({name_a}) + {title_b} ({name_b})")

        # Provedores de streaming BR disponíveis
        br_providers = c.get("flatrate_providers_br") or c.get("watch_providers_br", [])
        if br_providers:
            shared_tags.append(f"📺 No {br_providers[0]} (BR)")

        # Priorização de obras no 'Quero Ver' de ambos os usuários (chave composta)
        cand_key = (c.get("media_type", "movie"), c.get("id"))
        is_in_both_watchlists = cand_key in in_both_watchlists
        if is_in_both_watchlists:
            # Garante pontuação de destaque para o desejo mútuo explícito
            combined_score = round(min(100.0, max(combined_score, 85.0) + 10.0), 1)
            shared_tags.insert(0, "🎯 No Quero Ver de ambos")

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
            "in_both_watchlists": is_in_both_watchlists,
            "flatrate_providers_br": br_providers,
            "watch_providers_br": br_providers,
            "_candidate_vector": cand_vec,
            "embedding": cand_emb,
            "genre_ids": c.get("genre_ids", []),
        }
        scored_items.append(item_data)

    # 8. Diversificação volumétrica com Fast Greedy DPP (< 2ms) + prioridade 'Quero Ver'
    must_include = [it for it in scored_items if it.get("in_both_watchlists")]
    must_include.sort(key=lambda x: x["match_score"], reverse=True)

    other_items = [it for it in scored_items if not it.get("in_both_watchlists")]

    dpp_target = max(0, limit - len(must_include))
    dpp_selected = apply_fast_dpp_reranking(other_items, top_k=dpp_target, lambda_param=0.65) if (dpp_target > 0 and other_items) else []

    final_items = must_include + dpp_selected

    for it in final_items:
        it.pop("_candidate_vector", None)
        it.pop("embedding", None)

    return {
        "friend": {
            "id": str(user_b.id),
            "name": user_b.name,
            "username": user_b.username,
        },
        "items": final_items[:limit],
        "total": len(final_items),
    }
