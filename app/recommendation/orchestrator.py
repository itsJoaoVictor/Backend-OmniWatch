import asyncio
import httpx
import math
import numpy as np
from typing import List, Dict, Any, Optional, Set, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.users.models import User
from app.core.config import settings
from app.recommendation.matcher import calculate_match_score, compute_omnimatch_score
from app.recommendation.embeddings import build_text_for_embedding, generate_embeddings_batch, calculate_cosine_similarity
from app.recommendation.vector_builder import WEIGHTS, normalize_genre_to_canonical

# TMDB genre_id → pt-BR name (matches vector keys like "genre_ação")
GENRE_MAP = {
    28: "ação", 12: "aventura", 16: "animação", 35: "comédia", 80: "crime",
    99: "documentário", 18: "drama", 10751: "família", 14: "fantasia", 36: "história",
    27: "terror", 10402: "música", 9648: "mistério", 10749: "romance", 878: "ficção científica",
    10770: "cinema tv", 53: "thriller", 10752: "guerra", 37: "faroeste",
    10759: "ação", 10762: "infantil", 10763: "notícias", 10764: "reality",
    10765: "ficção científica", 10766: "novela", 10767: "talk", 10768: "guerra"
}

GENRE_MAP_REVERSE = {v: k for k, v in GENRE_MAP.items()}

# Keyword name → TMDB keyword ID (common ones for discover filtering)
KEYWORD_MAP = {
    "superhero": 9715, "super-herói": 9715,
    "based on novel": 818, "time travel": 4379, "space": 1526,
    "dystopia": 4565, "spy": 3691, "heist": 9748, "revenge": 161562,
    "zombie": 12377, "vampire": 12377, "magic": 9882,
    "artificial intelligence": 9882, "robot": 11451,
    "sports": 12655, "war": 10752,
}

TV_GENRE_MAP = {
    "ação": "10759", "aventura": "10759", "action & adventure": "10759",
    "ficção científica": "10765", "fantasia": "10765", "sci-fi & fantasy": "10765",
    "guerra": "10768", "política": "10768", "war & politics": "10768",
    "comédia": "35", "drama": "18", "animação": "16", "mistério": "9648", "crime": "80",
    "documentário": "99", "faroeste": "37", "família": "10751",
    "infantil": "10762", "kids": "10762", "notícias": "10763", "news": "10763",
    "reality": "10764", "novela": "10766", "soap": "10766",
    "talk": "10767", "talk show": "10767",
}

async def fetch_discover_candidates(
    user_vector: Dict[str, float],
    db=None,
    user_id: str = None,
    persona_id: Optional[str] = None,
    allowed_media_keys: Optional[Set[Tuple[str, int]]] = None,
) -> List[Dict[str, Any]]:
    """
    Recuperação Multi-Funil de Candidatos TMDB (Sub-etapa 2.1 + Ideia 9):
    4 funis assíncronos paralelos com fusão por postos recíprocos (RRF):
    - Funil 1 (35%): Grafo Comportamental Direto (/recommendations das 3-5 âncoras de ouro)
    - Funil 2 (35%): Discover Paramétrico Booleano Fino (conjunção AND ',' de gêneros, sem dealbreakers, vote_avg >= 6.8)
    - Funil 3 (15%): Franquias & Autoria Cruzada (/collection/{id} + /person/{id}/combined_credits de diretores)
    - Funil 4 (15%): Momentum & Streaming BR (/discover com watch_region=BR e flatrate)
    Reduz chamadas HTTP externas de 112 para 12-16 requisições consolidadas.
    """
    if not settings.TMDB_API_KEY:
        return []

    user_items = []
    golden_anchors = []
    known_collection_ids = set()
    signature_director_ids = set()
    dealbreaker_genre_ids = []
    live_genre_name_to_id: Dict[str, int] = {}

    genre_features = [(k.replace("genre_", ""), v) for k, v in user_vector.items() if k.startswith("genre_") and v > 0]
    genre_features.sort(key=lambda x: x[1], reverse=True)
    top_genres = [g for g, _ in genre_features[:6]]
    persona_focus_genres: Set[str] = set(top_genres[:3]) if persona_id else set()

    if db and user_id:
        try:
            from app.tracking.services import get_user_list
            user_items = await get_user_list(db, user_id)

            for list_item in user_items:
                media = list_item.media
                if not media:
                    continue

                if media.genres:
                    for g in media.genres:
                        if isinstance(g, dict):
                            _name = str(g.get("name", "")).lower()
                            _gid = g.get("id")
                            if _name and _gid:
                                live_genre_name_to_id[_name] = _gid

                if allowed_media_keys is not None and (media.media_type, media.tmdb_id) not in allowed_media_keys:
                    continue

                is_fav = getattr(list_item, "is_favorite", False)
                rating = getattr(list_item, "rating", 0.0) or 0.0
                rewatch = getattr(list_item, "rewatch_count", 0) or 0

                # Âncoras de ouro: favoritas, ratings >= 4.0, reassistidas, ou seeds explícitas da vibe recente
                if is_fav or rating >= 4.0 or rewatch > 0 or allowed_media_keys is not None:
                    priority = (3.0 if is_fav else 0.0) + (rating or 3.5) + (2.0 if rewatch > 0 else 0.0)
                    primary_g = "geral"
                    media_canons: List[str] = []
                    if media.genres and isinstance(media.genres, list):
                        for g_entry in media.genres:
                            canons = normalize_genre_to_canonical(g_entry)
                            if canons:
                                media_canons.extend(canons)
                        if media_canons:
                            primary_g = media_canons[0]
                    if persona_focus_genres:
                        if not (set(media_canons) & persona_focus_genres):
                            continue
                        for mc in media_canons:
                            if mc in persona_focus_genres:
                                primary_g = mc
                                break
                    golden_anchors.append((priority, media.media_type, media.tmdb_id, primary_g))

                if getattr(media, "collection_tmdb_id", None):
                    known_collection_ids.add(media.collection_tmdb_id)

                if media.crew_ids:
                    for entry in media.crew_ids:
                        pid = entry.get("person_id")
                        name = entry.get("name", "")
                        if pid and name and user_vector.get(f"director_{name.lower()}", 0.0) >= 1.5:
                            signature_director_ids.add(pid)
        except Exception:
            pass

    # Dealbreakers: gêneros com peso explicitamente negativo no perfil
    for k, v in user_vector.items():
        if k.startswith("genre_") and v < 0:
            canon = k.replace("genre_", "")
            gid = GENRE_MAP_REVERSE.get(canon)
            if gid:
                dealbreaker_genre_ids.append(str(gid))

    without_genres_str = ",".join(set(dealbreaker_genre_ids)) if dealbreaker_genre_ids else None

    # Seleção estratificada de âncoras de ouro por gênero primário (evita monotonia de um único gênero)
    golden_anchors.sort(key=lambda x: x[0], reverse=True)

    def _select_stratified_anchors(anchors_list: List[Any], max_count: int = 6) -> List[Tuple[float, str, int]]:
        selected: List[Tuple[float, str, int]] = []
        seen_ids: Set[int] = set()
        used_genres: Set[str] = set()
        # 1º passe: melhor âncora de cada gênero primário distinto
        for prio, mt, tid, pg in anchors_list:
            if tid not in seen_ids and pg not in used_genres:
                selected.append((prio, mt, tid))
                seen_ids.add(tid)
                used_genres.add(pg)
                if len(selected) >= max_count:
                    return selected
        # 2º passe: completa vagas restantes por prioridade
        for prio, mt, tid, _ in anchors_list:
            if tid not in seen_ids:
                selected.append((prio, mt, tid))
                seen_ids.add(tid)
                if len(selected) >= max_count:
                    break
        return selected

    top_movie_anchors = _select_stratified_anchors([a for a in golden_anchors if a[1] == "movie"], max_count=6)
    top_tv_anchors = _select_stratified_anchors([a for a in golden_anchors if a[1] == "tv"], max_count=6)
    top_anchors = top_movie_anchors + top_tv_anchors

    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {settings.TMDB_API_KEY}"
    }

    async def fetch_endpoint(_client: Any, endpoint: str, media_type: Optional[str], params: Dict[str, Any]) -> List[Dict[str, Any]]:
        from app.core.tmdb_client import tmdb_get_json
        default_params = {"language": "pt-BR", "page": 1}
        default_params.update(params)
        data = await tmdb_get_json(endpoint, params=default_params, timeout=12.0)
        if not data:
            return []
        if "results" in data:
            raw_list = data.get("results", [])
        elif "parts" in data:
            raw_list = data.get("parts", [])
            for r in raw_list:
                r["media_type"] = "movie"
        elif "crew" in data:
            raw_list = [c for c in data.get("crew", []) if c.get("job") == "Director"]
        else:
            raw_list = []

        for r in raw_list:
            if media_type:
                r["media_type"] = media_type
            elif "media_type" not in r:
                r["media_type"] = "movie" if "title" in r else "tv"
        return raw_list

    async with httpx.AsyncClient() as client:
        # === FUNIL 1: Grafo Comportamental Direto (35%) ===
        funnel_1_tasks = []
        for _, mtype, tid in top_anchors:
            endpoint = f"/{mtype}/{tid}/recommendations"
            funnel_1_tasks.append(fetch_endpoint(client, endpoint, mtype, {"page": 1}))

        # === FUNIL 2: Discover Paramétrico Multi-Eixo (35%) ===
        funnel_2_tasks = []

        top_m_ids = []
        for g in top_genres:
            gid = live_genre_name_to_id.get(g) or GENRE_MAP_REVERSE.get(g)
            if gid and str(gid) not in top_m_ids:
                top_m_ids.append(str(gid))

        top_tv_ids = []
        for g in top_genres:
            tid = TV_GENRE_MAP.get(g, str(GENRE_MAP_REVERSE.get(g, "")))
            if tid and str(tid) not in top_tv_ids:
                top_tv_ids.append(str(tid))

        # Cria múltiplos pares complementares de gêneros para explorar todas as facetas do usuário
        m_genre_queries = []
        if len(top_m_ids) >= 2:
            m_genre_queries.append(",".join(top_m_ids[:2]))
        elif top_m_ids:
            m_genre_queries.append(top_m_ids[0])
        if persona_id:
            if len(top_m_ids) >= 3:
                m_genre_queries.append(f"{top_m_ids[0]},{top_m_ids[2]}")
                m_genre_queries.append(f"{top_m_ids[1]},{top_m_ids[2]}")
            for single_gid in top_m_ids[:2]:
                if single_gid not in m_genre_queries:
                    m_genre_queries.append(single_gid)
        else:
            if len(top_m_ids) >= 4:
                m_genre_queries.append(",".join(top_m_ids[2:4]))
            if len(top_m_ids) >= 5:
                m_genre_queries.append(top_m_ids[4])

        tv_genre_queries = []
        if len(top_tv_ids) >= 2:
            tv_genre_queries.append(",".join(top_tv_ids[:2]))
        elif top_tv_ids:
            tv_genre_queries.append(top_tv_ids[0])
        if persona_id:
            if len(top_tv_ids) >= 3:
                tv_genre_queries.append(f"{top_tv_ids[0]},{top_tv_ids[2]}")
                tv_genre_queries.append(f"{top_tv_ids[1]},{top_tv_ids[2]}")
            for single_tid in top_tv_ids[:2]:
                if single_tid not in tv_genre_queries:
                    tv_genre_queries.append(single_tid)
        else:
            if len(top_tv_ids) >= 4:
                tv_genre_queries.append(",".join(top_tv_ids[2:4]))
            if len(top_tv_ids) >= 5:
                tv_genre_queries.append(top_tv_ids[4])

        lang_pref = None
        lang_features = [(k.replace("lang_", ""), v) for k, v in user_vector.items() if k.startswith("lang_") and v >= 2.5]
        lang_features.sort(key=lambda x: x[1], reverse=True)
        if lang_features and lang_features[0][0] != "en":
            lang_pref = lang_features[0][0]

        for idx_q, wg_m in enumerate(m_genre_queries):
            m_params = {
                "with_genres": wg_m,
                "vote_average.gte": 6.8 if idx_q == 0 else 6.6,
                "vote_count.gte": 150 if idx_q == 0 else 120,
                "sort_by": "popularity.desc",
                "page": 1
            }
            if without_genres_str:
                m_params["without_genres"] = without_genres_str
            if lang_pref and idx_q == 0:
                m_params["with_original_language"] = lang_pref
            funnel_2_tasks.append(fetch_endpoint(client, "/discover/movie", "movie", m_params))

        for idx_q, wg_tv in enumerate(tv_genre_queries):
            tv_params = {
                "with_genres": wg_tv,
                "vote_average.gte": 6.8 if idx_q == 0 else 6.6,
                "vote_count.gte": 80 if idx_q == 0 else 60,
                "sort_by": "popularity.desc",
                "page": 1
            }
            if without_genres_str:
                tv_params["without_genres"] = without_genres_str
            if lang_pref and idx_q == 0:
                tv_params["with_original_language"] = lang_pref
            funnel_2_tasks.append(fetch_endpoint(client, "/discover/tv", "tv", tv_params))

        # === FUNIL 3: Franquias & Autoria Cruzada (15%) ===
        funnel_3_tasks = []
        for col_id in list(known_collection_ids)[:3]:
            funnel_3_tasks.append(fetch_endpoint(client, f"/collection/{col_id}", "movie", {}))
        for p_id in list(signature_director_ids)[:3]:
            funnel_3_tasks.append(fetch_endpoint(client, f"/person/{p_id}/combined_credits", None, {}))

        # === FUNIL 4: Momentum & Streaming BR (15%) ===
        funnel_4_tasks = []
        funnel_4_tasks.append(fetch_endpoint(client, "/discover/movie", "movie", {
            "watch_region": "BR",
            "with_watch_monetization_types": "flatrate",
            "vote_average.gte": 6.5,
            "vote_count.gte": 100,
            "sort_by": "popularity.desc",
            "page": 1
        }))
        funnel_4_tasks.append(fetch_endpoint(client, "/discover/tv", "tv", {
            "watch_region": "BR",
            "with_watch_monetization_types": "flatrate",
            "vote_average.gte": 6.5,
            "vote_count.gte": 50,
            "sort_by": "popularity.desc",
            "page": 1
        }))
        if not persona_id and allowed_media_keys is None:
            funnel_4_tasks.append(fetch_endpoint(client, "/trending/all/week", None, {"page": 1}))

        res_f1, res_f2, res_f3, res_f4 = await asyncio.gather(
            asyncio.gather(*funnel_1_tasks, return_exceptions=True) if funnel_1_tasks else asyncio.sleep(0, result=[]),
            asyncio.gather(*funnel_2_tasks, return_exceptions=True) if funnel_2_tasks else asyncio.sleep(0, result=[]),
            asyncio.gather(*funnel_3_tasks, return_exceptions=True) if funnel_3_tasks else asyncio.sleep(0, result=[]),
            asyncio.gather(*funnel_4_tasks, return_exceptions=True) if funnel_4_tasks else asyncio.sleep(0, result=[]),
        )

    # === FUSÃO POR POSTOS RECÍPROCOS (RRF) ===
    k_rrf = 60.0
    w_f1 = 0.35
    w_f2 = 0.35
    w_f3 = 0.15
    w_f4 = 0.15

    rrf_scores: Dict[Tuple[str, int], float] = {}
    candidate_map: Dict[Tuple[str, int], Dict[str, Any]] = {}

    def ingest_funnel_results(results_batch: Any, weight: float):
        if not isinstance(results_batch, (list, tuple)):
            return
        for b in results_batch:
            if not isinstance(b, list):
                continue
            rank_by_type = {"movie": 0, "tv": 0}
            for c in b:
                cid = c.get("id")
                mtype = c.get("media_type", "movie")
                if not cid or mtype not in ("movie", "tv"):
                    continue

                # Filtro de tração mínima para candidatos gerais já lançados (elimina ruído com 0 ou 1 voto)
                vc = float(c.get("vote_count", 0) or 0)
                r_date = c.get("release_date") or c.get("first_air_date")
                is_past = bool(r_date and len(str(r_date)) >= 4 and str(r_date)[:4].isdigit())
                if is_past and vc < (20.0 if mtype == "movie" else 15.0):
                    continue

                rank_idx = rank_by_type[mtype]
                rank_by_type[mtype] += 1
                key = (mtype, cid)
                if key not in candidate_map:
                    candidate_map[key] = c
                rrf_scores[key] = rrf_scores.get(key, 0.0) + (weight / (k_rrf + rank_idx + 1))

    ingest_funnel_results(res_f1, w_f1)
    ingest_funnel_results(res_f2, w_f2)
    ingest_funnel_results(res_f3, w_f3)
    ingest_funnel_results(res_f4, w_f4)

    sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)
    movie_cands: List[Dict[str, Any]] = []
    tv_cands: List[Dict[str, Any]] = []
    for key in sorted_keys:
        c = candidate_map[key]
        c["_rrf_score"] = rrf_scores[key]
        if key[0] == "movie" and len(movie_cands) < 100:
            movie_cands.append(c)
        elif key[0] == "tv" and len(tv_cands) < 100:
            tv_cands.append(c)
        if len(movie_cands) >= 100 and len(tv_cands) >= 100:
            break

    return movie_cands + tv_cands


async def enrich_candidates_with_details(
    candidates: List[Dict[str, Any]],
    user_vector: Optional[Dict[str, float]] = None,
    db: Optional[AsyncSession] = None,
    seen_keys: Optional[set] = None,
    allow_collection_redirect: bool = True
) -> List[Dict[str, Any]]:
    """
    Enriquecimento 1-RTT TMDB (append_to_response=credits,keywords,watch/providers,belongs_to_collection)
    + Curadoria Collection Gate (redireciona para Filme 1 se franquia inédita - Sub-etapa 2.4)
    + Batch Embeddings OpenRouter (qwen/qwen3-embedding-8b gravados em Media - Sub-etapa 2.2).
    """
    from app.details.services import fetch_movie_details, fetch_tv_details
    import math

    if not candidates:
        return []

    movies = [c for c in candidates if c.get("media_type") == "movie"]
    tv = [c for c in candidates if c.get("media_type") == "tv"]

    def compute_rrf_ranking(pool: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not pool:
            return []

        def get_fast_affinity(cand: Dict[str, Any]) -> float:
            if not user_vector:
                return 0.0
            g_score = 0.0
            for g_id in cand.get("genre_ids", []):
                for canon in normalize_genre_to_canonical(g_id):
                    g_score += user_vector.get(f"genre_{canon}", 0.0)
            for g in cand.get("genres", []):
                for canon in normalize_genre_to_canonical(g):
                    g_score += user_vector.get(f"genre_{canon}", 0.0)

            lang = cand.get("original_language", "")
            lang_score = user_vector.get(f"lang_{lang}", 0.0) if lang else 0.0
            return g_score + 0.5 * lang_score

        def get_quality_score(cand: Dict[str, Any]) -> float:
            va = float(cand.get("vote_average") or 0.0)
            vc = float(cand.get("vote_count") or 0.0)
            damped_vote = (va * vc + 6.5 * 50.0) / (vc + 50.0)
            pop = float(cand.get("popularity") or 0.0)
            return damped_vote + 0.25 * math.log1p(max(0.0, pop))

        def get_collab_priority(cand: Dict[str, Any]) -> float:
            return 1.0 if cand.get("collaborative") else 0.0

        sorted_by_aff = sorted(pool, key=get_fast_affinity, reverse=True)
        aff_rank = {id(c): i + 1 for i, c in enumerate(sorted_by_aff)}

        sorted_by_qual = sorted(pool, key=get_quality_score, reverse=True)
        qual_rank = {id(c): i + 1 for i, c in enumerate(sorted_by_qual)}

        sorted_by_collab = sorted(pool, key=get_collab_priority, reverse=True)
        collab_rank = {id(c): i + 1 for i, c in enumerate(sorted_by_collab)}

        k_rrf = 60.0
        w_aff = 1.0 if user_vector else 0.0
        w_qual = 0.6
        w_collab = 0.8

        for c in pool:
            cid_ref = id(c)
            rrf = (
                (w_aff / (k_rrf + aff_rank[cid_ref])) +
                (w_qual / (k_rrf + qual_rank[cid_ref])) +
                (w_collab / (k_rrf + collab_rank[cid_ref]))
            )
            aff_val = get_fast_affinity(c)
            c["_rrf_score"] = rrf * (1.0 + max(0.0, aff_val) / 20.0)

        return sorted(pool, key=lambda c: c.get("_rrf_score", 0.0), reverse=True)

    sorted_movies = compute_rrf_ranking(movies)
    sorted_tv = compute_rrf_ranking(tv)

    top_candidates = sorted_movies[:85] + sorted_tv[:85]

    async def enrich_one(candidate: Dict[str, Any]) -> Dict[str, Any]:
        tmdb_id = candidate.get("id")
        media_type = candidate.get("media_type", "movie")
        try:
            if media_type == "movie":
                details = await fetch_movie_details(tmdb_id)
                # Curadoria Collection Gate (Sub-etapa 2.4): Redireciona para Filme 1 se franquia inédita
                if allow_collection_redirect and hasattr(details, "belongs_to_collection") and details.belongs_to_collection and seen_keys is not None:
                    col = details.belongs_to_collection
                    col_id = col.id if hasattr(col, "id") else (col.get("id") if isinstance(col, dict) else None)
                    if col_id:
                        try:
                            from app.details.services import fetch_collection_details
                            col_resp = await fetch_collection_details(col_id)
                            if col_resp and col_resp.parts:
                                sorted_parts = sorted(
                                    col_resp.parts,
                                    key=lambda p: (getattr(p, "release_date", None) or (p.get("release_date") if isinstance(p, dict) else "")) or "9999"
                                )
                                if sorted_parts:
                                    first_p = sorted_parts[0]
                                    first_id = getattr(first_p, "id", None) or (first_p.get("id") if isinstance(first_p, dict) else None)
                                    if first_id and first_id != tmdb_id and ("movie", first_id) not in seen_keys:
                                        first_details = await fetch_movie_details(first_id)
                                        details = first_details
                                        candidate["id"] = first_details.id
                                        candidate["_collection_redirect"] = True
                        except Exception:
                            pass

                candidate["title"] = getattr(details, "title", None) or getattr(details, "name", "")
                candidate["overview"] = getattr(details, "overview", "")
                candidate["poster_path"] = getattr(details, "poster_path", None)
                candidate["backdrop_path"] = getattr(details, "backdrop_path", None)
                candidate["vote_average"] = getattr(details, "vote_average", 0.0) or candidate.get("vote_average") or 0.0
                candidate["vote_count"] = getattr(details, "vote_count", 0) or candidate.get("vote_count") or 0
                candidate["release_date"] = getattr(details, "release_date", "") or candidate.get("release_date") or candidate.get("first_air_date") or ""
                candidate["cast"] = [c.name for c in details.credits.cast[:10]] if details.credits and details.credits.cast else []
                candidate["crew"] = [c.name for c in details.credits.crew if getattr(c, "job", "") in ("Director", "Creator")] if details.credits and details.credits.crew else []
                candidate["keywords_list"] = details.keywords if hasattr(details, "keywords") else []
                candidate["original_language"] = getattr(details, "original_language", "") or ""
                if getattr(details, "genres", None):
                    candidate["genre_ids"] = [
                        (getattr(g, "id", 0) if not isinstance(g, dict) else g.get("id", 0))
                        for g in details.genres
                        if (getattr(g, "id", None) if not isinstance(g, dict) else g.get("id"))
                    ]
                    candidate["genres"] = [
                        {
                            "id": (getattr(g, "id", 0) if not isinstance(g, dict) else g.get("id", 0)),
                            "name": (getattr(g, "name", "") if not isinstance(g, dict) else g.get("name", "")),
                        }
                        for g in details.genres
                        if (getattr(g, "name", None) if not isinstance(g, dict) else g.get("name"))
                    ]
                elif candidate.get("genre_ids") and not candidate.get("genres"):
                    candidate["genres"] = [
                        {"id": gid, "name": GENRE_MAP.get(gid, "").title()}
                        for gid in candidate["genre_ids"]
                        if gid in GENRE_MAP
                    ]
                if hasattr(details, "watch_providers") and details.watch_providers:
                    candidate["watch_providers_br"] = [wp.provider_name for wp in details.watch_providers if hasattr(wp, "provider_name")]
                col_obj = getattr(details, "belongs_to_collection", None)
                if col_obj:
                    candidate["belongs_to_collection"] = (
                        {"id": getattr(col_obj, "id", None), "name": getattr(col_obj, "name", "")}
                        if not isinstance(col_obj, dict) else col_obj
                    )
            else:
                details = await fetch_tv_details(tmdb_id)
                candidate["title"] = getattr(details, "title", None) or getattr(details, "name", "")
                candidate["overview"] = getattr(details, "overview", "")
                candidate["poster_path"] = getattr(details, "poster_path", None)
                candidate["backdrop_path"] = getattr(details, "backdrop_path", None)
                candidate["vote_average"] = getattr(details, "vote_average", 0.0) or candidate.get("vote_average") or 0.0
                candidate["vote_count"] = getattr(details, "vote_count", 0) or candidate.get("vote_count") or 0
                air_date = getattr(details, "first_air_date", "") or candidate.get("first_air_date") or candidate.get("release_date") or ""
                candidate["first_air_date"] = air_date
                candidate["release_date"] = air_date
                candidate["cast"] = [c.name for c in details.credits.cast[:10]] if details.credits and details.credits.cast else []
                candidate["crew"] = [c.name for c in details.credits.crew if getattr(c, "job", "") in ("Director", "Creator")] if details.credits and details.credits.crew else []
                candidate["keywords_list"] = details.keywords if hasattr(details, "keywords") else []
                candidate["original_language"] = getattr(details, "original_language", "") or ""
                if getattr(details, "genres", None):
                    candidate["genre_ids"] = [
                        (getattr(g, "id", 0) if not isinstance(g, dict) else g.get("id", 0))
                        for g in details.genres
                        if (getattr(g, "id", None) if not isinstance(g, dict) else g.get("id"))
                    ]
                    candidate["genres"] = [
                        {
                            "id": (getattr(g, "id", 0) if not isinstance(g, dict) else g.get("id", 0)),
                            "name": (getattr(g, "name", "") if not isinstance(g, dict) else g.get("name", "")),
                        }
                        for g in details.genres
                        if (getattr(g, "name", None) if not isinstance(g, dict) else g.get("name"))
                    ]
                elif candidate.get("genre_ids") and not candidate.get("genres"):
                    candidate["genres"] = [
                        {"id": gid, "name": GENRE_MAP.get(gid, "").title()}
                        for gid in candidate["genre_ids"]
                        if gid in GENRE_MAP
                    ]
                if hasattr(details, "watch_providers") and details.watch_providers:
                    candidate["watch_providers_br"] = [wp.provider_name for wp in details.watch_providers if hasattr(wp, "provider_name")]
        except Exception:
            candidate["cast"] = []
            candidate["crew"] = []
            candidate["keywords_list"] = []
            candidate["original_language"] = ""
        return candidate

    enriched = await asyncio.gather(*[enrich_one(c) for c in top_candidates])

    # === BATCH EMBEDDINGS OPENROUTER (Sub-etapa 2.2) ===
    if db:
        try:
            cand_keys = [(c.get("media_type", "movie"), c["id"]) for c in enriched if "id" in c]
            from app.media.models import Media as MediaModel
            cand_tmdb_ids = [k[1] for k in cand_keys]
            media_rows = (await db.execute(
                select(MediaModel)
                .where(MediaModel.tmdb_id.in_(cand_tmdb_ids))
            )).scalars().all()
            existing_media_map = {(row.media_type or "movie", row.tmdb_id): row for row in media_rows}
            existing_embs = {k: row.embedding for k, row in existing_media_map.items() if row.embedding is not None}

            cands_to_embed = []
            texts_to_embed = []
            for c in enriched:
                m_key = (c.get("media_type", "movie"), c.get("id"))
                # Sincroniza vote_average / vote_count / genres no Media local caso estivessem zerados
                row_m = existing_media_map.get(m_key)
                if row_m is not None:
                    if c.get("vote_average") and not row_m.vote_average:
                        row_m.vote_average = float(c.get("vote_average") or 0.0)
                    if c.get("vote_count") and not row_m.vote_count:
                        row_m.vote_count = int(c.get("vote_count") or 0)
                    if c.get("genres") and not row_m.genres:
                        row_m.genres = [
                            g if isinstance(g, dict) else {"id": 0, "name": str(g)}
                            for g in c.get("genres", [])
                        ]
                if m_key in existing_embs:
                    c["embedding"] = existing_embs[m_key]
                elif not c.get("embedding"):
                    cands_to_embed.append(c)
                    texts_to_embed.append(build_text_for_embedding(c))

            if texts_to_embed:
                batch_embeddings = await generate_embeddings_batch(texts_to_embed)
                added_keys = set()
                for cand_item, emb in zip(cands_to_embed, batch_embeddings):
                    if emb:
                        cand_item["embedding"] = emb
                        cid = cand_item["id"]
                        mtype = cand_item.get("media_type", "movie")
                        key = (mtype, cid)
                        if key in added_keys:
                            continue
                        added_keys.add(key)
                        existing_m = (await db.execute(
                            select(MediaModel).where(MediaModel.tmdb_id == cid, MediaModel.media_type == mtype)
                        )).scalars().first()
                        if existing_m:
                            existing_m.embedding = emb
                            if cand_item.get("popularity"):
                                existing_m.popularity = float(cand_item.get("popularity") or 0.0)
                            if cand_item.get("vote_average"):
                                existing_m.vote_average = float(cand_item.get("vote_average") or 0.0)
                            if cand_item.get("vote_count"):
                                existing_m.vote_count = int(cand_item.get("vote_count") or 0)
                            if cand_item.get("watch_providers_br"):
                                existing_m.watch_providers_br = cand_item.get("watch_providers_br")
                        else:
                            by_tmdb = (await db.execute(
                                select(MediaModel).where(MediaModel.tmdb_id == cid)
                            )).scalars().first()
                            if by_tmdb:
                                by_tmdb.embedding = emb
                            else:
                                new_m = MediaModel(
                                    tmdb_id=cid,
                                    media_type=mtype,
                                    title=cand_item.get("title") or "Mídia",
                                    poster_path=cand_item.get("poster_path"),
                                    backdrop_path=cand_item.get("backdrop_path"),
                                    original_language=cand_item.get("original_language"),
                                    genres=[
                                        g if isinstance(g, dict) else {"id": 0, "name": str(g)}
                                        for g in cand_item.get("genres", [])
                                    ] if isinstance(cand_item.get("genres"), list) else [],
                                    directors=cand_item.get("crew", []),
                                    main_cast=cand_item.get("cast", []),
                                    keywords=cand_item.get("keywords_list", []),
                                    popularity=float(cand_item.get("popularity", 0.0) or 0.0),
                                    vote_average=float(cand_item.get("vote_average", 0.0) or 0.0),
                                    vote_count=int(cand_item.get("vote_count", 0) or 0),
                                    belongs_to_collection=cand_item.get("belongs_to_collection"),
                                    watch_providers_br=cand_item.get("flatrate_providers_br") or cand_item.get("watch_providers_br", []),
                                    embedding=emb
                                )
                                db.add(new_m)
                await db.commit()
        except Exception as e:
            try:
                await db.rollback()
            except Exception:
                pass

    return list(enriched)


def build_candidate_vector(candidate: Dict[str, Any]) -> Dict[str, float]:
    """
    Builds a feature vector from a candidate dict.
    Supports shallow (genre_ids only), enriched, and collaborative candidates.
    """
    vector = {}

    m_type = candidate.get("media_type")
    if m_type:
        vector[f"type_{m_type}"] = WEIGHTS["media_type"]

    # Genres from genre_ids (shallow)
    for g_id in candidate.get("genre_ids", []):
        for canonical_name in normalize_genre_to_canonical(g_id):
            vector[f"genre_{canonical_name}"] = WEIGHTS["genre"]

    # Genres from enriched/collaborative list
    for g in candidate.get("genres", []):
        for canonical_name in normalize_genre_to_canonical(g):
            vector[f"genre_{canonical_name}"] = WEIGHTS["genre"]

    # Fase 2: language feature
    lang = candidate.get("original_language", "")
    if lang and lang != "en":
        vector[f"lang_{lang}"] = WEIGHTS["language"]

    # Fase 1: enriched cast
    for actor in candidate.get("cast", []):
        if actor:
            vector[f"cast_{actor.lower()}"] = WEIGHTS["cast"]

    # Fase 1: enriched crew (directors)
    for crew_member in candidate.get("crew", []):
        if crew_member:
            vector[f"director_{crew_member.lower()}"] = WEIGHTS["director"]

    # Fase 1: keywords
    for kw in candidate.get("keywords_list", []):
        if kw:
            vector[f"keyword_{kw.lower()}"] = WEIGHTS["keyword"]

    return vector


def calculate_item_similarity(cand_a: Dict[str, Any], cand_b: Dict[str, Any]) -> float:
    """
    Calcula a similaridade entre duas obras candidatas para o algoritmo MMR.
    1. Se ambas possuem embedding neural denso (Fase 6), usa similaridade cosseno direta.
    2. Caso contrário, calcula similaridade cosseno sobre os vetores de features esparsas.
    3. Fallback: sobreposição Jaccard de IDs de gêneros.
    """
    emb_a = cand_a.get("embedding")
    emb_b = cand_b.get("embedding")
    if emb_a and emb_b:
        from app.recommendation.embeddings import calculate_cosine_similarity
        return calculate_cosine_similarity(emb_a, emb_b)

    vec_a = cand_a.get("_candidate_vector")
    vec_b = cand_b.get("_candidate_vector")
    if vec_a and vec_b:
        common_keys = set(vec_a.keys()) & set(vec_b.keys())
        if not common_keys:
            return 0.0
        dot = sum(vec_a[k] * vec_b[k] for k in common_keys)
        norm_a = sum(v * v for v in vec_a.values()) ** 0.5
        norm_b = sum(v * v for v in vec_b.values()) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    canonical_a = set()
    for gid in cand_a.get("genre_ids", []):
        canonical_a.update(normalize_genre_to_canonical(gid))
    for g in cand_a.get("genres", []):
        canonical_a.update(normalize_genre_to_canonical(g))

    canonical_b = set()
    for gid in cand_b.get("genre_ids", []):
        canonical_b.update(normalize_genre_to_canonical(gid))
    for g in cand_b.get("genres", []):
        canonical_b.update(normalize_genre_to_canonical(g))

    if canonical_a and canonical_b:
        intersection = len(canonical_a & canonical_b)
        union = len(canonical_a | canonical_b)
        return float(intersection / union) if union > 0 else 0.0

    return 0.0


def apply_fast_dpp_reranking(
    candidates: List[Dict[str, Any]],
    top_k: int = 15,
    lambda_param: float = 0.65
) -> List[Dict[str, Any]]:
    """
    Aplica Fast Greedy DPP (Determinantal Point Processes) com decomposição incremental de Cholesky
    (Chen et al. 2018) para selecionar o subconjunto que maximiza simultaneamente a relevância
    e a diversidade volumétrica de subgêneros e estilos (Ideia 12) em tempo de execução < 2ms.

    Kernel L = diag(q) S diag(q), onde:
      q_i = exp(alpha * rel_i) com alpha = 0.5 * lambda / (1 - lambda)
      S_ij = similaridade semântico-estrutural entre obras
      Ganho marginal incremental = d_i^2 = L_ii - ||c_i||^2
    """
    if not candidates:
        return []
    if len(candidates) <= 1 or top_k <= 1:
        return candidates[:top_k]

    n = len(candidates)
    target_k = min(top_k, n)

    eff_lambda = max(0.01, min(0.99, lambda_param))
    alpha = 0.5 * (eff_lambda / (1.0 - eff_lambda))

    q = np.zeros(n, dtype=np.float64)
    for i, c in enumerate(candidates):
        rel = float(c.get("match_score", 50.0)) / 100.0
        q[i] = math.exp(alpha * rel)

    c_matrix = np.zeros((target_k, n), dtype=np.float64)
    d_sq = q ** 2

    sim_cache: Dict[Tuple[int, int], float] = {}
    def get_cand_sim(idx_a: int, idx_b: int) -> float:
        if idx_a == idx_b:
            return 1.0
        pair = (min(idx_a, idx_b), max(idx_a, idx_b))
        if pair not in sim_cache:
            sim = calculate_item_similarity(candidates[pair[0]], candidates[pair[1]])
            sim_cache[pair] = max(0.0, min(1.0, float(sim)))
        return sim_cache[pair]

    selected_indices: List[int] = []
    selected_mask = np.zeros(n, dtype=bool)

    for step in range(target_k):
        masked_d_sq = np.where(selected_mask, -np.inf, d_sq)
        best_idx = int(np.argmax(masked_d_sq))

        if masked_d_sq[best_idx] <= 1e-12:
            rem = [idx for idx in range(n) if not selected_mask[idx]]
            rem.sort(key=lambda idx: candidates[idx].get("match_score", 0), reverse=True)
            for r_idx in rem[:(target_k - step)]:
                selected_indices.append(r_idx)
                selected_mask[r_idx] = True
            break

        selected_indices.append(best_idx)
        selected_mask[best_idx] = True

        if len(selected_indices) >= target_k:
            break

        d_best = math.sqrt(masked_d_sq[best_idx])
        c_matrix[step, best_idx] = d_best

        for i in range(n):
            if not selected_mask[i]:
                s_ij = get_cand_sim(best_idx, i)
                l_ij = q[best_idx] * s_ij * q[i]

                if step > 0:
                    inner_prod = float(np.dot(c_matrix[:step, best_idx], c_matrix[:step, i]))
                else:
                    inner_prod = 0.0

                e_i = (l_ij - inner_prod) / d_best
                c_matrix[step, i] = e_i
                d_sq[i] = max(0.0, d_sq[i] - (e_i ** 2))

    return [candidates[i] for i in selected_indices]


# Alias para retrocompatibilidade com chamadas existentes
apply_mmr_reranking = apply_fast_dpp_reranking


def reciprocal_rank_fusion(
    channel_lists: List[Tuple[str, List[Dict[str, Any]], float]],
    k_constant: int = 60,
    top_n: int = 150
) -> List[Dict[str, Any]]:
    """
    Combina múltiplos canais heterogêneos de recuperação (Canal 1 Semântico ANN,
    Canal 2 TMDB Discover e Canal 3 Colaborativo ALS - Sub-etapa 5.2 + Ideia 9)
    através da fórmula canônica de Reciprocal Rank Fusion:
      RRF(d) = sum_{c in Canais} [ weight_c / (k + rank_c(d)) ]
    """
    scores: Dict[Tuple[str, int], float] = {}
    candidates_by_key: Dict[Tuple[str, int], Dict[str, Any]] = {}
    channel_tags: Dict[Tuple[str, int], List[str]] = {}

    for channel_name, cand_list, weight in channel_lists:
        for rank, cand in enumerate(cand_list, start=1):
            cid = cand.get("id")
            mtype = cand.get("media_type") or "movie"
            if not cid:
                continue
            key = (mtype, cid)

            rrf_gain = weight / (k_constant + rank)
            scores[key] = scores.get(key, 0.0) + rrf_gain

            if key not in candidates_by_key:
                candidates_by_key[key] = cand
                channel_tags[key] = []
            else:
                existing = candidates_by_key[key]
                if cand.get("collaborative"):
                    existing["collaborative"] = True
                    existing["collaborative_score"] = cand.get("collaborative_score", 0.0)
                if cand.get("semantic"):
                    existing["semantic"] = True
                    existing["semantic_score"] = cand.get("semantic_score", 0.0)
                if cand.get("embedding") and not existing.get("embedding"):
                    existing["embedding"] = cand.get("embedding")

            channel_tags[key].append(channel_name)

    sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
    merged: List[Dict[str, Any]] = []

    for k in sorted_keys[:top_n]:
        cand = candidates_by_key[k]
        cand["_rrf_score"] = round(scores[k], 5)
        cand["_channels"] = channel_tags.get(k, [])
        merged.append(cand)

    return merged


def _damped_quality(c: Dict[str, Any]) -> float:
    va = float(c.get("vote_average") or 0.0)
    vc = float(c.get("vote_count") or 0.0)
    return (va * vc + 6.5 * 50.0) / (vc + 50.0)


def _franchise_key(c: Dict[str, Any]) -> Optional[str]:
    import re
    col = c.get("belongs_to_collection")
    if isinstance(col, dict) and col.get("id"):
        key = f"col:{col['id']}"
    elif c.get("collection_tmdb_id"):
        key = f"col:{c['collection_tmdb_id']}"
    else:
        key = None
    kws = " ".join(str(k).lower() for k in (c.get("keywords_list") or []))
    if "marvel cinematic universe" in kws or "mcu" in kws.split():
        return "universe:mcu"
    if key:
        return key
    title = str(c.get("title") or c.get("name") or "").strip()
    if not title:
        return None
    base = title.split(":")[0].split(" - ")[0].strip()
    base = re.sub(
        r"\s+(?:[2-9]|ii|iii|iv|v|vi|vii|terceiro|para sempre|o filme|parte\s+\w+)$",
        "",
        base,
        flags=re.IGNORECASE,
    ).strip().lower()
    return f"title:{base}" if base else None


def _cap_per_franchise(items: List[Dict[str, Any]], limit: int, max_per: int = 2) -> List[Dict[str, Any]]:
    """Maximo `max_per` (2) obras por franquia/universo ou mesmo diretor principal, preservando a ordem."""
    counts: Dict[str, int] = {}
    dir_counts: Dict[str, int] = {}
    out: List[Dict[str, Any]] = []
    for c in items:
        fk = _franchise_key(c)
        if fk and counts.get(fk, 0) >= max_per:
            continue
        crew_list = c.get("crew") or []
        first_dir = ""
        if crew_list:
            d0 = crew_list[0]
            first_dir = (d0.get("name", "") if isinstance(d0, dict) else str(d0)).strip().lower()
        if first_dir and dir_counts.get(first_dir, 0) >= max_per:
            continue
        if fk:
            counts[fk] = counts.get(fk, 0) + 1
        if first_dir:
            dir_counts[first_dir] = dir_counts.get(first_dir, 0) + 1
        out.append(c)
        if len(out) >= limit:
            break
    return out


def _apply_quality_floor(pool: List[Dict[str, Any]], min_needed: int, mtype: str) -> List[Dict[str, Any]]:
    """Piso: nota >= 6.5 e >=100 votos (>=50 series). Relaxa se sobrar pouco."""
    min_votes = 100 if mtype == "movie" else 50
    ok = [c for c in pool if float(c.get("vote_average") or 0) >= 6.5 and int(c.get("vote_count") or 0) >= min_votes]
    if len(ok) >= min_needed:
        return ok
    relaxed = [c for c in pool if float(c.get("vote_average") or 0) >= 6.0 and int(c.get("vote_count") or 0) >= 30]
    return relaxed if len(relaxed) >= min_needed else (relaxed or pool)


# Pisos mínimos do range de compatibilidade [MIN% - 100%]
MIN_COMPATIBILITY_SCORE = 60.0
MIN_UPCOMING_COMPATIBILITY_SCORE = 55.0


def _pick_discovery(
    rest: List[Dict[str, Any]],
    user_vector: Dict[str, float],
    n: int,
    min_score: float = MIN_COMPATIBILITY_SCORE,
) -> List[Dict[str, Any]]:
    """Descoberta adjacente dentro do range de compatibilidade [min_score, 100%]."""
    if n <= 0 or not rest:
        return []
    top_genres = {k for k, _ in sorted(((k, v) for k, v in user_vector.items() if k.startswith("genre_")), key=lambda x: -x[1])[:4]}
    top_langs = {k for k, _ in sorted(((k, v) for k, v in user_vector.items() if k.startswith("lang_")), key=lambda x: -x[1])[:2]}

    def novelty(c: Dict[str, Any]) -> float:
        genres = set()
        for g in list(c.get("genre_ids", [])) + list(c.get("genres", [])):
            for canon in normalize_genre_to_canonical(g):
                genres.add(f"genre_{canon}")
        n_new = len(genres - top_genres)
        lang_new = 1 if f"lang_{c.get('original_language', '')}" not in top_langs else 0
        return n_new + 1.5 * lang_new

    in_range = [
        c for c in rest
        if float(c.get("match_score") or 0.0) >= min_score and novelty(c) > 0
    ]
    ranked = sorted(
        in_range,
        key=lambda c: float(c.get("match_score") or 0.0) * 0.65 + _damped_quality(c) * 3.5,
        reverse=True,
    )
    return ranked[:n]



async def get_personalized_recommendations(
    db: AsyncSession,
    user_id: str,
    top_k: int = 50,
    persona_id: Optional[str] = None,
    impressions_map: Optional[Dict[str, int]] = None
) -> List[Dict[str, Any]]:
    # Get user profile
    user_result = await db.execute(select(User).where(User.id == user_id))
    user = user_result.scalars().first()

    if not user or not user.feature_vector:
        return []  # Cold start

    user_vector = user.feature_vector
    user_embedding = user.embedding
    user_taste_clusters = user.taste_clusters or []

    # V2: Se uma Persona específica foi selecionada, direciona os vetores para o polo do cluster
    if persona_id and user_taste_clusters:
        target_persona = next((p for p in user_taste_clusters if str(p.get("id")) == str(persona_id)), None)
        if target_persona:
            user_vector = target_persona.get("feature_vector") or user_vector
            user_embedding = target_persona.get("centroid_embedding") or user_embedding
        discover_candidates = await fetch_discover_candidates(user_vector, db=db, user_id=user_id, persona_id=str(persona_id))
    elif user_taste_clusters and len(user_taste_clusters) > 1:
        # Pescaria Multi-Persona para o Carrossel Geral: busca candidatos para o perfil geral + cada faceta
        tasks = [fetch_discover_candidates(user_vector, db=db, user_id=user_id)]
        tasks.extend([
            fetch_discover_candidates(p.get("feature_vector") or user_vector, db=db, user_id=user_id, persona_id=p.get("id"))
            for p in user_taste_clusters
        ])
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Intercala em round-robin separadamente por formato (movie e tv) para nenhuma faceta dominar o topo
        def _round_robin_by_type(batches: List[Any], mtype: str) -> List[Dict[str, Any]]:
            lists = [
                [c for c in res if c.get("media_type", "movie") == mtype]
                for res in batches if isinstance(res, list)
            ]
            merged: List[Dict[str, Any]] = []
            seen_ids: Set[int] = set()
            max_len = max((len(lst) for lst in lists), default=0)
            for idx in range(max_len):
                for lst in lists:
                    if idx < len(lst):
                        c = lst[idx]
                        cid = c.get("id")
                        if cid and cid not in seen_ids:
                            seen_ids.add(cid)
                            merged.append(c)
            return merged

        discover_candidates = _round_robin_by_type(results, "movie") + _round_robin_by_type(results, "tv")
    else:
        discover_candidates = await fetch_discover_candidates(user_vector, db=db, user_id=user_id)

    # 2. Filter out already watched/watchlisted items and dismissed items usando chave canônica (media_type, tmdb_id)
    from app.tracking.services import get_user_list
    from app.recommendation.service import (
        get_active_dismissed_media_keys,
        get_user_dismissed_records,
        calculate_semantic_dismiss_penalty,
    )
    user_list = await get_user_list(db, user_id)
    seen_media_keys = {(item.media.media_type, item.media.tmdb_id) for item in user_list if item.media}
    dismissed_keys = await get_active_dismissed_media_keys(db, user_id)
    dismissed_records = await get_user_dismissed_records(db, user_id)
    dismissed_snapshots = [
        {
            "embedding": getattr(r, "embedding", None),
            "expires_at": getattr(r, "expires_at", None),
            "created_at": getattr(r, "created_at", None),
        }
        for r in dismissed_records
    ]
    seen_media_keys = seen_media_keys.union(dismissed_keys)

    # Cria snapshots imutáveis em memória para isolar do ciclo de vida da sessão SQLAlchemy
    user_item_snapshots = []
    for it in user_list:
        try:
            m = getattr(it, "media", None)
            ep_list = getattr(it, "episodes", []) or []
            ep_rewatches = sum((getattr(ep, "rewatch_count", 0) or 0) for ep in ep_list)
            user_item_snapshots.append({
                "title": getattr(m, "title", None) if m else None,
                "media_type": getattr(m, "media_type", "movie") if m else "movie",
                "embedding": getattr(m, "embedding", None) if m else None,
                "media_obj": m,
                "rating": getattr(it, "rating", None),
                "status": getattr(it, "status", None),
                "is_favorite": getattr(it, "is_favorite", False),
                "rewatch_count": getattr(it, "rewatch_count", 0) or 0,
                "ep_rewatches": ep_rewatches,
            })
        except Exception:
            pass

    filtered_discover_movies = [
        c for c in discover_candidates
        if c.get("media_type", "movie") == "movie" and ("movie", c.get("id")) not in seen_media_keys
    ]
    filtered_discover_tv = [
        c for c in discover_candidates
        if c.get("media_type") == "tv" and ("tv", c.get("id")) not in seen_media_keys
    ]

    # === CANAIS INDEPENDENTES POR SESSÃO: FILMES (movie) vs SÉRIES (tv) ===
    from app.recommendation.vector_search import search_semantic_candidates
    from app.recommendation.collaborative import fetch_collaborative_candidates, get_collaborative_rrf_weight

    async def _safe_semantic(mtype: str) -> List[Dict[str, Any]]:
        try:
            if not persona_id and user_taste_clusters and len(user_taste_clusters) > 1:
                sem_tasks = [
                    search_semantic_candidates(
                        db=db,
                        query_embedding=user_embedding,
                        excluded_keys=seen_media_keys,
                        limit=45,
                        media_type_filter=mtype,
                    )
                ]
                for p in user_taste_clusters:
                    p_emb = p.get("centroid_embedding")
                    if p_emb:
                        sem_tasks.append(
                            search_semantic_candidates(
                                db=db,
                                query_embedding=p_emb,
                                excluded_keys=seen_media_keys,
                                limit=35,
                                media_type_filter=mtype,
                            )
                        )
                sem_batches = await asyncio.gather(*sem_tasks, return_exceptions=True)
                merged_sem: List[Dict[str, Any]] = []
                seen_sem_ids: Set[int] = set()
                valid_lists = [b for b in sem_batches if isinstance(b, list)]
                max_len = max((len(lst) for lst in valid_lists), default=0)
                for idx in range(max_len):
                    for lst in valid_lists:
                        if idx < len(lst):
                            cand = lst[idx]
                            cid = cand.get("id")
                            if cid and cid not in seen_sem_ids:
                                seen_sem_ids.add(cid)
                                merged_sem.append(cand)
                return merged_sem[:80]

            return await search_semantic_candidates(
                db=db,
                query_embedding=user_embedding,
                excluded_keys=seen_media_keys,
                limit=60,
                media_type_filter=mtype,
            )
        except Exception as e:
            logger.warning(f"Erro no Canal Semântico ANN ({mtype}): {e}")
            return []

    async def _safe_collab(mtype: str) -> List[Dict[str, Any]]:
        try:
            return await fetch_collaborative_candidates(
                db=db,
                target_user_id=user_id,
                seen_media_keys=seen_media_keys,
                limit=45,
                media_type_filter=mtype,
            )
        except Exception as e:
            logger.warning(f"Erro no Canal Colaborativo ALS ({mtype}): {e}")
            return []

    semantic_movies, semantic_tv, collab_movies, collab_tv = await asyncio.gather(
        _safe_semantic("movie"),
        _safe_semantic("tv"),
        _safe_collab("movie"),
        _safe_collab("tv"),
    )

    collab_weight = get_collaborative_rrf_weight()

    # Fusão RRF independente para Filmes e para Séries (zero competição entre formatos)
    movie_channels = []
    if semantic_movies:
        movie_channels.append(("semantic", semantic_movies, 1.0))
    if filtered_discover_movies:
        movie_channels.append(("discover", filtered_discover_movies, 1.0))
    if collab_movies:
        movie_channels.append(("collaborative", collab_movies, collab_weight))

    tv_channels = []
    if semantic_tv:
        tv_channels.append(("semantic", semantic_tv, 1.0))
    if filtered_discover_tv:
        tv_channels.append(("discover", filtered_discover_tv, 1.0))
    if collab_tv:
        tv_channels.append(("collaborative", collab_tv, collab_weight))

    movie_candidates = reciprocal_rank_fusion(movie_channels, k_constant=60, top_n=110) if movie_channels else []
    tv_candidates = reciprocal_rank_fusion(tv_channels, k_constant=60, top_n=110) if tv_channels else []
    all_candidates = movie_candidates + tv_candidates

    if not all_candidates:
        return []

    seen_candidate_keys = {(c.get("media_type", "movie"), c.get("id")) for c in all_candidates if "id" in c}
    combined_seen_keys = seen_media_keys | seen_candidate_keys

    # 3. Enriquece apenas candidatos rasos que não possuem metadados completos (incluindo gêneros)
    to_enrich = [
        c for c in all_candidates
        if not (c.get("cast") and c.get("keywords_list") and (c.get("genres") or c.get("genre_ids")))
        or not c.get("vote_count")
    ]
    already_enriched = [
        c for c in all_candidates
        if (c.get("cast") and c.get("keywords_list") and (c.get("genres") or c.get("genre_ids")))
        and c.get("vote_count")
    ]

    enriched_discover = await enrich_candidates_with_details(
        to_enrich,
        user_vector=user_vector,
        db=db,
        seen_keys=combined_seen_keys
    ) if to_enrich else []

    # Garante que nenhum redirecionamento de franquia traga uma obra já vista e desduplica por (media_type, id)
    total_candidates = []
    seen_after_enrich = set()
    for c in (already_enriched + list(enriched_discover)):
        ck = (c.get("media_type", "movie"), c.get("id"))
        if ck[1] and ck not in seen_media_keys and ck not in seen_after_enrich:
            seen_after_enrich.add(ck)
            total_candidates.append(c)

    # Fase 6: Busca embeddings locais pré-calculados para os candidatos usando chave composta
    candidate_tmdb_ids = [c["id"] for c in total_candidates if "id" in c]
    if candidate_tmdb_ids:
        try:
            from app.media.models import Media as MediaModel
            media_records = await db.execute(
                select(MediaModel.media_type, MediaModel.tmdb_id, MediaModel.embedding,
                       MediaModel.vote_average, MediaModel.vote_count, MediaModel.release_date).where(
                    MediaModel.tmdb_id.in_(candidate_tmdb_ids)
                )
            )
            _rows = media_records.all()
            emb_map = {(row[0], row[1]): row[2] for row in _rows if row[2]}
            _meta = {(row[0], row[1]): row for row in _rows}
            for c in total_candidates:
                _r = _meta.get((c.get("media_type", "movie"), c.get("id")))
                if _r is not None:
                    if not c.get("vote_average") and _r[3]:
                        c["vote_average"] = float(_r[3])
                    if not c.get("vote_count") and _r[4]:
                        c["vote_count"] = int(_r[4])
                    if not c.get("release_date") and not c.get("first_air_date") and _r[5]:
                        c["release_date"] = str(_r[5])
            for c in total_candidates:
                m_key = (c.get("media_type", "movie"), c.get("id"))
                if m_key in emb_map and not c.get("embedding"):
                    c["embedding"] = emb_map[m_key]
        except Exception:
            pass

    # Filtra obras ainda não lançadas (que pertencem apenas à vitrine de Lançamentos Futuros)
    from datetime import datetime, timezone
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    released_candidates = []
    for c in total_candidates:
        r_date = str(c.get("release_date") or c.get("first_air_date") or "").strip()
        vc = int(c.get("vote_count") or 0)
        if r_date and len(r_date) >= 10 and r_date[:10] > today_str:
            continue
        if vc <= 0 and (not r_date or r_date[:4] >= today_str[:4]):
            continue
        released_candidates.append(c)
    total_candidates = released_candidates

    # 4. Score & Re-rank usando a Equação Canônica Única (OmniMatch v3 - Ideia 5)
    if user_embedding is None:
        user_embedding = user.embedding

    active_personas = user_taste_clusters if (not persona_id and user_taste_clusters and len(user_taste_clusters) > 1) else None
    target_persona = next((p for p in user_taste_clusters if str(p.get("id")) == str(persona_id)), None) if persona_id else None

    from app.recommendation.ranker import LightGBMRanker
    lgbm_ranker = LightGBMRanker()

    scored_candidates = []
    for c in total_candidates:
        candidate_vector = build_candidate_vector(c)
        c["_candidate_vector"] = candidate_vector
        cand_emb = c.get("embedding")

        omni = compute_omnimatch_score(
            user_vector=user_vector,
            candidate_vector=candidate_vector,
            user_embedding=user_embedding,
            candidate_embedding=cand_emb,
            target_persona=target_persona,
            active_personas=active_personas,
            vote_average=float(c.get("vote_average") or 0.0),
            vote_count=int(c.get("vote_count") or 0),
            user_item_count=len(user_list),
            candidate=c,
            user_items=user_item_snapshots,
        )

        final_score = omni["match_score"]
        top_tags = omni["match_tags"]

        # LGBMRanker Leak-Free re-ranking e calibração para o feed geral de explore (Sub-etapa 6.2 + Ideia 10)
        if not persona_id and lgbm_ranker.is_ready():
            try:
                final_score = lgbm_ranker.predict_score(
                    user=user,
                    candidate=c,
                    cand_emb=cand_emb,
                    base_omnimatch=final_score,
                    user_items=user_item_snapshots,
                    active_personas=active_personas,
                )
            except Exception:
                pass

        if c.get("collaborative") and "✨ Descoberta Colaborativa" not in top_tags:
            top_tags.append("✨ Descoberta Colaborativa")
        elif c.get("semantic") and c.get("semantic_score", 0) >= 70.0 and "✨ Afinidade Semântica" not in top_tags:
            top_tags.append("✨ Afinidade Semântica")

        # Aplica penalidade contextual suave se o candidato for semanticamente muito próximo (>=0.75) de uma obra dispensada
        if dismissed_snapshots and cand_emb:
            penalty = calculate_semantic_dismiss_penalty(cand_emb, dismissed_snapshots)
            if penalty > 0:
                final_score = max(0.0, round(final_score - penalty, 1))

        # Anti-Fadiga de Carrossel (Impression Freshness - Sub-etapa 4.3 + Ideia 10)
        # Registra a penalidade de rotação sem distorcer a % real de compatibilidade exibida ao usuário
        if impressions_map:
            c_type = c.get("media_type", "movie")
            c_id = c.get("id")
            imp_key = f"{c_type}:{c_id}"
            imp_count = impressions_map.get(imp_key, 0) or impressions_map.get(str(c_id), 0)
            if imp_count >= 2:
                repeats = imp_count - 1
                decay_penalty = min(15.0, repeats * 3.0)
                c["freshness_penalty"] = decay_penalty

        # Exposição de provedores de streaming BR (Sub-etapa 4.3 + Ideia 11)
        br_providers = c.get("flatrate_providers_br") or c.get("watch_providers_br", [])
        c["flatrate_providers_br"] = br_providers
        c["watch_providers_br"] = br_providers

        c["match_score"] = final_score
        c["match_tags"] = top_tags
        c["_best_persona_id"] = omni.get("best_persona_id")
        c["is_preliminary"] = omni.get("is_preliminary", False)

        scored_candidates.append(c)

    # 5. Aplica Reranking Garantindo Divisão 50/50 Exata
    target_each = top_k // 2
    
    def select_diverse(pool: List[Dict[str, Any]], target: int) -> List[Dict[str, Any]]:
        if not active_personas:
            return apply_mmr_reranking(pool, lambda_param=0.72, top_k=target)

        # Prioridade #1 (First-Pick): garante os maiores % de compatibilidade absolutos no topo do Explore
        sorted_by_score = sorted(pool, key=lambda x: x.get("match_score", 0), reverse=True)
        top_champions = _cap_per_franchise(sorted_by_score, min(6, max(2, target // 3)), max_per=2)
        selected = list(top_champions)
        selected_ids = {c["id"] for c in selected}

        num_personas = len(active_personas)
        quota = max(3, (target - len(selected)) // num_personas + 1)
        by_persona = {p.get("id", "general"): [] for p in active_personas}
        by_persona["general"] = []
        for c in pool:
            if c["id"] not in selected_ids:
                by_persona[c.get("_best_persona_id", "general")].append(c)

        for p in active_personas:
            p_pool = by_persona.get(p.get("id"), [])
            if p_pool:
                for item in apply_mmr_reranking(p_pool, lambda_param=0.72, top_k=quota):
                    if item["id"] not in selected_ids:
                        selected.append(item)
                        selected_ids.add(item["id"])

        if len(selected) < target:
            rem = [c for c in pool if c["id"] not in selected_ids]
            rem.sort(key=lambda x: x.get("match_score", 0), reverse=True)
            for item in rem[:(target - len(selected))]:
                selected.append(item)
                selected_ids.add(item["id"])
        return selected

    raw_movies = [c for c in scored_candidates if c.get("media_type") == "movie"]
    raw_tv = [c for c in scored_candidates if c.get("media_type") == "tv"]

    def select_mixed(
        pool: List[Dict[str, Any]],
        target: int,
        min_score: float = MIN_COMPATIBILITY_SCORE,
    ) -> List[Dict[str, Any]]:
        """Mix ~70% afinidade / ~20% descoberta adjacente / ~10% surpresa, estritamente dentro de [min_score, 100%]."""
        if not pool:
            return []
        in_range_pool = [c for c in pool if float(c.get("match_score") or 0.0) >= min_score]
        working_pool = in_range_pool if len(in_range_pool) >= target else pool

        n_aff = max(1, int(round(target * 0.7)))
        n_disc = int(round(target * 0.2))
        picked = _cap_per_franchise(select_diverse(working_pool, n_aff * 2), n_aff)
        chosen_ids = {c["id"] for c in picked}
        rest = [c for c in working_pool if c["id"] not in chosen_ids]
        disc = _pick_discovery(rest, user_vector, n_disc, min_score=min_score)
        picked.extend(disc)
        chosen_ids |= {c["id"] for c in disc}
        # surpresa: alta qualidade combinada com compatibilidade dentro do range [min_score, 100%]
        rest_in_range = [
            c for c in working_pool
            if c["id"] not in chosen_ids and float(c.get("match_score") or 0.0) >= min_score
        ]
        surprise = sorted(
            rest_in_range,
            key=lambda c: float(c.get("match_score") or 0.0) * 0.55 + _damped_quality(c) * 4.5,
            reverse=True,
        )[: max(0, target - len(picked))]
        picked.extend(surprise)
        return _cap_per_franchise(picked, target)

    def _finalize_media_section(
        raw_pool: List[Dict[str, Any]],
        target: int,
        mtype: str,
        min_score: float = MIN_COMPATIBILITY_SCORE,
    ) -> List[Dict[str, Any]]:
        """
        Seleciona, limita franquias e garante que toda sugestão fique no range [min_score, 100%].
        Se alguma sugestão cair abaixo de min_score ou faltar vaga, substitui por outro candidato do pool dentro do range.
        """
        if target <= 0 or not raw_pool:
            return []
        in_range_raw = [c for c in raw_pool if float(c.get("match_score") or 0.0) >= min_score]
        base_pool = in_range_raw if len(in_range_raw) >= target else raw_pool

        filtered_pool = _apply_quality_floor(base_pool, target, mtype)
        if persona_id:
            picked = _cap_per_franchise(select_diverse(filtered_pool, target * 2), target)
        else:
            picked = select_mixed(filtered_pool, target, min_score=min_score)

        # Remove qualquer candidato abaixo de min_score se houver substitutos dentro do range [min_score, 100%]
        picked_in_range = [c for c in picked if float(c.get("match_score") or 0.0) >= min_score]
        if len(in_range_raw) >= target:
            picked = picked_in_range

        picked.sort(key=lambda x: x.get("match_score", 0), reverse=True)
        picked = _cap_per_franchise(picked, target)

        # Backfill 1: substitui/completa prioritariamente com candidatos dentro do range [min_score, 100%]
        if len(picked) < target:
            have_ids = {c.get("id") for c in picked}
            left_in_range = [
                c for c in raw_pool
                if c.get("id") not in have_ids
                and float(c.get("match_score") or 0.0) >= min_score
                and float(c.get("vote_average") or 0) >= 6.0
                and int(c.get("vote_count") or 0) >= 30
            ]
            left_in_range.sort(key=lambda x: x.get("match_score", 0), reverse=True)
            picked = _cap_per_franchise(picked + left_in_range, target)

        # Backfill 2: qualquer candidato restante dentro do range [min_score, 100%]
        if len(picked) < target:
            have_ids = {c.get("id") for c in picked}
            left_range_any = [
                c for c in raw_pool
                if c.get("id") not in have_ids
                and float(c.get("match_score") or 0.0) >= min_score
            ]
            left_range_any.sort(key=lambda x: x.get("match_score", 0), reverse=True)
            picked = _cap_per_franchise(picked + left_range_any, target, max_per=3)

        # Fallback final graduado apenas se todo o pool do TMDB tiver menos de `target` obras >= min_score
        if len(picked) < target:
            have_ids = {c.get("id") for c in picked}
            left_any = [c for c in raw_pool if c.get("id") not in have_ids]
            left_any.sort(key=lambda x: x.get("match_score", 0), reverse=True)
            picked.extend(left_any[: target - len(picked)])

        picked.sort(key=lambda x: x.get("match_score", 0), reverse=True)
        return picked[:target]

    target_movies = top_k // 2
    target_tv = top_k - target_movies

    selected_movies = _finalize_media_section(raw_movies, target_movies, "movie")
    selected_tv = _finalize_media_section(raw_tv, target_tv, "tv")

    diverse_recommendations = selected_movies + selected_tv
    diverse_recommendations.sort(key=lambda x: x.get("match_score", 0), reverse=True)

    # Limpa dados internos temporários antes de retornar (O(1) por chave, eliminando embeddings vazados)
    for item in diverse_recommendations:
        item.pop("embedding", None)
        item.pop("_candidate_vector", None)
        item.pop("candidate_vector", None)
        item.pop("_best_persona_id", None)
        item.pop("_ml_score", None)
        item.pop("_rrf_score", None)
        item.pop("_channels", None)
        item.pop("cast", None)
        item.pop("crew", None)
        item.pop("softcore", None)
        item.pop("adult", None)

    return diverse_recommendations


async def get_persona_recommendations(
    db: AsyncSession,
    user_id: str,
    top_k_per_persona: int = 25,
    exclude_keys: Optional[Set[Tuple[str, int]]] = None
) -> List[Dict[str, Any]]:
    """
    V2 — Perfis Multi-Cluster:
    Retorna as múltiplas Personas de interesse do usuário, cada uma com seus
    metadados (nome, emoji, top gêneros) e seu carrossel de recomendações exclusivo.
    """
    user_result = await db.execute(select(User).where(User.id == user_id))
    user = user_result.scalars().first()
    if not user or not user.feature_vector:
        return []

    clusters = user.taste_clusters
    if not clusters:
        from app.recommendation.cluster_manager import compute_user_taste_clusters
        clusters = await compute_user_taste_clusters(db, user_id)

    if not clusters:
        return []

    persona_carousels = []
    excluded: Set[Tuple[str, int]] = set(exclude_keys or set())

    def _norm_title(c: Dict[str, Any]) -> str:
        return str(c.get("title") or c.get("name") or "").strip().lower()

    def _genre_set(genres_like) -> Set[str]:
        out: Set[str] = set()
        for g in genres_like or []:
            for canon in normalize_genre_to_canonical(g):
                out.add(canon)
        return out

    # 1) Busca candidatos de cada faceta (pool amplo para garantir o mínimo por aba)
    pool_size = max(70, top_k_per_persona * 3)
    raw_by_persona: Dict[Any, List[Dict[str, Any]]] = {}
    for p in clusters:
        raw_by_persona[p.get("id")] = await get_personalized_recommendations(
            db=db, user_id=user_id, top_k=pool_size, persona_id=p.get("id")
        )

    # 2) Atribuicao exclusiva com garantia de piso mínimo (top_k_per_persona):
    # Primeiro cada obra vai para a faceta com melhor coerencia tematica;
    # obras excedentes (> top_k_per_persona) não bloqueiam facetas menores, e o Passo 2
    # faz backfill a partir do pool coerente de cada faceta para garantir 25 sugestões por aba.
    persona_genres = {p.get("id"): _genre_set(p.get("top_genres", [])) for p in clusters}

    def _adjusted(c: Dict[str, Any], pid, min_score: float = MIN_COMPATIBILITY_SCORE) -> float:
        m_score = float(c.get("match_score", 0.0) or 0.0)
        if m_score < min_score:
            return -1.0  # fora do range [min_score, 100%]
        pg = persona_genres.get(pid) or set()
        cg = _genre_set(list(c.get("genre_ids", [])) + list(c.get("genres", [])))
        overlap = len(pg & cg) / max(1, len(pg))
        if pg and not (pg & cg):
            return -1.0  # incoerente com o tema da faceta
        return m_score * (0.6 + 0.4 * overlap) + 10.0 * overlap

    best: Dict[Tuple[str, str], Tuple[float, Any, Dict[str, Any]]] = {}
    all_unique_candidates: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for pid, recs_p in raw_by_persona.items():
        for c in recs_p:
            key = (c.get("media_type", "movie"), c.get("id"))
            if key in excluded:
                continue
            tkey = (c.get("media_type", "movie"), _norm_title(c))
            if tkey not in all_unique_candidates:
                all_unique_candidates[tkey] = dict(c)
            sc = _adjusted(c, pid)
            if sc < 0:
                continue
            if tkey not in best or sc > best[tkey][0]:
                best[tkey] = (sc, pid, dict(c))

    assigned: Dict[Any, List[Tuple[float, Dict[str, Any]]]] = {p.get("id"): [] for p in clusters}
    for sc, pid, c in best.values():
        assigned[pid].append((sc, c))

    final_by_persona: Dict[Any, List[Dict[str, Any]]] = {}
    used_keys: Set[Tuple[str, Any]] = set(excluded)
    used_titles: Set[Tuple[str, str]] = set()

    # Passo 1: Seleção inicial exclusiva (facetas com pool mais restrito escolhem primeiro para evitar canibalização)
    clusters_by_pool = sorted(clusters, key=lambda p: len(assigned.get(p.get("id"), [])))
    for p in clusters_by_pool:
        pid = p.get("id")
        ordered = [c for _, c in sorted(assigned[pid], key=lambda x: x[0], reverse=True)]
        capped = _cap_per_franchise(ordered, top_k_per_persona, max_per=2)
        final_by_persona[pid] = capped
        for item in capped:
            used_keys.add((item.get("media_type", "movie"), item.get("id")))
            used_titles.add((item.get("media_type", "movie"), _norm_title(item)))

    # Passo 2: Backfill garantido para facetas que ficaram abaixo de top_k_per_persona
    for p in clusters_by_pool:
        pid = p.get("id")
        current_list = final_by_persona[pid]
        if len(current_list) < top_k_per_persona:
            backfill_scored: List[Tuple[float, Dict[str, Any]]] = []
            seen_local_tkeys: Set[Tuple[str, str]] = set()

            # Prioriza o pool nativo da faceta + obras excedentes coerentes de outras facetas
            candidate_sources = list(raw_by_persona.get(pid, [])) + list(all_unique_candidates.values())
            for cand in candidate_sources:
                ckey = (cand.get("media_type", "movie"), cand.get("id"))
                tkey = (cand.get("media_type", "movie"), _norm_title(cand))
                if ckey in used_keys or tkey in used_titles or tkey in seen_local_tkeys:
                    continue
                sc = _adjusted(cand, pid)
                if sc < 0:
                    continue
                seen_local_tkeys.add(tkey)
                backfill_scored.append((sc, dict(cand)))

            backfill_scored.sort(key=lambda x: x[0], reverse=True)
            backfill_ordered = [c for _, c in backfill_scored]

            combined = _cap_per_franchise(current_list + backfill_ordered, top_k_per_persona, max_per=2)
            if len(combined) < top_k_per_persona:
                combined = _cap_per_franchise(current_list + backfill_ordered, top_k_per_persona, max_per=3)

            for item in combined:
                used_keys.add((item.get("media_type", "movie"), item.get("id")))
                used_titles.add((item.get("media_type", "movie"), _norm_title(item)))
            final_by_persona[pid] = combined

        final_by_persona[pid].sort(key=lambda x: float(x.get("match_score", 0.0)), reverse=True)

    for p in clusters:
        p_id = p.get("id")
        recs = final_by_persona.get(p_id, [])
        persona_carousels.append({
            "id": p_id,
            "name": p.get("name"),
            "emoji": p.get("emoji"),
            "tagline": p.get("tagline", ""),
            "item_count": p.get("item_count", 0),
            "top_genres": p.get("top_genres", []),
            "sample_titles": p.get("sample_titles", []),
            "recommendations": recs
        })

    return persona_carousels


async def fetch_upcoming_candidates(user_vector: Dict[str, float], db=None, user_id: str = None) -> List[Dict[str, Any]]:
    if not settings.TMDB_API_KEY:
        return []

    from datetime import datetime, timezone, timedelta
    today_dt = datetime.now(timezone.utc)
    today = today_dt.strftime("%Y-%m-%d")
    max_date = (today_dt + timedelta(days=365)).strftime("%Y-%m-%d")

    genre_features = [(k.replace("genre_", ""), v) for k, v in user_vector.items() if k.startswith("genre_")]
    genre_features.sort(key=lambda x: x[1], reverse=True)

    live_genre_name_to_id: Dict[str, int] = {}
    favorite_crew_ids: List[str] = []
    favorite_cast_ids: List[str] = []

    if db and user_id:
        try:
            from app.tracking.services import get_user_list
            user_items = await get_user_list(db, user_id)
            crew_counter: Dict[str, int] = {}
            cast_counter: Dict[str, int] = {}

            for list_item in user_items:
                media = list_item.media
                if not media:
                    continue

                if media.genres:
                    for g in media.genres:
                        if isinstance(g, dict):
                            _name = str(g.get("name", "")).lower()
                            _gid = g.get("id")
                            if _name and _gid:
                                live_genre_name_to_id[_name] = _gid

                weight = 2 if list_item.is_favorite or (list_item.rating and list_item.rating >= 4.0) else 1
                def _pid(entry):
                    if isinstance(entry, dict):
                        v = entry.get("person_id") or entry.get("id")
                        return str(v) if v else None
                    return str(entry) if entry else None

                if media.crew_ids and isinstance(media.crew_ids, list):
                    for entry in media.crew_ids[:5]:
                        cid = _pid(entry)
                        if cid:
                            crew_counter[cid] = crew_counter.get(cid, 0) + weight
                if media.cast_ids and isinstance(media.cast_ids, list):
                    for entry in media.cast_ids[:5]:
                        cid = _pid(entry)
                        if cid:
                            cast_counter[cid] = cast_counter.get(cid, 0) + weight

            favorite_crew_ids = [k for k, _ in sorted(crew_counter.items(), key=lambda x: x[1], reverse=True)[:5]]
            favorite_cast_ids = [k for k, _ in sorted(cast_counter.items(), key=lambda x: x[1], reverse=True)[:5]]
        except Exception:
            pass

    primary_genres = [g for g, v in genre_features[:5] if v > 0]
    secondary_genres = [g for g, v in genre_features[5:10] if v > 0]

    prim_movie_ids = [str(live_genre_name_to_id.get(g) or GENRE_MAP_REVERSE.get(g)) for g in primary_genres if (live_genre_name_to_id.get(g) or GENRE_MAP_REVERSE.get(g))]
    with_primary_movie_genres = "|".join(prim_movie_ids) if prim_movie_ids else None
    sec_movie_ids = [str(live_genre_name_to_id.get(g) or GENRE_MAP_REVERSE.get(g)) for g in secondary_genres if (live_genre_name_to_id.get(g) or GENRE_MAP_REVERSE.get(g))]
    with_sec_movie_genres = "|".join(sec_movie_ids) if sec_movie_ids else None

    prim_tv_ids = [TV_GENRE_MAP.get(g, str(GENRE_MAP_REVERSE.get(g, ""))) for g in primary_genres]
    prim_tv_ids = [x for x in prim_tv_ids if x]
    with_primary_tv_genres = "|".join(prim_tv_ids) if prim_tv_ids else None
    sec_tv_ids = [TV_GENRE_MAP.get(g, str(GENRE_MAP_REVERSE.get(g, ""))) for g in secondary_genres]
    sec_tv_ids = [x for x in sec_tv_ids if x]
    with_sec_tv_genres = "|".join(sec_tv_ids) if sec_tv_ids else None

    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {settings.TMDB_API_KEY}"
    }

    async def fetch_endpoint(_client: Any, endpoint: str, media_type: Optional[str], params: Dict[str, Any]) -> List[Dict[str, Any]]:
        from app.core.tmdb_client import tmdb_get_json
        default_params = {"language": "pt-BR", "page": 1}
        default_params.update(params)
        data = await tmdb_get_json(endpoint, params=default_params, timeout=12.0)
        if not data:
            return []
        results = data.get("results", [])
        for r in results:
            if media_type:
                r["media_type"] = media_type
            elif "media_type" not in r:
                r["media_type"] = "movie" if "title" in r else "tv"
        return results

    tasks = []
    async with httpx.AsyncClient() as client:
        # === 1. OBRAS ORIGINAIS INÉDITAS (Anti-Franquias / Sem Sequências) ===
        # Exclui tags de sequel (9663), franchise (209714), prequel (179431), spin-off (236407), cinematic universe (180547)
        if with_primary_movie_genres:
            for p in [1, 2]:
                tasks.append(fetch_endpoint(client, "/discover/movie", "movie", {
                    "primary_release_date.gte": today,
                    "primary_release_date.lte": max_date,
                    "without_keywords": "9663,209714,179431,236407,180547",
                    "with_genres": with_primary_movie_genres,
                    "sort_by": "popularity.desc",
                    "page": p
                }))

        # === 2. PRODUTORAS DE CINEMA AUTORAL / HISTÓRIAS ORIGINAIS DE PRESTÍGIO ===
        # A24 (41077), Neon (90886), Searchlight (43), Focus Features (10163)
        tasks.append(fetch_endpoint(client, "/discover/movie", "movie", {
            "primary_release_date.gte": today,
            "primary_release_date.lte": max_date,
            "with_companies": "41077|90886|43|10163",
            "sort_by": "popularity.desc",
            "page": 1
        }))

        # === 3. RADAR DE TALENTOS FAVORITOS (Diretores e Atores que o usuário mais consome) ===
        if favorite_crew_ids:
            tasks.append(fetch_endpoint(client, "/discover/movie", "movie", {
                "primary_release_date.gte": today,
                "primary_release_date.lte": max_date,
                "with_crew": "|".join(favorite_crew_ids),
                "sort_by": "popularity.desc",
                "page": 1
            }))
        if favorite_cast_ids:
            tasks.append(fetch_endpoint(client, "/discover/movie", "movie", {
                "primary_release_date.gte": today,
                "primary_release_date.lte": max_date,
                "with_cast": "|".join(favorite_cast_ids),
                "sort_by": "popularity.desc",
                "page": 1
            }))

        # === 4. ESTREIAS NO BRASIL (Cinema comercial/limitado e Streaming Digital) ===
        tasks.append(fetch_endpoint(client, "/discover/movie", "movie", {
            "region": "BR",
            "release_date.gte": today,
            "release_date.lte": max_date,
            "with_release_type": "2|3|4",
            "sort_by": "popularity.desc",
            "page": 1
        }))
        tasks.append(fetch_endpoint(client, "/movie/upcoming", "movie", {
            "region": "BR",
            "page": 1
        }))

        # === 5. DESCOBERTA GLOBAL POR GÊNEROS E POPULARIDADE ===
        if with_sec_movie_genres:
            tasks.append(fetch_endpoint(client, "/discover/movie", "movie", {
                "primary_release_date.gte": today,
                "primary_release_date.lte": max_date,
                "with_genres": with_sec_movie_genres,
                "sort_by": "popularity.desc",
                "page": 1
            }))
        for p in [1, 2]:
            tasks.append(fetch_endpoint(client, "/discover/movie", "movie", {
                "primary_release_date.gte": today,
                "primary_release_date.lte": max_date,
                "sort_by": "popularity.desc",
                "page": p
            }))

        # === 6. SÉRIES INÉDITAS DE FICÇÃO E MINISSÉRIES (com with_type=2|4) ===
        if with_primary_tv_genres:
            for p in [1, 2, 3]:
                tasks.append(fetch_endpoint(client, "/discover/tv", "tv", {
                    "first_air_date.gte": today,
                    "first_air_date.lte": max_date,
                    "include_null_first_air_dates": "false",
                    "with_type": "2|4",
                    "with_genres": with_primary_tv_genres,
                    "sort_by": "popularity.desc",
                    "page": p
                }))
        for single_tv_gid in prim_tv_ids[:3]:
            tasks.append(fetch_endpoint(client, "/discover/tv", "tv", {
                "first_air_date.gte": today,
                "first_air_date.lte": max_date,
                "include_null_first_air_dates": "false",
                "with_type": "2|4",
                "with_genres": single_tv_gid,
                "sort_by": "popularity.desc",
                "page": 1
            }))
        if with_sec_tv_genres:
            tasks.append(fetch_endpoint(client, "/discover/tv", "tv", {
                "first_air_date.gte": today,
                "first_air_date.lte": max_date,
                "include_null_first_air_dates": "false",
                "with_type": "2|4",
                "with_genres": with_sec_tv_genres,
                "sort_by": "popularity.desc",
                "page": 1
            }))
        for p in [1, 2, 3]:
            tasks.append(fetch_endpoint(client, "/discover/tv", "tv", {
                "first_air_date.gte": today,
                "first_air_date.lte": max_date,
                "include_null_first_air_dates": "false",
                "with_type": "2|4",
                "sort_by": "popularity.desc",
                "page": p
            }))

        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

    candidates = []
    seen_cand_keys = set()
    for batch in batch_results:
        if isinstance(batch, list):
            for c in batch:
                cid = c.get("id")
                mtype = c.get("media_type") or "movie"
                ckey = (mtype, cid)
                if not cid or ckey in seen_cand_keys:
                    continue

                r_date = c.get("release_date") if mtype == "movie" else c.get("first_air_date")
                if r_date and r_date >= today:
                    seen_cand_keys.add(ckey)
                    candidates.append(c)

    return candidates


async def get_upcoming_recommendations(
    db: AsyncSession,
    user_id: str,
    top_k_per_type: int = 25
) -> Dict[str, List[Dict[str, Any]]]:
    user_result = await db.execute(select(User).where(User.id == user_id))
    user = user_result.scalars().first()

    user_vector = user.feature_vector if user else {}
    user_embedding = user.embedding if user else None
    user_taste_clusters = user.taste_clusters if (user and user.taste_clusters) else []

    # Candidatos futuros
    upcoming_candidates = await fetch_upcoming_candidates(user_vector, db=db, user_id=user_id)

    # Filtrar itens que o usuário já adicionou na lista ou dispensou usando chave canônica (media_type, tmdb_id)
    from app.tracking.services import get_user_list
    from app.recommendation.service import get_active_dismissed_media_keys
    user_list = await get_user_list(db, user_id)
    seen_media_keys = {(item.media.media_type, item.media.tmdb_id) for item in user_list if item.media}
    dismissed_keys = await get_active_dismissed_media_keys(db, user_id)
    seen_media_keys = seen_media_keys.union(dismissed_keys)

    user_item_snapshots = []
    for it in user_list:
        try:
            m = getattr(it, "media", None)
            user_item_snapshots.append({
                "title": getattr(m, "title", None) if m else None,
                "media_type": getattr(m, "media_type", "movie") if m else "movie",
                "embedding": getattr(m, "embedding", None) if m else None,
                "media_obj": m,
                "rating": getattr(it, "rating", None),
                "status": getattr(it, "status", None),
                "is_favorite": getattr(it, "is_favorite", False),
                "rewatch_count": getattr(it, "rewatch_count", 0) or 0,
                "ep_rewatches": 0,
            })
        except Exception:
            pass

    filtered_candidates = [
        c for c in upcoming_candidates
        if (c.get("media_type", "movie"), c.get("id")) not in seen_media_keys
    ]

    if not filtered_candidates:
        return {"movies": [], "series": []}

    # Enriquecimento com detalhes (sem redirecionar para Filme 1 do passado em Próximos Lançamentos)
    enriched = await enrich_candidates_with_details(
        filtered_candidates,
        user_vector=user_vector,
        db=db,
        seen_keys=seen_media_keys,
        allow_collection_redirect=False
    )

    from datetime import datetime, timezone
    today_dt = datetime.now(timezone.utc)
    today_str = today_dt.strftime("%Y-%m-%d")

    def _is_future_release(date_val: Any) -> bool:
        if not date_val:
            return False
        d = str(date_val).strip()
        if len(d) == 4 and d.isdigit():
            return int(d) >= int(today_str[:4])
        if len(d) >= 10:
            return d[:10] >= today_str[:10]
        return d >= today_str

    # Filtro estrito: garante que somente obras com data de estreia futura permaneçam
    strictly_upcoming = [
        c for c in enriched
        if _is_future_release(c.get("release_date") if c.get("media_type") == "movie" else c.get("first_air_date"))
    ]

    active_personas = user_taste_clusters if len(user_taste_clusters) > 1 else None

    scored_movies = []
    scored_tv = []

    import re
    sequel_title_re = re.compile(r"(\b[2-9]\b|\b(?:ii|iii|iv|v|vi)\b|parte\s+(?:dois|três|2|3)|season\s+[2-9]|temporada\s+[2-9])", re.IGNORECASE)
    sequel_kw_set = {"sequel", "prequel", "spin off", "remake", "reboot"}

    for c in strictly_upcoming:
        candidate_vector = build_candidate_vector(c)
        c["_candidate_vector"] = candidate_vector
        cand_emb = c.get("embedding")

        omni = compute_omnimatch_score(
            user_vector=user_vector or {},
            candidate_vector=candidate_vector,
            user_embedding=user_embedding,
            candidate_embedding=cand_emb,
            active_personas=active_personas,
            vote_average=float(c.get("vote_average") or 0.0),
            vote_count=int(c.get("vote_count") or 0),
            user_item_count=len(user_list),
            candidate=c,
            user_items=user_item_snapshots,
        )

        top_tags = list(omni["match_tags"])
        kws_lower = {str(k.get("name") if isinstance(k, dict) else k).lower().strip() for k in (c.get("keywords_list") or [])}
        title_str = str(c.get("title") or c.get("name") or "")
        is_sequel_or_franchise = bool(
            c.get("belongs_to_collection")
            or (kws_lower & sequel_kw_set)
            or sequel_title_re.search(title_str)
        )
        if is_sequel_or_franchise:
            if "🎬 Franquia / Continuação" not in top_tags and len(top_tags) < 3:
                top_tags.append("🎬 Franquia / Continuação")
        elif "✨ História Original" not in top_tags:
            top_tags.append("✨ História Original")

        c["match_score"] = omni["match_score"]
        c["match_tags"] = top_tags[:3]
        c["is_preliminary"] = omni.get("is_preliminary", False)

        if c.get("media_type") == "movie":
            scored_movies.append(c)
        elif c.get("media_type") == "tv":
            scored_tv.append(c)

    # Ordenação decrescente pelo Match Score e priorização do range [MIN_UPCOMING_COMPATIBILITY_SCORE, 100%]
    scored_movies.sort(key=lambda x: x.get("match_score", 0), reverse=True)
    scored_tv.sort(key=lambda x: x.get("match_score", 0), reverse=True)

    def _select_upcoming_in_range(pool: List[Dict[str, Any]], target: int, min_score: float = MIN_UPCOMING_COMPATIBILITY_SCORE) -> List[Dict[str, Any]]:
        in_range = [c for c in pool if float(c.get("match_score") or 0.0) >= min_score]
        selected = _cap_per_franchise(in_range, target, max_per=2)
        if len(selected) < target and len(in_range) > len(selected):
            selected = _cap_per_franchise(in_range, target, max_per=3)
        if len(selected) < target:
            have_ids = {c.get("id") for c in selected}
            rem = [c for c in pool if c.get("id") not in have_ids]
            rem.sort(key=lambda x: x.get("match_score", 0), reverse=True)
            selected.extend(rem[: target - len(selected)])
        selected.sort(key=lambda x: x.get("match_score", 0), reverse=True)
        return selected[:target]

    final_up_movies = _select_upcoming_in_range(scored_movies, top_k_per_type)
    final_up_tv = _select_upcoming_in_range(scored_tv, top_k_per_type)

    # Limpeza de campos internos e embeddings
    for item in final_up_movies + final_up_tv:
        item.pop("embedding", None)
        item.pop("_candidate_vector", None)
        item.pop("candidate_vector", None)
        item.pop("_best_persona_id", None)
        item.pop("_ml_score", None)
        item.pop("_rrf_score", None)
        item.pop("cast", None)
        item.pop("crew", None)
        item.pop("softcore", None)
        item.pop("adult", None)

    return {
        "movies": final_up_movies,
        "series": final_up_tv,
        "tv": final_up_tv,
    }


async def get_recent_vibe_recommendations(
    db: AsyncSession,
    user_id: str,
    top_k: int = 25,
    exclude_keys: Optional[Set[Tuple[str, int]]] = None,
) -> List[Dict[str, Any]]:
    """
    Gera o carrossel 'Na Sua Vibe Recente' com janela móvel de 21 dias (Sub-etapa 4.1 + Ideia 5).
    Captura o humor pontual recente do usuário (ex: maratona de terror, ficção anos 80, animações)
    usando Rocchio de curto prazo e Fast Greedy DPP para diversificação volumétrica.
    """
    import uuid
    from app.tracking.models import UserListItem
    from app.recommendation.profile_manager import compute_recent_vibe_profile
    from sqlalchemy.orm import selectinload

    target_uuid = user_id if isinstance(user_id, uuid.UUID) else uuid.UUID(str(user_id))

    # 1. Carrega itens da lista do usuário
    stmt = (
        select(UserListItem)
        .options(selectinload(UserListItem.media), selectinload(UserListItem.episodes))
        .where(UserListItem.user_id == target_uuid)
    )
    res = await db.execute(stmt)
    list_items = res.scalars().all()

    if not list_items:
        return []

    # 2. Constrói perfil de memória curta Rocchio restrito a 'completed'/'watching' <= 30 dias
    vibe_vector, vibe_embedding, seed_keys = compute_recent_vibe_profile(
        list_items, half_life_days=14.0, max_age_days=30.0
    )

    if not vibe_vector or not seed_keys:
        return []

    # 3. Conjunto de obras já assistidas / dispensadas para filtragem
    seen_keys: Set[Tuple[str, int]] = set()
    for it in list_items:
        if it.media and getattr(it.media, "tmdb_id", None):
            m_type = getattr(it.media, "media_type", "movie") or "movie"
            seen_keys.add((m_type, it.media.tmdb_id))

    from app.recommendation.service import get_active_dismissed_media_keys
    from app.recommendation.models import UserRecommendation
    dismissed_keys = await get_active_dismissed_media_keys(db, str(user_id))
    excluded_keys = seen_keys.union(dismissed_keys)
    if exclude_keys:
        excluded_keys = excluded_keys.union(exclude_keys)

    # Evita duplicar itens que já estão na vitrine principal (explore_items)
    try:
        rec_row = (await db.execute(select(UserRecommendation.explore_items).where(UserRecommendation.user_id == target_uuid))).scalar_one_or_none()
        if rec_row and isinstance(rec_row, list):
            for exp_it in rec_row:
                if isinstance(exp_it, dict) and exp_it.get("id"):
                    excluded_keys.add((exp_it.get("media_type", "movie"), exp_it.get("id")))
    except Exception:
        pass

    seed_key_set = set(seed_keys)
    recent_item_snapshots = []
    for it in list_items:
        try:
            m = getattr(it, "media", None)
            if not m or (getattr(m, "media_type", "movie") or "movie", getattr(m, "tmdb_id", None)) not in seed_key_set:
                continue
            ep_list = getattr(it, "episodes", []) or []
            ep_rewatches = sum((getattr(ep, "rewatch_count", 0) or 0) for ep in ep_list)
            recent_item_snapshots.append({
                "title": getattr(m, "title", None),
                "media_type": getattr(m, "media_type", "movie") or "movie",
                "embedding": getattr(m, "embedding", None),
                "media_obj": m,
                "rating": getattr(it, "rating", None),
                "status": getattr(it, "status", None),
                "is_favorite": getattr(it, "is_favorite", False),
                "rewatch_count": getattr(it, "rewatch_count", 0) or 0,
                "ep_rewatches": ep_rewatches,
            })
        except Exception:
            pass

    # 4. Recuperação de Candidatos para a Vibe (âncoras restritas APENAS aos top 15 seeds <= 30 dias)
    # Funil A: Discover baseado exclusivamente no vetor e nas obras elegíveis da vibe (<= 30 dias)
    discover_candidates = await fetch_discover_candidates(
        vibe_vector,
        db=db,
        user_id=str(user_id),
        allowed_media_keys=seed_key_set,
    )

    # Funil B: Recomendações TMDB a partir dos top seeds recentes (com Bearer Auth + Retry contra 429/timeout)
    recommended_candidates: List[Dict[str, Any]] = []
    top_seeds = seed_keys[:6]
    if top_seeds:
        from app.core.tmdb_client import tmdb_get_json

        async def _fetch_seed_recs(m_type: str, s_id: int) -> List[Dict[str, Any]]:
            endpoint = "movie" if m_type == "movie" else "tv"
            data = await tmdb_get_json(
                f"/{endpoint}/{s_id}/recommendations",
                params={"language": "pt-BR", "page": 1},
                timeout=12.0,
            )
            if not data:
                return []
            out: List[Dict[str, Any]] = []
            for item in data.get("results", [])[:15]:
                item["media_type"] = m_type
                out.append(item)
            return out

        seed_batches = await asyncio.gather(*[_fetch_seed_recs(mt, sid) for mt, sid in top_seeds], return_exceptions=True)
        for batch in seed_batches:
            if isinstance(batch, list):
                recommended_candidates.extend(batch)

    # Junta e desduplica (prioriza seeds diretos da vibe recente + discover da vibe)
    combined_raw: List[Dict[str, Any]] = []
    fallback_raw: List[Dict[str, Any]] = []
    seen_in_vibe: Set[Tuple[str, int]] = set()

    for c in (recommended_candidates + discover_candidates):
        cid = c.get("id")
        ctype = c.get("media_type") or "movie"
        ckey = (ctype, cid)
        if not cid or ckey in excluded_keys or ckey in seen_in_vibe:
            continue

        v_count = c.get("vote_count", 0) or 0
        v_avg = c.get("vote_average", 0.0) or 0.0
        if v_count >= 60 and v_avg >= 5.8:
            seen_in_vibe.add(ckey)
            combined_raw.append(c)
        elif v_count >= 20 and v_avg >= 5.5:
            seen_in_vibe.add(ckey)
            fallback_raw.append(c)

    if len(combined_raw) < top_k * 2 and fallback_raw:
        combined_raw.extend(fallback_raw[: max(0, top_k * 2 - len(combined_raw))])

    if not combined_raw:
        return []

    # 5. Enriquecimento dos candidatos (1 RTT) + batch embeddings
    enriched = await enrich_candidates_with_details(
        combined_raw,
        user_vector=vibe_vector,
        db=db,
        seen_keys=excluded_keys
    )

    # 6. Pontuação de Afinidade com a Vibe
    scored_vibe: List[Dict[str, Any]] = []
    seen_after_enrich_vibe: Set[Tuple[str, int]] = set()
    for c in enriched:
        ckey = (c.get("media_type", "movie"), c.get("id"))
        if not ckey[1] or ckey in excluded_keys or ckey in seen_after_enrich_vibe:
            continue
        seen_after_enrich_vibe.add(ckey)

        cand_vec = build_candidate_vector(c)
        c["_candidate_vector"] = cand_vec
        cand_emb = c.get("embedding")

        omni = compute_omnimatch_score(
            user_vector=vibe_vector or {},
            candidate_vector=cand_vec,
            user_embedding=vibe_embedding,
            candidate_embedding=cand_emb,
            vote_average=float(c.get("vote_average") or 0.0),
            vote_count=int(c.get("vote_count") or 0),
            user_item_count=len(list_items),
            candidate=c,
            user_items=recent_item_snapshots,
        )

        score = omni["match_score"]
        tags = omni["match_tags"]
        if "✨ Na Sua Vibe" not in tags:
            tags.insert(0, "✨ Na Sua Vibe")

        br_providers = c.get("flatrate_providers_br") or c.get("watch_providers_br", [])
        c["flatrate_providers_br"] = br_providers
        c["watch_providers_br"] = br_providers
        c["match_score"] = score
        c["match_tags"] = tags[:3]
        scored_vibe.append(c)

    # 7. Filtra prioritariamente pelo range [MIN_COMPATIBILITY_SCORE, 100%] e aplica Fast Greedy DPP
    in_range_vibe = [c for c in scored_vibe if float(c.get("match_score") or 0.0) >= MIN_COMPATIBILITY_SCORE]
    vibe_pool = in_range_vibe if len(in_range_vibe) >= top_k else scored_vibe
    diverse_vibe = apply_fast_dpp_reranking(vibe_pool, top_k=top_k, lambda_param=0.65)
    if len(diverse_vibe) < top_k:
        have_vibe_ids = {(c.get("media_type", "movie"), c.get("id")) for c in diverse_vibe}
        rem_vibe = [c for c in scored_vibe if (c.get("media_type", "movie"), c.get("id")) not in have_vibe_ids]
        rem_vibe.sort(key=lambda x: x.get("match_score", 0), reverse=True)
        diverse_vibe.extend(rem_vibe[: top_k - len(diverse_vibe)])
    diverse_vibe.sort(key=lambda x: x.get("match_score", 0), reverse=True)

    for it in diverse_vibe:
        it.pop("embedding", None)
        it.pop("_candidate_vector", None)
        it.pop("candidate_vector", None)
        it.pop("_best_persona_id", None)
        it.pop("_ml_score", None)
        it.pop("_rrf_score", None)
        it.pop("cast", None)
        it.pop("crew", None)
        it.pop("softcore", None)
        it.pop("adult", None)

    return diverse_vibe


