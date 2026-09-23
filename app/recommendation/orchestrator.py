import asyncio
import httpx
from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.users.models import User
from app.core.config import settings
from app.recommendation.matcher import calculate_match_score
from app.recommendation.vector_builder import WEIGHTS

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
    "ficção científica": "10765", "sci-fi & fantasy": "10765",
    "comédia": "35", "drama": "18", "animação": "16", "mistério": "9648", "crime": "80"
}

async def fetch_discover_candidates(user_vector: Dict[str, float], db=None, user_id: str = None) -> List[Dict[str, Any]]:
    if not settings.TMDB_API_KEY:
        return []

    # --- Build smart filters from user profile ---
    genre_features = [(k.replace("genre_", ""), v) for k, v in user_vector.items() if k.startswith("genre_")]
    genre_features.sort(key=lambda x: x[1], reverse=True)

    # --- Busca a lista do usuário uma única vez para genres, cast_ids, crew_ids e PROPORÇÃO Filmes/Séries ---
    live_genre_name_to_id: Dict[str, int] = {}
    cast_person_scores: Dict[int, float] = {}
    crew_person_scores: Dict[int, float] = {}
    
    movie_count = 0
    tv_count = 0

    if db and user_id:
        try:
            from app.tracking.services import get_user_list
            user_items = await get_user_list(db, user_id)

            for list_item in user_items:
                media = list_item.media
                if not media:
                    continue
                    
                if media.media_type == "movie":
                    movie_count += 1
                elif media.media_type == "tv":
                    tv_count += 1

                if media.genres:
                    for g in media.genres:
                        if isinstance(g, dict):
                            _name = str(g.get("name", "")).lower()
                            _gid = g.get("id")
                            if _name and _gid:
                                live_genre_name_to_id[_name] = _gid

                if media.cast_ids:
                    for entry in media.cast_ids:
                        person_id = entry.get("person_id")
                        name = entry.get("name", "")
                        if not person_id or not name:
                            continue
                        feature_key = f"cast_{name.lower()}"
                        signal = user_vector.get(feature_key, 0.0)
                        if signal > 0:
                            cast_person_scores[person_id] = cast_person_scores.get(person_id, 0.0) + signal

                if media.crew_ids:
                    for entry in media.crew_ids:
                        person_id = entry.get("person_id")
                        name = entry.get("name", "")
                        if not person_id or not name:
                            continue
                        feature_key = f"director_{name.lower()}"
                        signal = user_vector.get(feature_key, 0.0)
                        if signal > 0:
                            crew_person_scores[person_id] = crew_person_scores.get(person_id, 0.0) + signal
        except Exception:
            pass
            
    # Proporção Dinâmica Filmes vs Séries
    total_items = movie_count + tv_count
    if total_items == 0:
        movie_prop = 0.5
        tv_prop = 0.5
    else:
        movie_prop = movie_count / total_items
        tv_prop = tv_count / total_items

    # Divisão dos gêneros do usuário (Top 5 primários, próximos 5 secundários)
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

    with_cast: Optional[str] = None
    with_crew: Optional[str] = None
    top_cast_ids = sorted(cast_person_scores.keys(), key=lambda pid: cast_person_scores[pid], reverse=True)[:3]
    top_crew_ids = sorted(crew_person_scores.keys(), key=lambda pid: crew_person_scores[pid], reverse=True)[:2]
    if top_cast_ids:
        with_cast = "|".join(str(pid) for pid in top_cast_ids)
    if top_crew_ids:
        with_crew = "|".join(str(pid) for pid in top_crew_ids)

    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {settings.TMDB_API_KEY}"
    }

    async def fetch_endpoint(client: httpx.AsyncClient, endpoint: str, media_type: Optional[str], params: Dict[str, Any]) -> List[Dict[str, Any]]:
        url = f"{settings.TMDB_BASE_URL}{endpoint}"
        default_params = {"language": "pt-BR", "page": 1}
        default_params.update(params)
        try:
            response = await client.get(url, headers=headers, params=default_params, timeout=10.0)
            response.raise_for_status()
            data = response.json()
            results = data.get("results", [])
            for r in results:
                if media_type:
                    r["media_type"] = media_type
                elif "media_type" not in r:
                    r["media_type"] = "movie" if "title" in r else "tv"
            return results
        except Exception:
            return []

    # Alocação Fixa (Garantindo abundância para os dois carrosséis dedicados)
    f1_movie_pages = 4
    f1_tv_pages = 4
    
    f2_movie_pages = 4
    f2_tv_pages = 4

    f5_movie_pages = 2
    f5_tv_pages = 2

    async with httpx.AsyncClient() as client:
        import random
        tasks = []
        
        def random_pages(count: int, start: int = 1, end: int = 15) -> List[int]:
            return random.sample(range(start, end + 1), min(count, end - start + 1))

        # Faceta 1: Top 5 Gêneros Principais
        if with_primary_movie_genres:
            for p in random_pages(f1_movie_pages, 1, 15):
                tasks.append(fetch_endpoint(client, "/discover/movie", "movie", {"with_genres": with_primary_movie_genres, "vote_count.gte": 200, "sort_by": "popularity.desc", "page": p}))
        if with_primary_tv_genres:
            for p in random_pages(f1_tv_pages, 1, 15):
                tasks.append(fetch_endpoint(client, "/discover/tv", "tv", {"with_genres": with_primary_tv_genres, "vote_count.gte": 100, "sort_by": "popularity.desc", "page": p}))

        # Faceta 2: Gêneros Secundários
        if with_sec_movie_genres:
            for p in random_pages(f2_movie_pages, 1, 15):
                tasks.append(fetch_endpoint(client, "/discover/movie", "movie", {"with_genres": with_sec_movie_genres, "vote_count.gte": 150, "sort_by": "popularity.desc", "page": p}))
        if with_sec_tv_genres:
            for p in random_pages(f2_tv_pages, 1, 15):
                tasks.append(fetch_endpoint(client, "/discover/tv", "tv", {"with_genres": with_sec_tv_genres, "vote_count.gte": 80, "sort_by": "popularity.desc", "page": p}))

        # Faceta 3: Elenco / Diretores Favoritos (Só Filmes no momento)
        if with_cast:
            tasks.append(fetch_endpoint(client, "/discover/movie", "movie", {"with_cast": with_cast, "sort_by": "popularity.desc", "page": 1}))
        if with_crew:
            tasks.append(fetch_endpoint(client, "/discover/movie", "movie", {"with_crew": with_crew, "sort_by": "popularity.desc", "page": 1}))

        # Faceta 4: Obras em Alta (Trending Geral)
        for p in random_pages(2, 1, 5):
            tasks.append(fetch_endpoint(client, "/trending/all/week", None, {"page": p}))

        # Faceta 5: Populares e Top Rated Globais
        for p in random_pages(f5_movie_pages, 1, 10):
            tasks.append(fetch_endpoint(client, "/movie/popular", "movie", {"page": p}))
            tasks.append(fetch_endpoint(client, "/movie/top_rated", "movie", {"page": p}))
        for p in random_pages(f5_tv_pages, 1, 10):
            tasks.append(fetch_endpoint(client, "/tv/popular", "tv", {"page": p}))
            tasks.append(fetch_endpoint(client, "/tv/top_rated", "tv", {"page": p}))

        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

    candidates = []
    seen_ids = set()
    for batch in batch_results:
        if isinstance(batch, list):
            for c in batch:
                cid = c.get("id")
                if cid and cid not in seen_ids:
                    seen_ids.add(cid)
                    candidates.append(c)

    return candidates


async def enrich_candidates_with_details(candidates: List[Dict[str, Any]], user_vector: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
    """
    Estágio 1 do Ranking: Aplica heurística rápida (ou ML futuro) em toda a piscina (400+ itens).
    Estágio 2: Enriquece via API apenas os Top 40 vencedores.
    """
    from app.details.services import fetch_movie_details, fetch_tv_details

    if not candidates:
        return []

    if user_vector:
        from app.recommendation.ranker import LightGBMRanker
        ranker = LightGBMRanker()
        # Mock de User temporario para usar no método predict_score
        dummy_user = User(feature_vector=user_vector)
        if user_vector is not None:
            # Não temos o embedding completo do usuario no enrich (vem só o dict), 
            # mas vamos passar para a preliminar de features rasas:
            pass

        if ranker.is_ready():
            features_batch = [ranker.extract_features(dummy_user, c, cand_emb=None) for c in candidates]
            import numpy as np
            try:
                probs = ranker.model.predict_proba(np.array(features_batch, dtype=np.float64))[:, 1]
                for c, p in zip(candidates, probs):
                    c["_ml_score"] = p * 100.0
                
                def preliminary_score(cand):
                    return cand.get("_ml_score", 0.0)
            except Exception as e:
                # Fallback seguro para heurística se a inferência do ML falhar
                def preliminary_score(cand):
                    return ranker.predict_score(dummy_user, cand, cand_emb=None)
        else:
            def preliminary_score(cand):
                return ranker.predict_score(dummy_user, cand, cand_emb=None)

        # Ao invés de rankear todos juntos, dividimos para garantir os dois carrosséis
        movies = [c for c in candidates if c.get("media_type") == "movie"]
        tv = [c for c in candidates if c.get("media_type") == "tv"]
        
        movies.sort(key=preliminary_score, reverse=True)
        tv.sort(key=preliminary_score, reverse=True)
        
        # Pega os Top 25 Filmes e Top 25 Séries para enviar ao Front
        top_candidates = movies[:25] + tv[:25]
        seen_enrich_ids = {c["id"] for c in top_candidates}
        remaining_candidates = [c for c in candidates if c["id"] not in seen_enrich_ids]
    else:
        movies = [c for c in candidates if c.get("media_type") == "movie"]
        tv = [c for c in candidates if c.get("media_type") == "tv"]
        top_candidates = movies[:25] + tv[:25]
        seen_enrich_ids = {c["id"] for c in top_candidates}
        remaining_candidates = [c for c in candidates if c["id"] not in seen_enrich_ids]

    async def enrich_one(candidate: Dict[str, Any]) -> Dict[str, Any]:
        tmdb_id = candidate.get("id")
        media_type = candidate.get("media_type", "movie")
        try:
            if media_type == "movie":
                details = await fetch_movie_details(tmdb_id)
                candidate["cast"] = [c.name for c in details.credits.cast[:10]] if details.credits else []
                candidate["crew"] = [c.name for c in details.credits.crew] if details.credits else []
                candidate["keywords_list"] = details.keywords if hasattr(details, "keywords") else []
                candidate["original_language"] = details.original_language or ""
            else:
                details = await fetch_tv_details(tmdb_id)
                candidate["cast"] = [c.name for c in details.credits.cast[:10]] if details.credits else []
                candidate["crew"] = [c.name for c in details.credits.crew] if details.credits else []
                candidate["keywords_list"] = details.keywords if hasattr(details, "keywords") else []
                candidate["original_language"] = details.original_language or ""
        except Exception:
            candidate["cast"] = []
            candidate["crew"] = []
            candidate["keywords_list"] = []
            candidate["original_language"] = ""
        return candidate

    enriched = await asyncio.gather(*[enrich_one(c) for c in top_candidates])
    return list(enriched) + remaining_candidates


def build_candidate_vector(candidate: Dict[str, Any]) -> Dict[str, float]:
    """
    Builds a feature vector from a candidate dict.
    Supports shallow (genre_ids only), enriched, and collaborative candidates.
    """
    vector = {}

    m_type = candidate.get("media_type")
    if m_type:
        vector[f"type_{m_type}"] = WEIGHTS["media_type"]

    # Genres from genre_ids (shallow) or enriched
    for g_id in candidate.get("genre_ids", []):
        if g_id == 10759:
            vector["genre_ação"] = WEIGHTS["genre"]
            vector["genre_aventura"] = WEIGHTS["genre"]
        elif g_id == 10765:
            vector["genre_ficção científica"] = WEIGHTS["genre"]
            vector["genre_fantasia"] = WEIGHTS["genre"]
        elif g_id == 10768:
            vector["genre_guerra"] = WEIGHTS["genre"]
        else:
            g_name = GENRE_MAP.get(g_id)
            if g_name:
                vector[f"genre_{g_name}"] = WEIGHTS["genre"]

    # Genres from enriched/collaborative list
    for g in candidate.get("genres", []):
        g_name = g.get("name", "").lower() if isinstance(g, dict) else str(g).lower()
        if "action & adventure" in g_name:
            vector["genre_ação"] = WEIGHTS["genre"]
            vector["genre_aventura"] = WEIGHTS["genre"]
        elif "sci-fi & fantasy" in g_name:
            vector["genre_ficção científica"] = WEIGHTS["genre"]
            vector["genre_fantasia"] = WEIGHTS["genre"]
        elif g_name:
            vector[f"genre_{g_name}"] = WEIGHTS["genre"]

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

    genres_a = set(cand_a.get("genre_ids", []))
    genres_b = set(cand_b.get("genre_ids", []))
    if genres_a and genres_b:
        intersection = len(genres_a & genres_b)
        union = len(genres_a | genres_b)
        return float(intersection / union) if union > 0 else 0.0

    return 0.0


def apply_mmr_reranking(
    candidates: List[Dict[str, Any]],
    lambda_param: float = 0.65,
    top_k: int = 15
) -> List[Dict[str, Any]]:
    """
    Aplica o algoritmo Maximal Marginal Relevance (MMR) para diversificar o carrossel de recomendações,
    evitando 'bolhas de filtro' (ex: quando todos os itens são de uma única franquia ou subgênero).

    Fórmula:
    MMR = argmax_{d_i in Remaining} [ lambda * Rel(d_i) - (1 - lambda) * max_{d_j in Selected} Sim(d_i, d_j) ]

    - lambda_param = 0.65: 65% relevância individual do usuário e 35% diversidade/novidade temática.
    - Se dois filmes forem extremamente parecidos, o segundo sofre penalidade de redundância,
      abrindo espaço para o próximo gênero ou estilo que o usuário também aprecia.
    """
    if not candidates:
        return []

    if len(candidates) <= 1:
        return candidates

    sim_cache: Dict[tuple, float] = {}

    def get_sim(item_a: Dict[str, Any], item_b: Dict[str, Any]) -> float:
        id_a = item_a.get("id")
        id_b = item_b.get("id")
        if id_a == id_b:
            return 1.0
        pair_key = (min(id_a, id_b), max(id_a, id_b))
        if pair_key not in sim_cache:
            sim_cache[pair_key] = calculate_item_similarity(item_a, item_b)
        return sim_cache[pair_key]

    remaining = list(candidates)
    selected = []

    # 1. Primeiro selecionado é o item de maior afinidade absoluta
    remaining.sort(key=lambda x: x.get("match_score", 0), reverse=True)
    selected.append(remaining.pop(0))

    # 2. Seleção gananciosa (greedy) dos próximos itens maximizando o score MMR
    target_count = min(top_k, len(candidates))

    while len(selected) < target_count and remaining:
        best_mmr_score = -float("inf")
        best_idx = 0

        for idx, cand in enumerate(remaining):
            rel = float(cand.get("match_score", 0)) / 100.0

            # Similaridade máxima com qualquer um dos itens já escolhidos para o carrossel
            max_sim_to_selected = max(get_sim(cand, sel_item) for sel_item in selected)

            # Equilíbrio MMR: relevância vs redundância
            mmr_score = (lambda_param * rel) - ((1.0 - lambda_param) * max_sim_to_selected)

            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_idx = idx

        selected.append(remaining.pop(best_idx))

    return selected


async def get_personalized_recommendations(
    db: AsyncSession,
    user_id: str,
    top_k: int = 50,
    persona_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    # Get user profile
    user_result = await db.execute(select(User).where(User.id == user_id))
    user = user_result.scalars().first()

    if not user or not user.feature_vector:
        return []  # Cold start

    user_vector = user.feature_vector
    user_embedding = user.embedding

    # V2: Se uma Persona específica foi selecionada, direciona os vetores para o polo do cluster
    if persona_id and user.taste_clusters:
        target_persona = next((p for p in user.taste_clusters if p.get("id") == persona_id), None)
        if target_persona:
            user_vector = target_persona.get("feature_vector") or user_vector
            user_embedding = target_persona.get("centroid_embedding") or user_embedding
        discover_candidates = await fetch_discover_candidates(user_vector, db=db, user_id=user_id)
    elif user.taste_clusters and len(user.taste_clusters) > 1:
        # Pescaria Multi-Persona para o Carrossel Geral: busca candidatos para cada faceta do usuario
        tasks = [
            fetch_discover_candidates(p.get("feature_vector") or user_vector, db=db, user_id=user_id)
            for p in user.taste_clusters
        ]
        tasks.append(fetch_discover_candidates(user_vector, db=db, user_id=user_id))
        results = await asyncio.gather(*tasks, return_exceptions=True)

        merged_discover = []
        seen_cand = set()
        for res in results:
            if isinstance(res, list):
                for c in res:
                    cid = c.get("id")
                    if cid and cid not in seen_cand:
                        merged_discover.append(c)
                        seen_cand.add(cid)
        discover_candidates = merged_discover
    else:
        discover_candidates = await fetch_discover_candidates(user_vector, db=db, user_id=user_id)

    # 2. Filter out already watched/watchlisted items
    from app.tracking.services import get_user_list
    user_list = await get_user_list(db, user_id)
    seen_tmdb_ids = {item.media.tmdb_id for item in user_list if item.media}

    filtered_discover = [c for c in discover_candidates if c.get("id") not in seen_tmdb_ids]

    # B) Fase 5: Collaborative Filtering candidates from similar users
    from app.recommendation.collaborative import fetch_collaborative_candidates
    try:
        collab_candidates = await fetch_collaborative_candidates(
            db=db,
            target_user_id=user_id,
            seen_tmdb_ids=seen_tmdb_ids,
            limit=10
        )
    except Exception:
        collab_candidates = []

    # Merge candidates: collaborative candidates are already local and detailed
    all_candidates = []
    seen_candidate_ids = set()

    for c in collab_candidates:
        cid = c.get("id")
        if cid and cid not in seen_candidate_ids:
            all_candidates.append(c)
            seen_candidate_ids.add(cid)

    for c in filtered_discover:
        cid = c.get("id")
        if cid and cid not in seen_candidate_ids:
            all_candidates.append(c)
            seen_candidate_ids.add(cid)

    if not all_candidates:
        return []

    # 3. Fase 1: Enrich discover candidates with full TMDB details
    to_enrich = [c for c in all_candidates if not c.get("collaborative")]
    already_enriched = [c for c in all_candidates if c.get("collaborative")]

    enriched_discover = await enrich_candidates_with_details(to_enrich, user_vector=user_vector) if to_enrich else []
    total_candidates = already_enriched + list(enriched_discover)

    # Fase 6: Busca embeddings locais pré-calculados para os candidatos
    candidate_tmdb_ids = [c["id"] for c in total_candidates if "id" in c]
    if candidate_tmdb_ids:
        try:
            from app.media.models import Media as MediaModel
            media_records = await db.execute(
                select(MediaModel.tmdb_id, MediaModel.embedding).where(MediaModel.tmdb_id.in_(candidate_tmdb_ids))
            )
            emb_map = {row[0]: row[1] for row in media_records.all() if row[1]}
            for c in total_candidates:
                if c.get("id") in emb_map and not c.get("embedding"):
                    c["embedding"] = emb_map[c["id"]]
        except Exception:
            pass

    # 4. Score & Re-rank using hybrid vectors (Affinity + Fase 6 Semantic Embeddings)
    from app.recommendation.embeddings import calculate_cosine_similarity
    if user_embedding is None:
        user_embedding = user.embedding

    active_personas = user.taste_clusters if (not persona_id and user.taste_clusters and len(user.taste_clusters) > 1) else None

    scored_candidates = []
    for c in total_candidates:
        candidate_vector = build_candidate_vector(c)
        c["_candidate_vector"] = candidate_vector
        cand_emb = c.get("embedding")

        if active_personas:
            # Score da Faceta Correspondente (Best Match entre as personas ativas do usuário)
            best_final_score = -1.0
            best_tags = []
            best_persona_id = "general"

            for p in active_personas:
                p_vec = p.get("feature_vector") or user_vector
                p_emb = p.get("centroid_embedding") or user_embedding
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
                    best_persona_id = p.get("id")

            final_score = best_final_score
            top_tags = best_tags
            c["_best_persona_id"] = best_persona_id
        else:
            affinity_score, top_tags = calculate_match_score(user_vector, candidate_vector)
            from app.recommendation.ranker import LightGBMRanker
            ranker = LightGBMRanker()
            dummy_user = User(feature_vector=user_vector, embedding=user_embedding)
            
            final_score = ranker.predict_score(dummy_user, c, cand_emb=cand_emb)
            
            if user_embedding and cand_emb:
                sem_sim = calculate_cosine_similarity(user_embedding, cand_emb)
                if sem_sim >= 0.60:
                    top_tags.append("✨ Sintonia Semântica")

        if c.get("collaborative") and "Gosto similar" not in top_tags:
            top_tags.append("Gosto similar")

        c["match_score"] = final_score
        c["match_tags"] = top_tags
        scored_candidates.append(c)

    # 5. Aplica Reranking Garantindo Divisão 50/50 Exata
    target_each = top_k // 2
    
    def select_diverse(pool: List[Dict[str, Any]], target: int) -> List[Dict[str, Any]]:
        if not active_personas:
            return apply_mmr_reranking(pool, lambda_param=0.65, top_k=target)
        
        num_personas = len(active_personas)
        quota = max(3, target // num_personas)
        by_persona = {p.get("id", "general"): [] for p in active_personas}
        by_persona["general"] = []
        for c in pool:
            by_persona[c.get("_best_persona_id", "general")].append(c)
            
        selected = []
        selected_ids = set()
        for p in active_personas:
            p_pool = by_persona.get(p.get("id"), [])
            if p_pool:
                for item in apply_mmr_reranking(p_pool, lambda_param=0.65, top_k=quota):
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

    movies_pool = [c for c in scored_candidates if c.get("media_type") == "movie"]
    tv_pool = [c for c in scored_candidates if c.get("media_type") == "tv"]
    
    selected_movies = select_diverse(movies_pool, target_each)
    selected_tv = select_diverse(tv_pool, target_each)
    
    if len(selected_movies) < target_each:
        rem_tv = [c for c in tv_pool if c not in selected_tv]
        selected_tv.extend(apply_mmr_reranking(rem_tv, lambda_param=0.65, top_k=target_each - len(selected_movies)))
    elif len(selected_tv) < target_each:
        rem_movies = [c for c in movies_pool if c not in selected_movies]
        selected_movies.extend(apply_mmr_reranking(rem_movies, lambda_param=0.65, top_k=target_each - len(selected_tv)))
        
    diverse_recommendations = selected_movies + selected_tv

    # Ordena o carrossel do maior % para o menor dentro do conjunto selecionado
    diverse_recommendations.sort(key=lambda x: x.get("match_score", 0), reverse=True)

    # Limpa dados internos temporários antes de retornar
    for item in diverse_recommendations:
        item.pop("_candidate_vector", None)
        item.pop("_best_persona_id", None)

    return diverse_recommendations


async def get_persona_recommendations(
    db: AsyncSession,
    user_id: str,
    top_k_per_persona: int = 15
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
    for p in clusters:
        p_id = p.get("id")
        recs = await get_personalized_recommendations(
            db=db,
            user_id=user_id,
            top_k=top_k_per_persona,
            persona_id=p_id
        )
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

