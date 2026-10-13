import math
import httpx
from typing import Dict, List, Optional, Set, Tuple
from fastapi import HTTPException
from app.core.config import settings
from app.core.cache import details_cache
from app.person.schemas import PersonDetailsResponse, PersonCastItem, PersonDiscoverResponse
from app.recommendation.orchestrator import GENRE_MAP
from app.recommendation.matcher import calculate_match_score
from app.recommendation.vector_builder import WEIGHTS

EXCLUDED_ACTING_GENRE_IDS = {10767, 10763, 10764, 99}  # Talk, News, Reality, Documentary
EXCLUDED_CHARACTER_TOKENS = (
    "self",
    "himself",
    "herself",
    "ele mesmo",
    "ela mesma",
    "uncredited",
    "não creditado",
    "archive footage",
    "archive sound",
    "cameo",
)


async def fetch_person_details(person_id: int) -> PersonDetailsResponse:
    if not settings.TMDB_API_KEY:
        raise HTTPException(status_code=500, detail="TMDB API key not configured")

    cache_key = f"details:person:v2:{person_id}"
    cached_data, is_stale = details_cache.get_with_status(cache_key)

    if not is_stale and cached_data:
        return PersonDetailsResponse(**cached_data)

    url = f"{settings.TMDB_BASE_URL}/person/{person_id}"
    params = {
        "language": "pt-BR",
        "append_to_response": "combined_credits"
    }

    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {settings.TMDB_API_KEY}"
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, params=params, timeout=10.0)
            if response.status_code == 404:
                raise HTTPException(status_code=404, detail="Pessoa não encontrada.")
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise HTTPException(status_code=404, detail="Pessoa não encontrada.")
        if cached_data:
            return PersonDetailsResponse(**cached_data)
        raise HTTPException(status_code=502, detail="Falha ao buscar detalhes no provedor externo.")
    except httpx.RequestError:
        if cached_data:
            return PersonDetailsResponse(**cached_data)
        raise HTTPException(status_code=502, detail="Falha de conexão ao buscar detalhes.")

    cast_data = data.get("combined_credits", {}).get("cast", [])
    crew_data = data.get("combined_credits", {}).get("crew", [])

    cast_list = []

    # Process cast
    for c in cast_data:
        media_type = c.get("media_type")
        if media_type not in ["movie", "tv"]:
            continue
        title = c.get("title") if media_type == "movie" else c.get("name")
        release = c.get("release_date") if media_type == "movie" else c.get("first_air_date")

        cast_list.append({
            "id": c.get("id"),
            "title": title or "",
            "character": c.get("character", ""),
            "department": "Acting",
            "poster_path": c.get("poster_path"),
            "media_type": media_type,
            "release_date": release,
            "popularity": float(c.get("popularity") or 0.0),
            "vote_count": int(c.get("vote_count") or 0),
            "vote_average": float(c.get("vote_average") or 0.0),
            "order": c.get("order"),
            "genre_ids": c.get("genre_ids") or [],
            "episode_count": c.get("episode_count"),
        })

    # Process crew
    for c in crew_data:
        media_type = c.get("media_type")
        if media_type not in ["movie", "tv"]:
            continue
        title = c.get("title") if media_type == "movie" else c.get("name")
        release = c.get("release_date") if media_type == "movie" else c.get("first_air_date")

        cast_list.append({
            "id": c.get("id"),
            "title": title or "",
            "job": c.get("job", ""),
            "department": c.get("department", ""),
            "poster_path": c.get("poster_path"),
            "media_type": media_type,
            "release_date": release,
            "popularity": float(c.get("popularity") or 0.0),
            "vote_count": int(c.get("vote_count") or 0),
            "vote_average": float(c.get("vote_average") or 0.0),
            "genre_ids": c.get("genre_ids") or [],
            "episode_count": c.get("episode_count"),
        })

    cast_list.sort(key=lambda x: x["popularity"], reverse=True)

    response_data = {
        "id": data.get("id"),
        "name": data.get("name", ""),
        "biography": data.get("biography", ""),
        "birthday": data.get("birthday"),
        "place_of_birth": data.get("place_of_birth"),
        "profile_path": data.get("profile_path"),
        "known_for_department": data.get("known_for_department", ""),
        "combined_credits": cast_list
    }

    details_cache.set(cache_key, response_data)

    return PersonDetailsResponse(**response_data)


def _compute_role_protagonism(item: PersonCastItem, target_dept: str) -> Tuple[float, str]:
    """
    Calcula o peso de protagonismo (0.0 a 1.0) e um rótulo amigável para o papel.
    Representa até 40% do Score de Sugestão.
    """
    dept = item.department or target_dept

    if dept == "Acting":
        if item.media_type == "tv":
            ep_count = item.episode_count or 0
            if ep_count >= 16:
                return 1.0, "Elenco Principal"
            if ep_count >= 8:
                return 0.90, "Protagonista"
            if ep_count >= 4:
                return 0.70, "Destaque na Série"
            if ep_count >= 2:
                return 0.35, "Participação"
            return 0.10, "Participação Especial"

        order = item.order
        if order is None:
            return 0.50, "Elenco"
        if order == 0:
            return 1.0, "Protagonista"
        if order == 1:
            return 0.95, "Protagonista"
        if order == 2:
            return 0.85, "Co-protagonista"
        if order <= 4:
            return 0.75, "Papel de Destaque"
        if order <= 7:
            return 0.55, "Coadjuvante"
        if order <= 12:
            return 0.30, "Elenco de Apoio"
        return 0.0, "Participação"

    job_lower = (item.job or "").lower()
    if dept == "Directing" or "director" in job_lower:
        if job_lower in ("director", "series director", "diretor"):
            return 1.0, "Direção"
        return 0.65, item.job or "Direção"

    if dept in ("Writing", "Creator") or job_lower in ("creator", "screenplay", "writer", "story"):
        return 0.95, item.job or "Roteiro / Criação"

    if dept == "Production":
        if job_lower == "producer":
            return 0.85, "Produção"
        return 0.60, item.job or "Produção"

    return 0.60, item.job or dept


def _compute_quality_relevance(item: PersonCastItem) -> float:
    """
    Calcula Qualidade + Relevância (0.0 a 1.0) combinando nota Bayesiana e alcance logarítmico.
    Representa até 35% do Score de Sugestão.
    Calibrado para que obras aclamadas (nota >= 8.0 e > 5.000 votos) atinjam ~0.95–1.0.
    """
    v = max(0, item.vote_count or 0)
    r = float(item.vote_average or 0.0)
    m = 250.0
    c = 6.8

    bayesian_rating = (v / (v + m)) * r + (m / (v + m)) * c
    rating_norm = max(0.0, min(1.0, (bayesian_rating - 5.2) / 3.0))

    log_votes = math.log10(max(1, v))
    reach_norm = max(0.0, min(1.0, (log_votes - 1.7) / 2.0))

    return 0.60 * rating_norm + 0.40 * reach_norm


def _score_single_vector_affinity(
    item_genre_keys: List[str],
    media_type: str,
    vec: Dict[str, float],
) -> Optional[float]:
    if not vec:
        return None

    user_genres = {k: v for k, v in vec.items() if k.startswith("genre_") and v > 0}
    if not user_genres:
        return None

    max_u_genre = max(user_genres.values()) or 1.0

    if not item_genre_keys:
        genre_ratio = 0.6
    else:
        ratios: List[float] = []
        penalties = 0.0
        for gk in item_genre_keys:
            u_val = vec.get(gk, 0.0)
            if u_val > 0:
                # Curva suave (x^0.75) para valorizar gêneros recorrentes sem exigir que sejam apenas o #1 absoluto
                raw_r = min(1.0, u_val / max_u_genre)
                ratios.append(math.pow(raw_r, 0.75))
            elif u_val < 0:
                penalties += (abs(u_val) / max_u_genre) * 0.35

        ratios.sort(reverse=True)
        if not ratios:
            genre_ratio = max(0.0, 0.25 - penalties)
        elif len(ratios) == 1:
            genre_ratio = max(0.0, min(1.0, ratios[0] - penalties))
        else:
            # Foca nos 2 melhores gêneros da obra para não punir filmes com 3-4 gêneros na TMDB
            top_blend = 0.70 * ratios[0] + 0.30 * ratios[1]
            genre_ratio = max(0.0, min(1.0, top_blend - penalties))

    type_key = f"type_{media_type}"
    u_type_val = vec.get(type_key, 0.0)
    type_ratio = 1.0 if u_type_val > 0 else (0.4 if u_type_val < 0 else 0.75)

    return 0.85 * genre_ratio + 0.15 * type_ratio


def _compute_user_genre_affinity(
    item: PersonCastItem,
    user_vector: Optional[Dict[str, float]],
    taste_clusters: Optional[List[Dict]] = None,
) -> Tuple[float, bool]:
    """
    Calcula a afinidade de gênero/formato do usuário (0.0 a 1.0) para os 25% do Score de Sugestão,
    avaliando tanto o vetor global quanto as Personas ativas (taste_clusters) e pegando o melhor encaixe.
    """
    item_genre_keys = [
        f"genre_{GENRE_MAP[gid]}"
        for gid in (item.genre_ids or [])
        if gid in GENRE_MAP
    ]

    candidate_vectors: List[Dict[str, float]] = []
    if taste_clusters:
        for cluster in taste_clusters:
            if isinstance(cluster, dict) and cluster.get("is_active", True):
                c_vec = cluster.get("feature_vector")
                if isinstance(c_vec, dict) and c_vec:
                    candidate_vectors.append(c_vec)

    if isinstance(user_vector, dict) and user_vector:
        candidate_vectors.append(user_vector)

    if not candidate_vectors:
        return 0.0, False

    best_affinity: Optional[float] = None
    for vec in candidate_vectors:
        score = _score_single_vector_affinity(item_genre_keys, item.media_type, vec)
        if score is not None:
            if best_affinity is None or score > best_affinity:
                best_affinity = score

    if best_affinity is None:
        return 0.0, False

    return best_affinity, True


def compute_person_discover_suggestions(
    person: PersonDetailsResponse,
    user_vector: Optional[Dict[str, float]] = None,
    taste_clusters: Optional[List[Dict]] = None,
    exclude_tmdb_ids: Optional[Set[int]] = None,
    limit: int = 12,
) -> PersonDiscoverResponse:
    """
    Score de Destaque Personalizado:
    - 40% Protagonismo (order / episode_count / job)
    - 35% Qualidade e Relevância (Nota Bayesiana + log10(vote_count))
    - 25% Afinidade de Gênero do Usuário (avaliada nas Personas + feature_vector)
    Filtra obras já assistidas/descartadas e aparições irrelevantes.
    """
    excluded = exclude_tmdb_ids or set()
    target_dept = person.known_for_department or "Acting"

    def _select_candidates(min_votes: int, strict_role: bool) -> List[PersonCastItem]:
        candidates_by_id: Dict[int, PersonCastItem] = {}
        for item in person.combined_credits:
            if not item.poster_path or item.id in excluded:
                continue
            if item.department != target_dept:
                continue
            if (item.vote_count or 0) < min_votes:
                continue

            if target_dept == "Acting":
                char_lower = (item.character or "").lower()
                if any(tok in char_lower for tok in EXCLUDED_CHARACTER_TOKENS):
                    continue
                if strict_role and any(gid in EXCLUDED_ACTING_GENRE_IDS for gid in (item.genre_ids or [])):
                    continue
                if item.media_type == "movie":
                    if item.order is not None and item.order > (10 if strict_role else 15):
                        continue
                elif item.media_type == "tv":
                    if strict_role and (item.episode_count is not None and item.episode_count <= 2):
                        continue

            existing = candidates_by_id.get(item.id)
            if not existing:
                candidates_by_id[item.id] = item
            else:
                # Mantém o crédito de maior relevância daquela obra
                if (item.order or 99) < (existing.order or 99):
                    candidates_by_id[item.id] = item

        return list(candidates_by_id.values())

    candidates = _select_candidates(min_votes=30, strict_role=True)
    if len(candidates) < 6:
        candidates = _select_candidates(min_votes=5, strict_role=False)

    scored_items: List[PersonCastItem] = []
    any_personalized = False

    for raw_item in candidates:
        role_norm, role_label = _compute_role_protagonism(raw_item, target_dept)
        if role_norm <= 0.0:
            continue

        quality_norm = _compute_quality_relevance(raw_item)
        affinity_norm, has_affinity = _compute_user_genre_affinity(
            raw_item, user_vector, taste_clusters
        )

        if has_affinity:
            any_personalized = True
            suggestion_score = (
                (role_norm * 40.0)
                + (quality_norm * 35.0)
                + (affinity_norm * 25.0)
            )
        else:
            # Se o usuário não estiver logado ou tiver vetor vazio, os 75% da Opção 2 escalam para 100%
            base_75 = (role_norm * 40.0) + (quality_norm * 35.0)
            suggestion_score = (base_75 / 75.0) * 100.0

        final_pct = round(max(0.0, min(100.0, suggestion_score)), 1)
        enriched = raw_item.model_copy(
            update={
                "suggestion_score": final_pct,
                "match_score": final_pct,
                "role_label": role_label,
            }
        )
        scored_items.append(enriched)

    scored_items.sort(
        key=lambda x: (x.suggestion_score or 0.0, x.vote_count or 0),
        reverse=True,
    )

    return PersonDiscoverResponse(
        items=scored_items[:limit],
        is_personalized=any_personalized,
    )


