import re
import httpx
from fastapi import HTTPException
from app.core.config import settings
from typing import List, Optional, Tuple
from app.core.cache import details_cache
from app.core.tmdb_client import get_tmdb_client, tmdb_request
from app.core.translator import (
    extract_latin_title_from_translations,
    is_non_latin,
    resolve_overview_pt_br,
    resolve_text_field_pt_br,
    translate_batch_to_pt_br,
    translate_to_pt_br,
)
from app.details.schemas import (
    MovieDetailsResponse, 
    TvSeriesDetailsResponse,
    GenreItem, 
    CastItem, 
    CrewItem, 
    CreditsResponse, 
    VideoItem,
    EpisodeItem,
    SeasonItem,
    WatchProviderItem,
    SeasonDetailsResponse,
    EpisodeDetailsResponse,
    CollectionResponse
)

_is_non_latin = is_non_latin

MOVIE_CREATIVE_JOBS = (
    "Director",
    "Screenplay",
    "Writer",
    "Author",
    "Novel",
    "Original Music Composer",
    "Director of Photography",
    "Producer",
    "Executive Producer",
)

TV_CREATIVE_JOBS = (
    "Director",
    "Series Director",
    "Executive Producer",
    "Writer",
    "Screenplay",
    "Original Music Composer",
)

_GENERIC_EP_PT_RE = re.compile(r"^epis[oó]dio\s+\d+$", re.IGNORECASE)
_GENERIC_EP_EN_RE = re.compile(r"^episode\s+\d+$", re.IGNORECASE)


def _needs_episode_name_fallback(name: Optional[str]) -> bool:
    if not name or not name.strip():
        return True
    cleaned = name.strip()
    return is_non_latin(cleaned) or bool(_GENERIC_EP_PT_RE.match(cleaned))


def _is_valid_fallback_episode_name(name: Optional[str]) -> bool:
    if not name or not name.strip():
        return False
    cleaned = name.strip()
    return not is_non_latin(cleaned) and not bool(_GENERIC_EP_EN_RE.match(cleaned))


def _extract_creative_crew(
    raw_crew: list,
    allowed_jobs: Tuple[str, ...],
    base_items: Optional[List[CrewItem]] = None,
    limit: int = 20,
) -> List[CrewItem]:
    result: List[CrewItem] = []
    seen = set()

    for item in base_items or []:
        key = (item.id, item.job)
        if key not in seen:
            seen.add(key)
            result.append(item)
            if len(result) >= limit:
                return result

    for job in allowed_jobs:
        for c in raw_crew or []:
            if c.get("job") != job or not c.get("id") or not c.get("name"):
                continue
            key = (c.get("id"), job)
            if key in seen:
                continue
            seen.add(key)
            result.append(
                CrewItem(
                    id=c.get("id"),
                    name=c.get("name"),
                    job=job,
                    profile_path=c.get("profile_path"),
                )
            )
            if len(result) >= limit:
                return result

    return result


async def _fetch_tmdb_en_fallback(url: str, timeout: float = 8.0) -> Optional[dict]:
    """Busca versão en-US leve (sem append_to_response) apenas quando faltar dado essencial."""
    try:
        en_resp = await tmdb_request(url, params={"language": "en-US"}, timeout=timeout)
        if en_resp.status_code == 200:
            return en_resp.json()
    except Exception:
        pass
    return None


async def _resolve_title_and_overview(url: str, data: dict, title_field: str) -> None:
    """Resolve título latino (caso ausente/CJK) e garante sinopse e tagline em pt-BR."""
    translations_payload = data.get("translations")
    en_data_fallback: Optional[dict] = None

    if not data.get(title_field) or _is_non_latin(data.get(title_field, "")):
        latin_title = extract_latin_title_from_translations(translations_payload, field=title_field)
        if latin_title:
            data[title_field] = latin_title
        else:
            en_data_fallback = await _fetch_tmdb_en_fallback(url)
            if en_data_fallback:
                en_title = en_data_fallback.get(title_field)
                if en_title and not _is_non_latin(en_title):
                    data[title_field] = en_title

    resolved_overview = await resolve_overview_pt_br(
        data.get("overview"),
        translations_payload=translations_payload,
        original_language=data.get("original_language"),
        fallback_en_overview=(en_data_fallback or {}).get("overview"),
    )
    if not resolved_overview and en_data_fallback is None:
        en_data_fallback = await _fetch_tmdb_en_fallback(url)
        if en_data_fallback and (en_data_fallback.get("overview") or "").strip():
            resolved_overview = await translate_to_pt_br(
                en_data_fallback["overview"], source_lang="en"
            )
    data["overview"] = resolved_overview

    data["tagline"] = await resolve_text_field_pt_br(
        data.get("tagline"),
        field="tagline",
        translations_payload=translations_payload,
        original_language=data.get("original_language"),
        fallback_en_value=(en_data_fallback or {}).get("tagline"),
    )


def _select_and_sort_videos(videos_data: list) -> List[VideoItem]:
    """
    Seleciona vídeos do YouTube priorizando Trailers (fallback para Teaser se não houver Trailer),
    ordenados por idioma (pt-BR -> pt -> en -> outros), oficialidade (official=True primeiro)
    e data de publicação mais recente, com deduplicação por chave.
    """
    valid = [
        v
        for v in (videos_data or [])
        if isinstance(v, dict)
        and v.get("site") == "YouTube"
        and v.get("id")
        and v.get("key")
    ]
    if not valid:
        return []

    candidates = [v for v in valid if v.get("type") == "Trailer"]
    if not candidates:
        candidates = [v for v in valid if v.get("type") == "Teaser"]
    if not candidates:
        return []

    def _lang_rank(v: dict) -> int:
        iso639 = (v.get("iso_639_1") or "").strip().lower()
        iso3166 = (v.get("iso_3166_1") or "").strip().upper()
        if iso639 == "pt" and iso3166 == "BR":
            return 0
        if iso639 == "pt":
            return 1
        if iso639 == "en":
            return 2
        return 3

    # Stable sort: newest published_at first, then (language rank, official flag)
    ordered = sorted(candidates, key=lambda v: str(v.get("published_at") or ""), reverse=True)
    ordered.sort(key=lambda v: (_lang_rank(v), 0 if v.get("official") is True else 1))

    seen_keys = set()
    result: List[VideoItem] = []
    for v in ordered:
        key = v.get("key")
        if key in seen_keys:
            continue
        seen_keys.add(key)
        result.append(
            VideoItem(
                id=str(v.get("id")),
                name=v.get("name") or "",
                key=str(key),
                site=str(v.get("site")),
                type=str(v.get("type")),
            )
        )
    return result


async def fetch_movie_details(movie_id: int) -> MovieDetailsResponse:
    if not settings.TMDB_API_KEY:
        raise HTTPException(status_code=500, detail="TMDB API key not configured")

    cache_key = f"details:movie:{movie_id}"
    cached_data, is_stale = details_cache.get_with_status(cache_key)

    if not is_stale and cached_data:
        return MovieDetailsResponse(**cached_data)

    url = f"{settings.TMDB_BASE_URL}/movie/{movie_id}"
    params = {
        "language": "pt-BR",
        "include_video_language": "pt-BR,pt,en,null",
        "append_to_response": "credits,videos,watch/providers,recommendations,keywords,translations"
    }

    try:
        response = await tmdb_request(url, params=params, timeout=12.0)
        if response.status_code == 404:
            raise HTTPException(status_code=404, detail="Filme não encontrado.")
        response.raise_for_status()
        data = response.json()
        await _resolve_title_and_overview(url, data, title_field="title")
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise HTTPException(status_code=404, detail="Filme não encontrado.")
        if cached_data:
            return MovieDetailsResponse(**cached_data)
        raise HTTPException(status_code=502, detail="Falha ao buscar detalhes do filme no provedor externo.")
    except httpx.RequestError as e:
        if cached_data:
            return MovieDetailsResponse(**cached_data)
        raise HTTPException(status_code=502, detail="Falha de connection ao buscar detalhes do filme no provedor externo.")

    # Normalize genres
    genres = [GenreItem(id=g.get("id"), name=g.get("name")) for g in data.get("genres", [])]

    # Normalize credits
    credits_data = data.get("credits", {})
    # Take first 15 cast members
    cast = [
        CastItem(
            id=c.get("id"),
            name=c.get("name"),
            character=c.get("character"),
            profile_path=c.get("profile_path")
        ) for c in credits_data.get("cast", [])[:15]
    ]
    # Equipe criativa principal ordenada por relevância (mantendo crew_ids apenas com Diretores)
    crew = _extract_creative_crew(credits_data.get("crew", []), MOVIE_CREATIVE_JOBS)
    credits_response = CreditsResponse(cast=cast, crew=crew)

    # Fase 4: person_id TMDB — estruturado para discover filtering e fotos
    cast_ids = [
        {"name": c.get("name"), "person_id": c.get("id"), "profile_path": c.get("profile_path")}
        for c in credits_data.get("cast", [])[:15]
        if c.get("id") and c.get("name")
    ]
    crew_ids = [
        {"name": c.get("name"), "person_id": c.get("id"), "profile_path": c.get("profile_path")}
        for c in credits_data.get("crew", [])
        if c.get("job") == "Director" and c.get("id") and c.get("name")
    ]

    # Normalize videos (Trailers com fallback para Teaser, ordenados por idioma e oficialidade)
    videos = _select_and_sort_videos(data.get("videos", {}).get("results", []))

    br_providers = data.get("watch/providers", {}).get("results", {}).get("BR", {})
    watch_providers_data = br_providers.get("flatrate", []) or br_providers.get("free", []) or br_providers.get("ads", [])
    watch_providers = [
        WatchProviderItem(
            provider_id=wp.get("provider_id"),
            provider_name=wp.get("provider_name"),
            logo_path=wp.get("logo_path")
        ) for wp in watch_providers_data if wp.get("provider_id") and wp.get("provider_name") and wp.get("logo_path")
    ]

    similar_data = data.get("recommendations", {}).get("results", [])
    similar = [
        {"id": s.get("id"), "title": s.get("title", s.get("name", "")), "poster_path": s.get("poster_path"), "vote_average": s.get("vote_average", 0.0)}
        for s in similar_data[:10]
    ]

    production_companies_data = data.get("production_companies", [])
    production_companies = [
        {
            "id": pc.get("id"),
            "name": pc.get("name"),
            "logo_path": pc.get("logo_path")
        } for pc in production_companies_data
    ]

    belongs_to_collection_data = data.get("belongs_to_collection")
    belongs_to_collection = None
    if belongs_to_collection_data:
        belongs_to_collection = {
            "id": belongs_to_collection_data.get("id"),
            "name": belongs_to_collection_data.get("name"),
            "poster_path": belongs_to_collection_data.get("poster_path"),
            "backdrop_path": belongs_to_collection_data.get("backdrop_path")
        }

    # Parse keywords (movies use "keywords.keywords", TV uses "keywords.results")
    keywords_raw = data.get("keywords", {}).get("keywords", [])
    keywords = [kw.get("name", "").lower() for kw in keywords_raw if kw.get("name")]

    response_data = {
        "id": data.get("id"),
        "title": data.get("title", ""),
        "original_title": data.get("original_title", ""),
        "original_language": data.get("original_language", ""),
        "overview": data.get("overview", ""),
        "tagline": data.get("tagline", ""),
        "poster_path": data.get("poster_path"),
        "backdrop_path": data.get("backdrop_path"),
        "release_date": data.get("release_date", ""),
        "status": data.get("status", ""),
        "runtime": data.get("runtime", 0),
        "budget": int(data.get("budget") or 0),
        "revenue": int(data.get("revenue") or 0),
        "vote_average": data.get("vote_average", 0.0),
        "vote_count": data.get("vote_count", 0) or 0,
        "genres": [g.model_dump() for g in genres],
        "production_companies": production_companies,
        "belongs_to_collection": belongs_to_collection,
        "credits": credits_response.model_dump(),
        "videos": [v.model_dump() for v in videos],
        "watch_providers": [wp.model_dump() for wp in watch_providers],
        "similar": similar,
        "keywords": keywords,
        # Fase 4: person_id TMDB para cada ator/diretor
        "cast_ids": cast_ids,
        "crew_ids": crew_ids,
    }

    details_cache.set(cache_key, response_data)

    return MovieDetailsResponse(**response_data)


async def fetch_tv_details(series_id: int) -> TvSeriesDetailsResponse:
    if not settings.TMDB_API_KEY:
        raise HTTPException(status_code=500, detail="TMDB API key not configured")

    cache_key = f"details:tv:{series_id}"
    cached_data, is_stale = details_cache.get_with_status(cache_key)

    if not is_stale and cached_data:
        return TvSeriesDetailsResponse(**cached_data)

    url = f"{settings.TMDB_BASE_URL}/tv/{series_id}"
    params = {
        "language": "pt-BR",
        "include_video_language": "pt-BR,pt,en,null",
        "append_to_response": "credits,videos,watch/providers,recommendations,keywords,translations"
    }

    try:
        response = await tmdb_request(url, params=params, timeout=12.0)
        if response.status_code == 404:
            raise HTTPException(status_code=404, detail="Série não encontrada.")
        response.raise_for_status()
        data = response.json()
        await _resolve_title_and_overview(url, data, title_field="name")
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise HTTPException(status_code=404, detail="Série não encontrada.")
        if cached_data:
            return TvSeriesDetailsResponse(**cached_data)
        raise HTTPException(status_code=502, detail="Falha ao buscar detalhes da série no provedor externo.")
    except httpx.RequestError as e:
        if cached_data:
            return TvSeriesDetailsResponse(**cached_data)
        raise HTTPException(status_code=502, detail="Falha de conexão ao buscar detalhes da série no provedor externo.")

    genres = [GenreItem(id=g.get("id"), name=g.get("name")) for g in data.get("genres", [])]

    credits_data = data.get("credits", {})
    cast = [
        CastItem(
            id=c.get("id"),
            name=c.get("name"),
            character=c.get("character"),
            profile_path=c.get("profile_path")
        ) for c in credits_data.get("cast", [])[:15]
    ]
    
    base_creators = [
        CrewItem(
            id=c.get("id"),
            name=c.get("name"),
            job="Creator",
            profile_path=c.get("profile_path")
        ) for c in data.get("created_by", []) if c.get("id") and c.get("name")
    ]
    crew = _extract_creative_crew(
        credits_data.get("crew", []),
        TV_CREATIVE_JOBS,
        base_items=base_creators,
    )
    credits_response = CreditsResponse(cast=cast, crew=crew)

    # Fase 4: person_id TMDB — estruturado para discover filtering e fotos
    cast_ids = [
        {"name": c.get("name"), "person_id": c.get("id"), "profile_path": c.get("profile_path")}
        for c in credits_data.get("cast", [])[:15]
        if c.get("id") and c.get("name")
    ]
    # Para TV, criadores vêm de 'created_by'; também incluímos diretores do crew
    crew_ids = [
        {"name": c.get("name"), "person_id": c.get("id"), "profile_path": c.get("profile_path")}
        for c in data.get("created_by", [])
        if c.get("id") and c.get("name")
    ]
    # Complementa com diretores do crew (Directors of TV episodes, etc.)
    for c in credits_data.get("crew", []):
        if c.get("job") in ("Director", "Series Director") and c.get("id") and c.get("name"):
            entry = {"name": c.get("name"), "person_id": c.get("id"), "profile_path": c.get("profile_path")}
            if not any(x.get("person_id") == c.get("id") for x in crew_ids):
                crew_ids.append(entry)

    videos = _select_and_sort_videos(data.get("videos", {}).get("results", []))
    
    br_providers = data.get("watch/providers", {}).get("results", {}).get("BR", {})
    watch_providers_data = br_providers.get("flatrate", []) or br_providers.get("free", []) or br_providers.get("ads", [])
    watch_providers = [
        WatchProviderItem(
            provider_id=wp.get("provider_id"),
            provider_name=wp.get("provider_name"),
            logo_path=wp.get("logo_path")
        ) for wp in watch_providers_data if wp.get("provider_id") and wp.get("provider_name") and wp.get("logo_path")
    ]

    def parse_episode(ep_data):
        if not ep_data:
            return None
        return EpisodeItem(
            id=ep_data.get("id"),
            name=ep_data.get("name"),
            air_date=ep_data.get("air_date"),
            episode_number=ep_data.get("episode_number"),
            season_number=ep_data.get("season_number")
        )

    last_episode = parse_episode(data.get("last_episode_to_air"))
    next_episode = parse_episode(data.get("next_episode_to_air"))
    
    seasons = []
    for s in data.get("seasons", []):
        seasons.append(SeasonItem(
            id=s.get("id"),
            name=s.get("name"),
            season_number=s.get("season_number"),
            episode_count=s.get("episode_count"),
            air_date=s.get("air_date"),
            poster_path=s.get("poster_path")
        ))

    similar_data = data.get("recommendations", {}).get("results", [])
    similar = [
        {"id": s.get("id"), "title": s.get("name", s.get("title", "")), "poster_path": s.get("poster_path"), "vote_average": s.get("vote_average", 0.0)}
        for s in similar_data[:10]
    ]

    networks_data = data.get("networks", [])
    networks = [
        {
            "id": net.get("id"),
            "name": net.get("name"),
            "logo_path": net.get("logo_path")
        } for net in networks_data
    ]

    # TV uses "keywords.results" (different from movie which uses "keywords.keywords")
    keywords_raw = data.get("keywords", {}).get("results", [])
    keywords = [kw.get("name", "").lower() for kw in keywords_raw if kw.get("name")]

    response_data = {
        "id": data.get("id"),
        "title": data.get("name", ""),
        "overview": data.get("overview", ""),
        "tagline": data.get("tagline", ""),
        "original_language": data.get("original_language", ""),
        "poster_path": data.get("poster_path"),
        "backdrop_path": data.get("backdrop_path"),
        "first_air_date": data.get("first_air_date"),
        "status": data.get("status", ""),
        "number_of_seasons": data.get("number_of_seasons", 0),
        "number_of_episodes": data.get("number_of_episodes", 0),
        "vote_average": data.get("vote_average", 0.0),
        "vote_count": data.get("vote_count", 0) or 0,
        "genres": [g.model_dump() for g in genres],
        "episode_run_time": data.get("episode_run_time", []),
        "networks": networks,
        "last_episode_to_air": last_episode.model_dump() if last_episode else None,
        "next_episode_to_air": next_episode.model_dump() if next_episode else None,
        "seasons": [s.model_dump() for s in seasons],
        "credits": credits_response.model_dump(),
        "videos": [v.model_dump() for v in videos],
        "watch_providers": [wp.model_dump() for wp in watch_providers],
        "similar": similar,
        "keywords": keywords,
        # Fase 4: person_id TMDB para cada ator/diretor
        "cast_ids": cast_ids,
        "crew_ids": crew_ids,
    }

    details_cache.set(cache_key, response_data)

    return TvSeriesDetailsResponse(**response_data)


async def fetch_season_details(series_id: int, season_number: int) -> SeasonDetailsResponse:
    if not settings.TMDB_API_KEY:
        raise HTTPException(status_code=500, detail="TMDB API key not configured")

    cache_key = f"details:tv:{series_id}:season:{season_number}"
    cached_data, is_stale = details_cache.get_with_status(cache_key)

    if not is_stale and cached_data:
        return SeasonDetailsResponse(**cached_data)

    url = f"{settings.TMDB_BASE_URL}/tv/{series_id}/season/{season_number}"
    params = {
        "language": "pt-BR",
        "append_to_response": "translations",
    }

    try:
        response = await tmdb_request(url, params=params, timeout=12.0)
        if response.status_code == 404:
            raise HTTPException(status_code=404, detail="Temporada não encontrada.")
        response.raise_for_status()
        data = response.json()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise HTTPException(status_code=404, detail="Temporada não encontrada.")
        if cached_data:
            return SeasonDetailsResponse(**cached_data)
        raise HTTPException(status_code=502, detail="Falha ao buscar detalhes da temporada no provedor externo.")
    except httpx.RequestError as e:
        if cached_data:
            return SeasonDetailsResponse(**cached_data)
        raise HTTPException(status_code=502, detail="Falha de conexão ao buscar detalhes da temporada no provedor externo.")

    season_overview = await resolve_overview_pt_br(
        data.get("overview"),
        translations_payload=data.get("translations"),
    )

    raw_episodes = data.get("episodes", [])
    missing_ep_indices = [
        idx
        for idx, ep in enumerate(raw_episodes)
        if not (ep.get("overview") or "").strip() or _is_non_latin(ep.get("overview", ""))
    ]
    missing_name_indices = [
        idx
        for idx, ep in enumerate(raw_episodes)
        if _needs_episode_name_fallback(ep.get("name"))
    ]

    if missing_ep_indices or not season_overview or missing_name_indices:
        en_season = await _fetch_tmdb_en_fallback(url, timeout=10.0)
        if en_season:
            if not season_overview and (en_season.get("overview") or "").strip():
                season_overview = await translate_to_pt_br(en_season["overview"], source_lang="en")

            en_eps_by_num = {
                ep.get("episode_number"): ep
                for ep in en_season.get("episodes", [])
                if ep.get("episode_number") is not None
            }

            if missing_name_indices:
                for idx in missing_name_indices:
                    ep_num = raw_episodes[idx].get("episode_number")
                    en_ep = en_eps_by_num.get(ep_num)
                    en_name = (en_ep or {}).get("name")
                    if _is_valid_fallback_episode_name(en_name):
                        raw_episodes[idx]["name"] = en_name.strip()

            if missing_ep_indices:
                to_translate: List[Tuple[int, str]] = []
                for idx in missing_ep_indices:
                    ep_num = raw_episodes[idx].get("episode_number")
                    en_ep = en_eps_by_num.get(ep_num)
                    en_ov = ((en_ep or {}).get("overview") or "").strip()
                    if en_ov:
                        to_translate.append((idx, en_ov))
                if to_translate:
                    translated_ovs = await translate_batch_to_pt_br(
                        [(txt, "en") for _, txt in to_translate]
                    )
                    for (idx, _), tr_ov in zip(to_translate, translated_ovs):
                        raw_episodes[idx]["overview"] = tr_ov

    episodes = []
    for ep in raw_episodes:
        episodes.append({
            "id": ep.get("id"),
            "name": ep.get("name", ""),
            "air_date": ep.get("air_date"),
            "episode_number": ep.get("episode_number"),
            "season_number": ep.get("season_number"),
            "overview": ep.get("overview"),
            "still_path": ep.get("still_path"),
            "vote_average": ep.get("vote_average"),
            "runtime": ep.get("runtime")
        })

    response_data = {
        "id": data.get("id"),
        "name": data.get("name", ""),
        "overview": season_overview,
        "season_number": data.get("season_number", 0),
        "poster_path": data.get("poster_path"),
        "air_date": data.get("air_date"),
        "episodes": episodes
    }

    details_cache.set(cache_key, response_data)

    return SeasonDetailsResponse(**response_data)


async def fetch_episode_details(series_id: int, season_number: int, episode_number: int) -> EpisodeDetailsResponse:
    if not settings.TMDB_API_KEY:
        raise HTTPException(status_code=500, detail="TMDB API key not configured")

    cache_key = f"details:tv:{series_id}:season:{season_number}:episode:{episode_number}"
    cached_data, is_stale = details_cache.get_with_status(cache_key)

    if not is_stale and cached_data:
        return EpisodeDetailsResponse(**cached_data)

    # Otimização Fast-Path: Verifica se a temporada completa já está em cache (apenas se a sinopse for válida)
    season_cache_key = f"details:tv:{series_id}:season:{season_number}"
    cached_season, is_season_stale = details_cache.get_with_status(season_cache_key)
    if not is_season_stale and cached_season and "episodes" in cached_season:
        for ep in cached_season["episodes"]:
            if ep.get("episode_number") == episode_number:
                cached_ep_ov = (ep.get("overview") or "").strip()
                if cached_ep_ov and not _is_non_latin(cached_ep_ov):
                    response_data = {
                        "id": ep.get("id"),
                        "name": ep.get("name", ""),
                        "air_date": ep.get("air_date"),
                        "episode_number": ep.get("episode_number", episode_number),
                        "season_number": ep.get("season_number", season_number),
                        "overview": cached_ep_ov,
                        "still_path": ep.get("still_path"),
                        "vote_average": ep.get("vote_average", 0.0),
                        "runtime": ep.get("runtime")
                    }
                    details_cache.set(cache_key, response_data)
                    return EpisodeDetailsResponse(**response_data)
                break

    url = f"{settings.TMDB_BASE_URL}/tv/{series_id}/season/{season_number}/episode/{episode_number}"
    params = {
        "language": "pt-BR",
        "append_to_response": "translations",
    }

    try:
        response = await tmdb_request(url, params=params, timeout=10.0)
        if response.status_code == 404:
            raise HTTPException(status_code=404, detail="Episódio não encontrado.")
        response.raise_for_status()
        data = response.json()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise HTTPException(status_code=404, detail="Episódio não encontrado.")
        if cached_data:
            return EpisodeDetailsResponse(**cached_data)
        raise HTTPException(status_code=502, detail="Falha ao buscar detalhes do episódio no provedor externo.")
    except httpx.RequestError as e:
        if cached_data:
            return EpisodeDetailsResponse(**cached_data)
        raise HTTPException(status_code=502, detail="Falha de conexão ao buscar detalhes do episódio no provedor externo.")

    en_ep: Optional[dict] = None
    resolved_overview = await resolve_overview_pt_br(
        data.get("overview"),
        translations_payload=data.get("translations"),
    )
    if not resolved_overview:
        en_ep = await _fetch_tmdb_en_fallback(url, timeout=8.0)
        if en_ep and (en_ep.get("overview") or "").strip():
            resolved_overview = await translate_to_pt_br(en_ep["overview"], source_lang="en")

    ep_name = (data.get("name") or "").strip()
    if _needs_episode_name_fallback(ep_name):
        cand_name = extract_latin_title_from_translations(data.get("translations"), field="name")
        if _is_valid_fallback_episode_name(cand_name):
            ep_name = cand_name.strip()
        else:
            if en_ep is None:
                en_ep = await _fetch_tmdb_en_fallback(url, timeout=8.0)
            if en_ep and _is_valid_fallback_episode_name(en_ep.get("name")):
                ep_name = en_ep["name"].strip()

    response_data = {
        "id": data.get("id"),
        "name": ep_name,
        "air_date": data.get("air_date"),
        "episode_number": data.get("episode_number", 0),
        "season_number": data.get("season_number", 0),
        "overview": resolved_overview,
        "still_path": data.get("still_path"),
        "vote_average": data.get("vote_average"),
        "runtime": data.get("runtime")
    }

    details_cache.set(cache_key, response_data)

    return EpisodeDetailsResponse(**response_data)

async def fetch_collection_details(collection_id: int) -> CollectionResponse:
    if not settings.TMDB_API_KEY:
        raise HTTPException(status_code=500, detail="TMDB API key not configured")

    cache_key = f"details:collection:{collection_id}"
    cached_data, is_stale = details_cache.get_with_status(cache_key)

    if not is_stale and cached_data:
        return CollectionResponse(**cached_data)

    url = f"{settings.TMDB_BASE_URL}/collection/{collection_id}"
    params = {
        "language": "pt-BR"
    }

    try:
        response = await tmdb_request(url, params=params, timeout=12.0)
        if response.status_code == 404:
            raise HTTPException(status_code=404, detail="Coleção não encontrada.")
        response.raise_for_status()
        data = response.json()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise HTTPException(status_code=404, detail="Coleção não encontrada.")
        if cached_data:
            return CollectionResponse(**cached_data)
        raise HTTPException(status_code=502, detail="Falha ao buscar detalhes da coleção no provedor externo.")
    except httpx.RequestError as e:
        if cached_data:
            return CollectionResponse(**cached_data)
        raise HTTPException(status_code=502, detail="Falha de conexão ao buscar detalhes da coleção no provedor externo.")

    resolved_col_ov = (data.get("overview") or "").strip()
    if not resolved_col_ov or _is_non_latin(resolved_col_ov):
        tr_payload = None
        try:
            tr_resp = await tmdb_request(
                f"{settings.TMDB_BASE_URL}/collection/{collection_id}/translations",
                timeout=8.0,
            )
            if tr_resp.status_code == 200:
                tr_payload = tr_resp.json()
        except Exception:
            pass
        resolved_col_ov = await resolve_overview_pt_br(
            data.get("overview"),
            translations_payload=tr_payload,
        )

    raw_parts = data.get("parts", [])
    need_en_name = not data.get("name") or _is_non_latin(data.get("name", ""))
    need_en_col_ov = not resolved_col_ov
    need_en_parts_title = any(_is_non_latin(p.get("title", "")) for p in raw_parts)
    missing_part_ov_indices = [
        idx
        for idx, p in enumerate(raw_parts)
        if not (p.get("overview") or "").strip() or _is_non_latin(p.get("overview", ""))
    ]

    en_parts_by_id = {}
    en_part_ovs_by_idx = {}
    if need_en_name or need_en_col_ov or need_en_parts_title or missing_part_ov_indices:
        en_data = await _fetch_tmdb_en_fallback(url, timeout=8.0)
        if en_data:
            if need_en_name:
                en_name = en_data.get("name")
                if en_name and not _is_non_latin(en_name):
                    data["name"] = en_name
            if need_en_col_ov and (en_data.get("overview") or "").strip():
                resolved_col_ov = await translate_to_pt_br(en_data["overview"], source_lang="en")

            en_parts_map = {ep["id"]: ep for ep in en_data.get("parts", []) if ep.get("id")}
            for ep_id, ep in en_parts_map.items():
                if ep.get("title") and not _is_non_latin(ep["title"]):
                    en_parts_by_id[ep_id] = ep["title"]

            if missing_part_ov_indices:
                to_translate: List[Tuple[int, str]] = []
                for idx in missing_part_ov_indices:
                    part_id = raw_parts[idx].get("id")
                    en_ep = en_parts_map.get(part_id) if part_id else None
                    en_ov = ((en_ep or {}).get("overview") or "").strip()
                    if en_ov:
                        to_translate.append((idx, en_ov))
                if to_translate:
                    translated_ovs = await translate_batch_to_pt_br(
                        [(txt, "en") for _, txt in to_translate]
                    )
                    for (idx, _), tr_ov in zip(to_translate, translated_ovs):
                        en_part_ovs_by_idx[idx] = tr_ov

    parts = []
    for idx, part in enumerate(raw_parts):
        p_title = part.get("title", "")
        if _is_non_latin(p_title) and part.get("id") in en_parts_by_id:
            p_title = en_parts_by_id[part["id"]]
        p_overview = en_part_ovs_by_idx.get(idx, (part.get("overview") or "").strip())
        parts.append({
            "id": part.get("id"),
            "title": p_title,
            "overview": p_overview,
            "poster_path": part.get("poster_path"),
            "backdrop_path": part.get("backdrop_path"),
            "release_date": part.get("release_date"),
            "vote_average": part.get("vote_average", 0.0)
        })

    response_data = {
        "id": data.get("id"),
        "name": data.get("name", ""),
        "overview": resolved_col_ov,
        "poster_path": data.get("poster_path"),
        "backdrop_path": data.get("backdrop_path"),
        "parts": parts
    }

    details_cache.set(cache_key, response_data)

    return CollectionResponse(**response_data)

