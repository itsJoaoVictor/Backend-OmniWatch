import httpx
from fastapi import HTTPException
from app.core.config import settings
from app.core.cache import details_cache
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
        "append_to_response": "credits,videos,watch/providers,recommendations,keywords"
    }
    
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {settings.TMDB_API_KEY}"
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, params=params, timeout=10.0)
            if response.status_code == 404:
                raise HTTPException(status_code=404, detail="Filme não encontrado.")
            response.raise_for_status()
            data = response.json()
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
    # Filter crew for Director
    crew = [
        CrewItem(
            id=c.get("id"),
            name=c.get("name"),
            job=c.get("job"),
            profile_path=c.get("profile_path")
        ) for c in credits_data.get("crew", []) if c.get("job") == "Director"
    ]
    credits_response = CreditsResponse(cast=cast, crew=crew)

    # Fase 4: person_id TMDB — estruturado para discover filtering
    cast_ids = [
        {"name": c.get("name"), "person_id": c.get("id")}
        for c in credits_data.get("cast", [])[:15]
        if c.get("id") and c.get("name")
    ]
    crew_ids = [
        {"name": c.get("name"), "person_id": c.get("id")}
        for c in credits_data.get("crew", [])
        if c.get("job") == "Director" and c.get("id") and c.get("name")
    ]

    # Normalize videos (filter for Trailers, prefer YouTube)
    videos_data = data.get("videos", {}).get("results", [])
    videos = [
        VideoItem(
            id=v.get("id"),
            name=v.get("name"),
            key=v.get("key"),
            site=v.get("site"),
            type=v.get("type")
        ) for v in videos_data if v.get("type") == "Trailer" and v.get("site") == "YouTube"
    ]

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
        "vote_average": data.get("vote_average", 0.0),
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
        "append_to_response": "credits,videos,watch/providers,recommendations,keywords"
    }
    
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {settings.TMDB_API_KEY}"
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, params=params, timeout=10.0)
            if response.status_code == 404:
                raise HTTPException(status_code=404, detail="Série não encontrada.")
            response.raise_for_status()
            data = response.json()
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
    
    # TV uses 'created_by' for creators or 'crew' for specific roles. Spec says "created_by" or "Creator" / "Executive Producer" in crew.
    # We will use 'created_by' as it's at the root and most accurate for TV creators on TMDB.
    crew = [
        CrewItem(
            id=c.get("id"),
            name=c.get("name"),
            job="Creator",
            profile_path=c.get("profile_path")
        ) for c in data.get("created_by", [])
    ]
    credits_response = CreditsResponse(cast=cast, crew=crew)

    # Fase 4: person_id TMDB — estruturado para discover filtering
    cast_ids = [
        {"name": c.get("name"), "person_id": c.get("id")}
        for c in credits_data.get("cast", [])[:15]
        if c.get("id") and c.get("name")
    ]
    # Para TV, criadores vêm de 'created_by'; também incluímos diretores do crew
    crew_ids = [
        {"name": c.get("name"), "person_id": c.get("id")}
        for c in data.get("created_by", [])
        if c.get("id") and c.get("name")
    ]
    # Complementa com diretores do crew (Directors of TV episodes, etc.)
    for c in credits_data.get("crew", []):
        if c.get("job") in ("Director", "Series Director") and c.get("id") and c.get("name"):
            entry = {"name": c.get("name"), "person_id": c.get("id")}
            if entry not in crew_ids:
                crew_ids.append(entry)

    videos_data = data.get("videos", {}).get("results", [])
    videos = [
        VideoItem(
            id=v.get("id"),
            name=v.get("name"),
            key=v.get("key"),
            site=v.get("site"),
            type=v.get("type")
        ) for v in videos_data if v.get("type") == "Trailer" and v.get("site") == "YouTube"
    ]
    
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
        "language": "pt-BR"
    }
    
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {settings.TMDB_API_KEY}"
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, params=params, timeout=10.0)
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

    episodes = []
    for ep in data.get("episodes", []):
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
        "overview": data.get("overview", ""),
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

    url = f"{settings.TMDB_BASE_URL}/tv/{series_id}/season/{season_number}/episode/{episode_number}"
    params = {
        "language": "pt-BR"
    }
    
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {settings.TMDB_API_KEY}"
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, params=params, timeout=10.0)
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

    response_data = {
        "id": data.get("id"),
        "name": data.get("name", ""),
        "air_date": data.get("air_date"),
        "episode_number": data.get("episode_number", 0),
        "season_number": data.get("season_number", 0),
        "overview": data.get("overview", ""),
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
    
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {settings.TMDB_API_KEY}"
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, params=params, timeout=10.0)
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

    parts = []
    for part in data.get("parts", []):
        parts.append({
            "id": part.get("id"),
            "title": part.get("title", ""),
            "overview": part.get("overview", ""),
            "poster_path": part.get("poster_path"),
            "backdrop_path": part.get("backdrop_path"),
            "release_date": part.get("release_date"),
            "vote_average": part.get("vote_average", 0.0)
        })

    response_data = {
        "id": data.get("id"),
        "name": data.get("name", ""),
        "overview": data.get("overview", ""),
        "poster_path": data.get("poster_path"),
        "backdrop_path": data.get("backdrop_path"),
        "parts": parts
    }

    details_cache.set(cache_key, response_data)

    return CollectionResponse(**response_data)

