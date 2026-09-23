import httpx
from fastapi import HTTPException
from app.core.config import settings
from app.core.cache import details_cache
from app.person.schemas import PersonDetailsResponse, PersonCastItem

async def fetch_person_details(person_id: int) -> PersonDetailsResponse:
    if not settings.TMDB_API_KEY:
        raise HTTPException(status_code=500, detail="TMDB API key not configured")

    cache_key = f"details:person:{person_id}"
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
    except httpx.RequestError as e:
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
            "popularity": c.get("popularity", 0.0),
            "vote_count": c.get("vote_count", 0),
            "order": c.get("order")
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
            "popularity": c.get("popularity", 0.0),
            "vote_count": c.get("vote_count", 0)
        })
        
    # Sort by popularity to show "Known For" best items, or by release date. 
    # Let's just sort by popularity descending to get the best roles first.
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
