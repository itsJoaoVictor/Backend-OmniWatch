import httpx
from fastapi import HTTPException
from app.core.config import settings
from app.core.cache import search_cache
from app.search.schemas import SearchMultiResponse, SearchItem
import urllib.parse

async def fetch_search_multi(query: str, page: int = 1) -> SearchMultiResponse:
    if not settings.TMDB_API_KEY:
        raise HTTPException(status_code=500, detail="TMDB API key not configured")

    cache_key = f"search:multi:{query}:{page}"
    cached_data, is_stale = search_cache.get_with_status(cache_key)

    if not is_stale and cached_data:
        return SearchMultiResponse(**cached_data)

    encoded_query = urllib.parse.quote(query)
    url = f"{settings.TMDB_BASE_URL}/search/multi"
    params = {
        "query": query, # httpx handles URL encoding automatically for params dict
        "include_adult": "false",
        "language": "pt-BR",
        "page": page
    }
    
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {settings.TMDB_API_KEY}"
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, params=params, timeout=10.0)
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPError as e:
        if cached_data:
            return SearchMultiResponse(**cached_data)
        raise HTTPException(status_code=502, detail="Failed to fetch search data from upstream provider.")

    normalized_results = []
    for item in data.get("results", []):
        media_type = item.get("media_type")
        
        if media_type == "movie":
            title = item.get("title")
            image_path = item.get("poster_path")
            date = item.get("release_date")
        elif media_type == "tv":
            title = item.get("name")
            image_path = item.get("poster_path")
            date = item.get("first_air_date")
        elif media_type == "person":
            title = item.get("name")
            image_path = item.get("profile_path")
            date = None
        else:
            continue
            
        normalized_results.append(SearchItem(
            id=item.get("id"),
            title=title or "",
            overview=item.get("overview"),
            image_path=image_path,
            media_type=media_type,
            date=date,
            popularity=item.get("popularity")
        ))
        
    response_data = {
        "page": data.get("page", 1),
        "results": [item.model_dump() for item in normalized_results],
        "total_pages": data.get("total_pages", 0),
        "total_results": data.get("total_results", 0)
    }

    search_cache.set(cache_key, response_data)

    return SearchMultiResponse(**response_data)
