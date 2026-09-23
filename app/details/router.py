from fastapi import APIRouter, Path
from app.details.schemas import MovieDetailsResponse, TvSeriesDetailsResponse, SeasonDetailsResponse, EpisodeDetailsResponse, CollectionResponse
from app.details.services import fetch_movie_details, fetch_tv_details, fetch_season_details, fetch_episode_details, fetch_collection_details

router = APIRouter(
    tags=["details"]
)

@router.get("/movies/{movie_id}", response_model=MovieDetailsResponse)
async def get_movie_details(
    movie_id: int = Path(..., description="O identificador único do filme")
):
    """
    Obter detalhes de um filme específico.
    """
    return await fetch_movie_details(movie_id)

@router.get("/tv/{series_id}", response_model=TvSeriesDetailsResponse)
async def get_tv_details(
    series_id: int = Path(..., description="O identificador único da série")
):
    """
    Obter detalhes de uma série específica.
    """
    return await fetch_tv_details(series_id)

@router.get("/tv/{series_id}/season/{season_number}", response_model=SeasonDetailsResponse)
async def get_season_details(
    series_id: int = Path(..., description="O identificador único da série"),
    season_number: int = Path(..., description="O número da temporada")
):
    """
    Obter detalhes de uma temporada de uma série específica, incluindo episódios.
    """
    return await fetch_season_details(series_id, season_number)

@router.get("/tv/{series_id}/season/{season_number}/episode/{episode_number}", response_model=EpisodeDetailsResponse)
async def get_episode_details(
    series_id: int = Path(..., description="O identificador único da série"),
    season_number: int = Path(..., description="O número da temporada"),
    episode_number: int = Path(..., description="O número do episódio")
):
    """
    Obter detalhes de um episódio de uma série específica.
    """
    return await fetch_episode_details(series_id, season_number, episode_number)

@router.get("/collections/{collection_id}", response_model=CollectionResponse)
async def get_collection_details(
    collection_id: int = Path(..., description="O identificador único da coleção")
):
    """
    Obter detalhes de uma coleção específica.
    """
    return await fetch_collection_details(collection_id)
