from fastapi import APIRouter, Path, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.core.database import get_db
from app.details.schemas import (
    MovieDetailsResponse,
    TvSeriesDetailsResponse,
    SeasonDetailsResponse,
    EpisodeDetailsResponse,
    CollectionResponse,
    UniverseRef,
)
from app.details.services import fetch_movie_details, fetch_tv_details, fetch_season_details, fetch_episode_details, fetch_collection_details
from app.media.models import Media
from app.universes.models import UniverseItem, Universe

router = APIRouter(
    tags=["details"]
)

@router.get("/movies/{movie_id}", response_model=MovieDetailsResponse)
async def get_movie_details(
    movie_id: int = Path(..., description="O identificador único do filme"),
    db: AsyncSession = Depends(get_db),
):
    """
    Obter detalhes de um filme específico.
    """
    details = await fetch_movie_details(movie_id)
    # Enriquece com universo se a mídia já foi associada
    stmt = select(Media.universe_key, Media.universe_name).where(Media.tmdb_id == movie_id, Media.media_type == "movie")
    res = await db.execute(stmt)
    row = res.first()
    if row and row[0] and row[0] not in ("none", ""):
        details.universe = UniverseRef(universe_key=row[0], name=row[1] or row[0])
    else:
        # Fallback para UniverseItem se já indexado
        u_stmt = select(Universe.universe_key, Universe.name).join(UniverseItem, UniverseItem.universe_id == Universe.id).where(UniverseItem.tmdb_id == movie_id, UniverseItem.media_type == "movie")
        u_res = await db.execute(u_stmt)
        u_row = u_res.first()
        if u_row:
            details.universe = UniverseRef(universe_key=u_row[0], name=u_row[1])
    return details

@router.get("/tv/{series_id}", response_model=TvSeriesDetailsResponse)
async def get_tv_details(
    series_id: int = Path(..., description="O identificador único da série"),
    db: AsyncSession = Depends(get_db),
):
    """
    Obter detalhes de uma série específica.
    """
    details = await fetch_tv_details(series_id)
    stmt = select(Media.universe_key, Media.universe_name).where(Media.tmdb_id == series_id, Media.media_type == "tv")
    res = await db.execute(stmt)
    row = res.first()
    if row and row[0] and row[0] not in ("none", ""):
        details.universe = UniverseRef(universe_key=row[0], name=row[1] or row[0])
    else:
        u_stmt = select(Universe.universe_key, Universe.name).join(UniverseItem, UniverseItem.universe_id == Universe.id).where(UniverseItem.tmdb_id == series_id, UniverseItem.media_type == "tv")
        u_res = await db.execute(u_stmt)
        u_row = u_res.first()
        if u_row:
            details.universe = UniverseRef(universe_key=u_row[0], name=u_row[1])
    return details

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
