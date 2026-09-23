from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class GenreItem(BaseModel):
    id: int
    name: str

class CastItem(BaseModel):
    id: int
    name: str
    character: str
    profile_path: Optional[str] = None

class CrewItem(BaseModel):
    id: int
    name: str
    job: str
    profile_path: Optional[str] = None

class CreditsResponse(BaseModel):
    cast: List[CastItem]
    crew: List[CrewItem]

class VideoItem(BaseModel):
    id: str
    name: str
    key: str
    site: str
    type: str

class WatchProviderItem(BaseModel):
    provider_id: int
    provider_name: str
    logo_path: str

class SimilarItem(BaseModel):
    id: int
    title: str
    poster_path: Optional[str] = None
    vote_average: float

class ProductionCompanyItem(BaseModel):
    id: int
    name: str
    logo_path: Optional[str] = None

class CollectionItem(BaseModel):
    id: int
    name: str
    poster_path: Optional[str] = None
    backdrop_path: Optional[str] = None

class CollectionPartItem(BaseModel):
    id: int
    title: str
    overview: str
    poster_path: Optional[str] = None
    backdrop_path: Optional[str] = None
    release_date: Optional[str] = None
    vote_average: float

class CollectionResponse(BaseModel):
    id: int
    name: str
    overview: str
    poster_path: Optional[str] = None
    backdrop_path: Optional[str] = None
    parts: List[CollectionPartItem]

class MovieDetailsResponse(BaseModel):
    id: int
    title: str
    original_title: str
    original_language: str
    overview: str
    tagline: str
    poster_path: Optional[str] = None
    backdrop_path: Optional[str] = None
    release_date: str
    status: str
    runtime: int
    vote_average: float
    genres: List[GenreItem]
    production_companies: List[ProductionCompanyItem] = []
    belongs_to_collection: Optional[CollectionItem] = None
    credits: CreditsResponse
    videos: List[VideoItem]
    watch_providers: List[WatchProviderItem] = []
    similar: List[SimilarItem] = []
    keywords: List[str] = []
    # Fase 4: person_id TMDB — lista de {"name": str, "person_id": int}
    cast_ids: Optional[List[Dict[str, Any]]] = None
    crew_ids: Optional[List[Dict[str, Any]]] = None

class EpisodeItem(BaseModel):
    id: int
    name: str
    air_date: Optional[str] = None
    episode_number: int
    season_number: int
    overview: Optional[str] = None
    still_path: Optional[str] = None
    vote_average: Optional[float] = None
    runtime: Optional[int] = None

class SeasonDetailsResponse(BaseModel):
    id: int
    name: str
    overview: Optional[str] = None
    season_number: int
    poster_path: Optional[str] = None
    air_date: Optional[str] = None
    episodes: List[EpisodeItem]

class EpisodeDetailsResponse(EpisodeItem):
    pass

class SeasonItem(BaseModel):
    id: int
    name: str
    season_number: int
    episode_count: int
    air_date: Optional[str] = None
    poster_path: Optional[str] = None

class NetworkItem(BaseModel):
    id: int
    name: str
    logo_path: Optional[str] = None

class TvSeriesDetailsResponse(BaseModel):
    id: int
    title: str
    overview: str
    tagline: str
    original_language: str = ""
    poster_path: Optional[str] = None
    backdrop_path: Optional[str] = None
    first_air_date: Optional[str] = None
    status: str
    number_of_seasons: int
    number_of_episodes: int
    vote_average: float
    genres: List[GenreItem]
    episode_run_time: List[int] = []
    networks: List[NetworkItem] = []
    last_episode_to_air: Optional[EpisodeItem] = None
    next_episode_to_air: Optional[EpisodeItem] = None
    seasons: List[SeasonItem]
    credits: CreditsResponse
    videos: List[VideoItem]
    watch_providers: List[WatchProviderItem] = []
    similar: List[SimilarItem] = []
    keywords: List[str] = []
    # Fase 4: person_id TMDB — lista de {"name": str, "person_id": int}
    cast_ids: Optional[List[Dict[str, Any]]] = None
    crew_ids: Optional[List[Dict[str, Any]]] = None
