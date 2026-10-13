from pydantic import BaseModel, Field
from typing import List, Optional

class PersonCastItem(BaseModel):
    id: int
    title: str
    character: Optional[str] = None
    job: Optional[str] = None
    department: Optional[str] = None
    poster_path: Optional[str] = None
    media_type: str # "movie" or "tv"
    release_date: Optional[str] = None
    popularity: float = 0.0
    vote_count: int = 0
    vote_average: float = 0.0
    order: Optional[int] = None
    genre_ids: List[int] = Field(default_factory=list)
    episode_count: Optional[int] = None
    suggestion_score: Optional[float] = None
    match_score: Optional[float] = None
    role_label: Optional[str] = None

class PersonDetailsResponse(BaseModel):
    id: int
    name: str
    biography: str
    birthday: Optional[str] = None
    place_of_birth: Optional[str] = None
    profile_path: Optional[str] = None
    known_for_department: str
    combined_credits: List[PersonCastItem]

class PersonDiscoverResponse(BaseModel):
    items: List[PersonCastItem]
    is_personalized: bool = False

