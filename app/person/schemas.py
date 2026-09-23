from pydantic import BaseModel
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
    vote_count: int = 0
    order: Optional[int] = None

class PersonDetailsResponse(BaseModel):
    id: int
    name: str
    biography: str
    birthday: Optional[str] = None
    place_of_birth: Optional[str] = None
    profile_path: Optional[str] = None
    known_for_department: str
    combined_credits: List[PersonCastItem]
