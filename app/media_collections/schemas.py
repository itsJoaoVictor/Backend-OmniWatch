from pydantic import BaseModel, UUID4
from typing import List, Optional
from datetime import datetime

class CollectionItemResponse(BaseModel):
    id: UUID4
    tmdb_id: int
    title: str
    release_date: Optional[str] = None
    poster_path: Optional[str] = None
    backdrop_path: Optional[str] = None
    media_id: Optional[UUID4] = None
    status: Optional[str] = None
    rating: Optional[float] = None

    class Config:
        from_attributes = True

class CollectionFollowResponse(BaseModel):
    success: bool
    message: str
    collection_id: UUID4
    tmdb_id: int
    name: str
    movies_added: int
    movies_already_in_list: int

class CollectionStatusResponse(BaseModel):
    is_following: bool
    auto_add: bool = True
    total_movies: int = 0
    watched_movies: int = 0
    collection_id: Optional[UUID4] = None

class UserCollectionDetailResponse(BaseModel):
    id: UUID4
    tmdb_id: int
    name: str
    overview: Optional[str] = None
    poster_path: Optional[str] = None
    backdrop_path: Optional[str] = None
    total_movies: int
    watched_movies: int
    completion_percentage: float
    items: List[CollectionItemResponse] = []
    created_at: datetime

    class Config:
        from_attributes = True
