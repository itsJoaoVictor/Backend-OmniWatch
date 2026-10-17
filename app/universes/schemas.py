from pydantic import BaseModel, UUID4
from typing import List, Optional
from datetime import datetime


class IgnoredItemPayload(BaseModel):
    tmdb_id: int
    media_type: str = "movie"


class UniverseFollowRequest(BaseModel):
    auto_add: bool = True
    ignored_items: List[IgnoredItemPayload] = []


class ToggleIgnoreItemRequest(BaseModel):
    tmdb_id: int
    media_type: str = "movie"
    is_ignored: bool


class UpdateUniverseSettingsRequest(BaseModel):
    auto_add: bool


class UniverseItemResponse(BaseModel):
    id: UUID4
    tmdb_id: int
    media_type: str = "movie"
    title: str
    release_date: Optional[str] = None
    poster_path: Optional[str] = None
    backdrop_path: Optional[str] = None
    collection_name: Optional[str] = None
    media_id: Optional[UUID4] = None
    status: Optional[str] = None
    rating: Optional[float] = None
    is_ignored: bool = False
    is_recap: bool = False

    class Config:
        from_attributes = True


class UniverseFollowResponse(BaseModel):
    success: bool
    message: str
    universe_id: UUID4
    universe_key: str
    name: str
    items_added: int
    items_already_in_list: int
    ignored_count: int = 0


class UniverseStatusResponse(BaseModel):
    is_following: bool
    auto_add: bool = True
    total_items: int = 0
    watched_items: int = 0
    movies_count: int = 0
    tv_count: int = 0
    active_items: int = 0
    ignored_items_count: int = 0
    universe_id: Optional[UUID4] = None


class UserUniverseDetailResponse(BaseModel):
    id: UUID4
    universe_key: str
    name: str
    overview: Optional[str] = None
    poster_path: Optional[str] = None
    backdrop_path: Optional[str] = None
    total_items: int
    watched_items: int
    active_items: int = 0
    ignored_items_count: int = 0
    movies_count: int = 0
    tv_count: int = 0
    collections_count: int = 0
    completion_percentage: float
    personal_completion_percentage: float = 0.0
    canonical_completion_percentage: float = 0.0
    auto_add: bool = True
    user_average_rating: Optional[float] = None
    rated_items_count: int = 0
    items: List[UniverseItemResponse] = []
    created_at: datetime

    class Config:
        from_attributes = True


class UniverseSuggestionResponse(BaseModel):
    id: UUID4
    universe_key: str
    name: str
    overview: Optional[str] = None
    poster_path: Optional[str] = None
    backdrop_path: Optional[str] = None
    total_items: int
    movies_count: int = 0
    tv_count: int = 0
    collections_count: int = 0
    items_in_list: int
    matched_titles: List[str] = []
    is_dismissed: bool = False
    is_reactivated: bool = False
    reactivation_reason: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True
