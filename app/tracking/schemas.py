from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from uuid import UUID
from app.media.schemas import MediaResponse

class UserListItemBase(BaseModel):
    status: Optional[str] = "plan_to_watch"
    rating: Optional[float] = None
    rewatch_count: Optional[int] = 0

class UserListItemCreate(UserListItemBase):
    tmdb_id: int
    media_type: str
    title: str = ""
    poster_path: Optional[str] = None
    backdrop_path: Optional[str] = None
    genres: Optional[List[str]] = []
    directors: Optional[List[str]] = []
    main_cast: Optional[List[str]] = []
    release_date: Optional[str] = None

class UserListItemUpdate(BaseModel):
    status: Optional[str] = None
    rating: Optional[float] = None
    rewatch_count: Optional[int] = None

class NextEpisodeInfo(BaseModel):
    season_number: int
    episode_number: int
    name: Optional[str] = None
    air_date: Optional[str] = None
    is_released: bool = True

class UserListItemResponse(UserListItemBase):
    id: UUID
    user_id: UUID
    media_id: UUID
    last_watched_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime
    media: MediaResponse
    is_up_to_date: Optional[bool] = None
    next_episode: Optional[NextEpisodeInfo] = None

    class Config:
        from_attributes = True

class UserEpisodeProgressCreate(BaseModel):
    season_number: int
    episode_number: int
    rating: Optional[float] = None

class UserEpisodeRatingUpdate(BaseModel):
    rating: Optional[float] = None

class UserEpisodeProgressResponse(BaseModel):
    id: UUID
    user_list_item_id: UUID
    season_number: int
    episode_number: int
    rating: Optional[float] = None
    watched_at: datetime

    class Config:
        from_attributes = True

class MediaRankingItem(BaseModel):
    rank: int
    id: UUID
    media_id: UUID
    tmdb_id: int
    title: str
    media_type: str
    poster_path: Optional[str] = None
    release_date: Optional[str] = None
    runtime: Optional[int] = 0
    genres: List[str] = []
    rating: Optional[float] = None
    rewatch_count: int = 0
    total_time_minutes: int = 0
    episodes_watched: int = 0
    status: str = "completed"
    last_watched_at: Optional[datetime] = None

class MediaRankingResponse(BaseModel):
    items: List[MediaRankingItem]
    total_items: int
    page: int
    page_size: int
    total_pages: int
    media_type: str
    sort_by: str
    period: str
