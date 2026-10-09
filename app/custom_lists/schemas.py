from typing import Optional, List
from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field

class CustomListItemCreate(BaseModel):
    tmdb_id: int
    media_type: str = Field(..., description="'movie' or 'tv'")
    title: str
    poster_path: Optional[str] = None
    backdrop_path: Optional[str] = None
    release_date: Optional[str] = None
    runtime: Optional[int] = 0
    note: Optional[str] = None
    position: Optional[int] = None

class CustomListItemUpdate(BaseModel):
    note: Optional[str] = None
    position: Optional[int] = None

class CustomListItemOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    list_id: UUID
    media_id: Optional[UUID] = None
    tmdb_id: int
    media_type: str
    title: str
    poster_path: Optional[str] = None
    backdrop_path: Optional[str] = None
    release_date: Optional[str] = None
    runtime: Optional[int] = 0
    position: int
    note: Optional[str] = None
    created_at: datetime

class CustomListCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=2000)
    is_ranked: bool = False
    cover_backdrop_path: Optional[str] = None
    cover_poster_path: Optional[str] = None

class CustomListUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=2000)
    is_ranked: Optional[bool] = None
    cover_backdrop_path: Optional[str] = None
    cover_poster_path: Optional[str] = None

class CustomListSummaryOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: UUID
    user_name: Optional[str] = None
    title: str
    description: Optional[str] = None
    is_ranked: bool
    cover_backdrop_path: Optional[str] = None
    cover_poster_path: Optional[str] = None
    items_count: int = 0
    preview_posters: List[str] = []
    created_at: datetime
    updated_at: datetime

class CustomListDetailOut(CustomListSummaryOut):
    items: List[CustomListItemOut] = []
    total_runtime_minutes: int = 0
    is_owner: bool = True

class ReorderItemsRequest(BaseModel):
    item_ids: List[UUID]

class MediaListMembership(BaseModel):
    list_id: UUID
    title: str
    contains_media: bool
    item_id: Optional[UUID] = None
