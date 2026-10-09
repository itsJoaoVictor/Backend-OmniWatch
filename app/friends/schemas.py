from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import uuid

class UserSummary(BaseModel):
    id: uuid.UUID
    name: str
    username: Optional[str] = None
    email: Optional[str] = None

    model_config = {"from_attributes": True}

class FriendUserResponse(BaseModel):
    id: uuid.UUID
    name: str
    username: Optional[str] = None
    email: Optional[str] = None
    friendship_id: uuid.UUID
    since: datetime

    model_config = {"from_attributes": True}

class FriendRequestItem(BaseModel):
    friendship_id: uuid.UUID
    user: UserSummary
    created_at: datetime

    model_config = {"from_attributes": True}

class FriendRequestsResponse(BaseModel):
    received: List[FriendRequestItem]
    sent: List[FriendRequestItem]

class SendFriendRequest(BaseModel):
    addressee_id: Optional[uuid.UUID] = None
    username: Optional[str] = None

class UserSearchResult(BaseModel):
    id: uuid.UUID
    name: str
    username: Optional[str] = None
    relationship_status: str # "none", "pending_sent", "pending_received", "friends"
    friendship_id: Optional[uuid.UUID] = None

    model_config = {"from_attributes": True}

class FriendFeedItem(BaseModel):
    id: str
    user_id: uuid.UUID
    user_name: str
    username: Optional[str] = None
    media_id: uuid.UUID
    tmdb_id: int
    media_type: str
    title: str
    poster_path: Optional[str] = None
    backdrop_path: Optional[str] = None
    rating: Optional[float] = None
    season_number: Optional[int] = None
    episode_numbers: List[int] = []
    episodes_label: Optional[str] = None
    watched_at: datetime
    action_type: str # "watched_movie" | "watched_episode" | "binge_watched"

    model_config = {"from_attributes": True}

class FriendFeedResponse(BaseModel):
    items: List[FriendFeedItem]
    total: int
    page: int
    limit: int
    has_more: bool
