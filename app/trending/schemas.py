from pydantic import BaseModel
from typing import List, Optional

class TrendingItem(BaseModel):
    id: int
    title: Optional[str] = None
    name: Optional[str] = None
    overview: str
    poster_path: Optional[str] = None
    media_type: str
    popularity: float
    vote_average: float
    release_date: Optional[str] = None
    first_air_date: Optional[str] = None

class TrendingResponse(BaseModel):
    page: int
    results: List[TrendingItem]
    total_pages: int
    total_results: int
