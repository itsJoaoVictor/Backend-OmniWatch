from typing import List, Optional
from pydantic import BaseModel

class SearchItem(BaseModel):
    id: int
    title: str
    overview: Optional[str] = None
    image_path: Optional[str] = None
    media_type: str
    date: Optional[str] = None
    popularity: Optional[float] = None

class SearchMultiResponse(BaseModel):
    page: int
    results: List[SearchItem]
    total_pages: int
    total_results: int
