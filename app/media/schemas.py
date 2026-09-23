from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from uuid import UUID

class MediaBase(BaseModel):
    tmdb_id: int
    media_type: str
    title: str
    poster_path: Optional[str] = None
    backdrop_path: Optional[str] = None
    original_language: Optional[str] = None
    # Lista de {"id": int, "name": str} (ou strings)
    genres: Optional[List[Union[Dict[str, Any], str]]] = []
    directors: Optional[List[str]] = []
    main_cast: Optional[List[str]] = []
    keywords: Optional[List[str]] = []
    # Fase 4: person_id TMDB — lista de {"name": str, "person_id": int}
    cast_ids: Optional[List[Dict[str, Any]]] = None
    crew_ids: Optional[List[Dict[str, Any]]] = None
    # Fase 6: embedding semântico (lista de floats)
    embedding: Optional[List[float]] = None
    release_date: Optional[str] = None
    runtime: Optional[int] = 0

class MediaCreate(MediaBase):
    pass

class MediaResponse(BaseModel):
    id: UUID
    tmdb_id: int
    media_type: str
    title: str
    poster_path: Optional[str] = None
    backdrop_path: Optional[str] = None
    original_language: Optional[str] = None
    genres: Optional[List[Union[Dict[str, Any], str]]] = []
    directors: Optional[List[str]] = []
    main_cast: Optional[List[str]] = []
    release_date: Optional[str] = None
    runtime: Optional[int] = 0
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
