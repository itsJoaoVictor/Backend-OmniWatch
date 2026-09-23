from fastapi import APIRouter
from app.person.schemas import PersonDetailsResponse
from app.person.services import fetch_person_details

router = APIRouter()

@router.get("/{person_id}", response_model=PersonDetailsResponse)
async def get_person_details(person_id: int):
    """
    Get details for a person (actor, director, etc.), including their credits.
    """
    return await fetch_person_details(person_id)
