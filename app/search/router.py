from fastapi import APIRouter, Query
from app.search.schemas import SearchMultiResponse
from app.search.services import fetch_search_multi

router = APIRouter()

@router.get("/multi", response_model=SearchMultiResponse, summary="Busca Multi", description="Busca filmes, séries e pessoas simultaneamente.")
async def search_multi(
    query: str = Query(..., description="Termo de busca em texto"),
    page: int = Query(1, description="Número da página para paginação", ge=1)
):
    return await fetch_search_multi(query=query, page=page)
