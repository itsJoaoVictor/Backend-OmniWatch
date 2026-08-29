from fastapi import APIRouter, Query
from app.trending.schemas import TrendingResponse
from app.trending.services import get_trending_week

router = APIRouter()

@router.get("", response_model=TrendingResponse)
async def get_trending(
    language: str = Query("pt-BR", description="Código de linguagem no padrão ISO-639-1-ISO-3166-1"),
    page: int = Query(1, description="Número da página para requisições de paginação")
):
    data = await get_trending_week(language, page)
    return data
