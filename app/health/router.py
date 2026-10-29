from fastapi import APIRouter, Request
from app.core.rate_limit import limiter
from app.health.schemas import HealthSummaryResponse
from app.health.services import get_system_health_summary

router = APIRouter()


@router.get("", response_model=HealthSummaryResponse)
@limiter.limit("30/minute")
async def get_health_status(request: Request):
    """
    Endpoint público de Health Check e Status Page do OmniWatch.
    Retorna a saúde em tempo real do banco de dados, cache, integrações externas (TMDB),
    motor de IA e workers em background, além do histórico de disponibilidade.
    """
    return await get_system_health_summary()
