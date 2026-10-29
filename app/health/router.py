from fastapi import APIRouter, Depends, Request
from app.admin.dependencies import get_current_admin_user
from app.core.rate_limit import limiter
from app.health.schemas import HealthSummaryResponse
from app.health.services import get_system_health_summary
from app.users.models import User

router = APIRouter()


@router.get("", response_model=HealthSummaryResponse)
@limiter.limit("30/minute")
async def get_health_status(
    request: Request,
    admin_user: User = Depends(get_current_admin_user),
):
    """
    Endpoint restrito ao administrador (jvgf98@gmail.com) para Health Check e Status Page do OmniWatch.
    Retorna a saúde em tempo real do banco de dados, cache, integrações externas (TMDB),
    motor de IA e workers em background, além do histórico de disponibilidade.
    """
    return await get_system_health_summary()
