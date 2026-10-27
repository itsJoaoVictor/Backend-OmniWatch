from datetime import datetime, date, timezone, timedelta
from typing import Optional
from app.core.config import settings, BRASILIA_TZ

def get_server_tz():
    """Retorna o ZoneInfo oficial do servidor (America/Sao_Paulo) com fallback para UTC-3."""
    try:
        return settings.tz
    except Exception:
        return BRASILIA_TZ

def get_server_now() -> datetime:
    """Retorna o datetime atual considerando o fuso horário oficial do servidor (Brasil - Horário de Brasília)."""
    try:
        return datetime.now(get_server_tz())
    except Exception:
        return datetime.now(timezone.utc)

def get_server_today() -> date:
    """Retorna a data civil atual do servidor no Brasil (YYYY-MM-DD)."""
    try:
        return datetime.now(get_server_tz()).date()
    except Exception:
        return datetime.now(timezone.utc).date()

def is_date_released(date_str: Optional[str]) -> bool:
    """
    Verifica se uma data (no formato 'YYYY-MM-DD') é válida e já ocorreu (ou seja, <= hoje no horário do servidor).
    Retorna False caso a data seja nula, vazia, inválida ou futura.
    """
    if not date_str or not isinstance(date_str, str):
        return False
    
    clean_date = date_str.strip()
    if not clean_date:
        return False
        
    try:
        release_date = datetime.strptime(clean_date[:10], "%Y-%m-%d").date()
        today_date = get_server_today()
        return release_date <= today_date
    except Exception:
        return False
