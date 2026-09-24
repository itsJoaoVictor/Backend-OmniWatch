from datetime import datetime, timezone
from typing import Optional

def is_date_released(date_str: Optional[str]) -> bool:
    """
    Verifica se uma data (no formato 'YYYY-MM-DD') é válida e já ocorreu (ou seja, <= hoje em UTC).
    Retorna False caso a data seja nula, vazia, inválida ou futura.
    """
    if not date_str or not isinstance(date_str, str):
        return False
    
    clean_date = date_str.strip()
    if not clean_date:
        return False
        
    try:
        release_date = datetime.strptime(clean_date[:10], "%Y-%m-%d").date()
        today_date = datetime.now(timezone.utc).date()
        return release_date <= today_date
    except (ValueError, TypeError):
        return False
