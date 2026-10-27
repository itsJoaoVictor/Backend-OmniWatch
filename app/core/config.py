import os
import time
from datetime import timezone, timedelta
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

load_dotenv()

# Configuração do fuso horário padrão do servidor (Brasil - Horário de Brasília)
SERVER_TIMEZONE = "America/Sao_Paulo"
BRASILIA_TZ = timezone(timedelta(hours=-3))

os.environ["TZ"] = SERVER_TIMEZONE
if hasattr(time, "tzset"):
    time.tzset()

class Settings:
    DEBUG: bool = os.getenv("DEBUG", "True").lower() in ("true", "1", "t")
    SERVER_TIMEZONE: str = SERVER_TIMEZONE
    TMDB_API_KEY: str = os.getenv("TMDB_API_KEY", "")
    TMDB_BASE_URL: str = "https://api.themoviedb.org/3"
    
    @property
    def tz(self):
        try:
            return ZoneInfo(self.SERVER_TIMEZONE)
        except Exception:
            return BRASILIA_TZ
    
    # Fase 6: Embeddings Semânticos via OpenRouter / OpenAI API
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY", "")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "deepseek/deepseek-v4-flash-0731")
    OPENAI_MODEL_Embedding: str = os.getenv("OPENAI_MODEL_Embedding", "qwen/qwen3-embedding-8b")

settings = Settings()
