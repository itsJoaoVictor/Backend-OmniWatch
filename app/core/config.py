import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    DEBUG: bool = os.getenv("DEBUG", "True").lower() in ("true", "1", "t")
    TMDB_API_KEY: str = os.getenv("TMDB_API_KEY", "")
    TMDB_BASE_URL: str = "https://api.themoviedb.org/3"
    
    # Fase 6: Embeddings Semânticos via OpenRouter / OpenAI API
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY", "")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "deepseek/deepseek-v4-flash-0731")
    OPENAI_MODEL_Embedding: str = os.getenv("OPENAI_MODEL_Embedding", "qwen/qwen3-embedding-8b")

settings = Settings()
