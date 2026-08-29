import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    TMDB_API_KEY: str = os.getenv("TMDB_API_KEY", "")
    TMDB_BASE_URL: str = "https://api.themoviedb.org/3"

settings = Settings()
