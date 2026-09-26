import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.users.router import router as users_router
from app.auth.router import router as auth_router
from app.trending.router import router as trending_router
from app.search.router import router as search_router
from app.details.router import router as details_router
from app.tracking.router import router as tracking_router
from app.media.router import router as media_router
from app.person.router import router as person_router
from app.calendar.router import router as calendar_router
from app.recommendation.router import router as recommendation_router
from app.recommendation.ranker import run_ranker_training_loop
from app.notifications.router import router as notifications_router
from app.media_collections.router import router as collections_router
from app.media_collections.services import run_collection_sync_loop
from app.calendar.sync import run_calendar_sync_loop, migrate_existing_future_media
from app.images.router import router as images_router

from app.core.rate_limit import limiter

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Sincroniza e migra títulos futuros para 'upcoming' automaticamente no startup
    await migrate_existing_future_media()

    # Start background tasks
    sync_task = asyncio.create_task(run_collection_sync_loop(interval_hours=24))
    ranker_task = asyncio.create_task(run_ranker_training_loop(interval_hours=24))
    cal_sync_task = asyncio.create_task(run_calendar_sync_loop(interval_hours=24))
    yield
    sync_task.cancel()
    ranker_task.cancel()
    cal_sync_task.cancel()
    try:
        await asyncio.gather(sync_task, ranker_task, cal_sync_task, return_exceptions=True)
    except Exception:
        pass

app = FastAPI(title="OmniWatch API", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://front-end-omni-watch-livid.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix="/api")
app.include_router(users_router, prefix="/api/users", tags=["users"])
app.include_router(trending_router, prefix="/api/trending", tags=["trending"])
app.include_router(search_router, prefix="/api/search", tags=["search"])
app.include_router(collections_router, prefix="/api")
app.include_router(details_router, prefix="/api")
app.include_router(tracking_router, prefix="/api", tags=["tracking"])
app.include_router(media_router, prefix="/api/media", tags=["media"])
app.include_router(person_router, prefix="/api/person", tags=["person"])
app.include_router(calendar_router, prefix="/api")
app.include_router(notifications_router, prefix="/api")
app.include_router(recommendation_router, prefix="/api/recommendations", tags=["recommendations"])
app.include_router(images_router, prefix="/api/images", tags=["images"])

@app.get("/")
def root():
    return {"message": "Welcome to OmniWatch API"}
