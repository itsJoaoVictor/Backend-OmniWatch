from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from app.users.router import router as users_router
from app.auth.router import router as auth_router

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="OmniWatch API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS Estrito (apenas o Frontend autorizado pode acessar)
# Para desenvolvimento local vamos liberar os hosts padrões do Next.js
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
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

@app.get("/")
def root():
    return {"message": "Welcome to OmniWatch API"}
