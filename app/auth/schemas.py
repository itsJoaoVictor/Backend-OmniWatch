from pydantic import BaseModel, Field

class LoginRequest(BaseModel):
    email: str = Field(..., description="E-mail ou nome de usuário")
    password: str
    remember_me: bool = False

class TokenResponse(BaseModel):
    message: str
    user: dict
