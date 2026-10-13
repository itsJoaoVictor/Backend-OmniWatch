from pydantic import BaseModel, EmailStr, Field, field_validator
import re

class LoginRequest(BaseModel):
    email: str = Field(..., description="E-mail ou nome de usuário")
    password: str
    remember_me: bool = False

class TokenResponse(BaseModel):
    message: str
    user: dict

class ForgotPasswordRequest(BaseModel):
    email: EmailStr = Field(..., description="Endereço de e-mail cadastrado")
    username: str = Field(..., min_length=1, description="Nome de usuário cadastrado")
    new_password: str = Field(..., min_length=8, description="Nova senha do usuário")

    @field_validator("email")
    @classmethod
    def normalize_email(cls, v: str) -> str:
        return v.strip().lower()

    @field_validator("username")
    @classmethod
    def normalize_username(cls, v: str) -> str:
        return v.strip().lower()

    @field_validator("new_password")
    @classmethod
    def validate_password_strength(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("A senha deve ter no mínimo 8 caracteres.")
        if not re.search(r"[A-Z]", v):
            raise ValueError("A senha deve conter pelo menos 1 letra maiúscula.")
        if not re.search(r"[0-9]", v):
            raise ValueError("A senha deve conter pelo menos 1 número.")
        if not re.search(r"[\W_]", v):
            raise ValueError("A senha deve conter pelo menos 1 caractere especial.")
        return v

