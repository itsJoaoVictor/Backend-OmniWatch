from pydantic import BaseModel, EmailStr, Field, field_validator
from datetime import datetime
import uuid
import re

import bleach

from typing import Optional

RESERVED_USERNAMES = {
    "admin", "administrator", "root", "system", "omniwatch", "moderator",
    "support", "help", "api", "auth", "login", "register", "null", "undefined", "me"
}

def validate_username_format(v: str) -> str:
    clean = v.strip().lower()
    if len(clean) < 3 or len(clean) > 30:
        raise ValueError("O nome de usuário deve ter entre 3 e 30 caracteres.")
    if not re.match(r"^[a-z0-9._-]+$", clean):
        raise ValueError("O nome de usuário pode conter apenas letras minúsculas, números, pontos (.), hífens (-) e sublinhados (_).")
    if clean in RESERVED_USERNAMES:
        raise ValueError("Este nome de usuário é reservado e não pode ser utilizado.")
    return clean

class UserCreate(BaseModel):
    name: str = Field(..., min_length=2, description="Nome completo ou de exibição")
    email: EmailStr = Field(..., description="Endereço de e-mail (usado para login)")
    password: str = Field(..., description="Senha do usuário")
    username: Optional[str] = Field(None, description="Nome de usuário único opcional")

    @field_validator('name')
    @classmethod
    def sanitize_name(cls, v: str) -> str:
        clean_name = bleach.clean(v, tags=[], attributes={}, strip=True)
        return clean_name.strip()

    @field_validator('email')
    @classmethod
    def normalize_email(cls, v: str) -> str:
        return v.strip().lower()

    @field_validator('username')
    @classmethod
    def validate_opt_username(cls, v: Optional[str]) -> Optional[str]:
        if v is None or not v.strip():
            return None
        return validate_username_format(v)

    @field_validator('password')
    @classmethod
    def validate_password_strength(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError('A senha deve ter no mínimo 8 caracteres.')
        if not re.search(r'[A-Z]', v):
            raise ValueError('A senha deve conter pelo menos 1 letra maiúscula.')
        if not re.search(r'[0-9]', v):
            raise ValueError('A senha deve conter pelo menos 1 número.')
        if not re.search(r'[\W_]', v):
            raise ValueError('A senha deve conter pelo menos 1 caractere especial.')
        return v

class UsernameUpdate(BaseModel):
    username: str = Field(..., min_length=3, max_length=30, description="Novo nome de usuário")

    @field_validator('username')
    @classmethod
    def validate_username(cls, v: str) -> str:
        return validate_username_format(v)

class UsernameCheckResponse(BaseModel):
    available: bool
    message: str

class UserResponse(BaseModel):
    id: uuid.UUID
    name: str
    email: EmailStr
    username: Optional[str] = None
    role: str
    created_at: datetime

    model_config = {'from_attributes': True}

