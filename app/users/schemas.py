from pydantic import BaseModel, EmailStr, Field, field_validator
from datetime import datetime
import uuid
import re

import bleach

class UserCreate(BaseModel):
    name: str = Field(..., min_length=2, description="Nome completo ou de exibição")
    email: EmailStr = Field(..., description="Endereço de e-mail (usado para login)")
    password: str = Field(..., description="Senha do usuário")

    @field_validator('name')
    @classmethod
    def sanitize_name(cls, v: str) -> str:
        clean_name = bleach.clean(v, tags=[], attributes={}, strip=True)
        return clean_name.strip()

    @field_validator('email')
    @classmethod
    def normalize_email(cls, v: str) -> str:
        return v.strip().lower()

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

class UserResponse(BaseModel):
    id: uuid.UUID
    name: str
    email: EmailStr
    role: str
    created_at: datetime

    model_config = {'from_attributes': True}
