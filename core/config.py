from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    environment: str = "development"
    port: int = 8000
    supabase_url: str | None = None
    supabase_publishable_key: str | None = None
    supabase_anon_key: str | None = None
    supabase_secret_key: str | None = None
    supabase_service_role_key: str | None = None
    gemini_api_key: str | None = None
    openai_api_key: str | None = None
    openrouter_api_key: str | None = None
    groq_api_key: str | None = None
    embedding_model: str = "gemini-embedding-2"
    embedding_dim: int = 768
    chunk_size: int = 800
    chunk_overlap: int = 150

    @property
    def supabase_api_key(self) -> str | None:
        return self.supabase_publishable_key or self.supabase_anon_key

    @property
    def supabase_admin_key(self) -> str | None:
        return self.supabase_secret_key or self.supabase_service_role_key


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


SettingsDep = Annotated[Settings, Depends(get_settings)]
