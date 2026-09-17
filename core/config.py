from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Local-mode settings. Embedding model is pinned; generate model is user-selected."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ollama_endpoint: str = "http://localhost:11434"
    ollama_model: str = "gemma4"
    ollama_embedding_model: str = "embeddinggemma"
    embedding_dim: int = 768
    ollama_embed_keep_alive: str = "30m"
    sqlite_path: str = "data/writr_local.db"
    usearch_path: str = "data/vectors.usearch"
    reference_chunk_size: int = 1200
    reference_chunk_overlap: int = 200


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
