"""Runtime settings, read from environment variables (prefix PAPERTRAIL_) or a .env file."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PAPERTRAIL_", env_file=".env", extra="ignore")

    database_url: str = "postgresql://papertrail:papertrail@localhost:5432/papertrail"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    llm_model: str = "claude-haiku-4-5-20251001"
    top_k: int = 5
    rerank_candidates: int = 20
    chunk_tokens: int = 400
    chunk_overlap: int = 50


@lru_cache
def get_settings() -> Settings:
    return Settings()
