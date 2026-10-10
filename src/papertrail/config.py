"""Runtime settings, read from environment variables (prefix PAPERTRAIL_) or a .env file."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PAPERTRAIL_", env_file=".env", extra="ignore")

    database_url: str = "postgresql://papertrail:papertrail@localhost:5432/papertrail"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    llm_model: str = "claude-haiku-4-5-20251001"
    judge_model: str = "claude-sonnet-5-5"  # faithfulness judge; deliberately not the answer model
    max_answer_tokens: int = 600
    # USD per million tokens, for cost estimates only. Update when the model or its price changes.
    llm_input_price_per_mtok: float = 1.0
    llm_output_price_per_mtok: float = 5.0
    judge_input_price_per_mtok: float = 3.0
    judge_output_price_per_mtok: float = 15.0
    retriever: str = "hybrid_header"
    top_k: int = 5
    rerank_candidates: int = 20
    chunk_words: int = 300
    chunk_overlap_words: int = 50


@lru_cache
def get_settings() -> Settings:
    return Settings()
