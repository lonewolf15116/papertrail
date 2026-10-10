"""Runtime settings, read from environment variables (prefix PAPERTRAIL_) or a .env file."""

from dataclasses import dataclass
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


@dataclass(frozen=True)
class LLMSpec:
    """Everything needed to build one LLM client and estimate what its calls cost."""

    provider: str
    model: str
    input_price_per_mtok: float  # USD per million tokens, for cost estimates only
    output_price_per_mtok: float
    key_env: str  # name of the environment variable that holds the provider's API key


@dataclass(frozen=True)
class ProviderDefaults:
    key_env: str
    answer: tuple[str, float, float]  # model, input and output USD per million tokens
    judge: tuple[str, float, float]


# Models and prices change often: override them with the PAPERTRAIL_LLM_* and PAPERTRAIL_JUDGE_*
# variables rather than editing this table. The judge is deliberately a different model from the
# answerer so the evaluation does not mark its own homework.
PROVIDERS: dict[str, ProviderDefaults] = {
    "anthropic": ProviderDefaults(
        key_env="ANTHROPIC_API_KEY",
        answer=("claude-haiku-4-5-20251001", 1.0, 5.0),
        judge=("claude-sonnet-5-5", 3.0, 15.0),
    ),
    "openai": ProviderDefaults(
        key_env="OPENAI_API_KEY",
        answer=("gpt-4.1-mini", 0.4, 1.6),
        judge=("gpt-4.1", 2.0, 8.0),
    ),
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PAPERTRAIL_", env_file=".env", extra="ignore")

    database_url: str = "postgresql://papertrail:papertrail@localhost:5432/papertrail"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # LLM backend: "anthropic" or "openai". The judge defaults to the same provider.
    llm_provider: str = "anthropic"
    judge_provider: str = ""
    # Empty / None means "use the provider's default" from PROVIDERS above.
    llm_model: str = ""
    judge_model: str = ""
    llm_input_price_per_mtok: float | None = None
    llm_output_price_per_mtok: float | None = None
    judge_input_price_per_mtok: float | None = None
    judge_output_price_per_mtok: float | None = None
    max_answer_tokens: int = 600
    retriever: str = "hybrid_header"
    top_k: int = 5
    rerank_candidates: int = 20
    chunk_words: int = 300
    chunk_overlap_words: int = 50

    def _spec(
        self,
        provider: str,
        role: str,
        model: str,
        price_in: float | None,
        price_out: float | None,
    ) -> LLMSpec:
        if provider not in PROVIDERS:
            raise ValueError(
                f"unknown LLM provider {provider!r}; choose one of {sorted(PROVIDERS)}"
            )
        defaults = PROVIDERS[provider]
        d_model, d_in, d_out = defaults.answer if role == "answer" else defaults.judge
        return LLMSpec(
            provider=provider,
            model=model or d_model,
            input_price_per_mtok=d_in if price_in is None else price_in,
            output_price_per_mtok=d_out if price_out is None else price_out,
            key_env=defaults.key_env,
        )

    def answer_llm(self) -> LLMSpec:
        return self._spec(
            self.llm_provider,
            "answer",
            self.llm_model,
            self.llm_input_price_per_mtok,
            self.llm_output_price_per_mtok,
        )

    def judge_llm(self) -> LLMSpec:
        return self._spec(
            self.judge_provider or self.llm_provider,
            "judge",
            self.judge_model,
            self.judge_input_price_per_mtok,
            self.judge_output_price_per_mtok,
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
