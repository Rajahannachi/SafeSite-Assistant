"""Central configuration, read from environment variables / the .env file.

Every tunable of the pipeline lives here so that no module hard-codes a model
name, a path or a secret.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

ROOT = Path(__file__).resolve().parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM (OpenRouter) -------------------------------------------------
    openrouter_api_key: SecretStr | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "nvidia/nemotron-3.5-lightning:free"
    # Comma-separated list in the env var, tried in order when the primary fails.
    llm_fallback_models: Annotated[list[str], NoDecode] = Field(
        default=["google/gemma-4-31b-it:free", "poolside/laguna-s-2.1:free"]
    )
    llm_temperature: float = 0.0
    llm_timeout_s: float = 45.0
    llm_max_attempts: int = 3  # attempts per model (1 call + 2 retries)

    # --- Embeddings & vector store ---------------------------------------
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    data_dir: Path = ROOT / "data"
    vectorstore_dir: Path = ROOT / "vectorstore"
    collection_name: str = "safesite"

    # --- Chunking ---------------------------------------------------------
    chunk_size: int = 800
    chunk_overlap: int = 120

    # --- Retrieval --------------------------------------------------------
    top_k: int = 4
    search_type: Literal["similarity", "mmr", "hybrid"] = "similarity"
    mmr_fetch_k: int = 20
    mmr_lambda: float = 0.6
    rerank: bool = False
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_candidates: int = 12

    # --- Conversation memory ---------------------------------------------
    max_history_turns: int = 4

    @field_validator("llm_fallback_models", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        if isinstance(value, str):
            return [m.strip() for m in value.split(",") if m.strip()]
        return value

    @field_validator("llm_model", "llm_fallback_models")
    @classmethod
    def _only_free_models(cls, value: str | list[str]) -> str | list[str]:
        """The exercise forbids paid APIs: refuse any model id without ':free'."""
        for model in [value] if isinstance(value, str) else value:
            if not model.endswith(":free"):
                raise ValueError(f"Model '{model}' is not a free OpenRouter model (must end with ':free').")
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
