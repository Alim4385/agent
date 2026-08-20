"""Mərkəzi konfiqurasiya. Bütün dəyərlər ətraf mühit dəyişənləri ilə override oluna bilər."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Config:
    # LLM server (OpenAI-uyğun API, məs. llama.cpp server, vLLM, LM Studio)
    llm_base_url: str = _env_str("LLM_BASE_URL", "http://127.0.0.1:8080/v1")
    chat_endpoint: str = "/chat/completions"
    embeddings_endpoint: str = "/embeddings"

    model_name: str = _env_str("LLM_MODEL", "local-model")
    temperature: float = _env_float("LLM_TEMPERATURE", 0.7)
    request_timeout: int = _env_int("LLM_TIMEOUT", 120)

    # Axtarış
    search_max_results: int = _env_int("SEARCH_MAX_RESULTS", 6)
    search_retries: int = _env_int("SEARCH_RETRIES", 2)
    search_backoff_seconds: float = _env_float("SEARCH_BACKOFF", 1.5)

    # Səhifə çəkmə (full-text fetch)
    fetch_enabled: bool = _env_str("FETCH_ENABLED", "true").lower() == "true"
    fetch_timeout: int = _env_int("FETCH_TIMEOUT", 8)
    fetch_max_chars: int = _env_int("FETCH_MAX_CHARS", 4000)
    fetch_concurrency: int = _env_int("FETCH_CONCURRENCY", 5)

    # Reranking
    rerank_top_k: int = _env_int("RERANK_TOP_K", 4)
    embeddings_enabled: bool = _env_str("EMBEDDINGS_ENABLED", "true").lower() == "true"

    # Söhbət
    max_history_turns: int = _env_int("MAX_HISTORY_TURNS", 6)  # user+assistant cütü

    # Streaming
    stream: bool = _env_str("LLM_STREAM", "true").lower() == "true"

    # Logging
    log_level: str = _env_str("LOG_LEVEL", "INFO")
    user_agent: str = "Mozilla/5.0 (compatible; WebAgent/1.0; +local-research-tool)"

    @property
    def chat_url(self) -> str:
        return self.llm_base_url.rstrip("/") + self.chat_endpoint

    @property
    def embeddings_url(self) -> str:
        return self.llm_base_url.rstrip("/") + self.embeddings_endpoint


CONFIG = Config()
