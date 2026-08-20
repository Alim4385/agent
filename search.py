"""DuckDuckGo üzərindən veb axtarışı — retry/backoff ilə."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List

from duckduckgo_search import DDGS
from duckduckgo_search.exceptions import DuckDuckGoSearchException

from .config import CONFIG
from .exceptions import SearchError
from .logging_setup import log


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    rank: int
    full_text: str = field(default="", repr=False)  # sonradan fetch.py doldurur

    @property
    def display_text(self) -> str:
        """Rerank/LLM-ə ötürüləcək mətn: tam mətn varsa onu, yoxdursa snippet-i istifadə et."""
        return self.full_text.strip() if self.full_text.strip() else self.snippet


def search_web(query: str) -> List[SearchResult]:
    """DuckDuckGo-da axtarış edir. Xəta zamanı exponential backoff ilə retry edir."""
    last_error: Exception | None = None

    for attempt in range(1, CONFIG.search_retries + 2):  # 1 ilkin cəhd + retries
        try:
            with DDGS() as ddgs:
                raw = list(ddgs.text(query, max_results=CONFIG.search_max_results))

            if not raw:
                log.warning("Axtarış nəticə qaytarmadı: %r", query)
                return []

            results = [
                SearchResult(
                    title=r.get("title", "") or "Başlıqsız",
                    url=r.get("href", "") or "",
                    snippet=r.get("body", "") or "",
                    rank=i,
                )
                for i, r in enumerate(raw, start=1)
            ]
            log.info("Axtarış uğurlu: %d nəticə tapıldı (%r)", len(results), query)
            return results

        except DuckDuckGoSearchException as e:
            last_error = e
            log.warning("Axtarış cəhdi %d uğursuz: %s", attempt, e)
        except Exception as e:  # gözlənilməz xətalar da tutulur
            last_error = e
            log.warning("Axtarış zamanı gözlənilməz xəta (cəhd %d): %s", attempt, e)

        if attempt <= CONFIG.search_retries:
            sleep_for = CONFIG.search_backoff_seconds * attempt
            time.sleep(sleep_for)

    raise SearchError(f"Axtarış {CONFIG.search_retries + 1} cəhddən sonra uğursuz oldu: {last_error}")
