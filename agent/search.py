"""DuckDuckGo üzərindən veb axtarışı — retry/backoff ilə."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List

try:  # yeni paket adı (ddgs >= 9) — köhnə duckduckgo-search ilə geriyə uyğunluq
    from ddgs import DDGS
except ImportError:  # pragma: no cover - quraşdırılmış versiyadan asılıdır
    from duckduckgo_search import DDGS

try:
    from ddgs.exceptions import DDGSException as _BaseDDGError
except Exception:  # noqa: BLE001
    try:
        from duckduckgo_search.exceptions import (
            DuckDuckGoSearchException as _BaseDDGError,
        )
    except Exception:  # noqa: BLE001
        _BaseDDGError = Exception

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


def _field(item, *names: str) -> str:
    """Nəticə dict (köhnə API) və ya obyekt (yeni `ddgs` API) ola bilər."""
    for name in names:
        if isinstance(item, dict):
            value = item.get(name)
        else:
            value = getattr(item, name, None)
        if value:
            return str(value)
    return ""


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
                    title=_field(r, "title") or "Başlıqsız",
                    url=_field(r, "href") or _field(r, "url"),
                    snippet=_field(r, "body") or _field(r, "snippet"),
                    rank=i,
                )
                for i, r in enumerate(raw, start=1)
            ]
            results = [r for r in results if r.url]
            # Eyni link bir neçə dəfə gələ bilər — təkrarları çıxarırıq.
            unique: List[SearchResult] = []
            seen = set()
            for r in results:
                if r.url in seen:
                    continue
                seen.add(r.url)
                r.rank = len(unique) + 1
                unique.append(r)

            log.info("Axtarış uğurlu: %d nəticə tapıldı (%r)", len(unique), query)
            return unique

        except _BaseDDGError as e:
            last_error = e
            log.warning("Axtarış cəhdi %d uğursuz: %s", attempt, e)
        except Exception as e:  # gözlənilməz xətalar da tutulur
            last_error = e
            log.warning("Axtarış zamanı gözlənilməz xəta (cəhd %d): %s", attempt, e)

        if attempt <= CONFIG.search_retries:
            sleep_for = CONFIG.search_backoff_seconds * attempt
            time.sleep(sleep_for)

    raise SearchError(f"Axtarış {CONFIG.search_retries + 1} cəhddən sonra uğursuz oldu: {last_error}")
