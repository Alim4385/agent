"""Axtarış nəticələrindəki linklərdən tam səhifə mətnini paralel şəkildə çəkir.

Snippet-lər çox vaxt kifayət qədər kontekst vermir. Bu modul əsl səhifəyə girib
reklam/menyu/footer kimi 'zibil' hissələri təmizləyib əsl məzmunu çıxarır.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import requests

try:
    import trafilatura
    _HAS_TRAFILATURA = True
except ImportError:  # kitabxana quraşdırılmayıbsa, snippet-lə davam edəcəyik
    _HAS_TRAFILATURA = False

from .config import CONFIG
from .logging_setup import log
from .search import SearchResult


def _fetch_one(result: SearchResult) -> None:
    """Tək bir nəticənin tam mətnini çəkib SearchResult.full_text-ə yazır. Xəta halında sükutla keçir."""
    if not result.url:
        return
    try:
        resp = requests.get(
            result.url,
            timeout=CONFIG.fetch_timeout,
            headers={"User-Agent": CONFIG.user_agent},
        )
        resp.raise_for_status()

        if _HAS_TRAFILATURA:
            text = trafilatura.extract(resp.text, include_comments=False, include_tables=False)
        else:
            text = None

        if not text:
            # trafilatura heç nə tapmadısa, snippet ilə davam et
            return

        result.full_text = text[: CONFIG.fetch_max_chars]
        log.debug("Tam mətn çəkildi (%d simvol): %s", len(result.full_text), result.url)

    except requests.exceptions.RequestException as e:
        log.debug("Səhifə çəkilə bilmədi (%s): %s", result.url, e)
    except Exception as e:
        log.debug("Gözlənilməz fetch xətası (%s): %s", result.url, e)


def enrich_with_full_text(results: List[SearchResult]) -> List[SearchResult]:
    """Bütün nəticələr üçün paralel şəkildə tam mətn çəkməyə çalışır.

    Uğursuz olanlar sadəcə öz snippet-i ilə qalır — bütün proses dayanmır.
    """
    if not CONFIG.fetch_enabled or not results:
        return results

    with ThreadPoolExecutor(max_workers=CONFIG.fetch_concurrency) as pool:
        futures = {pool.submit(_fetch_one, r): r for r in results}
        for future in as_completed(futures):
            future.result()  # istisnalar artıq _fetch_one içində tutulur

    return results
