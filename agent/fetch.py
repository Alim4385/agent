"""Axtarış nəticələrindəki linklərdən tam səhifə mətnini paralel şəkildə çəkir.

Snippet-lər çox vaxt kifayət qədər kontekst vermir. Bu modul əsl səhifəyə girib
naviqasiya/reklam/footer kimi "zibil" hissələri təmizləyib əsl məzmunu çıxarır.

Bütün iş standart kitabxana ilə görülür: `urllib` (net.py) + `html.parser`
(html_text.py). Heç bir xarici paket tələb olunmur.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from . import net
from .config import CONFIG
from .html_text import extract_text
from .logging_setup import log
from .search import SearchResult

# Bu content-type-lar mətn deyil — çəkməyə dəyməz
_SKIP_CONTENT_TYPES = ("image/", "video/", "audio/", "application/pdf", "application/zip")


def fetch_page_text(url: str, *, timeout: float | None = None) -> str:
    """Bir səhifənin əsas mətnini qaytarır. Uğursuzluqda boş sətir."""
    timeout = CONFIG.fetch_timeout if timeout is None else timeout
    resp = net.get_text(url, timeout=timeout, max_bytes=CONFIG.fetch_max_bytes)

    content_type = resp.content_type.lower()
    if any(content_type.startswith(t) for t in _SKIP_CONTENT_TYPES):
        log.debug("Mətn olmayan content-type, keçilir (%s): %s", content_type, url)
        return ""

    return extract_text(resp.text)


def _fetch_one(result: SearchResult) -> None:
    """Tək nəticənin tam mətnini çəkib `full_text`-ə yazır. Xəta halında sükutla keçir."""
    if not result.url:
        return
    try:
        text = fetch_page_text(result.url)
        if not text:
            return  # mətn çıxarıla bilmədi — snippet ilə davam edirik
        result.full_text = text[: CONFIG.fetch_max_chars]
        log.debug("Tam mətn çəkildi (%d simvol): %s", len(result.full_text), result.url)
    except net.NetworkError as e:
        log.debug("Səhifə çəkilə bilmədi (%s): %s", result.url, e)
    except net.HttpError as e:
        log.debug("Səhifə HTTP xətası qaytardı (%s): %s", result.url, e)
    except Exception as e:  # noqa: BLE001 — bir səhifə bütün prosesi dayandırmamalıdır
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
