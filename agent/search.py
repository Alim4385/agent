"""DuckDuckGo üzərindən veb axtarışı — üçüncü tərəf paketi olmadan.

Əvvəl `ddgs`/`duckduckgo-search` paketi istifadə olunurdu; indi eyni işi
standart kitabxana görür: DuckDuckGo-nun HTML interfeysinə POST sorğusu
göndərilir və nəticələr `html.parser` ilə oxunur.

İki backend sırayla yoxlanılır (biri bot-yoxlaması qaytarsa, digəri işlədilir):
  • https://html.duckduckgo.com/html/
  • https://lite.duckduckgo.com/lite/
"""

from __future__ import annotations

import html as html_lib
import time
import urllib.parse
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Dict, List, Optional

from . import net
from .config import CONFIG
from .exceptions import SearchError
from .logging_setup import log

SEARCH_BACKENDS = (
    "https://html.duckduckgo.com/html/",
    "https://lite.duckduckgo.com/lite/",
)

# Bot-yoxlaması/captcha əlamətləri — belə cavab nəticə sayılmır
_BLOCK_MARKERS = ("anomaly", "captcha", "unusual traffic", "are you a robot")


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


# ---------------------------------------------------------------------------
# HTML parse
# ---------------------------------------------------------------------------

def _clean_url(href: Optional[str]) -> str:
    """DDG yönləndirmə linkini (`//duckduckgo.com/l/?uddg=...`) əsl ünvana çevirir."""
    href = html_lib.unescape(href or "").strip()
    if not href:
        return ""
    if href.startswith("//"):
        href = "https:" + href
    elif href.startswith("/"):
        return ""  # daxili (reklam/parametr) linklər

    parsed = urllib.parse.urlparse(href)
    if not parsed.scheme.startswith("http"):
        return ""
    if parsed.netloc.endswith("duckduckgo.com"):
        if parsed.path.startswith("/l/"):
            query = urllib.parse.parse_qs(parsed.query)
            target = query.get("uddg") or query.get("u")
            if target:
                return urllib.parse.unquote(target[0])
        return ""  # reklam (y.js) və digər daxili linklər
    return href


class _DDGParser(HTMLParser):
    """Həm `html.duckduckgo.com`, həm də `lite.duckduckgo.com` markupunu oxuyur.

    html versiyası: <a class="result__a">başlıq</a> + <a/div class="result__snippet">
    lite versiyası: <a class="result-link">başlıq</a> + <td class="result-snippet">
    """

    TITLE_CLASSES = {"result__a", "result-link"}
    SNIPPET_CLASSES = {"result__snippet", "result-snippet"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.items: List[Dict[str, str]] = []
        self._mode: Optional[str] = None  # "title" | "snippet"
        self._buf: List[str] = []
        self._href = ""
        self._depth = 0
        self._accept_snippet = True  # reklam/skip edilən nəticənin snippet-i yazılmasın

    def _start(self, mode: str, href: str = "") -> None:
        # Yeni hissə başlayanda əvvəlkini yekunlaşdırırıq (buf-u təmizləyərək)
        if self._mode == "title":
            self._finish_title()
        elif self._mode == "snippet":
            self._finish_snippet()
        self._mode, self._buf, self._href, self._depth = mode, [], href, 0

    def _finish_title(self) -> None:
        title = " ".join("".join(self._buf).split())
        url = _clean_url(self._href)
        self._buf = []
        if title and url:
            self.items.append({"title": title, "url": url, "snippet": ""})
            self._accept_snippet = True
        else:
            self._accept_snippet = False  # reklam və ya yararsız link

    def _finish_snippet(self) -> None:
        snippet = " ".join("".join(self._buf).split())
        self._buf = []
        if self.items and snippet and self._accept_snippet:
            self.items[-1]["snippet"] = snippet

    def handle_starttag(self, tag, attrs):
        attr = {k: (v or "") for k, v in attrs}
        classes = set((attr.get("class") or "").split())

        if self._mode == "snippet" and tag in ("a", "div", "td", "span"):
            self._depth += 1

        if tag == "a" and classes & self.TITLE_CLASSES:
            self._start("title", attr.get("href", ""))
        elif classes & self.SNIPPET_CLASSES and self._mode != "snippet":
            self._start("snippet")

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag):
        if self._mode == "title" and tag == "a":
            self._finish_title()
            self._mode = None
        elif self._mode == "snippet" and tag in ("a", "div", "td", "span"):
            if self._depth > 0:
                self._depth -= 1
            else:
                self._finish_snippet()
                self._mode = None

    def handle_data(self, data):
        if self._mode:
            self._buf.append(data)

    def close(self) -> None:  # noqa: D401 — HTMLParser.close uyğunluğu
        super().close()
        if self._mode == "title":
            self._finish_title()
        elif self._mode == "snippet":
            self._finish_snippet()
        self._mode = None


def parse_results(page: str, limit: int) -> List[SearchResult]:
    """Axtarış HTML səhifəsindən nəticələri çıxarır (təkrarlar və boş linklər atılır)."""
    parser = _DDGParser()
    try:
        parser.feed(page or "")
        parser.close()
    except Exception as e:  # noqa: BLE001 — pozulmuş HTML axtarışı dayandırmasın
        log.debug("Axtarış HTML-i parse edilə bilmədi: %s", e)
        return []

    results: List[SearchResult] = []
    seen = set()
    for item in parser.items:
        url = item["url"]
        if not url or url in seen:
            continue
        seen.add(url)
        results.append(
            SearchResult(
                title=item["title"] or "Başlıqsız",
                url=url,
                snippet=item["snippet"],
                rank=len(results) + 1,
            )
        )
        if len(results) >= limit:
            break
    return results


# ---------------------------------------------------------------------------
# Şəbəkə
# ---------------------------------------------------------------------------

def _fetch_backend(backend_url: str, query: str) -> str:
    """Backend-dən HTML cavabını çəkir (testlərdə bu funksiya əvəzlənir).

    Əvvəl POST, nəticə görünmürsə eyni backend GET ilə də yoxlanılır — DDG
    bəzən metodkandan asılı olaraq fərqli cavab qaytarır.
    """
    form = {"q": query, "kl": "wt-wt"}
    response = net.post_form(
        backend_url,
        form,
        timeout=CONFIG.search_timeout,
        max_bytes=3_000_000,
        headers={"Referer": "https://duckduckgo.com/"},
    )
    page = response.text
    if _has_result_markup(page):
        return page

    get_url = backend_url + "?" + urllib.parse.urlencode({"q": query, "kl": "wt-wt"})
    return net.get_text(get_url, timeout=CONFIG.search_timeout, max_bytes=3_000_000).text


def _has_result_markup(page: str) -> bool:
    return ("result__a" in page) or ("result-link" in page)


def _looks_blocked(page: str) -> bool:
    lowered = (page or "").lower()
    return any(marker in lowered for marker in _BLOCK_MARKERS)


def search_web(query: str) -> List[SearchResult]:
    """DuckDuckGo-da axtarış edir; backend-lər və cəhdlər üzrə retry/backoff tətbiq edir."""
    last_error: Optional[Exception] = None

    for attempt in range(1, CONFIG.search_retries + 2):  # 1 ilkin cəhd + retries
        for backend in SEARCH_BACKENDS:
            try:
                page = _fetch_backend(backend, query)
            except (net.NetworkError, net.HttpError) as e:
                last_error = e
                log.warning("Axtarış backend-i əlçatmaz (%s): %s", backend, e)
                continue
            except Exception as e:  # noqa: BLE001
                last_error = e
                log.warning("Axtarış zamanı gözlənilməz xəta (%s): %s", backend, e)
                continue

            if _looks_blocked(page):
                last_error = SearchError("DuckDuckGo bot yoxlaması qaytardı")
                log.warning("Axtarış backend-i bot yoxlaması qaytardı: %s", backend)
                continue

            results = parse_results(page, CONFIG.search_max_results)
            if results:
                log.info("Axtarış uğurlu: %d nəticə (%r, backend: %s)", len(results), query, backend)
                return results

            last_error = SearchError(f"{backend} — nəticə tapılmadı")
            log.debug("Backend nəticə qaytarmadı: %s", backend)

        if attempt <= CONFIG.search_retries:
            sleep_for = CONFIG.search_backoff_seconds * attempt
            log.info("Növbəti cəhddən əvvəl %.1f saniyə gözlənilir", sleep_for)
            time.sleep(sleep_for)

    message = f"Axtarış {CONFIG.search_retries + 1} cəhddən sonra uğursuz oldu: {last_error}"
    log.error("%s", message)
    raise SearchError(message)
