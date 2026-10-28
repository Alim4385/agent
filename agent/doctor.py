"""`python -m agent.doctor` — sistemin bütün mərhələlərini canlı yoxlayır.

Hər mərhələ üçün ✓ / ✗ nəticə və problem halında konkret işarə göstərilir:
  1) mühit (Python versiyası, xarici paket tələb olunmur)
  2) LLM serveri (/chat/completions)
  3) embeddings endpoint-i (rerank üçün, məcburi deyil)
  4) real veb axtarışı (DuckDuckGo HTML interfeysi)
  5) real səhifə çəkmə + mətn çıxarma (daxili html_text)
  6) tam pipeline (axtarış → fetch → rerank → LLM)

İstifadə:
    python -m agent.doctor              # bütün yoxlamalar
    python -m agent.doctor --offline    # yalnız lokal yoxlamalar (şəbəkəsiz)
    python -m agent.doctor --query "..."  # axtarış üçün xüsusi sorğu
"""

from __future__ import annotations

import argparse
import time
from typing import Callable, List, Tuple

from .config import CONFIG

OK = "✓"
FAIL = "✗"
SKIP = "–"
WARN = "!"  # məcburi olmayan yoxlama

TEST_QUERY = "Azərbaycan paytaxtı Bakı"


class Report:
    def __init__(self) -> None:
        self.results: List[Tuple[str, str, str]] = []
        self.optional: set = set()

    def add_optional(self, name: str, status: str, detail: str = "") -> None:
        self.optional.add(name)
        self.add(name, status, detail)

    def add(self, name: str, status: str, detail: str = "") -> None:
        self.results.append((name, status, detail))
        line = f"[{status}] {name}"
        if detail:
            line += f" — {detail}"
        print(line, flush=True)

    def check(self, name: str, fn: Callable[[], str]) -> bool:
        """fn uğurlu olsa onun qaytardığı detalı yazır, xəta olsa ✗ qeyd edir."""
        started = time.time()
        try:
            detail = fn()
            self.add(name, OK, f"{detail} ({time.time() - started:.1f}s)")
            return True
        except Exception as e:  # noqa: BLE001 — hər xəta hesabatda göstərilir
            self.add(name, FAIL, f"{type(e).__name__}: {e}")
            return False

    def check_optional(self, name: str, fn: Callable[[], str]) -> bool:
        """Məcburi olmayan yoxlama: uğursuzluq `!` kimi göstərilir, işi dayandırmır."""
        started = time.time()
        try:
            detail = fn()
            self.add(name, OK, f"{detail} ({time.time() - started:.1f}s)")
            return True
        except Exception as e:  # noqa: BLE001
            self.add_optional(name, WARN, f"{e} ({time.time() - started:.1f}s)")
            return False

    @property
    def failed(self) -> List[str]:
        return [
            name
            for name, status, _ in self.results
            if status == FAIL and name not in self.optional
        ]

    @property
    def warned(self) -> List[str]:
        return [
            name
            for name, status, _ in self.results
            if status == WARN or (status == FAIL and name in self.optional)
        ]

    def summary(self) -> int:
        print("\n" + "-" * 60)
        if self.warned and not self.failed:
            print("Nəticə: kritik xəta yoxdur — bunlar işi dayandırmır:")
            for name in self.warned:
                print(f"  ! {name}")
        if not self.failed:
            print("Nəticə: bütün yoxlamalar keçdi ✓")
            return 0
        print("Nəticə: uğursuz yoxlamalar:")
        for name in self.failed:
            print(f"  - {name}")
        print("\nİşarələr:")
        print("  • LLM serveri: serverin işlədiyini və LLM_BASE_URL-i yoxlayın")
        print("  • Axtarış: internet/DPI/proxy maneəsi ola bilər (şirkət şəbəkəsi?)")
        print("  • Python 3.10+ lazımdır; xarici paket tələb olunmur (stdlib kifayətdir)")
        print("  • Səhifə çəkmə: bəzi saytlar bot UA-sını bloklayır (USER_AGENT dəyişənini dəyişin)")
        return 1


def _check_deps() -> str:
    """Xarici paket tələb olunmur — yalnız Python versiyası və stdlib modulları yoxlanılır."""
    import sys

    if sys.version_info < (3, 10):
        raise RuntimeError(
            f"Python 3.10+ lazımdır, hazırda {sys.version_info.major}.{sys.version_info.minor}"
        )
    required = ("urllib.request", "html.parser", "json", "gzip", "zlib", "concurrent.futures")
    missing = []
    for module in required:
        try:
            __import__(module)
        except ImportError:  # pragma: no cover — stdlib həmişə mövcuddur
            missing.append(module)
    if missing:
        raise ImportError("standart kitabxana natamamdır: " + ", ".join(missing))
    return f"Python {sys.version.split()[0]}, xarici paket tələb olunmur (stdlib)"


def _check_llm() -> str:
    from .llm_client import chat_completion

    answer = chat_completion(
        [{"role": "user", "content": "Yalnız 'OK' sözünü yaz."}]
    )
    answer = (answer or "").strip()
    if not answer:
        raise RuntimeError("server boş cavab qaytardı")
    return f"model cavab verdi: {answer[:40]!r}"


def _check_embeddings() -> str:
    if not CONFIG.embeddings_enabled:
        raise RuntimeError("EMBEDDINGS_ENABLED=false — rerank TF-IDF ilə işləyəcək")
    from .rerank import _get_embeddings

    vectors = _get_embeddings(["test"])
    if not vectors:
        raise RuntimeError("endpoint cavab vermədi — rerank TF-IDF/açar-söz üsuluna keçəcək")
    return f"{len(vectors)} vektor, ölçü {len(vectors[0])}"


def _check_search(query: str) -> str:
    from .search import search_web

    results = search_web(query)
    if not results:
        raise RuntimeError("heç bir nəticə qaytarmadı")
    top = results[0]
    return f"{len(results)} nəticə; ilk: {top.title[:40]!r} — {top.url}"


def _check_fetch(query: str) -> str:
    from .fetch import enrich_with_full_text
    from .search import search_web

    results = search_web(query)[:3]
    if not results:
        raise RuntimeError("axtarış nəticəsi yoxdur, fetch yoxlanıla bilmədi")

    enriched = enrich_with_full_text(results)
    with_text = [r for r in enriched if r.full_text]
    if not with_text:
        raise RuntimeError("3 səhifədən heç birindən mətn çıxarıla bilmədi")
    return f"{len(with_text)}/{len(enriched)} səhifədən mətn çıxarıldı"


def _check_pipeline(query: str) -> str:
    from .history import ConversationHistory
    from .pipeline import answer

    content, sources = answer(query, ConversationHistory(), stream=False)
    if not (content or "").strip():
        raise RuntimeError("LLM boş cavab qaytardı")
    return f"{len(sources)} mənbə, cavab {len(content)} simvol"


def run(offline: bool = False, query: str = TEST_QUERY) -> int:
    print(f"Konfiqurasiya: LLM={CONFIG.chat_url} | model={CONFIG.model_name}")
    print(f"Yoxlama sorğusu: {query!r}\n")

    report = Report()
    report.check("Asılılıqlar", _check_deps)

    if offline:
        report.add("LLM serveri", SKIP, "--offline rejimi")
        report.add("Embeddings", SKIP, "--offline rejimi")
        report.add("Veb axtarışı", SKIP, "--offline rejimi")
        report.add("Səhifə çəkmə", SKIP, "--offline rejimi")
        report.add("Tam pipeline", SKIP, "--offline rejimi")
        return report.summary()

    llm_ok = report.check("LLM serveri", _check_llm)
    report.check_optional("Embeddings (istəyə bağlı)", _check_embeddings)
    search_ok = report.check("Veb axtarışı (DuckDuckGo HTML)", lambda: _check_search(query))
    if search_ok:
        report.check("Səhifə çəkmə + mətn çıxarma", lambda: _check_fetch(query))
    else:
        report.add("Səhifə çəkmə + mətn çıxarma", SKIP, "axtarış nəticəsi olmadığı üçün keçildi")

    if llm_ok:
        report.check("Tam pipeline", lambda: _check_pipeline(query))
    else:
        report.add("Tam pipeline", SKIP, "LLM serveri əlçatan olmadığı üçün keçildi")

    if not search_ok:
        print(
            "\nQeyd: axtarış alınmırsa, agent yenə də işləyir — sadəcə mənbəsiz cavab verir\n"
            "və bunu log-da xəbərdarlıq kimi yazır."
        )
    return report.summary()


def main() -> None:
    parser = argparse.ArgumentParser(description="Web agent üçün diaqnostika")
    parser.add_argument("--offline", action="store_true", help="Şəbəkə yoxlamalarını keç")
    parser.add_argument("--query", default=TEST_QUERY, help="Test üçün axtarış sorğusu")
    args = parser.parse_args()
    raise SystemExit(run(offline=args.offline, query=args.query))


if __name__ == "__main__":
    main()
