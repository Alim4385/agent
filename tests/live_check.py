"""Canlı yoxlama: real internetdə axtarış + real səhifə çəkmə.

Bu skript mock istifadə etmir — DuckDuckGo-ya real sorğu göndərir və tapılan
linklərdən bir neçəsini real olaraq çəkib trafilatura ilə mətn çıxarır.
İnternetə çıxışı olan maşında işə salın:

    python tests/live_check.py "Azərbaycan paytaxtı"
    python tests/live_check.py --query "..." --pages 3

Çıxış kodu: hər şey qaydasındadırsa 0, problem varsa 1.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Skripti istənilən qovluqdan işə salmaq üçün repo kökünü sys.path-ə əlavə edirik.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.config import CONFIG
from agent.fetch import enrich_with_full_text
from agent.logging_setup import log
from agent.pipeline import format_sources
from agent.rerank import rerank
from agent.search import search_web


def main() -> int:
    parser = argparse.ArgumentParser(description="Veb hissəsinin canlı yoxlanması")
    parser.add_argument("--query", default="Azərbaycan paytaxtı Bakı")
    parser.add_argument("--pages", type=int, default=3, help="Neçə səhifə çəkiləcək")
    args = parser.parse_args()

    failures = 0
    print(f"Konfiqurasiya: nəticə limiti {CONFIG.search_max_results}, "
          f"fetch {'aktiv' if CONFIG.fetch_enabled else 'deaktiv'}, "
          f"top_k {CONFIG.rerank_top_k}\n")

    # 1) Axtarış
    started = time.time()
    try:
        results = search_web(args.query)
    except Exception as e:  # noqa: BLE001
        print(f"✗ Axtarış alınmadı: {type(e).__name__}: {e}")
        return 1
    print(f"✓ Axtarış: {len(results)} nəticə ({time.time() - started:.1f}s)")
    for r in results:
        print(f"   [{r.rank}] {r.title[:60]} — {r.url}")

    if not results:
        print("✗ Nəticə yoxdur — axtarış işləmir.")
        return 1

    # 2) Tam səhifə çəkmə
    subset = results[: args.pages]
    started = time.time()
    enriched = enrich_with_full_text(subset)
    filled = [r for r in enriched if r.full_text]
    print(f"\n{'✓' if filled else '✗'} Səhifə çəkmə: {len(filled)}/{len(enriched)} "
          f"səhifədən mətn çıxarıldı ({time.time() - started:.1f}s)")
    for r in enriched:
        size = len(r.full_text)
        status = f"{size} simvol" if size else "snippet istifadə olunur"
        print(f"   [{r.rank}] {status} — {r.url}")
    if not filled:
        failures += 1

    # 3) Rerank
    ranked = rerank(args.query, results)
    print(f"\n✓ Rerank: {len(ranked)} mənbə seçildi")
    print(format_sources(ranked))

    print("\n" + "-" * 60)
    if failures:
        print("Nəticə: veb hissəsində problem var (yuxarıdakı ✗ sətrinə baxın).")
        print("İpucu: `python -m agent.doctor` daha ətraflı diaqnostika verir.")
        return 1
    print("Nəticə: axtarış, səhifə çəkmə və rerank işləyir ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
