"""Bütün mərhələləri birləşdirən əsas pipeline: axtarış -> tam mətn -> rerank -> LLM."""

from __future__ import annotations

from typing import Generator, List, Tuple

from .config import CONFIG
from .exceptions import SearchError, WebAgentError
from .fetch import enrich_with_full_text
from .history import ConversationHistory
from .llm_client import chat_completion, chat_completion_stream
from .logging_setup import log
from .rerank import rerank
from .search import SearchResult, search_web

SYSTEM_PROMPT_TEMPLATE = (
    "Sən internetdə axtarış aparan dəqiq və dürüst tədqiqat köməkçisisən. "
    "Aşağıda nömrələnmiş mənbələr verilib. Yalnız bu mənbələrə əsaslanaraq "
    "Azərbaycan dilində cavab ver. Hər faktı yazanda mənbəni [1], [2] kimi qeyd et. "
    "Əgər mənbələrdə cavab üçün kifayət qədər məlumat yoxdursa, bunu açıq şəkildə de "
    "və fərziyyə etmə.\n\n"
    "Mənbələr:\n{context}"
)


def _build_context(results: List[SearchResult]) -> str:
    if not results:
        return "Heç bir mənbə tapılmadı."
    parts = []
    for r in results:
        parts.append(f"[{r.rank}] {r.title}\nLink: {r.url}\n{r.display_text}")
    return "\n\n".join(parts)


def gather_sources(query: str) -> List[SearchResult]:
    """Axtarış -> tam mətn -> rerank zəncirini işlədir. Xəta halında boş siyahı qaytarmır, yuxarı ötürür."""
    results = search_web(query)
    if not results:
        return []

    results = enrich_with_full_text(results)
    top_results = rerank(query, results)
    return top_results


def answer(
    query: str, history: ConversationHistory, stream: bool = None
) -> Tuple[str, List[SearchResult]] | Tuple[Generator[str, None, None], List[SearchResult]]:
    """Sualı cavablandırır.

    Hər iki halda (stream, sources) tuple qaytarır:
      - stream=True:  sources dərhal mövcuddur, generator isə tokenləri ardıcıl verir.
        Çağıran tərəf generatoru tükətdikdən sonra tam mətni history.add_assistant()
        ilə özü əlavə etməlidir.
      - stream=False: (cavab_mətni, sources) — hər ikisi hazır.
    """
    if stream is None:
        stream = CONFIG.stream

    try:
        sources = gather_sources(query)
    except SearchError as e:
        log.error("Axtarış tamamilə uğursuz oldu: %s", e)
        sources = []

    context = _build_context(sources)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)

    messages = [{"role": "system", "content": system_prompt}] + history.as_list() + [
        {"role": "user", "content": query}
    ]

    if stream:
        return chat_completion_stream(messages), sources
    else:
        content = chat_completion(messages)
        return content, sources


def format_sources(sources: List[SearchResult]) -> str:
    if not sources:
        return ""
    lines = [f"[{r.rank}] {r.title} — {r.url}" for r in sources]
    return "📎 Mənbələr:\n" + "\n".join(lines)
