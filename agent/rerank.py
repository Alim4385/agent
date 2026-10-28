"""Axtarış nəticələrini sualla uyğunluğuna görə yenidən sıralayır.

Üç səviyyəli strategiya (yuxarıdan aşağı, ilk işləyəni istifadə edir):
  1) LLM serverinin /embeddings endpoint-i (varsa) — semantik oxşarlıq
  2) Təmiz Python TF-IDF + cosine similarity (sklearn tələb etmir)
  3) Sadə açar-söz üst-üstə düşməsi — son çarə

Bütün hesablamalar standart kitabxana ilə aparılır (math, re, collections).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Optional

from . import net
from .config import CONFIG
from .logging_setup import log
from .search import SearchResult

# Azərbaycan/türk/ingilis dilləri üçün kiçik stopword siyahısı
_STOPWORDS = frozenset(
    """
    və və ya ilə üçün ki də da isə bu o bir nə necə hansı harada kim çox az ən
    amma ancaq həm həmçinin belə ona onu onun öz özü bütün bəzi hər heç
    the a an and or of to in for on with is are was were be been this that
    these those it its as at by from not no but if then than so such
    """.split()
)

_WORD_RE = re.compile(r"[^\W\d_]+|\d+", re.UNICODE)


# ---------------------------------------------------------------------------
# Tokenizasiya və TF-IDF
# ---------------------------------------------------------------------------

def tokenize(text: str) -> List[str]:
    """Mətni kiçik hərfli sözlərə bölür; stopword və 1 hərfli sözləri atır."""
    words = (match.group(0).lower() for match in _WORD_RE.finditer(text or ""))
    return [w for w in words if len(w) > 1 and w not in _STOPWORDS]


def _tfidf_vectors(docs: List[List[str]]) -> List[Dict[str, float]]:
    """Sənədlər üçün L2-normallaşdırılmış TF-IDF vektorları (sparse dict)."""
    total = len(docs)
    doc_freq: Counter = Counter()
    for tokens in docs:
        doc_freq.update(set(tokens))

    idf = {term: math.log((1 + total) / (1 + freq)) + 1.0 for term, freq in doc_freq.items()}

    vectors: List[Dict[str, float]] = []
    for tokens in docs:
        counts = Counter(tokens)
        vector = {
            term: (1.0 + math.log(count)) * idf.get(term, 1.0)
            for term, count in counts.items()
            if count > 0
        }
        norm = math.sqrt(sum(v * v for v in vector.values()))
        if norm > 0:
            vector = {term: value / norm for term, value in vector.items()}
        vectors.append(vector)
    return vectors


def _sparse_cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    """Normallaşdırılmış vektorlar üçün cosine — yalnız ortaq terminlər gəzilir."""
    if len(a) > len(b):
        a, b = b, a
    return sum(value * b[term] for term, value in a.items() if term in b)


# ---------------------------------------------------------------------------
# Embeddings (LLM serveri dəstəkləyirsə)
# ---------------------------------------------------------------------------

def _get_embeddings(texts: List[str]) -> Optional[List[List[float]]]:
    """LLM serverindən embedding almağa çalışır. Endpoint yoxdursa None qaytarır."""
    try:
        data = net.post_json(
            CONFIG.embeddings_url,
            {"model": CONFIG.model_name, "input": texts},
            timeout=min(CONFIG.request_timeout, 30),
        )
        items = data["data"]
        # Bəzi serverlər nəticələri sırasız qaytarır — "index" varsa ona görə düzürük.
        if all(isinstance(item, dict) and "index" in item for item in items):
            items = sorted(items, key=lambda item: item["index"])
        return [item["embedding"] for item in items]
    except Exception as e:  # noqa: BLE001 — endpoint yoxdursa sükutla keçirik
        log.debug("Embeddings endpoint mövcud deyil ya da xəta verdi: %s", e)
        return None


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Strategiyalar
# ---------------------------------------------------------------------------

def _rerank_with_embeddings(query: str, results: List[SearchResult]) -> Optional[List[SearchResult]]:
    texts = [query] + [r.display_text for r in results]
    embeddings = _get_embeddings(texts)
    if embeddings is None or len(embeddings) != len(texts):
        return None

    query_vec, doc_vecs = embeddings[0], embeddings[1:]
    scored = list(zip(results, (_cosine(query_vec, vec) for vec in doc_vecs)))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    log.info("Reranking: embeddings üsulu istifadə olundu")
    return [result for result, _ in scored]


def _rerank_with_tfidf(query: str, results: List[SearchResult]) -> Optional[List[SearchResult]]:
    try:
        corpus = [tokenize(query)] + [tokenize(r.display_text) for r in results]
        if not corpus[0]:
            return None
        vectors = _tfidf_vectors(corpus)
        query_vec, doc_vecs = vectors[0], vectors[1:]
        scored = list(zip(results, (_sparse_cosine(query_vec, vec) for vec in doc_vecs)))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        log.info("Reranking: TF-IDF üsulu istifadə olundu")
        return [result for result, _ in scored]
    except Exception as e:  # noqa: BLE001
        log.debug("TF-IDF reranking uğursuz oldu: %s", e)
        return None


def _rerank_with_keyword_overlap(query: str, results: List[SearchResult]) -> List[SearchResult]:
    query_words = set(tokenize(query)) or set(query.lower().split())

    def score(r: SearchResult) -> int:
        return len(query_words & set(tokenize(r.display_text)))

    log.info("Reranking: sadə açar-söz üst-üstə düşmə üsulu istifadə olundu")
    return sorted(results, key=score, reverse=True)


def rerank(query: str, results: List[SearchResult]) -> List[SearchResult]:
    """Nəticələri sualla uyğunluğuna görə sıralayıb ən yaxşı top_k-nı qaytarır."""
    if not results:
        return results

    ranked: Optional[List[SearchResult]] = None
    if CONFIG.embeddings_enabled:
        ranked = _rerank_with_embeddings(query, results)
    if ranked is None:
        ranked = _rerank_with_tfidf(query, results)
    if ranked is None:
        ranked = _rerank_with_keyword_overlap(query, results)

    top = ranked[: CONFIG.rerank_top_k]
    # Sitat nömrələri ([1], [2] ...) kontekstdəki sıra ilə üst-üstə düşməlidir.
    for new_rank, result in enumerate(top, start=1):
        result.rank = new_rank
    return top
