"""Axtarış nəticələrini sualla uyğunluğuna görə yenidən sıralayır.

Üç səviyyəli strategiya (yuxarıdan aşağı, ilk işləyəni istifadə edir):
  1) LLM serverinin /embeddings endpoint-i (varsa) — semantik oxşarlıq
  2) scikit-learn TF-IDF + cosine similarity (yerli, sürətli)
  3) sadə açar-söz üst-üstə düşməsi (heç bir asılılıq tələb etməyən son çarə)
"""

from __future__ import annotations

import math
from typing import List

import requests

from .config import CONFIG
from .logging_setup import log
from .search import SearchResult

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


def _get_embeddings(texts: List[str]) -> List[List[float]] | None:
    """LLM serverindən embedding almağa çalışır. Endpoint yoxdursa None qaytarır."""
    try:
        resp = requests.post(
            CONFIG.embeddings_url,
            json={"model": CONFIG.model_name, "input": texts},
            timeout=CONFIG.request_timeout,
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        return [item["embedding"] for item in data]
    except Exception as e:
        log.debug("Embeddings endpoint mövcud deyil ya da xəta verdi: %s", e)
        return None


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _rerank_with_embeddings(query: str, results: List[SearchResult]) -> List[SearchResult] | None:
    texts = [query] + [r.display_text for r in results]
    embeddings = _get_embeddings(texts)
    if embeddings is None:
        return None

    query_vec, doc_vecs = embeddings[0], embeddings[1:]
    scored = list(zip(results, (_cosine(query_vec, v) for v in doc_vecs)))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    log.info("Reranking: embeddings üsulu istifadə olundu")
    return [r for r, _ in scored]


def _rerank_with_tfidf(query: str, results: List[SearchResult]) -> List[SearchResult] | None:
    if not _HAS_SKLEARN:
        return None
    try:
        corpus = [query] + [r.display_text for r in results]
        vectorizer = TfidfVectorizer(max_features=4096)
        matrix = vectorizer.fit_transform(corpus)
        sims = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
        scored = sorted(zip(results, sims), key=lambda pair: pair[1], reverse=True)
        log.info("Reranking: TF-IDF üsulu istifadə olundu")
        return [r for r, _ in scored]
    except Exception as e:
        log.debug("TF-IDF reranking uğursuz oldu: %s", e)
        return None


def _rerank_with_keyword_overlap(query: str, results: List[SearchResult]) -> List[SearchResult]:
    query_words = set(query.lower().split())

    def score(r: SearchResult) -> int:
        text_words = set(r.display_text.lower().split())
        return len(query_words & text_words)

    log.info("Reranking: sadə açar-söz üst-üstə düşmə üsulu istifadə olundu")
    return sorted(results, key=score, reverse=True)


def rerank(query: str, results: List[SearchResult]) -> List[SearchResult]:
    """Nəticələri sualla uyğunluğuna görə sıralayıb ən yaxşı top_k-nı qaytarır."""
    if not results:
        return results

    ranked = None
    if CONFIG.embeddings_enabled:
        ranked = _rerank_with_embeddings(query, results)
    if ranked is None:
        ranked = _rerank_with_tfidf(query, results)
    if ranked is None:
        ranked = _rerank_with_keyword_overlap(query, results)

    return ranked[: CONFIG.rerank_top_k]
