"""OpenAI-uyğun /chat/completions endpoint-i ilə əlaqə (streaming və qeyri-streaming)."""

from __future__ import annotations

import json
from typing import Generator, List, Dict

import requests

from .config import CONFIG
from .exceptions import LLMConnectionError, LLMResponseError
from .logging_setup import log


def _build_payload(messages: List[Dict[str, str]], stream: bool) -> dict:
    return {
        "model": CONFIG.model_name,
        "messages": messages,
        "temperature": CONFIG.temperature,
        "stream": stream,
    }


def chat_completion(messages: List[Dict[str, str]]) -> str:
    """Qeyri-streaming çağırış: tam cavabı bir dəfəyə qaytarır."""
    try:
        resp = requests.post(
            CONFIG.chat_url,
            json=_build_payload(messages, stream=False),
            timeout=CONFIG.request_timeout,
        )
        resp.raise_for_status()
    except requests.exceptions.ConnectionError as e:
        raise LLMConnectionError(
            f"LLM serverinə qoşulmaq mümkün olmadı ({CONFIG.chat_url}). Server işə düşüb? Detal: {e}"
        ) from e
    except requests.exceptions.Timeout as e:
        raise LLMConnectionError(f"LLM server {CONFIG.request_timeout}s ərzində cavab vermədi.") from e
    except requests.exceptions.HTTPError as e:
        raise LLMResponseError(f"LLM server HTTP xətası qaytardı: {e}") from e

    try:
        return resp.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, ValueError) as e:
        raise LLMResponseError(f"LLM cavabı gözlənilməz formatdadır: {e}") from e


def chat_completion_stream(messages: List[Dict[str, str]]) -> Generator[str, None, None]:
    """Streaming çağırış: server-sent events formatında tokenləri ardıcıl yield edir.

    Server SSE dəstəkləmirsə (adi JSON qaytarırsa), avtomatik qeyri-streaming
    rejiminə keçir ki, istifadəçi heç nə itirməsin.
    """
    try:
        resp = requests.post(
            CONFIG.chat_url,
            json=_build_payload(messages, stream=True),
            timeout=CONFIG.request_timeout,
            stream=True,
        )
        resp.raise_for_status()
    except requests.exceptions.ConnectionError as e:
        raise LLMConnectionError(
            f"LLM serverinə qoşulmaq mümkün olmadı ({CONFIG.chat_url}). Server işə düşüb? Detal: {e}"
        ) from e
    except requests.exceptions.Timeout as e:
        raise LLMConnectionError(f"LLM server {CONFIG.request_timeout}s ərzində cavab vermədi.") from e
    except requests.exceptions.HTTPError as e:
        raise LLMResponseError(f"LLM server HTTP xətası qaytardı: {e}") from e

    content_type = resp.headers.get("content-type", "")

    if "text/event-stream" not in content_type:
        # Server streaming dəstəkləmir — tam JSON kimi emal et
        try:
            data = resp.json()
            yield data["choices"][0]["message"]["content"]
            return
        except (ValueError, KeyError, IndexError) as e:
            raise LLMResponseError(f"LLM cavabı emal edilə bilmədi: {e}") from e

    got_any_chunk = False
    for raw_line in resp.iter_lines(decode_unicode=True):
        if not raw_line or not raw_line.startswith("data:"):
            continue
        chunk = raw_line[len("data:"):].strip()
        if chunk == "[DONE]":
            break
        try:
            payload = json.loads(chunk)
            delta = payload["choices"][0].get("delta", {})
            token = delta.get("content")
            if token:
                got_any_chunk = True
                yield token
        except (ValueError, KeyError, IndexError) as e:
            log.debug("Stream chunk parse edilə bilmədi, ötürülür: %s (%r)", e, chunk)
            continue

    if not got_any_chunk:
        log.warning("Streaming heç bir token qaytarmadı — server konfiqurasiyasını yoxlayın.")
