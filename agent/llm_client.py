"""OpenAI-uyğun /chat/completions endpoint-i ilə əlaqə (streaming və qeyri-streaming).

`requests` əvəzinə yalnız standart kitabxana (`urllib`) istifadə olunur — SSE
axını `http.client` cavabını sətir-sətir oxumaqla emal edilir.
"""

from __future__ import annotations

import json
from typing import Dict, Generator, List

from . import net
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


def _connection_message(exc: Exception) -> str:
    return (
        f"LLM serverinə qoşulmaq mümkün olmadı ({CONFIG.chat_url}). "
        f"Server işə düşüb? Detal: {exc}"
    )


def chat_completion(messages: List[Dict[str, str]]) -> str:
    """Qeyri-streaming çağırış: tam cavabı bir dəfəyə qaytarır."""
    try:
        data = net.post_json(
            CONFIG.chat_url,
            _build_payload(messages, stream=False),
            timeout=CONFIG.request_timeout,
        )
    except net.NetworkError as e:
        raise LLMConnectionError(_connection_message(e)) from e
    except net.HttpError as e:
        raise LLMResponseError(f"LLM server HTTP xətası qaytardı: {e}") from e
    except ValueError as e:  # json parse
        raise LLMResponseError(f"LLM cavabı JSON kimi oxuna bilmədi: {e}") from e

    try:
        return data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as e:
        raise LLMResponseError(f"LLM cavabı gözlənilməz formatdadır: {e}") from e


def chat_completion_stream(messages: List[Dict[str, str]]) -> Generator[str, None, None]:
    """Streaming çağırış: server-sent events formatında tokenləri ardıcıl yield edir.

    Server SSE dəstəkləmirsə (adi JSON qaytarırsa), avtomatik qeyri-streaming
    rejiminə keçir ki, istifadəçi heç nə itirməsin.
    """
    try:
        resp, content_type = net.open_stream(
            CONFIG.chat_url,
            _build_payload(messages, stream=True),
            timeout=CONFIG.request_timeout,
        )
    except net.NetworkError as e:
        raise LLMConnectionError(_connection_message(e)) from e
    except net.HttpError as e:
        raise LLMResponseError(f"LLM server HTTP xətası qaytardı: {e}") from e

    try:
        if "text/event-stream" not in (content_type or "").lower():
            # Server streaming dəstəkləmir — tam JSON kimi emal et
            raw = resp.read()
            resp.close()
            try:
                data = json.loads(raw.decode("utf-8", "replace"))
                yield data["choices"][0]["message"]["content"] or ""
            except (ValueError, KeyError, IndexError, TypeError) as e:
                raise LLMResponseError(f"LLM cavabı emal edilə bilmədi: {e}") from e
            return

        got_any_chunk = False
        try:
            for raw_line in resp:
                line = raw_line.decode("utf-8", "replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                chunk = line[len("data:"):].strip()
                if chunk == "[DONE]":
                    break
                try:
                    payload = json.loads(chunk)
                    delta = payload["choices"][0].get("delta") or {}
                    token = delta.get("content")
                    if token:
                        got_any_chunk = True
                        yield token
                except (ValueError, KeyError, IndexError) as e:
                    log.debug("Stream chunk parse edilə bilmədi, ötürülür: %s (%r)", e, chunk)
                    continue
        except OSError as e:
            # Axın ortasında qopma — göstərilən mətn itməsin deyə konkret xəta
            log.warning("Streaming zamanı bağlantı kəsildi: %s", e)
            raise LLMConnectionError(f"LLM axını kəsildi: {e}") from e
        finally:
            try:
                resp.close()
            except Exception:  # noqa: BLE001
                pass

        if not got_any_chunk:
            log.warning("Streaming heç bir token qaytarmadı — server konfiqurasiyasını yoxlayın.")
    finally:
        try:
            resp.close()
        except Exception:  # noqa: BLE001
            pass
