"""Şəbəkə qatı — yalnız Python standart kitabxanası (urllib) ilə.

Heç bir üçüncü tərəf paketi istifadə olunmur: `requests` əvəzinə `urllib.request`,
gzip/deflate açma `gzip`/`zlib` ilə, charset aşkarlanması isə əl ilə edilir.
Proxy dəstəyi avtomatikdir (urllib `http_proxy`/`https_proxy` dəyişənlərinə baxır).
"""

from __future__ import annotations

import gzip
import json
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
import zlib
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

from .config import CONFIG

DEFAULT_MAX_BYTES = 2_000_000  # 2 MB — səhifələr üçün kifayətdir, yaddaşı qoruyur

_META_CHARSET_RE = re.compile(rb"""charset\s*=\s*["']?\s*([\w\-]+)""", re.IGNORECASE)


class NetworkError(Exception):
    """Qoşulma/DNS/timeout kimi şəbəkə səviyyəli problemlər."""


class HttpError(Exception):
    """Server 4xx/5xx status qaytardı."""

    def __init__(self, status: int, url: str, body: str = "") -> None:
        self.status = status
        self.url = url
        self.body = body
        super().__init__(f"HTTP {status} — {url}")


@dataclass
class Response:
    url: str
    status: int
    headers: Mapping[str, str]
    content: bytes

    @property
    def content_type(self) -> str:
        return self.headers.get("content-type", "")

    @property
    def text(self) -> str:
        return decode_bytes(self.content, self.content_type)


# ---------------------------------------------------------------------------
# Dekodlaşdırma köməkçiləri
# ---------------------------------------------------------------------------

def _decompress(raw: bytes, encoding: str) -> bytes:
    encoding = (encoding or "").lower().strip()
    if not encoding or encoding == "identity":
        return raw
    try:
        if "gzip" in encoding:
            return gzip.decompress(raw)
        if "deflate" in encoding:
            try:
                return zlib.decompress(raw)
            except zlib.error:  # bəzi serverlər "raw" deflate göndərir
                return zlib.decompress(raw, -zlib.MAX_WBITS)
    except (OSError, zlib.error):
        return raw  # açıla bilmirsə, xam baytları saxlayırıq
    return raw


def _charset_from_content_type(content_type: str) -> Optional[str]:
    match = re.search(r"charset\s*=\s*[\"']?([\w\-]+)", content_type or "", re.IGNORECASE)
    return match.group(1) if match else None


def decode_bytes(raw: bytes, content_type: str = "") -> str:
    """Baytları mətnə çevirir: header → <meta charset> → utf-8 (təxmin)."""
    candidates = []
    header_charset = _charset_from_content_type(content_type)
    if header_charset:
        candidates.append(header_charset)
    meta = _META_CHARSET_RE.search(raw[:4096])
    if meta:
        candidates.append(meta.group(1).decode("ascii", "ignore"))
    candidates.extend(["utf-8", "cp1251", "iso-8859-9"])

    for charset in candidates:
        try:
            return raw.decode(charset, errors="strict")
        except (LookupError, UnicodeDecodeError):
            continue
    return raw.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Sorğu qurucuları
# ---------------------------------------------------------------------------

def _build_request(
    method: Optional[str],
    url: str,
    form: Optional[Dict[str, Any]] = None,
    json_body: Any = None,
    headers: Optional[Mapping[str, str]] = None,
) -> urllib.request.Request:
    hdrs = {
        "User-Agent": CONFIG.user_agent,
        "Accept-Language": "az,tr;q=0.9,en;q=0.8,ru;q=0.6",
        "Accept-Encoding": "gzip, deflate",
    }
    body: Optional[bytes] = None
    if json_body is not None:
        body = json.dumps(json_body).encode("utf-8")
        hdrs["Content-Type"] = "application/json"
        hdrs.setdefault("Accept", "application/json")
    elif form is not None:
        body = urllib.parse.urlencode(form).encode("utf-8")
        hdrs["Content-Type"] = "application/x-www-form-urlencoded"
        hdrs.setdefault("Accept", "text/html,application/xhtml+xml")

    if headers:
        hdrs.update({k: v for k, v in headers.items() if v is not None})

    return urllib.request.Request(
        url, data=body, headers=hdrs, method=method or ("POST" if body is not None else "GET")
    )


def _describe_network_error(exc: BaseException, url: str, timeout: float) -> NetworkError:
    reason = getattr(exc, "reason", exc)
    if isinstance(reason, (socket.timeout, TimeoutError)):
        return NetworkError(f"{url} — {timeout:g} saniyə ərzində cavab gəlmədi (timeout)")
    # Yaygın səbəbləri oxunaqlı şəklə salırıq
    hints = {
        "Network is unreachable": "şəbəkə əlçatmazdır (internet/proxy/DPI maneəsi ola bilər)",
        "Name or service not known": "domen adı həll olunmadı (DNS problemi)",
        "Temporary failure in name resolution": "DNS müvəqqəti əlçatmazdır",
        "Connection refused": "bağlantı rədd edildi (xidmət işləmir?)",
        "certificate verify failed": "SSL sertifikatı yoxlanıla bilmədi (köhnə CA paketi?)",
    }
    message = str(reason)
    for needle, hint in hints.items():
        if needle.lower() in message.lower():
            return NetworkError(f"{url} — {hint}: {message}")
    return NetworkError(f"{url} — {message}")


def request(
    method: Optional[str],
    url: str,
    *,
    form: Optional[Dict[str, Any]] = None,
    json_body: Any = None,
    headers: Optional[Mapping[str, str]] = None,
    timeout: Optional[float] = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> Response:
    """Sorğu göndərir və tam cavabı oxuyur (redirect-lər urllib tərəfindən izlənir)."""
    timeout = CONFIG.fetch_timeout if timeout is None else timeout
    req = _build_request(method, url, form, json_body, headers)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read(max_bytes)
            headers_dict = {k.lower(): v for k, v in resp.headers.items()}
            status = getattr(resp, "status", 200)
            final_url = resp.geturl() or url
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read(2048).decode("utf-8", "replace")
        except Exception:  # noqa: BLE001 — cavab gövdəsi oxunmursa əhəmiyyətli deyil
            pass
        finally:
            try:
                e.close()
            except Exception:  # noqa: BLE001
                pass
        raise HttpError(e.code, url, body) from e
    except (urllib.error.URLError, socket.timeout, TimeoutError, OSError) as e:
        raise _describe_network_error(e, url, timeout) from e

    raw = _decompress(raw, headers_dict.get("content-encoding", ""))
    return Response(url=final_url, status=status, headers=headers_dict, content=raw)


def get_text(url: str, *, timeout: Optional[float] = None, max_bytes: int = DEFAULT_MAX_BYTES) -> Response:
    return request("GET", url, timeout=timeout, max_bytes=max_bytes)


def post_form(
    url: str,
    form: Dict[str, Any],
    *,
    timeout: Optional[float] = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    headers: Optional[Mapping[str, str]] = None,
) -> Response:
    return request("POST", url, form=form, timeout=timeout, max_bytes=max_bytes, headers=headers)


def post_json(
    url: str,
    payload: Any,
    *,
    timeout: Optional[float] = None,
    max_bytes: int = 16_000_000,
) -> Any:
    """JSON göndərib JSON cavabı parse edir. Parse xətası ValueError kimi ötürülür."""
    resp = request("POST", url, json_body=payload, timeout=timeout, max_bytes=max_bytes)
    return json.loads(resp.text)


def open_stream(
    url: str, payload: Any, *, timeout: Optional[float] = None
) -> Tuple[Any, str]:
    """Streaming cavabı açır: (file-like response, content-type).

    Çağıran tərəf cavabı `for line in resp:` ilə sətir-sətir oxuyub `resp.close()`
    etməlidir. Xəta halında NetworkError / HttpError qaldırılır.
    """
    timeout = CONFIG.request_timeout if timeout is None else timeout
    req = _build_request("POST", url, json_body=payload, headers={"Accept": "text/event-stream"})
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read(2048).decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            pass
        finally:
            try:
                e.close()
            except Exception:  # noqa: BLE001
                pass
        raise HttpError(e.code, url, body) from e
    except (urllib.error.URLError, socket.timeout, TimeoutError, OSError) as e:
        raise _describe_network_error(e, url, timeout) from e

    return resp, resp.headers.get("content-type", "") or ""
