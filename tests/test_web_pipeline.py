"""Tam yoxlama — üçüncü tərəf paketləri olmadan, internetsiz.

Testlər lokal HTTP serverlər qaldırır: biri "veb saytları" (real HTML, gzip
sıxılması, 404), biri isə OpenAI-uyğun LLM serveri (SSE streaming, tam JSON və
/embeddings). Axtarış üçün DuckDuckGo HTML nümunələri (html + lite markup)
istifadə olunur. Beləliklə şəbəkə olmadan da bütün zəncir real kodla işlədilir.

İşə salma (repo kökündən):

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import ast
import gzip
import json
import pathlib
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from agent import pipeline
from agent.config import CONFIG
from agent.exceptions import LLMConnectionError, SearchError
from agent.fetch import enrich_with_full_text, fetch_page_text
from agent.history import ConversationHistory
from agent.html_text import extract_text
from agent.llm_client import chat_completion, chat_completion_stream
from agent.rerank import rerank, tokenize
from agent.search import SearchResult, parse_results, search_web

# ---------------------------------------------------------------------------
# Nümunə DuckDuckGo cavabları
# ---------------------------------------------------------------------------

DDG_HTML = """
<html><body>
<div class="result results_links results_links_deep web-result">
  <div class="links_main links_deep result__body">
    <h2 class="result__title">
      <a rel="nofollow" class="result__a"
         href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fbaki&amp;rut=abc">Bakı — Vikipediya</a>
    </h2>
    <a class="result__snippet" href="https://example.com/baki">Bakı Azərbaycanın <b>paytaxtıdır</b>.</a>
  </div>
</div>
<div class="result results_links web-result">
  <div class="result__body">
    <h2><a rel="nofollow" class="result__a" href="https://duckduckgo.com/y.js?ad_provider=x">REKLAM: endirim</a></h2>
    <a class="result__snippet">Reklam mətni</a>
  </div>
</div>
<div class="result results_links web-result">
  <div class="result__body">
    <h2><a rel="nofollow" class="result__a"
           href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fplov">Plov resepti</a></h2>
    <div class="result__snippet">Düyü və ətlə hazırlanan <a href="https://example.org/plov">plov</a> resepti.</div>
  </div>
</div>
</body></html>
"""

DDG_LITE = """
<html><body><table>
  <tr><td><a rel="nofollow" href="https://example.net/lite" class="result-link">Lite nəticə</a></td></tr>
  <tr><td class="result-snippet">Lite backend-dən gələn snippet.</td></tr>
</table></body></html>
"""

ARTICLE_BAKU = """
<html><head><title>Bakı - Nümunə Sayt</title>
<style>.x{color:red}</style><script>var a = 1;</script></head>
<body>
<header><a href="/">Ana səhifə</a> <a href="/xeber">Xəbərlər</a> <a href="/elaqe">Əlaqə</a></header>
<nav><ul><li><a href="/a">Siyasət</a></li><li><a href="/b">İqtisadiyyat</a></li></ul></nav>
<div class="content">
  <article>
    <h1>Bakının əhalisi</h1>
    <p>Bakı Azərbaycanın paytaxtıdır və ölkənin ən böyük şəhəridir.
    Şəhərin əhalisi təxminən 2,3 milyon nəfərdir.</p>
    <p>Bakı Xəzər dənizinin qərb sahilində, Abşeron yarımadasında yerləşir.</p>
  </article>
  <aside>REKLAM: Bu ay endirimlər! Klikləyin və qazanın!</aside>
</div>
<footer>© 2026 Nümunə Sayt</footer>
</body></html>
"""


# ---------------------------------------------------------------------------
# Test veb saytı
# ---------------------------------------------------------------------------

class SiteHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # test zamanı səssiz
        pass

    def _send(self, body: bytes, status: int = 200, content_type: str = "text/html; charset=utf-8",
              encoding: str = "") -> None:
        self.send_response(status)
        self.send_header("content-type", content_type)
        if encoding:
            self.send_header("content-encoding", encoding)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path == "/baki":
            self._send(ARTICLE_BAKU.encode("utf-8"))
        elif self.path == "/gzip":
            self._send(gzip.compress(ARTICLE_BAKU.encode("utf-8")), encoding="gzip")
        elif self.path == "/cp1251":
            page = ("<html><body><article><p>Баку — столица Азербайджана, "
                    "население около 2,3 миллиона человек.</p></article></body></html>")
            self._send(page.encode("cp1251"), content_type="text/html; charset=windows-1251")
        elif self.path == "/binary":
            self._send(b"%PDF-1.4 fake", content_type="application/pdf")
        elif self.path == "/redirect":
            self.send_response(302)
            self.send_header("location", "/baki")
            self.send_header("content-length", "0")
            self.end_headers()
        elif self.path == "/hang":
            import time
            time.sleep(3)
            self._send(b"<html><body>gec</body></html>")
        else:
            self._send("<html><body>404 — tapılmadı</body></html>".encode(), status=404)


# ---------------------------------------------------------------------------
# Test LLM serveri (OpenAI-uyğun)
# ---------------------------------------------------------------------------

class LLMHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    received: list = []

    def log_message(self, *args):
        pass

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("content-length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        LLMHandler.received.append((self.path, payload))

        if self.path.endswith("/embeddings"):
            data = {"data": [
                {"index": i, "embedding": [1.0, float(len(text) % 5)]}
                for i, text in enumerate(payload["input"])
            ]}
            self._json(data)
            return

        if payload.get("stream"):
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("transfer-encoding", "chunked")
            self.end_headers()
            for token in ["Bakı", " Azərbaycanın", " paytaxtıdır", " [1]."]:
                chunk = "data: " + json.dumps({"choices": [{"delta": {"content": token}}]}) + "\n\n"
                raw = chunk.encode()
                self.wfile.write(f"{len(raw):x}\r\n".encode() + raw + b"\r\n")
                self.wfile.flush()
            raw = b"data: [DONE]\n\n"
            self.wfile.write(f"{len(raw):x}\r\n".encode() + raw + b"\r\n0\r\n\r\n")
            self.wfile.flush()
            return

        self._json({"choices": [{"message": {"content": "Bakı Azərbaycanın paytaxtıdır [1]."}}]})

    def _json(self, obj) -> None:
        raw = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def _serve(handler) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


# ---------------------------------------------------------------------------
# Testlər
# ---------------------------------------------------------------------------

class WebPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.site = _serve(SiteHandler)
        cls.llm = _serve(LLMHandler)
        cls.site_url = f"http://127.0.0.1:{cls.site.server_address[1]}"
        cls.llm_url = f"http://127.0.0.1:{cls.llm.server_address[1]}/v1"

        cls._saved = {k: getattr(CONFIG, k) for k in
                      ("llm_base_url", "fetch_enabled", "embeddings_enabled",
                       "rerank_top_k", "search_backoff_seconds", "max_history_turns",
                       "fetch_timeout")}
        object.__setattr__(CONFIG, "llm_base_url", cls.llm_url)
        object.__setattr__(CONFIG, "fetch_enabled", True)
        object.__setattr__(CONFIG, "rerank_top_k", 4)
        object.__setattr__(CONFIG, "search_backoff_seconds", 0.0)

    @classmethod
    def tearDownClass(cls) -> None:
        for key, value in cls._saved.items():
            object.__setattr__(CONFIG, key, value)
        for server in (cls.site, cls.llm):
            server.shutdown()
            server.server_close()

    # -- 0) paket asılılığı yoxdur ---------------------------------------
    def test_00_no_third_party_imports(self) -> None:
        """Layihə kodu yalnız standart kitabxananı import edir."""
        stdlib = set(__import__("sys").stdlib_module_names)
        package_dir = pathlib.Path(__file__).resolve().parent.parent / "agent"
        offenders = []

        for path in sorted(package_dir.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    names = [node.module]
                for name in names:
                    root = name.split(".")[0]
                    if root not in stdlib and root != "agent":
                        offenders.append(f"{path.name}: {name}")

        self.assertEqual(offenders, [], f"xarici paket importu tapıldı: {offenders}")

    # -- 1) axtarış HTML parse -------------------------------------------
    def test_01_parse_ddg_html_results(self) -> None:
        results = parse_results(DDG_HTML, 10)
        self.assertEqual([r.title for r in results], ["Bakı — Vikipediya", "Plov resepti"])
        # yönləndirmə linki əsl ünvana çevrilməlidir
        self.assertEqual(results[0].url, "https://example.com/baki")
        self.assertEqual(results[0].snippet, "Bakı Azərbaycanın paytaxtıdır.")
        self.assertEqual(results[1].url, "https://example.org/plov")
        self.assertEqual([r.rank for r in results], [1, 2])

    def test_02_parse_ddg_lite_results(self) -> None:
        results = parse_results(DDG_LITE, 10)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].url, "https://example.net/lite")
        self.assertEqual(results[0].snippet, "Lite backend-dən gələn snippet.")

    def test_03_search_backend_fallback_and_dedupe(self) -> None:
        """Birinci backend bloklanırsa, ikincisi işlədilir; təkrar linklər atılır."""
        import agent.search as search_module

        calls = []

        def fake_fetch(url, query):
            calls.append(url)
            if url == search_module.SEARCH_BACKENDS[0]:
                return "<html><body>Our systems have detected unusual traffic</body></html>"
            return DDG_LITE

        original = search_module._fetch_backend
        search_module._fetch_backend = fake_fetch
        try:
            results = search_web("Bakı")
        finally:
            search_module._fetch_backend = original

        self.assertEqual(results[0].url, "https://example.net/lite")
        self.assertEqual(len(calls), 2)

    def test_04_search_total_failure_raises(self) -> None:
        import agent.search as search_module
        from agent import net

        def boom(url, query):
            raise net.NetworkError("şəbəkə yoxdur")

        original = search_module._fetch_backend
        search_module._fetch_backend = boom
        try:
            with self.assertRaises(SearchError):
                search_web("test")
        finally:
            search_module._fetch_backend = original

    # -- 2) HTML → mətn ---------------------------------------------------
    def test_05_extract_text_drops_boilerplate(self) -> None:
        text = extract_text(ARTICLE_BAKU)
        self.assertIn("2,3 milyon", text)
        self.assertIn("Bakının əhalisi", text)          # başlıq saxlanılır
        self.assertNotIn("REKLAM", text)                # reklam atılır
        self.assertNotIn("Xəbərlər", text)              # header/menyu atılır
        self.assertNotIn("©", text)                     # footer atılır
        self.assertNotIn("var a = 1", text)             # script atılır

    def test_06_extract_text_edge_cases(self) -> None:
        self.assertEqual(extract_text(""), "")
        self.assertEqual(extract_text("<nav><a href='/'>Ana</a></nav>"), "")
        self.assertEqual(extract_text("<div><p>Qısa mətn.</p></div>"), "Qısa mətn.")

    def test_06b_void_tags_do_not_swallow_page(self) -> None:
        """<input>/<img> kimi bağlanmayan teqlər qalan səhifəni "udmamalıdır"."""
        page = """<html><body><form><input type="text" name="q"><button>Axtar</button></form>
        <article><p>Bu mətn input teqindən sonra gəlir və görünməlidir. İkinci cümlə də var.</p></article>
        <img src="x.png"><p>Bu da img teqindən sonra gələn mətndir və itməməlidir.</p></body></html>"""
        text = extract_text(page)
        self.assertIn("input teqindən sonra", text)
        self.assertIn("img teqindən sonra", text)

    def test_06c_fallback_for_unusual_markup(self) -> None:
        """Qeyri-standart markupda sadə ehtiyat rejim işə düşür."""
        page = "<div>" + "<span>cümlə</span> " * 40 + "</div>"
        text = extract_text(page)
        self.assertGreater(len(text), 100)

    # -- 3) fetch ---------------------------------------------------------
    def test_07_fetch_real_pages(self) -> None:
        baku = fetch_page_text(f"{self.site_url}/baki")
        self.assertIn("2,3 milyon", baku)

        gz = fetch_page_text(f"{self.site_url}/gzip")   # gzip açılmalıdır
        self.assertIn("2,3 milyon", gz)

        cp = fetch_page_text(f"{self.site_url}/cp1251")  # charset aşkarlanması
        self.assertIn("Баку", cp)
        self.assertIn("2,3 миллиона", cp)

        self.assertEqual(fetch_page_text(f"{self.site_url}/binary"), "")  # PDF keçilir)
        self.assertEqual(fetch_page_text(f"{self.site_url}/redirect"), baku)  # redirect izlənir

    def test_08_enrich_handles_broken_pages(self) -> None:
        results = [
            SearchResult("Bakı", f"{self.site_url}/baki", "qısa snippet", 1),
            SearchResult("Sınıq", f"{self.site_url}/yoxdur", "qısa snippet", 2),
            SearchResult("Ölü host", "http://127.0.0.1:1/x", "qısa snippet", 3),
        ]
        enriched = enrich_with_full_text(results)
        self.assertIn("2,3 milyon", enriched[0].full_text)
        for broken in enriched[1:]:  # 404 və qoşulmayan host proqramı dayandırmır
            self.assertEqual(broken.full_text, "")
            self.assertEqual(broken.display_text, "qısa snippet")

    # -- 4) rerank --------------------------------------------------------
    def test_09_tokenize_and_tfidf_rerank(self) -> None:
        self.assertIn("bakı", tokenize("Bakı və Azərbaycan"))
        self.assertNotIn("və", tokenize("Bakı və Azərbaycan"))  # stopword atılır

        object.__setattr__(CONFIG, "embeddings_enabled", False)
        try:
            results = [
                SearchResult("Resept", "https://example.org/plov",
                             "Düyü yuyulur, qazanda qovrulur, plov bişirilir.", 1),
                SearchResult("Bakı", "https://example.org/baki",
                             "Bakı Azərbaycanın paytaxtıdır, əhalisi 2,3 milyon nəfərdir.", 2),
                SearchResult("Digər", "https://example.org/x", "Tamamilə aidiyyətsiz mətn.", 3),
            ]
            ranked = rerank("Bakı Azərbaycan paytaxtı əhalisi", results)
            self.assertEqual(ranked[0].url, "https://example.org/baki")
            self.assertEqual([r.rank for r in ranked], [1, 2, 3])
        finally:
            object.__setattr__(CONFIG, "embeddings_enabled", True)

    def test_10_rerank_uses_embeddings_when_available(self) -> None:
        object.__setattr__(CONFIG, "embeddings_enabled", True)
        LLMHandler.received.clear()
        results = [
            SearchResult("A", "https://example.org/a", "Bakı paytaxtdır.", 1),
            SearchResult("B", "https://example.org/b", "Plov resepti: düyü və ət.", 2),
        ]
        ranked = rerank("Bakı paytaxt", results)
        self.assertEqual([r.rank for r in ranked], [1, 2])
        self.assertTrue(any(path.endswith("/embeddings") for path, _ in LLMHandler.received))

    # -- 5) pipeline ------------------------------------------------------
    def test_11_pipeline_builds_context_with_sources(self) -> None:
        original = pipeline.search_web
        pipeline.search_web = lambda q: [
            SearchResult("Bakı", f"{self.site_url}/baki", "snippet", 1),
            SearchResult("Plov", f"{self.site_url}/baki?x=1", "snippet", 2),
        ]
        try:
            content, sources = pipeline.answer("Bakı haradadır?", ConversationHistory(), stream=False)
        finally:
            pipeline.search_web = original

        self.assertIn("Bakı", content)
        self.assertTrue(sources)
        self.assertEqual([s.rank for s in sources], list(range(1, len(sources) + 1)))

        system_prompt = LLMHandler.received[-1][1]["messages"][0]["content"]
        self.assertIn("Mənbələr:", system_prompt)
        self.assertIn("[1]", system_prompt)
        self.assertIn("2,3 milyon", system_prompt)   # fetch edilmiş mətn kontekstdədir
        self.assertIn("Azərbaycan dilində", system_prompt)
        self.assertIn("Mənbələr", pipeline.format_sources(sources))

    def test_12_pipeline_survives_search_failure(self) -> None:
        def boom(query):
            raise SearchError("test: şəbəkə yoxdur")

        original = pipeline.search_web
        pipeline.search_web = boom
        try:
            content, sources = pipeline.answer("sual", ConversationHistory(), stream=False)
        finally:
            pipeline.search_web = original

        self.assertEqual(sources, [])
        self.assertIn("Bakı", content)

    # -- 6) LLM klient ----------------------------------------------------
    def test_13_streaming_and_non_streaming(self) -> None:
        messages = [{"role": "user", "content": "Salam"}]
        full = chat_completion(messages)
        self.assertIn("paytaxtıdır", full)

        chunks = list(chat_completion_stream(messages))
        self.assertGreater(len(chunks), 1)
        self.assertEqual("".join(chunks), "Bakı Azərbaycanın paytaxtıdır [1].")

    def test_14_bad_llm_url_raises_connection_error(self) -> None:
        object.__setattr__(CONFIG, "llm_base_url", "http://127.0.0.1:1/v1")
        try:
            with self.assertRaises(LLMConnectionError):
                chat_completion([{"role": "user", "content": "x"}])
            with self.assertRaises(LLMConnectionError):
                list(chat_completion_stream([{"role": "user", "content": "x"}]))
        finally:
            object.__setattr__(CONFIG, "llm_base_url", self.llm_url)

    # -- 7) tarixçə -------------------------------------------------------
    def test_15_history_trimming(self) -> None:
        object.__setattr__(CONFIG, "max_history_turns", 2)
        try:
            history = ConversationHistory()
            for i in range(5):
                history.add_user(f"sual {i}")
                history.add_assistant(f"cavab {i}")
            self.assertEqual(len(history.as_list()), 4)
            self.assertEqual(history.as_list()[-1]["content"], "cavab 4")
        finally:
            object.__setattr__(CONFIG, "max_history_turns", 6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
