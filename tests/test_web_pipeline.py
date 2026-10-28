"""Veb hissəsinin (axtarış → fetch → rerank → LLM) tam yoxlanması — internetsiz.

Testlər lokal HTTP serverlər qaldırır: biri "veb saytları" (reklam/menyu ilə
birlikdə real HTML), biri isə OpenAI-uyğun LLM serveri (həm streaming, həm tam
JSON cavab). Beləliklə şəbəkə olmadan da bütün zəncir real kodla işlədilir.

İşə salma (repo kökündən):

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from agent import pipeline
from agent.config import CONFIG
from agent.fetch import enrich_with_full_text
from agent.history import ConversationHistory
from agent.llm_client import chat_completion, chat_completion_stream
from agent.rerank import rerank
from agent.search import SearchResult, search_web

# ---------------------------------------------------------------------------
# Test "veb saytları"
# ---------------------------------------------------------------------------

ARTICLE_BAKU = """
<div class="site">
  <header><a href="/">Ana səhifə</a> | <a href="/xeber">Xəbərlər</a></header>
  <article>
    <h1>Bakının əhalisi</h1>
    <p>Bakı Azərbaycanın paytaxtıdır və ölkənin ən böyük şəhəridir.
    Şəhərin əhalisi təxminən 2,3 milyon nəfərdir.</p>
    <p>Bakı Xəzər dənizinin qərb sahilində, Abşeron yarımadasında yerləşir.</p>
  </article>
  <aside>REKLAM: Bu ay endirimlər! Klikləyin və qazanın!</aside>
  <footer>© 2026 Nümunə Sayt</footer>
</div>
"""

ARTICLE_COOKING = """
<html><body>
  <nav>Menyu: Reseptlər | Mətbəx | Əlaqə</nav>
  <main>
    <h1>Plov necə bişirilir</h1>
    <p>Düyü əvvəlcə yuyulur, sonra qazanda yağda qovrulur və su əlavə edilir.</p>
  </main>
  <div class="ads">REKLAM: endirim endirim endirim</div>
</body></html>
"""


class SiteHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # test zamanı səssiz
        pass

    def do_GET(self):  # noqa: N802
        if self.path == "/baki":
            body, status = ARTICLE_BAKU, 200
        elif self.path == "/plov":
            body, status = ARTICLE_COOKING, 200
        else:
            body, status = "<html><body>404</body></html>", 404

        raw = body.encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "text/html; charset=utf-8")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def _serve(handler) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


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
            data = {"data": [{"index": i, "embedding": [1.0, float(len(t) % 5)]}
                             for i, t in enumerate(payload["input"])]}
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

        # Qlobal konfiqurasiyanı testə uyğunlaşdırırıq (CLI-dakı kimi object.__setattr__).
        cls._saved = {k: getattr(CONFIG, k) for k in
                      ("llm_base_url", "fetch_enabled", "embeddings_enabled", "rerank_top_k")}
        object.__setattr__(CONFIG, "llm_base_url", cls.llm_url)
        object.__setattr__(CONFIG, "fetch_enabled", True)
        object.__setattr__(CONFIG, "rerank_top_k", 4)

    @classmethod
    def tearDownClass(cls) -> None:
        for key, value in cls._saved.items():
            object.__setattr__(CONFIG, key, value)
        for server in (cls.site, cls.llm):
            server.shutdown()
            server.server_close()

    # -- axtarış ---------------------------------------------------------
    def test_01_search_maps_either_api_shape(self) -> None:
        """search_web həm dict (köhnə API), həm obyekt (yeni ddgs) nəticələrini oxuyur."""

        class DictDDGS:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def text(self, query, max_results=None):
                return [
                    {"title": "Bakı", "href": self_url() + "/baki", "body": "paytaxt"},
                    # təkrar link — çıxarılmalıdır
                    {"title": "Bakı (təkrar)", "href": self_url() + "/baki", "body": "təkrar"},
                    # linki olmayan nəticə — atılmalıdır
                    {"title": "Link yoxdur", "href": "", "body": "..."},
                    {"title": "Plov", "href": self_url() + "/plov", "body": "resept"},
                ]

        class ObjDDGS(DictDDGS):
            def text(self, query, max_results=None):
                class R:
                    def __init__(self, title, href, body):
                        self.title, self.href, self.body = title, href, body

                return [R("Bakı", self_url() + "/baki", "paytaxt")]

        import agent.search as search_module

        original = search_module.DDGS
        try:
            search_module.DDGS = DictDDGS
            results = search_web("Bakı")
            self.assertEqual([r.title for r in results], ["Bakı", "Plov"])
            self.assertEqual([r.rank for r in results], [1, 2])

            search_module.DDGS = ObjDDGS
            results = search_web("Bakı")
            self.assertEqual(results[0].title, "Bakı")
            self.assertTrue(results[0].url.endswith("/baki"))
        finally:
            search_module.DDGS = original

    def test_02_search_failure_raises_search_error(self) -> None:
        import agent.search as search_module
        from agent.exceptions import SearchError

        class FailingDDGS:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def text(self, query, max_results=None):
                raise RuntimeError("şəbəkə yoxdur")

        original = search_module.DDGS
        saved_backoff = CONFIG.search_backoff_seconds
        object.__setattr__(CONFIG, "search_backoff_seconds", 0.0)
        try:
            search_module.DDGS = FailingDDGS
            with self.assertRaises(SearchError):
                search_web("test")
        finally:
            search_module.DDGS = original
            object.__setattr__(CONFIG, "search_backoff_seconds", saved_backoff)

    # -- fetch -----------------------------------------------------------
    def test_03_fetch_extracts_article_and_drops_ads(self) -> None:
        results = [
            SearchResult("Bakı", f"{self.site_url}/baki", "qısa snippet", 1),
            SearchResult("Plov", f"{self.site_url}/plov", "qısa snippet", 2),
            SearchResult("Sınıq", f"{self.site_url}/yoxdur", "qısa snippet", 3),
        ]
        enriched = enrich_with_full_text(results)
        baku = enriched[0]

        self.assertIn("2,3 milyon", baku.full_text)          # əsl məzmun çıxarıldı
        self.assertNotIn("REKLAM", baku.full_text)            # reklam təmizləndi
        self.assertIn("Plov", enriched[1].full_text + enriched[1].snippet)
        # 404 olan səhifə çökmür, snippet ilə qalır
        self.assertEqual(enriched[2].full_text, "")
        self.assertEqual(enriched[2].display_text, "qısa snippet")

    # -- rerank ----------------------------------------------------------
    def test_04_rerank_orders_by_relevance_and_renumbers(self) -> None:
        object.__setattr__(CONFIG, "embeddings_enabled", False)  # TF-IDF / açar-söz yolu
        try:
            results = [
                SearchResult("Resept", f"{self.site_url}/plov",
                             "Düyü yuyulur, qazanda qovrulur, plov bişirilir.", 1),
                SearchResult("Bakı", f"{self.site_url}/baki",
                             "Bakı Azərbaycanın paytaxtıdır, əhalisi 2,3 milyon nəfərdir.", 2),
                SearchResult("Digər", f"{self.site_url}/x", "Tamamilə aidiyyətsiz mətn.", 3),
            ]
            ranked = rerank("Bakı Azərbaycan paytaxtı əhalisi", results)
            self.assertEqual(ranked[0].url, f"{self.site_url}/baki")
            self.assertEqual([r.rank for r in ranked], [1, 2, 3])  # sitat nömrələri düzgün
        finally:
            object.__setattr__(CONFIG, "embeddings_enabled", True)

    def test_05_rerank_with_embeddings_uses_server(self) -> None:
        object.__setattr__(CONFIG, "embeddings_enabled", True)
        LLMHandler.received.clear()
        results = [
            SearchResult("A", f"{self.site_url}/baki", "Bakı paytaxtdır.", 1),
            SearchResult("B", f"{self.site_url}/plov", "Plov resepti: düyü və ət.", 2),
        ]
        ranked = rerank("Bakı paytaxt", results)
        self.assertEqual(len(ranked), 2)
        self.assertEqual([r.rank for r in ranked], [1, 2])
        self.assertTrue(
            any(path.endswith("/embeddings") for path, _ in LLMHandler.received),
            "embeddings endpoint çağırılmalı idi",
        )

    # -- pipeline --------------------------------------------------------
    def test_06_pipeline_builds_context_with_sources(self) -> None:
        original = pipeline.search_web
        pipeline.search_web = lambda q: [
            SearchResult("Bakı", f"{self.site_url}/baki", "snippet", 1),
            SearchResult("Plov", f"{self.site_url}/plov", "snippet", 2),
        ]
        try:
            history = ConversationHistory()
            content, sources = pipeline.answer("Bakı haradadır?", history, stream=False)
        finally:
            pipeline.search_web = original

        self.assertIn("Bakı", content)
        self.assertTrue(sources, "mənbələr qaytarılmalı idi")
        self.assertEqual([s.rank for s in sources], list(range(1, len(sources) + 1)))

        system_prompt = LLMHandler.received[-1][1]["messages"][0]["content"]
        self.assertIn("Mənbələr:", system_prompt)
        self.assertIn("[1]", system_prompt)          # kontekst nömrəli sitatlar
        self.assertIn("2,3 milyon", system_prompt)   # fetch edilmiş tam mətn kontekstdədir
        self.assertIn("Azərbaycan dilində", system_prompt)

        block = pipeline.format_sources(sources)
        self.assertIn("Mənbələr", block)

    def test_07_search_failure_does_not_crash_pipeline(self) -> None:
        from agent.exceptions import SearchError

        def boom(query):
            raise SearchError("test: şəbəkə yoxdur")

        original = pipeline.search_web
        pipeline.search_web = boom
        try:
            content, sources = pipeline.answer("sual", ConversationHistory(), stream=False)
        finally:
            pipeline.search_web = original

        self.assertEqual(sources, [])
        self.assertIn("Bakı", content)  # LLM yenə cavab verir, proqram çökmür

    # -- LLM klient ------------------------------------------------------
    def test_08_streaming_and_non_streaming(self) -> None:
        messages = [{"role": "user", "content": "Salam"}]

        full = chat_completion(messages)
        self.assertIn("paytaxtıdır", full)

        chunks = list(chat_completion_stream(messages))
        self.assertGreater(len(chunks), 1)
        self.assertEqual("".join(chunks), "Bakı Azərbaycanın paytaxtıdır [1].")

    def test_09_bad_llm_url_raises_connection_error(self) -> None:
        from agent.exceptions import LLMConnectionError

        object.__setattr__(CONFIG, "llm_base_url", "http://127.0.0.1:1/v1")
        try:
            with self.assertRaises(LLMConnectionError):
                chat_completion([{"role": "user", "content": "x"}])
        finally:
            object.__setattr__(CONFIG, "llm_base_url", self.llm_url)

    def test_10_history_trimming(self) -> None:
        saved = CONFIG.max_history_turns
        object.__setattr__(CONFIG, "max_history_turns", 2)
        try:
            history = ConversationHistory()
            for i in range(5):
                history.add_user(f"sual {i}")
                history.add_assistant(f"cavab {i}")
            self.assertEqual(len(history.as_list()), 4)  # 2 cüt
            self.assertEqual(history.as_list()[-1]["content"], "cavab 4")
        finally:
            object.__setattr__(CONFIG, "max_history_turns", saved)


def self_url() -> str:
    """Axtarış stub-u üçün lokal saytın ünvanı (test sinfindən asılı olmadan)."""
    return WebPipelineTests.site_url


if __name__ == "__main__":
    unittest.main(verbosity=2)
