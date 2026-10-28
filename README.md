# Web Agent — Axtarışla Gücləndirilmiş Yerli LLM Aksesuarı

Yerli LLM serverinizi (llama.cpp server, vLLM, LM Studio və s. — OpenAI-uyğun
`/v1/chat/completions` endpoint-i verən istənilən server) real-vaxt veb
axtarışı ilə birləşdirən modul strukturlu agent.

> **Sıfır asılılıq:** layihə **heç bir xarici paket tələb etmir** — hər şey
> Python 3.10+ standart kitabxanası ilə yazılıb. `pip install` lazım deyil,
> `python run.py` kifayətdir. (Əvvəl `requests`, `duckduckgo-search`,
> `trafilatura`, `scikit-learn` istifadə olunurdu — hamısı çıxarıldı.)

## Pipeline

```
Sual
  │
  ▼
[1] search.py     → DuckDuckGo HTML interfeysində axtarış (urllib + html.parser),
  │                  iki backend (html → lite) + retry/backoff
  ▼
[2] fetch.py      → Hər linkin tam mətni paralel çəkilir (net.py + html_text.py:
  │                  naviqasiya/reklam/footer təmizlənir; alınmasa snippet qalır)
  ▼
[3] rerank.py     → Nəticələr suala uyğunluğa görə yenidən sıralanır:
  │                  embeddings (server dəstəkləsə) → təmiz Python TF-IDF → açar-söz
  ▼
[4] pipeline.py   → Ən yaxşı N mənbədən kontekst qurur, LLM-ə göndərir
  │
  ▼
[5] llm_client.py → Streaming (SSE, token-token) və ya tam cavab (urllib)
  │
  ▼
Cavab + mənbə siyahısı ([1], [2] sitatları ilə)
```

## Nə istifadə olunur (hamısı standart kitabxana)

| Ehtiyac | Əvvəl | İndi |
|---|---|---|
| HTTP sorğuları | `requests` | `urllib.request` + `gzip`/`zlib` + charset aşkarlanması → `agent/net.py` |
| Veb axtarışı | `ddgs` | `html.duckduckgo.com` / `lite.duckduckgo.com` + `html.parser` → `agent/search.py` |
| Mətn çıxarma | `trafilatura` (+`lxml`) | öz blok/link-sıxlığı əsaslı çıxarıcı → `agent/html_text.py` |
| Reranking | `scikit-learn` | təmiz Python TF-IDF + cosine → `agent/rerank.py` |
| LLM çağırışı | `requests` | `urllib` + əl ilə SSE parse → `agent/llm_client.py` |

## İşə salma

```bash
python run.py          # və ya: python -m agent
```

Heç bir quraşdırma addımı yoxdur. İstəsəniz paket kimi də qura bilərsiniz
(`pip install -e .`, sonra `web-agent` əmri) — amma bu da xarici paket çəkmir.

Seçimlər:

| Bayraq         | Təsvir                                              |
|----------------|------------------------------------------------------|
| `--no-stream`  | Streaming-i deaktiv edir, tam cavabı bir dəfəyə göstərir |
| `--no-fetch`   | Tam səhifə çəkməni deaktiv edir (sürətli, snippet-lə kifayətlənir) |
| `--top-k N`    | Rerank sonrası LLM-ə neçə mənbə ötürüləcəyini təyin edir |
| `--model NAME` | Server üzərindəki model adını göstərir              |
| `--doctor`     | Sistemin bütün mərhələlərini yoxlayıb çıxır         |

## Yoxlama (diaqnostika)

```bash
# 1) Bütün mərhələlər: mühit, LLM, embeddings, real axtarış, fetch, pipeline
python -m agent.doctor
python -m agent.doctor --offline      # yalnız şəbəkəsiz yoxlamalar

# 2) Canlı yoxlama — real DuckDuckGo + real səhifələr
python tests/live_check.py --query "Azərbaycan paytaxtı" --pages 3

# 3) Oflayn testlər — 18 test, lokal saytlar + lokal LLM serveri ilə
python -m unittest discover -s tests -v
```

Testlərin əhatəsi (18 test): DDG HTML/lite parse edilməsi, reklam linklərinin atılması,
backend fallback, axtarış xətası → `SearchError`, HTML-dən reklam/menyu/footer
təmizlənməsi, gzip + `windows-1251` charset, 404/ölü host davranışı, TF-IDF
rerank və sitat nömrələri, embeddings rerank, kontekst qurulması, streaming və
qeyri-streaming cavab, səhv LLM ünvanı, tarixçənin budanması, **və layihədə
xarici paket importunun olmaması**.

## Konfiqurasiya

Bütün parametrlər ətraf mühit dəyişənləri ilə idarə olunur — bax `.env.example`.
Layihə kökündə `.env` faylı yaratsanız (`cp .env.example .env`), dəyərlər
avtomatik oxunur; artıq `export` edilmiş dəyişənlər həmişə üstün gəlir.

Əsas dəyişənlər:

- `LLM_BASE_URL` — serverin ünvanı (default: `http://127.0.0.1:8080/v1`)
- `LLM_MODEL` — model adı (default: `local-model`)
- `SEARCH_MAX_RESULTS` — neçə xam nəticə axtarılsın (default: 6)
- `SEARCH_TIMEOUT` — axtarış backend-i üçün timeout (default: 15 s)
- `RERANK_TOP_K` — rerank sonrası neçəsi LLM-ə getsin (default: 4)
- `EMBEDDINGS_ENABLED` — server `/embeddings` dəstəkləyirsə semantik rerank aktivləşir
- `FETCH_MAX_BYTES` — səhifədən ən çox neçə bayt oxunacaq (default: 2 MB)
- `USER_AGENT` — bəzi saytlar bot UA-sını bloklayır; lazım gələrsə dəyişin
- `LOG_LEVEL` — `DEBUG` ilə hər addımın detalları görünür

Proxy lazımdırsa standart `http_proxy` / `https_proxy` dəyişənlərindən istifadə
olunur (urllib avtomatik oxuyur).

## Modul strukturu

```
.
├── run.py                        # Giriş nöqtəsi (python run.py)
├── pyproject.toml                # Paket metadatası (dependencies = [])
├── .env.example                  # Konfiqurasiya nümunəsi
├── tests/
│   ├── test_web_pipeline.py      # 18 oflayn test
│   └── live_check.py             # Real internet yoxlaması
└── agent/                        # Python paketi (__init__.py, __main__.py)
    ├── config.py                 # Mərkəzi konfiqurasiya (.env + env override)
    ├── exceptions.py              # Layihəyə xas xəta tipləri
    ├── logging_setup.py           # Struktur logging
    ├── net.py                     # urllib əsaslı HTTP (gzip, charset, proxy, SSE açma)
    ├── html_text.py               # HTML → əsas mətn (reklam/menyu təmizləyici)
    ├── search.py                  # DuckDuckGo HTML axtarışı + retry/backoff
    ├── fetch.py                   # Paralel tam-səhifə mətn çıxarma
    ├── rerank.py                  # embeddings → TF-IDF → açar-söz sıralaması
    ├── llm_client.py              # Streaming/qeyri-streaming LLM çağırışı
    ├── history.py                 # Söhbət tarixçəsi (avtomatik budama)
    ├── pipeline.py                # Bütün mərhələləri birləşdirən orkestrasiya
    ├── doctor.py                  # Diaqnostika (python -m agent.doctor)
    └── cli.py                     # İnteraktiv REPL
```

## Genişləndirmə fikirləri

- **Keşləmə**: eyni sual üçün axtarışı təkrarlamamaq üçün `functools.lru_cache`
  kifayətdir (paket lazım deyil).
- **Çoxlu axtarış provideri**: `search.py`-da `SEARCH_BACKENDS` siyahısına yeni
  HTML endpoint + parser əlavə etmək kifayətdir.
- **Cross-encoder reranker**: `rerank.py`-a Cohere Rerank və ya yerli
  `bge-reranker` modelini əlavə etmək dəqiqliyi artırar.
- **Struktur çıxış**: LLM-dən JSON formatda cavab tələb edib UI-a bağlamaq olar.

## Məhdudiyyətlər

- DuckDuckGo ictimai HTML interfeysi bəzən bot-yoxlaması (captcha) qaytarır.
  Bu halda ikinci backend yoxlanılır; hər ikisi bloklanarsa agent mənbəsiz cavab
  verir və log-da xəbərdarlıq yazır (proqram çökmür).
- `html_text` sadə evristikadır: trafilatura qədər dəqiq deyil, amma 15+ MB
  asılılıq tələb etmir. Lazım gələrsə `FETCH_ENABLED=false` ilə söndürün.
- JavaScript ilə yüklənən səhifələrdən mətn çıxarılmır (brauzer yoxdur).
