# Web Agent — Axtarışla Gücləndirilmiş Yerli LLM Aksesuarı

Yerli LLM serverinizi (llama.cpp server, vLLM, LM Studio və s. — OpenAI-uyğun
`/v1/chat/completions` endpoint-i verən istənilən server) real-vaxt veb
axtarışı ilə birləşdirən modul strukturlu agent.

## Pipeline

```
Sual
  │
  ▼
[1] search.py     → DuckDuckGo-da axtarış (retry + exponential backoff)
  │
  ▼
[2] fetch.py       → Hər linkin tam mətnini paralel çəkir (trafilatura ilə
  │                   reklam/menyu təmizlənir; uğursuz olsa snippet qalır)
  ▼
[3] rerank.py       → Nəticələri suala uyğunluğa görə yenidən sıralayır:
  │                   embeddings (server dəstəkləsə) → TF-IDF → açar-söz
  ▼
[4] pipeline.py     → Ən yaxşı N mənbədən kontekst qurur, LLM-ə göndərir
  │
  ▼
[5] llm_client.py   → Streaming (token-token) və ya tam cavab
  │
  ▼
Cavab + mənbə siyahısı ([1], [2] sitatları ilə)
```

## Quraşdırma

```bash
pip install -r requirements.txt --break-system-packages
```

## İşə salma

```bash
python run.py
```

Seçimlər:

| Bayraq         | Təsvir                                              |
|----------------|------------------------------------------------------|
| `--no-stream`  | Streaming-i deaktiv edir, tam cavabı bir dəfəyə göstərir |
| `--no-fetch`   | Tam səhifə çəkməni deaktiv edir (sürətli, snippet-lə kifayətlənir) |
| `--top-k N`    | Rerank sonrası LLM-ə neçə mənbə ötürüləcəyini təyin edir |
| `--model NAME` | Server üzərindəki model adını göstərir              |

## Konfiqurasiya

Bütün parametrlər ətraf mühit dəyişənləri ilə idarə olunur — bax `.env.example`.
Əsas dəyişənlər:

- `LLM_BASE_URL` — serverin ünvanı (default: `http://127.0.0.1:8080/v1`)
- `SEARCH_MAX_RESULTS` — neçə xam nəticə axtarılsın (default: 6)
- `RERANK_TOP_K` — rerank sonrası neçəsi LLM-ə getsin (default: 4)
- `EMBEDDINGS_ENABLED` — server `/embeddings` dəstəkləyirsə semantik rerank aktivləşir

## Modul strukturu

```
agent/
├── config.py          # Mərkəzi konfiqurasiya (env-var override dəstəyi ilə)
├── exceptions.py       # Layihəyə xas xəta tipləri
├── logging_setup.py    # Struktur logging
├── search.py            # DuckDuckGo axtarışı + retry/backoff
├── fetch.py              # Paralel tam-səhifə mətn çıxarma (trafilatura)
├── rerank.py              # 3-səviyyəli reranking strategiyası
├── llm_client.py           # Streaming/qeyri-streaming LLM çağırışı
├── history.py               # Söhbət tarixçəsi (avtomatik budama)
├── pipeline.py                # Bütün mərhələləri birləşdirən orkestrasiya
└── cli.py                      # İnteraktiv REPL
run.py                            # Giriş nöqtəsi
```

## Genişləndirmə fikirləri

- **Keşləmə**: eyni sual üçün axtarışı təkrarlamamaq üçün `functools.lru_cache`
  və ya Redis əlavə edilə bilər.
- **Çoxlu axtarış provideri**: `search.py`-ı abstraksiya edib Brave/Tavily/SerpAPI
  arasında seçim vermək mümkündür.
- **Cross-encoder reranker**: `rerank.py`-a Cohere Rerank və ya yerli
  `bge-reranker` modelini əlavə etmək dəqiqliyi artırar.
- **Struktur çıxış**: LLM-dən JSON formatda cavab tələb edib UI-a bağlamaq olar.
