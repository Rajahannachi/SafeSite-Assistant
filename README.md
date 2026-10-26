# SafeSite Assistant

A RAG chatbot that answers construction-site safety questions **only** from the company's safety documents, cites every fact as `[file, p.X]`, and refuses with *"I could not find this in the safety documents."* when the answer is not there.

Stack: Python 3.11 · LangChain (LCEL) · local `all-MiniLM-L6-v2` embeddings · Chroma on disk · free OpenRouter LLM (`:free`) · FastAPI · Streamlit · Docker.

## Setup (4 commands)

```bash
python -m venv .venv && .venv\Scripts\activate        # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env                                 # then put your OPENROUTER_API_KEY in .env
python ingest.py && python -m uvicorn api:app
```

Then open http://localhost:8000/docs, or:

```bash
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d "{\"question\": \"Which PPE is required for concrete cutting?\"}"
```

Optional: `streamlit run streamlit_app.py` (chat UI, needs the API running) · `docker compose up --build` (API on :8000, UI on :8501).

## Architecture

```
                         INGESTION (ingest.py, offline, idempotent)
 data/*.pdf|md|txt --> PyPDFLoader / TextLoader --> RecursiveCharacterTextSplitter --> all-MiniLM-L6-v2
                         (metadata: source, page)      (800 chars, 120 overlap)          (local, 384-d)
                                                                                              |
                                                                                              v
                                                                         Chroma (vectorstore/, SHA-256 chunk ids)
                                                                                              |
                         QUERY (rag_chain.py + api.py, per request)                           | top-4 chunks
 POST /ask --> [condense follow-up if history] --> retriever (similarity k=4 | MMR | hybrid BM25+vector | + rerank)
           --> format_docs ([source: file, p.X] headers) --> prompt --> LLM (:free, retry + fallbacks)
           --> StrOutputParser --> {"answer", "sources": [{file, page, excerpt}], "latency_ms"}
```

| File | Role |
|---|---|
| `config.py` | All settings from `.env` (pydantic-settings); rejects any model id not ending in `:free` |
| `store.py` | Shared embedding model + Chroma instance |
| `ingest.py` | Load, split, embed, sync the vector store (no duplicates on re-run) |
| `rag_chain.py` | Prompt, retrievers (similarity / MMR / hybrid / rerank), LLM with retries + fallbacks, LCEL chain, citation parsing |
| `api.py` | FastAPI: `/ask`, `/ask/stream` (SSE), `/health`, `/sessions/{id}`; error mapping |
| `streamlit_app.py` | Chat UI with streaming and follow-up questions |
| `eval/` | `questions.json`, `run_eval.py`, `compare_retrieval.py`, generated results |
| `data/` | The 5 safety documents |
| `docs/` | Illustrated technical dossier (PDF: screenshots, diagrams, workflows) |

## Design choices

**Dataset.** Five documents written for a fictional company (BuildRight), modelled on public OSHA / INRS guidance: general site regulations, PPE rules, scaffolding & work at height (3 PDFs, 3 pages each), incident reporting procedure (Markdown) and a cut-off saw manual (TXT). The 3 PDFs are real multi-page files, so the PDF loader and real page numbers are exercised. Writing them ourselves lets the evaluation know the ground truth exactly.

**Chunking: `chunk_size=800`, `chunk_overlap=120` (characters).**
- `all-MiniLM-L6-v2` truncates its input at 256 word-pieces (~1,000 characters of English). 800 characters (~180 tokens) stays under that limit, so no chunk is silently cut when embedded.
- One safety rule or sub-section is typically 300-800 characters: an 800-char chunk holds a complete rule with its numbers, which is what the questions ask for.
- 4 chunks x 800 chars = ~3,200 characters (~800 tokens) of context: cheap and fast on free models, and small enough that the model is not distracted.
- 120 characters (15 %) of overlap keeps a sentence that straddles a boundary readable in at least one chunk.
- Separators try Markdown headings and blank lines first, so chunks follow the document structure.

**Idempotent ingestion.** Each chunk id is `sha256(source | page | start_index | text)`. On re-run, ids already in Chroma are skipped, new ones added and ids no longer produced (edited or deleted documents) removed. `--reset` rebuilds from scratch.

**Retrieval.** Default is plain similarity (k=4), chosen from the measured comparison below; MMR (fetch_k=20, lambda=0.6) is available. Hybrid (BM25 + vectors fused with Reciprocal Rank Fusion) and cross-encoder re-ranking (`ms-marco-MiniLM-L-6-v2`, local) are switchable in `.env`. See the comparison below.

**Grounding.** The system prompt forbids outside knowledge, requires a `[file, p.X]` citation after each fact and fixes the exact refusal sentence. Every passage in the context is prefixed with `[source: file, p.X]`, so citing is a copy operation. The API returns only the cited chunks as sources, and no sources on a refusal.

**Robustness.** `ChatOpenAI(timeout=45, max_retries=0).with_retry(...)` makes up to 3 attempts per model on 429 / timeout / connection / 5xx errors, with exponential back-off and jitter (~1 s, then ~2 s); `.with_fallbacks([...])` then moves to the next free model. If everything fails the API answers 503 + `Retry-After` (rate limit), 504 (timeout) or 502. Caveat found while testing the UI: `.with_retry()` only wraps `invoke`/`batch`, not streaming, so `/ask/stream` retries transient errors itself, and only before the first token has been sent.

**Bonus features.** Conversational memory (follow-ups are rewritten into standalone questions before retrieval, per `session_id`), streaming (SSE), hybrid search, re-ranking, Streamlit UI, Docker.

## Similarity vs MMR

`python eval/compare_retrieval.py` prints the top-4 chunks of `similarity`, `mmr` and `hybrid` for the 3 example questions; full table in [eval/retrieval_comparison.md](eval/retrieval_comparison.md).

| Question | Similarity | MMR (`lambda_mult=0.6`) | Hybrid |
|---|---|---|---|
| Scaffold guardrail height | Answer chunk (p.2) first; 4 relevant chunks | Ranks 1-3 identical; rank 4 is the **document header** instead of a useful chunk | Answer chunk first; 4 relevant chunks |
| PPE for concrete cutting | PPE table **and** the saw manual's PPE section: both sources | Keeps the PPE table but **drops the saw manual's PPE section** for the manual header and "Starting the machine" | PPE table + saw manual PPE section |
| Incident, first 24 h | 4 relevant chunks of the incident procedure | Rank 4 replaced by an **off-topic** housekeeping chunk | Same off-topic chunk at rank 4 |

**Comment.** I expected MMR's diversity to help (for instance to bring in both sources for concrete cutting). The measurement says otherwise: similarity already retrieved both, and MMR swapped a relevant chunk for a less useful one in 2 of 3 questions. On a small, non-redundant corpus (43 chunks) the diversity penalty mostly adds noise; MMR would pay off on a large corpus with many near-duplicate pages. **The default was therefore changed from `mmr` to `similarity`**; MMR and hybrid remain available through `SEARCH_TYPE`.

## Evaluation

`eval/questions.json`: 10 questions (8 answerable, 2 not: the CEO's salary and the price of the CS-350 saw, a trap because the saw *is* in the documents). `eval/run_eval.py` grades automatically: an answer is correct when it is not a refusal and every key-fact regex matches (e.g. `2 m` but not `1.2 m`); a refusal is correct when it is the exact refusal sentence.

**Runs of 25/10/2026**: k=4, `nvidia/nemotron-3.5-lightning:free`, chunks 800/120; only the retriever changes.

| Retriever | Accuracy | Answerable | Refusals | Expected file cited | Hit@4 | Median latency | Mean | Max |
|---|---|---|---|---|---|---|---|---|
| **Similarity (default)** | **10/10** | 8/8 | 2/2 | 8/8 | 8/8 | 11.5 s | 24.6 s | 106.5 s |
| MMR | 10/10 | 8/8 | 2/2 | 8/8 | 8/8 | 8.9 s | 17.1 s | 62.1 s |
| Hybrid | 10/10 | 8/8 | 2/2 | 8/8 | 8/8 | 12.1 s | 13.4 s | 31.5 s |
| BM25 only (first run, before embeddings were installed) | 10/10 | 8/8 | 2/2 | 8/8 | 8/8 | 6.3 s | 9.3 s | 19.3 s |

Full answers: `eval/results_<retriever>.json` / `.md`. Live API answers to the 3 example questions + the refused one, and UI screenshots: [docs/SafeSite_Assistant_Technical_Dossier.pdf](docs/SafeSite_Assistant_Technical_Dossier.pdf) (section 2).

**Honest analysis.**
- Grounding and refusals are solid: 8/8 exact refusals across runs, including the q10 trap; every number is copied exactly.
- Reading the answers one by one shows weaknesses the metrics miss: some bullets are not cited (BM25 run, q05) or citations are bunched at the end (similarity run, q02); in a follow-up test the model wrote "regardless of the measured level", slightly beyond the text; the API can return two sources with the same file and page.
- Latency is dominated by the free LLM and very irregular (spikes of 53-106 s that move between questions across runs). They are probably 429 retries or fallbacks, but this could not be verified because the answering model is not logged, so latency differences between retrievers mean nothing.
- The set does not separate the retrievers (all 10/10); only the rank-by-rank comparison does. 10 questions is a small sample (95 % CI of the success rate ≈ 69-100 %), the questions reuse the documents' wording, and each configuration ran once.

Also verified: ingestion is idempotent (second run: `added: 0, unchanged: 43`); the API with FastAPI's TestClient and live with uvicorn (`/health`, 422 on an empty question, refusal with `sources: []`, SSE streaming, follow-up question with memory); the fallback chain with the primary model forced to a 404; the Streamlit UI in real use (screenshots in the illustrated dossier). Not run: re-ranking (cross-encoder not downloaded) and Docker.

## Limits and next improvements

- Free models: 5-100 s latency, rate limits, and models disappear (the exercise's Llama 3.3 70B, Mistral Small 3.2 and Gemma 3 were no longer free on 24/10/2026; the defaults were picked by probing the current `:free` list).
- OpenRouter's free tier allows 50 requests per day without credits (`free-models-per-day`); it was exhausted on 25/10 after 4 evaluation runs, after which every model returns 429 until the daily reset. Retrying that error is pointless; detecting it and failing fast is a next step.
- Sessions and the BM25 index live in memory; English-only embeddings; MD/TXT always cited as `p.1`; no OCR for scanned PDFs; citations are checked for presence, not for faithfulness; no authentication.
- Next: log which model answered and how many retries; a harder evaluation set (50+ questions, paraphrases, French, multi-document) with an LLM judge, to actually separate similarity, hybrid and re-ranking; sentence-level citation verification; merge same-page sources; multilingual embeddings; a relevance threshold that refuses without calling the LLM; Redis sessions and API keys; OCR; tracing.

Commands assume the virtual environment is activated (`.venv\Scripts\activate`); otherwise `python` is the global interpreter and imports fail.

