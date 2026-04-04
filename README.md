# Smart Research Assistant

Streamlit app for **PDF Q&A with citations**, backed by **LangChain**, **Google Gemini** (chat + embeddings), and **Pinecone**. When uploads are not enough, the agent can call **Tavily** for web search. **Chat and vectors are scoped per browser tab** (Streamlit session + Pinecone namespace).

## Features

- Index multiple PDFs with chunk-level metadata for **Sources** blocks in answers  
- **FlashRank** reranking on top of vector search (optional)  
- **Optional multi-query** retrieval for harder questions (extra LLM calls)  
- **Retrieval scope** in the sidebar: restrict search to selected PDFs via Pinecone metadata on `source`  
- **Tool-calling agent**: document search first, web when needed (prompt-driven, not a separate router)

**Upload limits:** up to **5** PDFs per batch, **≤10 pages** each — tune with `MAX_UPLOAD_FILES` and `MAX_PAGES_PER_FILE` in `.env`.

## Prerequisites

- **Python 3.10+** recommended  
- Accounts / keys: [Google AI Studio](https://aistudio.google.com/apikey) (Gemini), [Pinecone](https://app.pinecone.io/), [Tavily](https://tavily.com)

## Quick start

```bash
cd rag_pipeline
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set at least `GOOGLE_API_KEY` (or `GEMINI_API_KEY`), `PINECONE_API_KEY`, and `TAVILY_API_KEY`. Then:

```bash
streamlit run streamlit_app.py
```

`streamlit_app.py` adds `src/` to `sys.path`, so you do not need `PYTHONPATH` for the UI.

**Pinecone:** By default the app **creates** the index on first use if it is missing (serverless; dimension/metric/region from `.env`). Set `PINECONE_AUTO_CREATE_INDEX=false` to create the index yourself in the Pinecone console. If creation fails, try another region (e.g. `PINECONE_REGION=us-west-2`).

## Using the app

1. **Upload** PDFs (within the limits above) and click **Index PDFs & enable chat**.  
2. **Chat** in step 2; answers should end with a **`### Sources`** section when documents were used.  
3. **Sidebar → Retrieval scope → Included documents**  
   - **Nothing selected:** search runs over the **full** indexed corpus (no Pinecone metadata filter).  
   - **One or more PDFs selected:** queries use `{"source": {"$in": […]}}` so only those files are searched.

Optional copy and knobs: files under **`prompts/`** and variables in **`src/core/config.py`** (defaults apply if omitted).

### Configuration reference (`.env`)

| Variable | Role |
|----------|------|
| `GOOGLE_API_KEY` or `GEMINI_API_KEY` | Required — Gemini chat + embeddings |
| `PINECONE_API_KEY` | Required |
| `PINECONE_INDEX_NAME` | Index name (default `rag-research-assistant`) |
| `TAVILY_API_KEY` | Required — web search tool |
| `GEMINI_CHAT_MODEL` | Chat model id (default `gemini-2.5-flash`) |
| `GEMINI_EMBEDDING_MODEL` | Embedding model (default `gemini-embedding-001`) |
| `GEMINI_MAX_RETRIES` | Chat client retries, `0`–`8` (default `1`) |
| `MAX_UPLOAD_FILES`, `MAX_PAGES_PER_FILE` | Upload limits |
| `RETRIEVAL_K`, `RETRIEVAL_FETCH_MULTIPLIER`, `USE_RERANK`, `USE_MULTI_QUERY` | Retrieval + rerank + multi-query |
| `CHUNK_SIZE`, `CHUNK_OVERLAP` | PDF splitting |
| `AGENT_MAX_ITERATIONS` | Max tool–model loops per reply |
| `PINECONE_AUTO_CREATE_INDEX` | Create serverless index if missing (default on) |
| `PINECONE_CLOUD`, `PINECONE_REGION` | Serverless spec (defaults `aws` / `us-east-1`) |
| `PINECONE_EMBEDDING_DIMENSION`, `PINECONE_METRIC` | Must match index (defaults `768` / `cosine`) |

**Prompt overrides (optional):** set inline text or a file path — `AGENT_SYSTEM_PROMPT` / `AGENT_SYSTEM_PROMPT_FILE`, `TAVILY_TOOL_DESCRIPTION` / `…_FILE`, `DOCUMENT_SEARCH_TOOL_DESCRIPTION` / `…_FILE`. Defaults use `prompts/*.txt`.

## Project layout

| Path | Role |
|------|------|
| `src/core/` | Pipeline: `config`, `ingestion`, `vectorstore`, `retrieval`, `agent`, `metrics` |
| `src/ui/` | Streamlit: `layout.py` (chrome, sidebar, assistant UI), `flows.py` (index + chat) |
| `streamlit_app.py` | Entry point: path setup, settings, wires layout + flows |

## How it works (short)

1. **`config.py`** — Loads `.env`, optional `prompts/`, model and Pinecone settings.  
2. **`ingestion.py`** — `PyPDFLoader` + `RecursiveCharacterTextSplitter`; metadata includes `source`, `page`, `session_id`, `chunk_index`, `citation_label`.  
3. **`vectorstore.py`** — Gemini embeddings + `PineconeVectorStore`, **namespace per session**.  
4. **`retrieval.py`** — Similarity search, optional **metadata filter**, optional **FlashRank** rerank (`RETRIEVAL_K` × `RETRIEVAL_FETCH_MULTIPLIER`, cap 50), optional **`MultiQueryRetriever`**.  
5. **`agent.py`** — `create_tool_calling_agent` + `AgentExecutor`: document tool + Tavily; capped by `AGENT_MAX_ITERATIONS`.

## RAG tuning (defaults in `.env.example`)

### Chunking

| Setting | Default | Notes |
|---------|---------|--------|
| `CHUNK_SIZE` | `1200` | Balance context vs. precision per hit |
| `CHUNK_OVERLAP` | `180` | Reduces broken thoughts at boundaries |
| Splitter | `RecursiveCharacterTextSplitter` | Prefers `\n\n`, `\n`, `. `, then space |

### Retrieval

1. **Embeddings** — Same model for queries and chunks (`GEMINI_EMBEDDING_MODEL`; default `gemini-embedding-001`). Keep `PINECONE_EMBEDDING_DIMENSION` aligned with the index (768 by default).  
2. **Rerank (on by default)** — Fetches up to `RETRIEVAL_K * RETRIEVAL_FETCH_MULTIPLIER` (max 50), reranks with [FlashRank](https://github.com/PrithivirajDamodaran/FlashRank) locally. Set **`USE_RERANK=false`** for pure vector order.  
3. **Metadata filter** — Only when the sidebar multiselect has **at least one** document selected (see [Using the app](#using-the-app)).  
4. **Multi-query** — Set **`USE_MULTI_QUERY=true`** for paraphrased search queries (extra Gemini calls per retrieval).

### Agent routing

Behavior comes from **`prompts/agent_system.txt`** and tool descriptions (e.g. **`prompts/tavily_tool_description.txt`**): **document search first** for substantive Q&A; **Tavily after** if the corpus is insufficient or the user wants live/external information.

### Problems we hit (reference)

| Issue | Mitigation |
|-------|------------|
| Slow / hidden Gemini **429** | **`GEMINI_MAX_RETRIES`** (default `1`); use `0` for no retry |
| Invalid chat model id | **`_normalize_gemini_chat_model()`** in `config.py` (e.g. `gemini-2.5` → `gemini-2.5-flash`) |
| Web tool overused | Tighter system + Tavily tool prompts: corpus-first |
| Noisy citations | Prompts: citation-free body, then **`### Sources`** after **`---`** |
| Imports from `src/` | `streamlit_app.py` prepends `src/` to `sys.path` |

### Possible next step

**Conversational query rewriting** — Short follow-ups (“What about the masks?”) embed poorly without context. Rewriting with recent history into a standalone search query would help multi-turn recall.

## Troubleshooting

- **Pinecone / index:** Check index name, region, embedding dimension, and metric vs `.env`.  
- **Embedding `404`:** Prefer **`gemini-embedding-001`**; re-index after changing embedding settings.  
- **Chat `429`:** Billing, different **`GEMINI_CHAT_MODEL`**, or lower **`AGENT_MAX_ITERATIONS`**. See [Gemini rate limits](https://ai.google.dev/gemini-api/docs/rate-limits).  
- **Tools + streaming errors:** Try another **`GEMINI_CHAT_MODEL`**. (Streaming is on in code — change `agent.py` if you need it off.)  
- **`tavily` import:** `pip install tavily-python` (in `requirements.txt`).  
- **Tool-calling quirks:** Try `GEMINI_CHAT_MODEL=gemini-1.5-flash`.

## License

Use and modify freely for your project.
