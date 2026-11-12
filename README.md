# Agentic Chatbot System for Document Querying Analysis and Summarization
My capstone project.

The Python package is `doc_agent` (distribution `doc-agent`). It is a LangGraph chatbot that routes each
question to a short chat, document question-answering, analysis, or summarization. By default it runs fully offline
with a fake LLM, a fake embedder and an in-memory vector store; with a Gemini key it ingests PDF, DOCX,
TXT and MD files and answers with source citations.

## Setup (Windows PowerShell)

```powershell
py -m venv .venv ; .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
copy .env.example .env      # all settings are DOC_AGENT_* variables; real env vars override .env
```

On bash: `python3 -m venv .venv && source .venv/bin/activate`.

### Using Gemini (free tier) with a persistent Chroma store

1. Create an API key in [Google AI Studio](https://aistudio.google.com/apikey).
2. Edit `.env` (it is git-ignored; never commit it or paste the key anywhere else):

   ```
   DOC_AGENT_LLM_PROVIDER=gemini
   DOC_AGENT_LLM_MODEL=gemini-3.5-flash-lite
   DOC_AGENT_EMBEDDING_PROVIDER=gemini
   DOC_AGENT_EMBEDDING_MODEL=gemini-embedding-001
   DOC_AGENT_VECTOR_STORE=chroma
   DOC_AGENT_PERSIST_DIR=.doc_agent_data
   GEMINI_API_KEY=<your key>
   ```

3. Run `python -m doc_agent path\to\report.pdf path\to\notes.docx` and ask questions.

The chat LLM, the embedding model and the vector store are three independent settings. Vectors are
stored under `DOC_AGENT_PERSIST_DIR` together with the name of the embedding model that produced them.
**Changing the embedding provider or model requires re-ingesting**: the app refuses to start with an
incompatible store; delete the persistence directory (`.doc_agent_data`) and ingest again. Free-tier
Gemini data may be used by Google for training, so use only public or sample documents.

## LangSmith tracing

Tracing is off by default. To enable it, set `LANGSMITH_TRACING=true`, put your key in
`LANGSMITH_API_KEY`, and choose `LANGSMITH_PROJECT` (defaults to `doc-agent`). LangGraph runs and
Gemini chat and embedding calls then appear as traces with provider/model metadata and latency;
token counts are included when the SDK response exposes them. Trace inputs and outputs are masked
for configured API keys.

Tracing uploads prompts and document text to LangSmith. Use only public or sample documents when it
is enabled. Tests always disable tracing and remove the LangSmith key.

## LangGraph Studio

The application supports Python 3.10+, but the current LangGraph CLI requires Python 3.11 or newer.
In a Python 3.11+ environment, install the optional Studio tools and start the local server:

```powershell
pip install -e ".[studio]"
langgraph dev
```

The LangGraph CLI needs a LangSmith account and API key to connect to Studio; use
`LANGSMITH_API_KEY` from `.env`. Tracing remains off unless `LANGSMITH_TRACING=true`. Studio runs the
same graph as the CLI (`rewrite -> router -> chat | qa | analysis | summarize`), with the server's
own persistence rather than the CLI's SQLite checkpointer. `langgraph.json` registers
`doc_agent.studio:graph`.

To regenerate the offline graph diagram, run:

```powershell
python scripts\export_graph.py
```

This writes `docs/graph.mmd`.

## Usage

```powershell
python -m doc_agent data\sample.txt     # or: doc-agent data\sample.txt
```

Ask questions at the `you>` prompt (`What is Scrum?`, `Summarize the document`,
`Compare LangGraph and Scrum`); type `quit` to exit. Supported files: `.txt`, `.md`, `.pdf`, `.docx`
(scanned PDFs need OCR and are not supported). Question-answering replies print a `Sources:` block with
one `[n] file#position` line per chunk given to the model; `[n]` matches the citations in the answer.

Options:

- `--thread <id>` (default `cli`): conversations are saved in SQLite
  (`DOC_AGENT_CHECKPOINT_DB`, default `.doc_agent_data\checkpoints.sqlite`). Restart with the same id to
  continue that conversation; use another id for a fresh one.
- `-v` / `--verbose`: log the routing decision and `node=<name> duration_ms=<n>` for every node to stderr.

**Follow-up questions.** Before routing, a `rewrite` step turns a follow-up such as "How long are they?"
into a standalone question ("How long are Scrum sprints?") using the last
`DOC_AGENT_REWRITE_HISTORY_TURNS` messages (default 4; `0` turns it off). The router, question
answering and analysis use the rewritten question. Cost: the first turn of a conversation makes no
extra LLM call; every follow-up turn makes one more. If that call fails, the original question is used.

**Summaries of long documents.** `Summarize the document` summarizes every chunk in parallel and merges
the summaries in groups until one remains (see [ADR-0007](docs/adr/0007-summarization-strategy.md)).
Tune it with `DOC_AGENT_SUMMARIZE_WORKERS` (default 2) and `DOC_AGENT_SUMMARIZE_GROUP_SIZE` (default 8).
It costs about one LLM call per chunk, so on the Gemini free tier (15 requests per minute) a document of
100 chunks takes several minutes (about 6.5 minutes measured), but it completes with the default retry
settings.

**Errors and retries.** There is exactly one retry layer: the provider adapters (chat and embeddings)
wrap every call in `call_with_retry`; nodes add none. Rate limits (429) and temporary server errors
(500, 502, 503, 504) are retried up to `DOC_AGENT_LLM_MAX_RETRIES` times (default 8, so at most 9
attempts per call). A 429 waits for the delay the server asks for (a "retry in 5.4s" hint or a
`Retry-After` header, at most 60 s per wait); other errors use exponential backoff (1 s, 2 s, 4 s, ...,
also at most 60 s), both with a little jitter. Authentication, missing-key and other client errors are
never retried. If the model is still unavailable, the chatbot replies "The language model is
unavailable. Please try again." instead of crashing; after an outage starts, that can take a few
minutes because the waits add up (at most 8 x 60 s per call). API key values are masked in error text.

Development checks (all offline; no API key needed):

```powershell
ruff check .
pytest -q
```

Tests that call a real provider are marked `@pytest.mark.real` and skipped by default. To run them
(they use `GEMINI_API_KEY` from the environment or `.env`):

```powershell
$env:DOC_AGENT_RUN_REAL = "1" ; pytest -q -m real
```

### Pre-commit

```powershell
pre-commit install            # run the hooks on every commit
pre-commit run --all-files    # run them once on the whole repo
```

Hooks: `ruff` (lint), `ruff-format`, trailing whitespace, end-of-file fixer.

### Pinned dependencies

`requirements.lock` pins the exact versions used for development. To reinstall from it:

```powershell
pip install -r requirements.lock
pip install -e . --no-deps
```

To regenerate it, create a fresh venv, run `pip install -e ".[dev]"`, then:

```powershell
python -c "import subprocess,sys; open('requirements.lock','w',encoding='utf-8',newline='\n').write(subprocess.run([sys.executable,'-m','pip','freeze','--exclude-editable'],capture_output=True,text=True,check=True).stdout)"
```

## Layout

```
src/doc_agent/
  config.py       Settings + load_settings() (DOC_AGENT_* variables, optional .env)
  state.py        AgentState (messages, standalone_question, intent, context, sources, answer)
  prompts.py      prompt templates ("### TASK: <name>", then "---", then payload)
  graph.py        build_graph(services, settings, checkpointer=None): rewrite -> router ->
                  chat | qa | analysis | summarize; SQLite conversation memory and per-node timing logs
  studio.py       module-level graph for LangGraph Studio (no application checkpointer)
  cli.py          command-line chat loop (--thread, --verbose)
  nodes/          rewrite.py router.py chat.py qa.py analysis.py summarize.py _utils.py (timing, error guard)
  services/       base.py (Chunk + interfaces) llm.py embeddings.py vectorstore.py ingestion.py
                  errors.py (LLMError, call_with_retry) container.py
tests/            offline tests (real-provider tests are opt-in); tests/data/intents.jsonl
                  holds the router examples
data/sample.txt   demo document
```

The six `chat` examples in `tests/data/intents.jsonl` are agent-drafted and await human review.

Nodes depend only on the interfaces in `services/base.py` (plus the provider-neutral
`services/errors.py`, see [ADR-0009](docs/adr/0009-nodes-may-use-error-helpers.md)); `services/container.py`
is the only place that chooses concrete providers.

## Extension recipes

- **New capability:** add one node file in `nodes/` with a `make_<name>(llm, store, ...)` factory, one
  `add_node` and one router-map entry in `graph.py`, and update the router prompt in `prompts.py`.
- **New LLM provider (for example OpenAI, xAI, Groq):** add one adapter class in `services/llm.py` that
  implements `invoke(prompt) -> str` and wraps provider exceptions with `call_with_retry` (they become
  `LLMError`, flagged `retryable` for rate limits and temporary server errors); add one entry to
  `LLM_PROVIDERS`; add the SDK to `pyproject.toml`; put its key variable
  (placeholder only) in `.env.example`. Then set `DOC_AGENT_LLM_PROVIDER` and `DOC_AGENT_LLM_MODEL`. Nodes,
  prompts and the graph do not change.
- **New embedding provider:** add an adapter implementing `embed(texts)` and one entry in `EMBEDDERS`
  (`services/embeddings.py`). Existing stores must be re-ingested.
- **New vector store:** implement the `VectorStore` Protocol and add one entry in `VECTOR_STORES`
  (`services/container.py`).
- **New document type:** add a `Path -> str` loader to `LOADERS` in `services/ingestion.py`.
