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

## Browser UI

Install the optional UI dependencies:

```powershell
pip install -e ".[ui]"
```

The UI shows conversation history backed by the API checkpointer. Documents and questions are sent
to the configured LLM provider; review its data-handling policy before uploading sensitive material.
The HTTP client timeout defaults to 660 seconds and must be longer than
`DOC_AGENT_SUMMARIZE_TIMEOUT_S` (default 600 seconds). Override it with `DOC_AGENT_UI_TIMEOUT_S` if
you change the server's summarize budget.

**API mode (two terminals).** In the first terminal, start FastAPI:

```powershell
.venv\Scripts\Activate.ps1
uvicorn doc_agent.api.main:app --reload
```

In the second terminal, start Streamlit:

```powershell
.venv\Scripts\Activate.ps1
$env:DOC_AGENT_MODE = "http"
$env:DOC_AGENT_API_URL = "http://127.0.0.1:8000"
streamlit run src\doc_agent\ui\app.py
```

**Embedded mode (one command, no separate API server).** Set the desired provider and persistence
settings in `.env`, then run:

```powershell
.venv\Scripts\Activate.ps1
$env:DOC_AGENT_MODE = "embedded"; streamlit run src\doc_agent\ui\app.py
```

The embedded UI calls `DocAgentApp` in-process and uses the same document, chat, and conversation
history interface as HTTP mode. Set `DOC_AGENT_MODE=embedded` in `.env` to start it with only
`streamlit run src\doc_agent\ui\app.py`.

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

To regenerate it, create a fresh venv, run `pip install -e ".[dev,ui]"`, then:

```powershell
python -c "import subprocess,sys; open('requirements.lock','w',encoding='utf-8',newline='\n').write(subprocess.run([sys.executable,'-m','pip','freeze','--exclude-editable'],capture_output=True,text=True,check=True).stdout)"
```

## Layout

```
src/doc_agent/
  ui/app.py       Streamlit UI (HTTP or in-process client)
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

## Running the evaluation

The evaluation runner reads `eval/dataset.jsonl`, ingests supported documents from a directory,
and writes metrics plus per-question results as JSON. Review **every reference answer** against
the source paper before relying on these draft results; see [eval/README.md](eval/README.md).
Place `2611.13436v1.pdf` in `data\` for the commands below, or pass the directory that contains the
PDF to `--docs`.

Baseline (summarization questions are skipped by default):

```powershell
python -m doc_agent.eval --dataset eval\dataset.jsonl --docs data --out eval\results\baseline.json
```

Run an experiment with a different retrieval count and chunk size:

```powershell
python -m doc_agent.eval --dataset eval\dataset.jsonl --docs data --k 8 --chunk-size 800 --chunk-overlap 50 --out eval\results\k8-chunk800.json
```

The runner supports `--include-summarize`, `--limit N`, `--k`, `--chunk-size`, `--chunk-overlap`,
`--judge`, and `--out`. It prints an estimated and actual number of pipeline LLM calls; these counts
exclude embedding requests and optional judge calls. The JSON records the selected chunk settings,
retrieval `k`, provider and model, intent accuracy, retrieval hit rate, invalid citation rate, and
average latency per question and node. LLM-judged scores are `null` unless `--judge` is requested.

To request Ragas scores, install its optional dependency and configure a real Gemini model and key:

```powershell
pip install -e ".[eval]"
python -m doc_agent.eval --dataset eval\dataset.jsonl --docs data --judge --out eval\results\judged.json
```

Judging uses the Ragas collections API with Gemini for faithfulness, answer relevance, and context
relevance. It adds provider calls; tracing should remain off unless you explicitly want prompts and
document excerpts sent to LangSmith. Without Ragas or a configured Gemini judge, the judge metrics
remain `null` and the output explains why. See the [Ragas Gemini integration](https://docs.ragas.io/en/stable/howtos/integrations/gemini/)
and [metric documentation](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/).

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
