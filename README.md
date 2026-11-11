# Agentic Chatbot System for Document Querying Analysis and Summarization
My capstone project.

The Python package is `doc_agent` (distribution `doc-agent`). It is a LangGraph chatbot that routes each
question to document question-answering, analysis, or summarization. The baseline runs fully offline with
a fake LLM and an in-memory vector store.

## Setup (Windows PowerShell)

```powershell
py -m venv .venv ; .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
copy .env.example .env      # optional; all settings are DOC_AGENT_* variables
```

On bash: `python3 -m venv .venv && source .venv/bin/activate`.

## Usage

```powershell
python -m doc_agent data\sample.txt     # or: doc-agent data\sample.txt
```

Ask questions at the `you>` prompt (`What is Scrum?`, `Summarize the document`,
`Compare LangGraph and Scrum`); type `quit` to exit. Supported files: `.txt`, `.md`.

Development checks:

```powershell
ruff check .
pytest -q
```

## Layout

```
src/doc_agent/
  config.py       Settings + load_settings() (DOC_AGENT_* environment variables)
  state.py        AgentState (messages, intent, context, answer)
  prompts.py      prompt templates ("### TASK: <name>", then "---", then payload)
  graph.py        build_graph(services, settings): router -> qa | analysis | summarize
  cli.py          command-line chat loop
  nodes/          router.py qa.py analysis.py summarize.py
  services/       base.py (interfaces) llm.py vectorstore.py ingestion.py container.py
tests/            offline tests
data/sample.txt   demo document
```

Nodes depend only on the interfaces in `services/base.py`; `services/container.py` is the only place that
chooses concrete providers.

## Extension recipes

- **New capability:** add one node file in `nodes/` with a `make_<name>(llm, store, ...)` factory, one
  `add_node` and one router-map entry in `graph.py`, and update the router prompt in `prompts.py`.
- **New LLM or vector store:** implement the Protocol from `services/base.py`, register it in
  `_make_llm` / `_make_store` in `services/container.py`, and document its settings in `.env.example`.
