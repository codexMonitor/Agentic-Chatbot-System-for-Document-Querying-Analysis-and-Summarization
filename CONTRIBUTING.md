# Contributing

Short notes for working on this repo. The full implementation spec is `CAPSTONE_PROCEDURES.md`.

## Invariants (never break)

- **I1.** Files in `nodes/` import only from `services/base.py` (interfaces), `prompts.py`, `state.py`, `config.py`. Never a concrete provider or provider SDK.
- **I2.** Only `services/container.py` chooses concrete providers (from `Settings`).
- **I3.** Each node is created by a factory `make_<name>(llm, store, ...)` that returns `run(state) -> dict`.
- **I4.** Prompts live in `prompts.py` as templates beginning `### TASK: <name>` followed by a line `---`. The offline `FakeLLM` relies on this format.
- **I5.** Every test must run offline with no network and no API key. Tests that need a real provider use `@pytest.mark.real` and are skipped unless `DOC_AGENT_RUN_REAL=1`.
- **I6.** All configuration comes from `Settings` / `DOC_AGENT_*` environment variables, documented in `.env.example`.
- **I7.** Adding a capability = one new node file + one `add_node` + one entry in the router map + a router-prompt update. No edits elsewhere.

## Commands

PowerShell (Windows) and bash differ only in venv activation:
```powershell
py -m venv .venv ; .venv\Scripts\Activate.ps1      # bash: python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
ruff check .
pytest -q
python -m doc_agent data\sample.txt                 # bash: data/sample.txt
```
Use `pathlib` for all paths. Read and write text files as UTF-8 explicitly.

## Definition of Done (every task)

- [ ] Work is on a local branch `phase-N/<short-name>` (Phase 0: `phase-0/foundation`); commits use `feat:`, `fix:`, `chore:`, `docs:`, `test:`, `refactor:`. Nothing is pushed by the agent (rule 8).
- [ ] `ruff check .` clean and `pytest -q` green.
- [ ] New behavior has at least one offline test.
- [ ] Invariants I1-I7 still hold.
- [ ] README / `.env.example` / docs updated if usage or settings changed.
