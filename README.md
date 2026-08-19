# Med-Evidence

**Clinician / student pharmacology guide.** Retrieval is **frozen**. Grounded query + generation live in a **separate** clinician Streamlit app; the operator lab stays ingest/eval only.

Two contracts, one ingest pipeline:

- **Ingest / retrieve** are document-agnostic: any uploaded PDF, TXT, MD, XML, or NXML.
- **Official evaluation** used StatPearls pharmacology only.

## Documentation

| Doc | Contents |
| --- | --- |
| [docs/user-guide.md](docs/user-guide.md) | Setup, clinician app, operator lab |
| [docs/architecture.md](docs/architecture.md) | System layout, data flow, packages |
| [docs/technical-report.md](docs/technical-report.md) | Frozen stack, generation contract, configs |
| [docs/evaluation.md](docs/evaluation.md) | Retrieval bakeoff record |

## Quick start

Product code should read:

1. [`configs/winning.yaml`](configs/winning.yaml) — locked chunk / embed / store / hybrid+rerank recipe (`FrozenStack`).
2. [`configs/serving.yaml`](configs/serving.example.yaml) — local ingest `job_id` (copy from the example).

Then call `clinical_rag.query.RetrievalSession` and `clinical_rag.generate.GroundedEngine`.

```python
from clinical_rag.query import RetrievalSession, citation_blocks

session = RetrievalSession.open()
hits = session.retrieve("What are the contraindications for Ampicillin?")
context = citation_blocks(hits)
```

```bash
uv run python scripts/retrieve.py "What are the contraindications for Ampicillin?"
```

## Clinician app (query + grounded generation)

```bash
uv sync --extra generate --extra dev
# copy .env.example → .env and set at least one LLM key (or run a local OpenAI-compatible server)
uv run streamlit run src/clinical_rag/ui/clinician_app.py --server.address 127.0.0.1 --server.port 8502
```

Demo buttons pull a **random** question from in-scope / refusal / out-of-corpus pools. The chat input stays mounted every rerun. The main panel shows the locked retrieval scorecard (MRR, P@k, R@k, nDCG, latency) plus per-turn / session generation heuristics (faithfulness, citation accuracy). Responsible AI checklist and disclaimer live in the sidebar.

## Operator lab (ingest / eval only)

```bash
uv sync --extra dev
uv run streamlit run src/clinical_rag/ui/streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

Tabs: ingest, retrieval eval, try-query (smoke), winning.yaml inspector. Not the clinician UI.

## Frozen stack (do not retune)

| Knob | Locked value |
| ---- | ------------ |
| Chunk | `section_aware` 400 / 0.12 / 520, `prefix_section_title: true` |
| Embed | `BAAI/bge-small-en-v1.5` |
| Store | Chroma |
| Retrieve | Hybrid BM25 0.7 / 0.3 + MiniLM cross-encoder rerank |

Re-running `scripts/lock_winning_combo.py` resets `winning.yaml` to the 2026-08-18 bakeoff (prefix **off**). Do not run it unless you intend that.

## Data

Gold templates: `data/eval/statpearls_pharmacology_templates.jsonl`

StatPearls NXML (not committed): symlink `data/pharmacology_data` → your local dump.

## Tests

```bash
uv run pytest
```
