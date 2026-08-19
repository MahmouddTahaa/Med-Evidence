# Architecture

Med-Evidence is a clinician / student pharmacology guide: **document-agnostic ingest**, **frozen hybrid retrieval**, and **grounded generation** in a separate chat app. The operator Streamlit lab is ingest/eval only — not the product UI.

## System context

```mermaid
flowchart TB
  clinician[Clinician / student]
  operator[Operator]
  med[Med-Evidence]
  llms[LLM providers]
  chroma[(Chroma)]
  disk[(winning.yaml + serving.yaml + jobs/)]

  clinician -->|chat :8502| med
  operator -->|lab :8501| med
  med --> llms
  med --> chroma
  med --> disk
```

## Two processes, one package

| App | Entry | Port | Role |
| --- | --- | --- | --- |
| Clinician | `src/clinical_rag/ui/clinician_app.py` | 8502 | Query + grounded generation |
| Operator lab | `src/clinical_rag/ui/streamlit_app.py` | 8501 | Ingest, retrieval eval, smoke query |

Both import `clinical_rag`. The clinician app must not import `eval.grid` or retune `configs/winning.yaml`.

```mermaid
flowchart LR
  subgraph product [Product path]
    C[clinician_app :8502]
    GE[GroundedEngine]
    RS[RetrievalSession]
    C --> GE --> RS
  end
  subgraph lab [Operator lab]
    L[streamlit_app :8501]
    IN[run_ingest]
    EV[eval runner]
    L --> IN
    L --> EV
  end
  RS --> RR[run_retrieve]
  EV --> RR
  W[winning.yaml] --> RS
  W --> IN
  S[serving.yaml job_id] --> RS
```

## End-to-end data flow

```mermaid
flowchart TD
  upload[Upload PDF / TXT / MD / XML / NXML] --> parse[Parser router]
  parse --> clean[Cleanup + quality]
  clean --> chunk[Chunker — product: section_aware + title prefix]
  chunk --> embed[Embedder — BGE-small]
  embed --> store[Chroma collection]
  chunk --> artifacts[artifacts/jobs/job_id/chunks.json]
  store --> retrieve[Hybrid retrieve]
  artifacts --> sparse[BM25 sparse index]
  sparse --> retrieve
  retrieve --> rrf[Weighted RRF]
  rrf --> rerank[MiniLM cross-encoder]
  rerank --> hits[SmokeHit list + metadata]
  hits --> cite[citation_blocks]
  cite --> gen[GroundedEngine]
  gen --> ui[Clinician UI — 4-block answer]
```

## Package map

| Package | Responsibility |
| --- | --- |
| `clinical_rag.parsing` | Media-type router, PDF/OCR, XML/NXML, text/JSON |
| `clinical_rag.chunking` | Lab strategies; product uses `section_aware` |
| `clinical_rag.indexing` | Embed + Chroma / Qdrant stores |
| `clinical_rag.retrieval` | Dense, BM25, RRF, rerank; lab-only sibling/parent |
| `clinical_rag.pipeline` | `run_ingest`, smoke query |
| `clinical_rag.stack` | `FrozenStack` from `winning.yaml` + serving pointer |
| `clinical_rag.query` | `RetrievalSession`, `citation_blocks` |
| `clinical_rag.generate` | Guardrails, weak retrieve path, prompts, `GroundedEngine` |
| `clinical_rag.adapters` | Embedders, stores, LLM cascade |
| `clinical_rag.eval` | Lab metrics and freeze — not product UI |
| `clinical_rag.ui` | Two Streamlit apps |

**Import rule:** product code loads the stack via `FrozenStack` / `RetrievalSession`, calls one ranker (`run_retrieve`), and always prompts with citation metadata — never bare strings.

## Config and artifacts

```mermaid
flowchart LR
  winning[configs/winning.yaml] -->|recipe| stack[FrozenStack]
  serving[configs/serving.yaml] -->|job_id| session[RetrievalSession]
  stack --> session
  job[artifacts/jobs/job_id/] --> session
  job --> report[report.json]
  job --> chunks[chunks.json]
  session --> chroma[(Chroma persist_dir)]
```

| Path | Meaning |
| --- | --- |
| `configs/winning.yaml` | Locked parser / chunk / embed / store / retrieve knobs |
| `configs/serving.yaml` | Machine-local `job_id` (copy from `serving.example.yaml`) |
| `artifacts/jobs/<job_id>/` | Ingest report, chunks, collection pointer |
| `data/eval/statpearls_pharmacology_templates.jsonl` | Official gold templates |
| `.env` | API keys and `GENERATION__*` knobs (not committed) |

## Product turn (clinician)

```mermaid
sequenceDiagram
  participant U as User
  participant UI as clinician_app
  participant E as GroundedEngine
  participant R as RetrievalSession
  participant L as LLM cascade

  U->>UI: question
  UI->>E: turn(TurnRequest)
  E->>E: classify input
  alt Refuse
    E-->>UI: structured refusal
  else Allowed / NeedsCaution
    E->>R: retrieve / weak-path
    R-->>E: hits
    alt still weak
      E-->>UI: Case C + top chunks
    else strong enough
      E->>L: grounded 4-block prompt
      L-->>E: tagged answer
      E->>E: bind citations to this turn's hits
      E-->>UI: TurnResult
    end
  end
```

## Contracts (two, on purpose)

1. **Ingest / retrieve** accept any supported upload format.
2. **Official evaluation** scores StatPearls pharmacology only.

Full bakeoff: [evaluation.md](evaluation.md). Engineering contract and knobs: [technical-report.md](technical-report.md). How to run the apps: [user-guide.md](user-guide.md).
