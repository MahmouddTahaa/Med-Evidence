# Med-Evidence

Clinician / student pharmacology guide. Ingest is document-agnostic (PDF, TXT, MD, XML, NXML); official evaluation uses StatPearls pharmacology only.

```bash
uv sync --extra dev
```

Copy `.env.example` to `.env`. After ingest, copy `configs/serving.example.yaml` to `configs/serving.yaml` and set the local `job_id`.
