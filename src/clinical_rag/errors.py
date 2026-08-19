class IngestError(Exception):
    """Operator-facing ingest failure (legal gate, parse, empty corpus)."""


class QueryError(Exception):
    """Clinician-facing generation / provider failure (fail closed)."""
