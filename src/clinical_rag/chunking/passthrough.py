from clinical_rag.errors import IngestError
from clinical_rag.schemas import ParsedDocument


def chunk_passthrough(parsed: ParsedDocument) -> None:
    raise IngestError("passthrough is only valid for pre-chunked JSON")
