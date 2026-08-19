"""Product query surface: frozen stack + one retrieval session + citation blocks.

Generation lives in `clinical_rag.generate`. Do not import eval grid or Streamlit
from here.
"""

from clinical_rag.query.context import RetrievedChunk, citation_blocks
from clinical_rag.query.session import RetrievalSession
from clinical_rag.stack import (
    SERVING_PATH,
    WINNING_PATH,
    FrozenStack,
    ServingPointer,
    load_frozen_stack,
    load_serving_pointer,
)

__all__ = [
    "FrozenStack",
    "RetrievedChunk",
    "RetrievalSession",
    "SERVING_PATH",
    "ServingPointer",
    "WINNING_PATH",
    "citation_blocks",
    "load_frozen_stack",
    "load_serving_pointer",
]
