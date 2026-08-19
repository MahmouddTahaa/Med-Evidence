import pytest

from clinical_rag.errors import IngestError
from clinical_rag.indexing.embed import assert_query_embedder_matches_index
from tests.helpers import HashEmbedder


def test_mismatched_query_model_raises():
    embedder = HashEmbedder()
    with pytest.raises(IngestError, match="does not match index model"):
        assert_query_embedder_matches_index(embedder, "BAAI/bge-m3")


def test_matching_query_model_ok():
    embedder = HashEmbedder()
    assert_query_embedder_matches_index(embedder, "hash-test")
