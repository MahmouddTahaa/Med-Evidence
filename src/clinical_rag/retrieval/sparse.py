from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from clinical_rag.errors import IngestError
from clinical_rag.schemas import KeywordMethod, SmokeHit

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _hit_from_row(row: dict[str, Any], score: float) -> SmokeHit:
    page = row.get("page_number")
    return SmokeHit(
        score=round(float(score), 6),
        text=str(row.get("text") or ""),
        document_name=str(row.get("document_name") or ""),
        section_title=str(row.get("section_title") or ""),
        page_number=None if page in (None, -1, "-1") else int(page),
        chunk_id=str(row.get("chunk_id") or ""),
        extraction_method=str(row.get("extraction_method") or ""),
        source_url=str(row.get("source_url") or ""),
        token_count=int(row.get("token_count") or 0),
    )


class SparseIndex:
    """In-process BM25Okapi or TF-IDF+cosine over job chunks.json rows."""

    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        method: KeywordMethod = KeywordMethod.bm25,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        if not rows:
            raise IngestError("SparseIndex requires at least one chunk")
        self.method = method
        self.k1 = k1
        self.b = b
        self._rows = rows
        self._docs: list[list[str]] = [tokenize(str(r.get("text") or "")) for r in rows]
        self._n = len(self._docs)
        self._doc_len = [len(d) for d in self._docs]
        self._avgdl = (sum(self._doc_len) / self._n) if self._n else 0.0
        self._df: Counter[str] = Counter()
        for doc in self._docs:
            self._df.update(set(doc))
        self._idf: dict[str, float] = {}
        for term, df in self._df.items():
            # Robertson–Walker IDF (rank-bm25 style)
            self._idf[term] = math.log(1.0 + (self._n - df + 0.5) / (df + 0.5))
        self._tfidf_vecs: list[dict[str, float]] | None = None
        if method is KeywordMethod.tfidf:
            self._tfidf_vecs = [self._tfidf_vector(doc) for doc in self._docs]

    @classmethod
    def from_chunks(
        cls,
        chunks: list[dict[str, Any]],
        *,
        method: KeywordMethod = KeywordMethod.bm25,
    ) -> SparseIndex:
        return cls(chunks, method=method)

    def _tfidf_vector(self, doc: list[str]) -> dict[str, float]:
        if not doc:
            return {}
        tf = Counter(doc)
        length = len(doc)
        vec: dict[str, float] = {}
        for term, count in tf.items():
            idf = math.log((1.0 + self._n) / (1.0 + self._df[term])) + 1.0
            vec[term] = (count / length) * idf
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    def _bm25_score(self, query_tokens: list[str], doc_idx: int) -> float:
        doc = self._docs[doc_idx]
        if not doc:
            return 0.0
        tf = Counter(doc)
        dl = self._doc_len[doc_idx]
        score = 0.0
        for term in query_tokens:
            if term not in tf:
                continue
            freq = tf[term]
            idf = self._idf.get(term, 0.0)
            denom = freq + self.k1 * (1.0 - self.b + self.b * dl / (self._avgdl or 1.0))
            score += idf * (freq * (self.k1 + 1.0)) / denom
        return score

    def _tfidf_score(self, query_tokens: list[str], doc_idx: int) -> float:
        assert self._tfidf_vecs is not None
        q_vec = self._tfidf_vector(query_tokens)
        if not q_vec:
            return 0.0
        d_vec = self._tfidf_vecs[doc_idx]
        return sum(q_vec[t] * d_vec.get(t, 0.0) for t in q_vec)

    def query(self, text: str, top_k: int) -> list[SmokeHit]:
        if top_k <= 0:
            return []
        q_tokens = tokenize(text)
        if not q_tokens:
            return []
        scored: list[tuple[float, int]] = []
        for i in range(self._n):
            if self.method is KeywordMethod.bm25:
                s = self._bm25_score(q_tokens, i)
            else:
                s = self._tfidf_score(q_tokens, i)
            if s > 0:
                scored.append((s, i))
        scored.sort(key=lambda x: (-x[0], x[1]))
        hits: list[SmokeHit] = []
        for score, idx in scored[:top_k]:
            hits.append(_hit_from_row(self._rows[idx], score))
        return hits
