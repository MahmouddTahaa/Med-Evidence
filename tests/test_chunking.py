from pathlib import Path

import pytest

from clinical_rag.chunking import chunk_parsed
from clinical_rag.chunking import semantic as semantic_mod
from clinical_rag.parsing.text_plain import parse_plain
from clinical_rag.schemas import ChunkConfig, IngestJobConfig, StrategyId
from tests.helpers import HashEmbedder, raw_doc


def _config(tmp_path: Path, strategy: StrategyId, **chunk_kw) -> IngestJobConfig:
    md = tmp_path / "guide.md"
    md.write_text(
        "# Intro\n\nShort intro paragraph.\n\n"
        "## Treatment\n\nGive drug A then reassess. " * 40
        + "\n\n## Follow-up\n\nSee the patient again in two weeks.\n",
        encoding="utf-8",
    )
    raw = raw_doc(md, doc_id="guide")
    parsed = parse_plain(raw)
    cfg = IngestJobConfig(
        corpus_id="t",
        job_id="job1",
        files=[raw],
        chunk=ChunkConfig(strategy_id=strategy, **chunk_kw),
    )
    return cfg, raw, parsed


def _assert_basic_chunks(chunks, *, strategy: StrategyId) -> None:
    assert chunks
    assert all(c.chunk_id.startswith(f"guide-{strategy.value}-") for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert all(c.page_number is not None for c in chunks)
    assert all(c.section_title for c in chunks)
    assert all(c.token_count > 0 for c in chunks)


def test_prefix_section_title_on_later_windows(tmp_path: Path):
    cfg, raw, parsed = _config(
        tmp_path,
        StrategyId.section_aware,
        target_tokens=40,
        max_tokens=50,
        prefix_section_title=True,
    )
    chunks = chunk_parsed(parsed, raw, cfg)
    treatment = [c for c in chunks if c.section_title == "Treatment"]
    assert len(treatment) >= 2
    assert all(c.text.lstrip().lower().startswith("treatment") for c in treatment)


def test_prefix_section_title_helper_skips_when_already_present():
    from clinical_rag.chunking.base import apply_section_title_prefix

    already = "Administration\nDose 40 mg."
    assert apply_section_title_prefix(already, "Administration") == already
    assert apply_section_title_prefix("Dose 40 mg.", "Administration") == "Administration\nDose 40 mg."
    assert ChunkConfig().prefix_section_title is False


def test_section_aware_keeps_headings(tmp_path: Path):
    cfg, raw, parsed = _config(tmp_path, StrategyId.section_aware)
    chunks = chunk_parsed(parsed, raw, cfg)
    titles = {c.section_title for c in chunks}
    assert "Treatment" in titles or "Intro" in titles
    assert all(c.chunk_id.startswith("guide-section_aware-") for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_fixed_windows_have_stable_ids(tmp_path: Path):
    cfg, raw, parsed = _config(tmp_path, StrategyId.fixed, target_tokens=80, overlap_ratio=0.12)
    a = chunk_parsed(parsed, raw, cfg)
    b = chunk_parsed(parsed, raw, cfg)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert all(c.token_count > 0 for c in a)


def test_hierarchical_indexes_children(tmp_path: Path):
    cfg, raw, parsed = _config(
        tmp_path,
        StrategyId.hierarchical,
        parent_tokens=120,
        child_tokens=50,
        overlap_ratio=0.12,
    )
    chunks = chunk_parsed(parsed, raw, cfg)
    assert chunks
    assert all(c.parent_chunk_id for c in chunks)
    assert all(c.parent_chunk_id.startswith("guide-parent-") for c in chunks)


@pytest.mark.parametrize(
    "strategy",
    [
        StrategyId.langchain_recursive,
        StrategyId.langchain_token,
        StrategyId.langchain_markdown,
    ],
)
def test_langchain_strategies_emit_cited_chunks(tmp_path: Path, strategy: StrategyId):
    cfg, raw, parsed = _config(tmp_path, strategy, target_tokens=80, overlap_ratio=0.12)
    a = chunk_parsed(parsed, raw, cfg)
    b = chunk_parsed(parsed, raw, cfg)
    _assert_basic_chunks(a, strategy=strategy)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    if strategy is StrategyId.langchain_markdown:
        titles = {c.section_title for c in a}
        assert titles & {"Intro", "Treatment", "Follow-up"}


def test_semantic_strategy_monkeypatched_encoder(tmp_path: Path):
    encoder = HashEmbedder()
    semantic_mod.set_semantic_encode_override(encoder.encode)
    try:
        cfg, raw, parsed = _config(
            tmp_path,
            StrategyId.semantic,
            target_tokens=80,
            max_tokens=120,
            overlap_ratio=0.12,
        )
        a = chunk_parsed(parsed, raw, cfg)
        b = chunk_parsed(parsed, raw, cfg)
        _assert_basic_chunks(a, strategy=StrategyId.semantic)
        assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    finally:
        semantic_mod.set_semantic_encode_override(None)
