from clinical_rag.config import resolve_embed_config
from clinical_rag.schemas import EmbedConfig


def test_resolve_embed_low_vram_uses_small_model(monkeypatch):
    monkeypatch.setattr("clinical_rag.config.cuda_total_memory_gb", lambda device_index=0: 3.66)
    monkeypatch.setattr("clinical_rag.config.detect_device", lambda requested: "cuda")

    resolved, warnings = resolve_embed_config(EmbedConfig())
    assert resolved.model_id == "BAAI/bge-small-en-v1.5"
    assert resolved.device == "cuda"
    assert resolved.batch_size <= 4
    assert warnings


def test_resolve_embed_keeps_m3_on_large_vram(monkeypatch):
    monkeypatch.setattr("clinical_rag.config.cuda_total_memory_gb", lambda device_index=0: 12.0)
    monkeypatch.setattr("clinical_rag.config.detect_device", lambda requested: "cuda")

    resolved, warnings = resolve_embed_config(EmbedConfig())
    assert resolved.model_id == "BAAI/bge-m3"
    assert not warnings
