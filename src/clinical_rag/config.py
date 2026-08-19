from __future__ import annotations

import re
import uuid
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from clinical_rag.schemas import (
    ChromaConfig,
    ChunkConfig,
    EmbedConfig,
    ParserConfig,
    QdrantConfig,
    RetrievalConfig,
    SmokeQueryConfig,
    VectorStoreKind,
)


class GenerationSettings(BaseModel):
    """Non-secret generation defaults. Secrets stay in .env only."""

    weak_score: float = Field(default=0.35, ge=0.0, le=1.0)
    prompt_k: int = Field(default=5, ge=1, le=20)
    memory_turns: int = Field(default=6, ge=0, le=20)
    multi_query_n: int = Field(default=3, ge=2, le=5)
    gemini_model: str = "gemini-2.5-flash"
    openai_model: str = "gpt-4o-mini"
    groq_model: str = "llama-3.3-70b-versatile"
    anthropic_model: str = "claude-sonnet-4-0"
    local_base_url: str = "http://127.0.0.1:11434/v1"
    local_model: str = "llama3.2"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )

    smoke_query: SmokeQueryConfig = SmokeQueryConfig()
    parser: ParserConfig = ParserConfig()
    chunk: ChunkConfig = ChunkConfig()
    embed: EmbedConfig = EmbedConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    vector_store: VectorStoreKind = VectorStoreKind.chroma
    chroma: ChromaConfig = ChromaConfig()
    qdrant: QdrantConfig = QdrantConfig()
    uploads_dir: Path = Path("data/uploads")
    jobs_dir: Path = Path("artifacts/jobs")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def store_kwargs(settings: Settings | None = None) -> dict:
    """Connection configs for build_store / smoke query (no secrets logged here)."""
    cfg = settings or get_settings()
    return {
        "chroma": cfg.chroma,
        "qdrant": cfg.qdrant,
    }


def make_job_id() -> str:
    return uuid.uuid4().hex[:12]


def make_doc_id(filename: str) -> str:
    stem = Path(filename).stem
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", stem).strip("-").lower()[:40]
    return slug or "doc"


def embed_model_slug(model_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", model_id)


def collection_name(corpus_id: str, strategy_id: str, embed_model_id: str) -> str:
    """Chroma/Qdrant-safe name: 3–512 chars, start/end alphanumeric."""
    raw = f"{corpus_id}__{strategy_id}__{embed_model_slug(embed_model_id)}"
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", raw)
    if len(name) < 3:
        name = f"col_{name}"
    name = name[:512].strip("._-")
    if len(name) < 3:
        name = f"col_{name}" if name else "col_default"
    if not name[0].isalnum():
        name = f"c{name}"
    if not name[-1].isalnum():
        name = f"{name}x"
    return name


def detect_device(requested: str) -> str:
    if requested and requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def cuda_total_memory_gb(device_index: int = 0) -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        props = torch.cuda.get_device_properties(device_index)
        return props.total_memory / (1024**3)
    except Exception:
        return None


# bge-m3 needs ~2.3 GiB for weights alone; leave headroom for activations and other GPU processes.
_MIN_VRAM_GB_FOR_BGE_M3 = 6.0
_MIN_RAM_GB_FOR_BGE_M3 = 10.0
_LARGE_EMBED_MODELS = frozenset({"BAAI/bge-m3"})


def _available_ram_gb() -> float | None:
    try:
        import psutil

        return psutil.virtual_memory().available / (1024**3)
    except Exception:
        try:
            meminfo = Path("/proc/meminfo").read_text(encoding="utf-8")
        except OSError:
            return None
        for line in meminfo.splitlines():
            if line.startswith("MemAvailable:"):
                kb = float(line.split()[1])
                return kb / (1024**2)
        return None


def resolve_embed_config(cfg: EmbedConfig) -> tuple[EmbedConfig, list[str]]:
    """Pick a model/device/batch_size that fits this machine when device is auto."""
    device = detect_device(cfg.device)
    model_id = cfg.model_id
    batch_size = cfg.batch_size
    warnings: list[str] = []

    vram_gb = cuda_total_memory_gb() if device == "cuda" else None
    ram_gb = _available_ram_gb() if device == "cpu" else None
    if model_id in _LARGE_EMBED_MODELS and vram_gb is not None and vram_gb < _MIN_VRAM_GB_FOR_BGE_M3:
        fallback = cfg.fallback_model_id
        if fallback and fallback != model_id:
            warnings.append(
                f"GPU has {vram_gb:.1f} GiB VRAM; switching embed model from {model_id} "
                f"to {fallback} (bge-m3 needs ~6 GiB on CUDA)."
            )
            model_id = fallback
            batch_size = min(batch_size, 4)
        else:
            warnings.append(
                f"GPU has {vram_gb:.1f} GiB VRAM; moving {model_id} to CPU to avoid CUDA OOM."
            )
            device = "cpu"
            batch_size = min(batch_size, 8)
            ram_gb = _available_ram_gb()

    if model_id in _LARGE_EMBED_MODELS and device == "cpu" and ram_gb is not None and ram_gb < _MIN_RAM_GB_FOR_BGE_M3:
        fallback = cfg.fallback_model_id
        if fallback and fallback != model_id:
            warnings.append(
                f"Only {ram_gb:.1f} GiB RAM free; switching embed model from {model_id} "
                f"to {fallback}."
            )
            model_id = fallback
            batch_size = min(batch_size, 8)

    if device == "cuda" and vram_gb is not None and vram_gb < 4.5:
        # 4 GiB laptop GPUs (e.g. RTX 3050 Ti): tiny batches only for large models.
        if model_id in _LARGE_EMBED_MODELS:
            batch_size = min(batch_size, 2)
        else:
            batch_size = min(batch_size, 16)

    resolved = EmbedConfig(
        model_id=model_id,
        fallback_model_id=cfg.fallback_model_id,
        device=device,
        batch_size=batch_size,
        provider=cfg.provider,
    )
    return resolved, warnings
