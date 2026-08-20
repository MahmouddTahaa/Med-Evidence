"""Export the frozen product-lock index for deployment (trimmed Chroma + job artifacts)."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import chromadb

JOB_ID = "ae69f99b47b7"
COLLECTION = (
    "statpearls_pharmacology_titleprefix__section_aware__BAAI_bge-small-en-v1.5"
)
SRC_CHROMA = Path("artifacts/indexes/chroma")
SRC_JOB = Path("artifacts/jobs") / JOB_ID
DEFAULT_OUT = Path("artifacts/product-lock")
PERSIST_DIR = "artifacts/product-lock/indexes/chroma"
BATCH = 500


def copy_collection(src_dir: Path, dst_dir: Path, name: str) -> int:
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    src = chromadb.PersistentClient(path=str(src_dir))
    dst = chromadb.PersistentClient(path=str(dst_dir))
    col = src.get_collection(name)
    meta = col.metadata or {}
    total = col.count()
    if total == 0:
        raise SystemExit(f"Source collection {name!r} is empty")

    out = dst.get_or_create_collection(name=name, metadata=meta)
    offset = 0
    while offset < total:
        batch = col.get(
            limit=BATCH,
            offset=offset,
            include=["embeddings", "documents", "metadatas"],
        )
        ids = batch["ids"]
        if not ids:
            break
        out.upsert(
            ids=ids,
            embeddings=batch["embeddings"],
            documents=batch["documents"],
            metadatas=batch["metadatas"],
        )
        offset += len(ids)
        print(f"  copied {offset}/{total}")
    return total


def main() -> None:
    p = argparse.ArgumentParser(description="Export trimmed product-lock index bundle")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    if not SRC_JOB.is_dir() or not (SRC_JOB / "chunks.json").is_file():
        raise SystemExit(f"Missing job dir: {SRC_JOB}")
    if not SRC_CHROMA.is_dir():
        raise SystemExit(f"Missing source Chroma: {SRC_CHROMA}")

    out_root = args.out
    job_dst = out_root / "jobs" / JOB_ID
    chroma_dst = out_root / "indexes" / "chroma"

    if out_root.exists():
        shutil.rmtree(out_root)
    job_dst.mkdir(parents=True)

    print("Copying job artifacts…")
    shutil.copy2(SRC_JOB / "report.json", job_dst / "report.json")
    shutil.copy2(SRC_JOB / "chunks.json", job_dst / "chunks.json")

    report = json.loads((job_dst / "report.json").read_text(encoding="utf-8"))
    combo = report.setdefault("combo", {})
    store = combo.setdefault("store", {})
    store["persist_dir"] = PERSIST_DIR
    (job_dst / "report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Exporting Chroma collection…")
    n = copy_collection(SRC_CHROMA, chroma_dst, COLLECTION)
    print(f"Exported {n} vectors to {out_root}")

    job_mb = sum(f.stat().st_size for f in job_dst.rglob("*") if f.is_file()) / 1024**2
    chroma_mb = sum(f.stat().st_size for f in chroma_dst.rglob("*") if f.is_file()) / 1024**2
    print(f"Bundle size: job {job_mb:.1f} MB + chroma {chroma_mb:.1f} MB")


if __name__ == "__main__":
    main()
