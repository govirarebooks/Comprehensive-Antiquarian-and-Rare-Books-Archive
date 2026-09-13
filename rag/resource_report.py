#!/usr/bin/env python3
"""Report derived scholar-cache footprint and catalogue/RAG resource usage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    files = {
        "dataset": ROOT / "govi-rare-books-academic-dataset.json",
        "scholar_index": ROOT / "rag" / "scholar_index.json",
        "bibliographic_analysis": ROOT / "rag" / "bibliographic_analysis.json",
        "external_knowledge": ROOT / "rag" / "external_knowledge" / "external_knowledge.json",
        "scholar_context": ROOT / "rag" / "external_knowledge" / "scholar_context_index.json",
        "ai_index": ROOT / "rag" / "ai_index.npz",
        "chunk_index": ROOT / "rag" / "chunk_index.npz",
        "vector_index": ROOT / "rag" / "vector_index.npz",
    }
    report = {name: size_bytes(path) for name, path in files.items()}
    report["derived_total_bytes"] = sum(report[k] for k in report if k != "dataset")
    report["all_total_bytes"] = sum(report.values())

    ext_path = files["external_knowledge"]
    if ext_path.exists():
        payload = json.loads(ext_path.read_text(encoding="utf-8"))
        report["external_counts"] = payload.get("counts", {})

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return

    print("GOVI RESOURCE REPORT")
    for key, value in report.items():
        if key.endswith("bytes") and isinstance(value, int):
            print(f"{key}: {value / 1024 / 1024:.2f} MB")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
