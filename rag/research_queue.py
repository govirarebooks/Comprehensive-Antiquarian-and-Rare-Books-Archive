#!/usr/bin/env python3
"""Manage the incremental external-scholar research queue.

The queue is derived operational state. It never changes the Govi source dataset.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "govi-rare-books-academic-dataset.json"
CACHE = ROOT / "rag" / "external_knowledge" / "external_knowledge.json"


def load(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def enqueue_record(record_id: str) -> int:
    dataset = load(DATASET, [])
    record = next((r for r in dataset if str(r.get("id")) == str(record_id)), None)
    if not record:
        raise SystemExit(f"Record non trovato: {record_id}")

    entities = []
    for field in ("authors", "related_names", "publishers"):
        for item in record.get(field, []) or []:
            name = item.get("name") if isinstance(item, dict) else item
            if name and name not in entities:
                entities.append(name)

    cache = load(CACHE, {"entities": {}})
    changed = []
    for name in entities:
        payload = cache.get("entities", {}).get(name)
        if not payload or payload.get("status") in {"pending", "partial"}:
            changed.append(name)

    print(json.dumps({
        "record_id": record_id,
        "book": record.get("title"),
        "entities": entities,
        "queue_candidates": changed,
        "queued_at": datetime.now(timezone.utc).isoformat(),
    }, ensure_ascii=False, indent=2))
    return 0


def status() -> int:
    cache = load(CACHE, {"entities": {}, "counts": {}})
    counts = cache.get("counts", {})
    print(json.dumps(counts, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def process(names: list[str], offline: bool = False, refresh: bool = False) -> int:
    if not names:
        print("Nessuna entità specificata.")
        return 0
    cmd = [
        sys.executable,
        str(ROOT / "rag" / "build_external_knowledge.py"),
        "--scope", "authors,related_names,publishers",
        "--names", "||".join(names),
    ]
    if offline:
        cmd.append("--offline")
    if refresh:
        cmd.append("--refresh")
    return subprocess.call(cmd, cwd=ROOT)


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("enqueue-record")
    p.add_argument("record_id")

    sub.add_parser("status")

    p = sub.add_parser("process")
    p.add_argument("names", nargs="*")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--refresh", action="store_true")

    args = parser.parse_args()
    if args.command == "enqueue-record":
        return enqueue_record(args.record_id)
    if args.command == "status":
        return status()
    if args.command == "process":
        return process(args.names, offline=args.offline, refresh=args.refresh)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
