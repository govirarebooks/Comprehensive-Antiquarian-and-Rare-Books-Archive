#!/usr/bin/env python3
"""Run the Govi scholar research cycle for one catalogue record.

Cycle:
1. Read the Govi record (authoritative facts).
2. Resolve/research its external entities.
3. Rebuild the scholar-context layer.
4. Return a dossier combining catalogue facts, external context and Govi links.

The script intentionally does not mutate the Govi source dataset.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "govi-rare-books-academic-dataset.json"
ENGINE = ROOT / "rag" / "scholar_engine.py"


def load_dataset():
    return json.loads(DATASET.read_text(encoding="utf-8"))


def get_record(records, record_id):
    for r in records:
        if str(r.get("id")) == str(record_id):
            return r
    raise ValueError(f"Record non trovato: {record_id}")


def entity_names(record):
    names = []
    for field in ("authors", "related_names"):
        for item in record.get(field, []) or []:
            name = item.get("name") if isinstance(item, dict) else item
            if name and name not in names:
                names.append(name)
    return names


def run(cmd):
    print("$", " ".join(str(x) for x in cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("book")
    parser.add_argument("--queue-only", action="store_true", help="Queue external research without network calls")
    args = parser.parse_args()

    records = load_dataset()
    record = get_record(records, args.book)
    names = entity_names(record)

    # External research is deliberately incremental and scoped to the current work.
    if names:
        cmd = [
            sys.executable,
            str(ROOT / "rag" / "build_external_knowledge.py"),
            "--scope", "authors,related_names",
            "--names", "||".join(names),
        ]
        if args.queue_only:
            cmd.append("--offline")
        run(cmd)
    run([sys.executable, str(ROOT / "rag" / "build_scholar_context.py")])

    print("=" * 80)
    print("GOVI SCHOLAR RESEARCH CYCLE")
    print("=" * 80)
    print("Book:", record.get("title"))
    print("ID:", record.get("id"))
    print("External entities researched:", len(names))
    print("\n--- BOOK DOSSIER ---")
    run([sys.executable, str(ENGINE), "book-dossier", str(record.get("id"))])
    print("\n--- SIMILAR BOOKS ---")
    run([sys.executable, str(ENGINE), "book-similar", str(record.get("id")), "--limit", "8"])


if __name__ == "__main__":
    main()
