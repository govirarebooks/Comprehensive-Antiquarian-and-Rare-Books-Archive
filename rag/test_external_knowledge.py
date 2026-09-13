#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "govi-rare-books-academic-dataset.json"
OUT = ROOT / "rag" / "external_knowledge" / "external_knowledge.json"


def main() -> None:
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    assert isinstance(dataset, list) and len(dataset) == 146
    assert OUT.exists(), "Run build_external_knowledge.py first"
    payload = json.loads(OUT.read_text(encoding="utf-8"))
    assert payload.get("version") == 1
    assert payload.get("source_policy", {}).get("catalogue_truth") == "govi-rare-books-academic-dataset.json"
    entities = payload.get("entities") or {}
    assert entities
    # External knowledge is explicitly separate from catalogue facts.
    sample = next(iter(entities.values()))
    assert "sources" in sample
    assert "scholar_context" in sample
    assert "catalogue_fact" not in sample
    print("GOVI EXTERNAL KNOWLEDGE TEST")
    print("PASS")
    print(f"Records: {len(dataset)}")
    print(f"External entities cached: {len(entities)}")


if __name__ == "__main__":
    main()
