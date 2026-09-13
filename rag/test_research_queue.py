#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import build_external_knowledge as bek


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        old = bek.OUTPUT_PATH
        try:
            bek.OUTPUT_PATH = Path(td) / "external_knowledge.json"
            payload = {"version": 1, "entities": {}}
            bek.save_json(bek.OUTPUT_PATH, payload)
            pending = bek.mark_pending("Corvus Andreas", ["author"], None, "offline_mode")
            assert pending["status"] == "pending"
            assert pending["next_retry_at"]
            assert pending["attempts"] == 1
            bek.save_json(bek.OUTPUT_PATH, {"version": 1, "entities": {"Corvus Andreas": pending}})
            loaded = json.loads(bek.OUTPUT_PATH.read_text(encoding="utf-8"))
            assert loaded["entities"]["Corvus Andreas"]["status"] == "pending"
        finally:
            bek.OUTPUT_PATH = old
    print("GOVI RESEARCH QUEUE TEST")
    print("PASS")


if __name__ == "__main__":
    main()
