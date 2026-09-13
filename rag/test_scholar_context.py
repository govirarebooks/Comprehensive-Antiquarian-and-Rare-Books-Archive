#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag.build_scholar_context import build_context


def main():
    dataset = [
        {"id": "1", "authors": [{"name": "Author A"}]},
        {"id": "2", "authors": [{"name": "Author B"}]},
    ]
    scholar = {
        "authors": {
            "Author A": {"records": ["1"]},
            "Author B": {"records": ["2"]},
        }
    }
    external = {
        "entities": {
            "Author A": {
                "sources": {
                    "wikidata": {
                        "id": "Q1",
                        "label": "Author A",
                        "claims": {"influenced_by": [{"id": "Q2", "label": "Author B"}]},
                    }
                },
                "scholar_context": {"influenced_by": ["Author B"]},
            },
            "Author B": {
                "sources": {
                    "wikidata": {"id": "Q2", "label": "Author B", "claims": {}}
                },
                "scholar_context": {},
            },
        }
    }
    result = build_context(dataset, scholar, external)
    assert result["counts"]["authors_with_external_context"] == 2
    links = result["external_relations"]
    assert any(x["source_author"] == "Author A" and x["target_author"] == "Author B" for x in links)
    assert result["records"]["1"]["external_context_available"] is True
    print("GOVI SCHOLAR CONTEXT TEST")
    print("PASS")
    print("External author links:", len(links))


if __name__ == "__main__":
    main()
