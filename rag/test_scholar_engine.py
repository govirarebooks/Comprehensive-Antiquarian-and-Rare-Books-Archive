#!/usr/bin/env python3

from __future__ import annotations

import re

from scholar_engine import ScholarEngine


CORVO_ID = "01KXXEC0QDA8AG7AHBDN2Q72DA"
MENGOLI_ID = "01KZ5SM6GHNTGP2RZXV7CBDS69"


def main():
    engine = ScholarEngine()

    assert len(engine.records) == 146
    assert len(engine.authors) == 157
    assert len(engine.topics) == 95

    source, similar_authors = engine.author_similarity(
        "Corvus Andreas",
        limit=8,
    )

    assert source == "Corvus Andreas"
    assert any(
        item["author"] == "Camerarius Joachim"
        for item in similar_authors
    ), "Camerarius should emerge as a meaningful scholarly connection for Corvo."

    profile = engine.collector_profile(CORVO_ID)
    assert profile["id"] == CORVO_ID

    serialized = str(profile)
    assert "The Dal Gesù brothers" not in serialized
    assert "In 1514 Grüninger" not in serialized
    assert "only one copy is recorded" in serialized

    _, similar_books = engine.book_similarity(
        CORVO_ID,
        limit=8,
    )

    assert similar_books
    assert all("score" in item for item in similar_books)
    assert any("conceptual_links" in item or item.get("shared_concepts") for item in similar_books)

    # Taxonomy regression: natural philosophy is not natural magic.
    assert "natural_magic" not in engine.srecords[MENGOLI_ID]["concepts"]
    assert "natural_philosophy" in engine.srecords[MENGOLI_ID]["concepts"]

    print("GOVI SCHOLAR ENGINE TEST")
    print("PASS")
    print(f"Records: {len(engine.records)}")
    print(f"Authors: {len(engine.authors)}")
    print(f"Topics: {len(engine.topics)}")
    print("Corvus → Camerarius connection: PASS")
    print("Collector-signal scope test: PASS")
    print("Book similarity: PASS")
    print("Natural philosophy ≠ natural magic: PASS")


if __name__ == "__main__":
    main()
