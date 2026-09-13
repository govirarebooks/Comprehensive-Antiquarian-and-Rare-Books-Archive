#!/usr/bin/env python3
"""Combine Govi scholar graph with external contextual knowledge.

This layer is explicitly derived. Govi's catalogue remains the source of truth.
External knowledge can add historical/intellectual context and discover links
between Govi authors, but it never changes catalogue facts.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "govi-rare-books-academic-dataset.json"
SCHOLAR_PATH = ROOT / "rag" / "scholar_index.json"
EXTERNAL_PATH = ROOT / "rag" / "external_knowledge" / "external_knowledge.json"
OUTPUT_PATH = ROOT / "rag" / "external_knowledge" / "scholar_context_index.json"


def norm(v: Any) -> str:
    text = str(v or "").lower()
    text = re.sub(r"[^\wÀ-ÿ]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def load(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def build_context(dataset, scholar, external):
    scholar_authors = scholar.get("authors", {})
    external_entities = external.get("entities", {})

    name_to_qid = {}
    qid_to_name = {}
    for name, payload in external_entities.items():
        wd = (payload.get("sources") or {}).get("wikidata") or {}
        qid = wd.get("id")
        if qid:
            qid_to_name[qid] = name
            name_to_qid[norm(name)] = qid
            label = wd.get("label")
            if label:
                qid_to_name[qid] = name

    # Map normalised Govi author names to their externally-resolved QIDs.
    author_qids = {}
    for author in scholar_authors:
        qid = name_to_qid.get(norm(author))
        if qid:
            author_qids[author] = qid

    qid_to_govi_author = {qid: author for author, qid in author_qids.items()}

    authors = {}
    external_relations = []

    for author, scholar_author in scholar_authors.items():
        ext = external_entities.get(author)
        if not ext:
            authors[author] = {
                "catalogue_author": True,
                "external_available": False,
                "context": {},
                "govi_crosslinks": [],
            }
            continue

        wd = (ext.get("sources") or {}).get("wikidata") or {}
        context = ext.get("scholar_context") or {}
        crosslinks = []

        claim_map = wd.get("claims") or {}
        for key in ("influenced_by", "influenced", "affiliations", "memberships", "educated_at", "employer"):
            for item in claim_map_for_context(claim_map, key):
                qid = item.get("id")
                target_author = qid_to_govi_author.get(qid)
                if target_author and target_author != author:
                    direction = "influenced_by" if key == "influenced_by" else "contextual_relation"
                    crosslinks.append({
                        "target_author": target_author,
                        "type": direction,
                        "external_property": key,
                        "external_entity_id": qid,
                        "evidence_source": "Wikidata",
                    })
                    external_relations.append({
                        "source_author": author,
                        "target_author": target_author,
                        "type": direction,
                        "external_property": key,
                        "external_entity_id": qid,
                        "evidence_source": "Wikidata",
                    })

        authors[author] = {
            "catalogue_author": True,
            "external_available": True,
            "external_entity_id": wd.get("id"),
            "external_label": wd.get("label"),
            "external_description": wd.get("description"),
            "wikipedia": (ext.get("sources") or {}).get("wikipedia"),
            "dbpedia": (ext.get("sources") or {}).get("dbpedia"),
            "context": context,
            "govi_crosslinks": unique_crosslinks(crosslinks),
        }

    record_context = {}
    for record in dataset:
        rid = str(record.get("id"))
        author_names = [
            a.get("name") if isinstance(a, dict) else a
            for a in record.get("authors", []) or []
        ]
        contexts = [authors[a] for a in author_names if a in authors and authors[a].get("external_available")]
        record_context[rid] = {
            "authors": author_names,
            "external_context_available": bool(contexts),
            "author_context": contexts,
            "govi_external_crosslinks": sorted(
                {
                    json.dumps(link, ensure_ascii=False, sort_keys=True)
                    for context in contexts
                    for link in context.get("govi_crosslinks", [])
                }
            ),
        }
        record_context[rid]["govi_external_crosslinks"] = [json.loads(x) for x in record_context[rid]["govi_external_crosslinks"]]

    return {
        "version": 1,
        "catalogue_truth": "govi-rare-books-academic-dataset.json",
        "external_context_cache": str(EXTERNAL_PATH.name),
        "authors": authors,
        "records": record_context,
        "external_relations": unique_crosslinks(external_relations),
        "counts": {
            "authors": len(authors),
            "authors_with_external_context": sum(1 for x in authors.values() if x.get("external_available")),
            "external_author_links": len(unique_crosslinks(external_relations)),
            "records": len(record_context),
        },
    }


def claim_map_for_context(claim_map: dict, key: str):
    # Claim names in the raw Wikidata payload are snake-case labels from
    # build_external_knowledge.py, but tolerate future variants.
    return claim_map.get(key, []) if isinstance(claim_map.get(key, []), list) else []


def unique_crosslinks(rows):
    seen = set()
    out = []
    for row in rows:
        key = json.dumps(row, ensure_ascii=False, sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append(row)
    return out


def main():
    dataset = load(DATASET_PATH, [])
    scholar = load(SCHOLAR_PATH, {})
    external = load(EXTERNAL_PATH, {"entities": {}})
    if not external.get("entities"):
        print("No external knowledge cache found; run: python rag/build_external_knowledge.py")
        return 0
    payload = build_context(dataset, scholar, external)
    save(OUTPUT_PATH, payload)
    print("=" * 80)
    print("GOVI SCHOLAR CONTEXT")
    print("=" * 80)
    print(f"Authors: {payload['counts']['authors']}")
    print(f"Authors with external context: {payload['counts']['authors_with_external_context']}")
    print(f"External author links: {payload['counts']['external_author_links']}")
    print(f"Records: {payload['counts']['records']}")
    print(f"Output: {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
