#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "govi-rare-books-academic-dataset.json"
SCHOLAR_PATH = ROOT / "rag" / "scholar_index.json"
CONTEXT_PATH = ROOT / "rag" / "external_knowledge" / "scholar_context_index.json"
AI_INDEX_PATH = ROOT / "rag" / "ai_index.npz"


GENERIC_CONCEPTS = {"rarity", "printing", "religion", "humanism", "medicine", "history_of_science", "illustration", "censorship"}

CONCEPT_TERMS = {
    "astrology": {"astrology", "astrological", "astrologer", "planet", "zodiac", "eclipse", "conjunction"},
    "divination": {"divination", "divinatory", "prophecy", "oracle", "prognostication", "prediction", "practica"},
    "omens": {"omen", "omens", "portent", "signs", "foreshadow", "future calamities"},
    "chiromancy": {"chiromancy", "palmistry", "palm reading", "palmists"},
    "physiognomy": {"physiognomy", "physiognomic"},
    "natural_magic": {"natural magic", "occult sciences", "magic", "magical"},
    "witchcraft": {"witchcraft", "witches", "witch trials", "sorcery", "witch craze"},
    "medicine": {"medicine", "medical", "physician", "surgery", "anatomy"},
    "printing": {"printer", "printing", "press", "typography", "bookshop", "bookseller"},
    "humanism": {"humanist", "humanism", "renaissance", "classical", "philology"},
    "religion": {"church", "theology", "religious", "reformation", "catholic", "lutheran", "protestant"},
}

CONCEPT_RELATIONS = {
    # Strong intellectual neighbourhoods: these are not assertions of identity,
    # but useful scholarly bridges for finding non-obvious analogues.
    "astrology": {"astrology": 1.0, "divination": 0.95, "omens": 0.90, "astronomy": 0.86, "natural_magic": 0.82, "history_of_science": 0.72},
    "astronomy": {"astronomy": 1.0, "astrology": 0.86, "history_of_science": 0.90, "natural_philosophy": 0.82},
    "divination": {"divination": 1.0, "astrology": 0.95, "omens": 0.92, "natural_magic": 0.84, "witchcraft": 0.68},
    "omens": {"omens": 1.0, "divination": 0.92, "astrology": 0.90, "natural_magic": 0.78, "witchcraft": 0.66},
    "chiromancy": {"chiromancy": 1.0, "physiognomy": 0.90, "divination": 0.86, "natural_magic": 0.74, "medicine": 0.65},
    "physiognomy": {"physiognomy": 1.0, "chiromancy": 0.90, "divination": 0.84, "medicine": 0.62, "natural_magic": 0.70},
    "natural_magic": {"natural_magic": 1.0, "astrology": 0.82, "divination": 0.84, "omens": 0.78, "witchcraft": 0.72, "chiromancy": 0.74, "medicine": 0.68, "history_of_science": 0.58},
    "witchcraft": {"witchcraft": 1.0, "natural_magic": 0.72, "divination": 0.68, "omens": 0.66, "religion": 0.60},
    "medicine": {"medicine": 1.0, "chiromancy": 0.65, "physiognomy": 0.62, "natural_magic": 0.68, "history_of_science": 0.78},
    "history_of_science": {"history_of_science": 1.0, "medicine": 0.78, "astronomy": 0.90, "astrology": 0.72, "natural_philosophy": 0.92, "natural_magic": 0.58},
    "natural_philosophy": {"natural_philosophy": 1.0, "history_of_science": 0.92, "astronomy": 0.82, "medicine": 0.70, "natural_magic": 0.28},
    "printing": {"printing": 1.0, "humanism": 0.42},
    "humanism": {"humanism": 1.0, "printing": 0.42, "religion": 0.25},
    "religion": {"religion": 1.0, "witchcraft": 0.60, "censorship": 0.55, "humanism": 0.25},
    "censorship": {"censorship": 1.0, "religion": 0.55, "witchcraft": 0.48},
}




def norm(text):
    text = str(text or "").lower()
    text = re.sub(r"[^\wÀ-ÿ]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def tokens(text):
    return {t for t in norm(text).split() if len(t) >= 3}


def jaccard(a, b):
    a = set(a); b = set(b)
    if not a and not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def conceptual_affinity(source_concepts, target_concepts):
    source = [c for c in source_concepts if c in CONCEPT_RELATIONS]
    target = [c for c in target_concepts if c in CONCEPT_RELATIONS]
    if not source or not target:
        return 0.0, []

    best_pairs = []
    best_score = 0.0
    for left in source:
        for right in target:
            score = CONCEPT_RELATIONS.get(left, {}).get(right, 0.0)
            if score <= 0:
                continue
            pair = {"from": left, "to": right, "strength": round(float(score), 2)}
            if score > best_score + 1e-9:
                best_score = score
                best_pairs = [pair]
            elif abs(score - best_score) < 1e-9:
                best_pairs.append(pair)

    # Exact multi-concept overlap is a stronger signal than a single bridge.
    exact = set(source) & set(target)
    exact_score = min(1.0, len(exact) / 2.0)
    if exact_score >= best_score:
        best_score = exact_score
        best_pairs = [
            {"from": concept, "to": concept, "strength": 1.0}
            for concept in sorted(exact)
        ]

    return float(best_score), best_pairs[:4]


def cosine(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    den = np.linalg.norm(a) * np.linalg.norm(b)
    if den == 0:
        return 0.0
    return float(np.dot(a, b) / den)


def load_data():
    with DATASET_PATH.open("r", encoding="utf-8") as f:
        dataset = json.load(f)
    with SCHOLAR_PATH.open("r", encoding="utf-8") as f:
        scholar = json.load(f)
    ai = np.load(AI_INDEX_PATH, allow_pickle=True)
    context = json.loads(CONTEXT_PATH.read_text(encoding="utf-8")) if CONTEXT_PATH.exists() else {"authors": {}, "records": {}, "external_relations": []}
    return dataset, scholar, ai, context


class ScholarEngine:
    def __init__(self):
        dataset, scholar, ai, context = load_data()
        self.dataset = dataset
        self.scholar = scholar
        self.records = {str(r.get("id")): r for r in dataset}
        self.srecords = scholar["records"]
        self.authors = scholar["authors"]
        self.publishers = scholar["publishers"]
        self.related = scholar["related_names"]
        self.topics = scholar["topics"]
        self.emb_full = {str(i): e for i, e in zip(ai["ids"], ai["embedding_full"])}
        self.emb_int = {str(i): e for i, e in zip(ai["ids"], ai["embedding_intellectual"])}
        self.emb_bib = {str(i): e for i, e in zip(ai["ids"], ai["embedding_bibliographic"])}
        self.context = context
        self.external_authors = context.get("authors", {}) or {}
        self.external_relations = context.get("external_relations", []) or []
        self.external_author_links = defaultdict(list)
        for link in self.external_relations:
            self.external_author_links[norm(link.get("source_author"))].append(link)
            self.external_author_links[norm(link.get("target_author"))].append(link)

        self.author_to_records = defaultdict(list)
        for rid, rec in self.srecords.items():
            for a in rec.get("authors", []):
                self.author_to_records[norm(a)].append(rid)

    def resolve_author(self, query):
        q = norm(query)
        exact = [name for name in self.authors if norm(name) == q]
        if exact:
            return exact[0]
        contains = [name for name in self.authors if q in norm(name) or norm(name) in q]
        if contains:
            return sorted(contains, key=len)[0]
        qtok = tokens(query)
        scored = []
        for name in self.authors:
            score = jaccard(qtok, tokens(name))
            if score > 0:
                scored.append((score, name))
        return max(scored)[1] if scored else None

    def author_vector(self, author_name, emb_map):
        ids = self.author_to_records.get(norm(author_name), [])
        vectors = [emb_map[rid] for rid in ids if rid in emb_map]
        if not vectors:
            return None
        v = np.mean(np.asarray(vectors, dtype=np.float32), axis=0)
        n = np.linalg.norm(v)
        return v / n if n else v

    def author_similarity(self, query, limit=8):
        source = self.resolve_author(query)
        if not source:
            raise ValueError(f"Autore non trovato nel catalogo: {query}")

        src = self.authors[source]
        src_vec = self.author_vector(source, self.emb_int)
        src_topics = src.get("topics", [])
        src_concepts = [c for c in src.get("concepts", []) if c not in GENERIC_CONCEPTS]
        src_related = src.get("related_names", [])
        src_publishers = src.get("publishers", [])
        src_periods = src.get("periods", [])

        results = []
        for name, a in self.authors.items():
            if name == source:
                continue
            vec = self.author_vector(name, self.emb_int)
            semantic = cosine(src_vec, vec) if src_vec is not None and vec is not None else 0.0
            topic = jaccard(src_topics, a.get("topics", []))
            other_concepts = [c for c in a.get("concepts", []) if c not in GENERIC_CONCEPTS]
            concept_exact = jaccard(src_concepts, other_concepts)
            concept_bridge, concept_links = conceptual_affinity(src_concepts, other_concepts)
            concept = max(concept_exact, concept_bridge)
            related = jaccard(src_related, a.get("related_names", []))
            pubs = jaccard(src_publishers, a.get("publishers", []))
            periods = jaccard(src_periods, a.get("periods", []))
            shared_topics = sorted(set(src_topics) & set(a.get("topics", [])))
            shared_concepts = sorted(set(src_concepts) & set(a.get("concepts", [])))
            shared_related = sorted(set(src_related) & set(a.get("related_names", [])))

            ext_links = [
                link for link in self.external_author_links.get(norm(source), [])
                if norm(link.get("source_author")) == norm(source) and norm(link.get("target_author")) == norm(name)
            ] + [
                link for link in self.external_author_links.get(norm(name), [])
                if norm(link.get("source_author")) == norm(name) and norm(link.get("target_author")) == norm(source)
            ]
            ext_context_src = self.external_authors.get(source, {}).get("context", {})
            ext_context_target = self.external_authors.get(name, {}).get("context", {})
            external_context_fields = ("movements", "affiliations", "memberships", "occupations", "educated_at")
            ext_overlap = 0.0
            ext_overlap_details = []
            for field in external_context_fields:
                left = set(ext_context_src.get(field, []) or [])
                right = set(ext_context_target.get(field, []) or [])
                overlap = jaccard(left, right)
                if overlap > ext_overlap:
                    ext_overlap = overlap
                    ext_overlap_details = [{"field": field, "shared": sorted(left & right)[:8]}]

            external_bridge = 1.0 if ext_links else ext_overlap

            # A scholar-level author analogy must have an intelligible bridge.
            # Shared century/topic alone is context, not a meaningful relationship.
            meaningful_bridge = max(
                concept_exact,
                concept_bridge,
                related,
                pubs,
                external_bridge,
            )
            if meaningful_bridge <= 0 and semantic < 0.62:
                continue

            # Semantic similarity helps order candidates, but explicit scholar
            # relationships must dominate the interpretation.
            score = (
                0.30*semantic
                + 0.24*concept_exact
                + 0.18*concept_bridge
                + 0.10*related
                + 0.08*pubs
                + 0.05*topic
                + 0.05*periods
                + 0.16*min(1.0, len(ext_links))
                + 0.04*ext_overlap
            )
            if meaningful_bridge <= 0:
                score *= 0.80

            relationship_type = "semantic_proximity"
            relationship_strength = round(float(semantic), 3)
            relationship_details = []
            if ext_links:
                relationship_type = "external_historical_link"
                relationship_strength = round(min(1.0, 0.72 + 0.10 * len(ext_links)), 3)
                relationship_details = [{"type": "external_historical_link", "links": ext_links[:4]}]
            elif concept_exact > 0:
                relationship_type = "shared_intellectual_concepts"
                relationship_strength = round(float(max(concept_exact, concept_bridge)), 3)
                relationship_details = [{"type": "shared_concept", "concepts": shared_concepts[:8]}]
            elif concept_links:
                relationship_type = "conceptual_bridge"
                relationship_strength = round(float(concept_bridge), 3)
                relationship_details = [{"type": "conceptual_bridge", "links": concept_links}]
            elif related > 0:
                relationship_type = "shared_related_name"
                relationship_strength = round(float(related), 3)
                relationship_details = [{"type": "shared_related_name", "names": shared_related[:8]}]
            elif pubs > 0:
                relationship_type = "shared_publisher"
                relationship_strength = round(float(pubs), 3)
                relationship_details = [{"type": "shared_publisher", "publishers": sorted(set(src_publishers) & set(a.get("publishers", [])))[:8]}]

            results.append({
                "author": name,
                "score": round(float(score), 4),
                "semantic": round(float(semantic), 4),
                "relationship_type": relationship_type,
                "relationship_strength": relationship_strength,
                "relationship_details": relationship_details,
                "shared_topics": shared_topics[:10],
                "shared_concepts": shared_concepts[:10],
                "shared_related_names": shared_related[:10],
                "shared_periods": sorted(set(src_periods) & set(a.get("periods", []))),
                "records": len(a.get("records", [])),
                "external_context": {
                    "available": bool(self.external_authors.get(name, {}).get("external_available")),
                    "description": self.external_authors.get(name, {}).get("external_description"),
                    "shared_context": ext_overlap_details,
                },
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return source, results[:limit]

    def book_similarity(self, record_id, limit=8):
        rid = self._resolve_record(record_id)
        if not rid:
            raise ValueError(f"Libro non trovato: {record_id}")
        src = self.srecords[rid]
        results = []
        for other_id, other in self.srecords.items():
            if other_id == rid:
                continue
            full = cosine(self.emb_full[rid], self.emb_full[other_id])
            intellectual = cosine(self.emb_int[rid], self.emb_int[other_id])
            bibliographic = cosine(self.emb_bib[rid], self.emb_bib[other_id])
            topic = jaccard(src.get("topics", []), other.get("topics", []))
            authors = jaccard(src.get("authors", []), other.get("authors", []))
            related = jaccard(src.get("related_names", []), other.get("related_names", []))
            external_book_links = []
            for sa in src.get("authors", []):
                for ta in other.get("authors", []):
                    for link in self.external_author_links.get(norm(sa), []):
                        if norm(link.get("source_author")) == norm(sa) and norm(link.get("target_author")) == norm(ta):
                            external_book_links.append(link)
                        elif norm(link.get("source_author")) == norm(ta) and norm(link.get("target_author")) == norm(sa):
                            external_book_links.append(link)
            src_concepts = [c for c in src.get("concepts", []) if c not in GENERIC_CONCEPTS]
            other_concepts = [c for c in other.get("concepts", []) if c not in GENERIC_CONCEPTS]
            concept_exact = jaccard(src_concepts, other_concepts)
            concept_bridge, concept_links = conceptual_affinity(src_concepts, other_concepts)

            source_year = src.get("publication_year")
            target_year = other.get("publication_year")
            if isinstance(source_year, int) and isinstance(target_year, int):
                year_proximity = math.exp(-abs(source_year - target_year) / 75.0)
            else:
                year_proximity = 0.0
            period_score = 1.0 if src.get("period") == other.get("period") else 0.0
            place_score = 1.0 if src.get("publication_place") and src.get("publication_place") == other.get("publication_place") else 0.0

            # Same century is contextual only. A book needs a real intellectual
            # or bibliographical bridge to qualify as a scholarly analogue.
            meaningful = max(concept_bridge, concept_exact, topic, authors, related, place_score, 1.0 if external_book_links else 0.0)
            if meaningful <= 0 and intellectual < 0.62:
                continue

            score = (
                0.34 * intellectual
                + 0.16 * full
                + 0.05 * bibliographic
                + 0.18 * concept_bridge
                + 0.10 * concept_exact
                + 0.08 * topic
                + 0.04 * authors
                + 0.02 * related
                + 0.01 * place_score
                + 0.02 * year_proximity
                + 0.10 * min(1.0, len(external_book_links))
            )

            reasons = []
            if concept_exact > 0:
                reasons.append({"type": "shared_concepts", "concepts": sorted(set(src_concepts) & set(other_concepts))[:6]})
            elif concept_links:
                reasons.append({"type": "conceptual_bridge", "links": concept_links})
            if topic > 0:
                reasons.append({"type": "shared_topics", "topics": sorted(set(src.get("topics", [])) & set(other.get("topics", [])))[:6]})
            if authors > 0:
                reasons.append({"type": "shared_authors", "authors": sorted(set(src.get("authors", [])) & set(other.get("authors", [])))})
            if related > 0:
                reasons.append({"type": "shared_related_names", "names": sorted(set(src.get("related_names", [])) & set(other.get("related_names", [])))})
            if place_score > 0:
                reasons.append({"type": "shared_publication_place", "place": src.get("publication_place")})
            if period_score > 0:
                reasons.append({"type": "shared_period", "period": src.get("period")})
            if year_proximity > 0.75:
                reasons.append({"type": "chronological_proximity", "years": [source_year, target_year]})
            if external_book_links:
                reasons.append({"type": "external_historical_link", "links": external_book_links[:4]})

            results.append({
                "id": other_id,
                "title": other["title"],
                "score": round(float(score), 4),
                "shared_topics": sorted(set(src.get("topics", [])) & set(other.get("topics", [])))[:10],
                "shared_concepts": sorted(set(src_concepts) & set(other_concepts))[:10],
                "conceptual_links": concept_links,
                "shared_authors": sorted(set(src.get("authors", [])) & set(other.get("authors", []))),
                "shared_related_names": sorted(set(src.get("related_names", [])) & set(other.get("related_names", []))),
                "period": other.get("period"),
                "publication_year": other.get("publication_year"),
                "connection_reasons": reasons[:5],
                "external_historical_links": external_book_links[:4],
            })
        results.sort(key=lambda x: x["score"], reverse=True)
        return rid, results[:limit]

    def query(self, query, limit=8):
        qtok = tokens(query)
        qconcepts = {c for c, terms in CONCEPT_TERMS.items() if any(term in norm(query) for term in terms)}
        # Lightweight lexical + catalog semantic search over existing AI vectors.
        # Query vector is intentionally handled by the calling recommendation engine;
        # this method focuses on scholar graph expansion from text concepts.
        rows = []
        for rid, rec in self.srecords.items():
            text = " ".join([rec.get("title", ""), " ".join(rec.get("topics", [])), " ".join(rec.get("concepts", []))])
            ttok = tokens(text)
            lexical = len(qtok & ttok) / max(1, len(qtok))
            concept = len(qconcepts & set(rec.get("concepts", []))) / max(1, len(qconcepts)) if qconcepts else 0.0
            topic_hits = [t for t in rec.get("topics", []) if tokens(t) & qtok]
            score = 0.65*concept + 0.25*lexical + 0.10*min(1.0, len(topic_hits)/2)
            if score > 0:
                rows.append({"id": rid, "title": rec["title"], "score": round(score,4), "concepts": rec.get("concepts", []), "topics": rec.get("topics", []), "period": rec.get("period")})
        rows.sort(key=lambda x: x["score"], reverse=True)
        return rows[:limit]

    def collector_profile(self, record_id):
        rid = self._resolve_record(record_id)
        if not rid:
            raise ValueError(f"Libro non trovato: {record_id}")
        rec = self.srecords[rid]
        return {
            "id": rid,
            "title": rec["title"],
            "authors": rec.get("authors", []),
            "period": rec.get("period"),
            "concepts": rec.get("concepts", []),
            "collector_signals": rec.get("collector_signals", []),
            "topics": rec.get("topics", []),
        }

    def author_dossier(self, query):
        source = self.resolve_author(query)
        if not source:
            raise ValueError(f"Autore non trovato nel catalogo: {query}")
        author = self.authors[source]
        external = self.external_authors.get(source, {})
        return {
            "author": source,
            "catalogue_records": author.get("records", []),
            "catalogue_topics": author.get("topics", []),
            "catalogue_concepts": author.get("concepts", []),
            "catalogue_related_names": author.get("related_names", []),
            "external_context_available": bool(external.get("external_available")),
            "external_entity_id": external.get("external_entity_id"),
            "external_description": external.get("external_description"),
            "scholar_context": external.get("context", {}),
            "wikipedia": external.get("wikipedia"),
            "govi_crosslinks": external.get("govi_crosslinks", []),
        }

    def book_dossier(self, record_id):
        rid = self._resolve_record(record_id)
        if not rid:
            raise ValueError(f"Libro non trovato: {record_id}")
        rec = self.records[rid]
        return {
            "id": rid,
            "title": rec.get("title"),
            "catalogue_record": rec,
            "scholar_context": self.context.get("records", {}).get(rid, {}),
        }

    def _resolve_record(self, query):
        raw = str(query or "").strip()
        q = norm(raw)
        for rid, r in self.srecords.items():
            if str(rid).strip().lower() == q:
                return rid
            title = norm(r.get("title", ""))
            if q == title or q in title:
                return rid
        return None


def main():
    parser = argparse.ArgumentParser(description="Govi Scholar Engine")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("author-similar")
    p.add_argument("author")
    p.add_argument("--limit", type=int, default=8)

    p = sub.add_parser("book-similar")
    p.add_argument("book")
    p.add_argument("--limit", type=int, default=8)

    p = sub.add_parser("query")
    p.add_argument("query")
    p.add_argument("--limit", type=int, default=8)

    p = sub.add_parser("collector")
    p.add_argument("book")

    p = sub.add_parser("author-dossier")
    p.add_argument("author")

    p = sub.add_parser("book-dossier")
    p.add_argument("book")

    args = parser.parse_args()
    engine = ScholarEngine()

    if args.command == "author-similar":
        source, results = engine.author_similarity(args.author, args.limit)
        print(json.dumps({"author": source, "results": results}, ensure_ascii=False, indent=2))
    elif args.command == "book-similar":
        source, results = engine.book_similarity(args.book, args.limit)
        print(json.dumps({"book_id": source, "results": results}, ensure_ascii=False, indent=2))
    elif args.command == "query":
        print(json.dumps({"query": args.query, "results": engine.query(args.query, args.limit)}, ensure_ascii=False, indent=2))
    elif args.command == "author-dossier":
        print(json.dumps(engine.author_dossier(args.author), ensure_ascii=False, indent=2))
    elif args.command == "book-dossier":
        print(json.dumps(engine.book_dossier(args.book), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(engine.collector_profile(args.book), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
