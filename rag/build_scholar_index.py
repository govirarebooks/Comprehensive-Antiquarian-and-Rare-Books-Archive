#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "govi-rare-books-academic-dataset.json"
BIBLIO_PATH = ROOT / "rag" / "bibliographic_analysis.json"
AI_INDEX_PATH = ROOT / "rag" / "ai_index.npz"
OUTPUT_JSON = ROOT / "rag" / "scholar_index.json"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

CONCEPTS = {
    "astrology": ["astrology", "astrological", "astrologer", "horoscope", "zodiac", "horary", "judicial astrology"],
    "astronomy": ["astronomy", "astronomical", "celestial", "eclipse", "conjunction", "planetary", "stars", "star catalog"],
    "divination": ["divination", "divinatory", "prophecy", "oracle", "prognostication", "prediction", "practica"],
    "omens": ["omen", "omens", "portent", "signs", "foreshadow", "future calamities"],
    "chiromancy": ["chiromancy", "palmistry", "palm reading", "palmists", "reading of hands"],
    "physiognomy": ["physiognomy", "physiognomic", "face reading"],
    "natural_magic": ["natural magic", "occult sciences", "occult", "magic", "magical", "natural magick"],
    "natural_philosophy": ["natural philosophy", "natural philosopher", "physics", "mechanics", "mathematical physics"],
    "witchcraft": ["witchcraft", "witches", "witch trials", "sorcery", "witch craze"],
    "medicine": ["medicine", "medical", "physician", "surgery", "anatomy"],
    "history_of_science": ["mathematics", "science", "scientific", "experimental", "scientific revolution", "natural sciences"],
    "printing": ["printer", "printing", "press", "typography", "typographic", "bookshop", "bookseller"],
    "humanism": ["humanist", "humanism", "renaissance", "classical", "philology"],
    "religion": ["church", "theology", "religious", "reformation", "catholic", "lutheran", "protestant", "calvinist"],
    "censorship": ["censored", "expurged", "index", "forbidden books", "prohibited", "suppressed"],
    "illustration": ["woodcut", "engraving", "illustrated", "illustrations", "plates", "iconographic"],
    "provenance": ["provenance", "ex-libris", "bookplate", "collection of", "library of", "belonged to", "owned by"],
    "rarity": ["extremely rare", "very rare", "very scarce", "only one copy", "only two copies", "scarce copy", "rare first edition"],
}

COLLECTOR_PATTERNS = {
    "first_edition": [r"\bfirst edition\b", r"\bprima edizione\b"],
    "first_issue": [r"\bfirst issue\b", r"\bfirst state\b"],
    "first_in_language": [r"\bfirst edition in (?:Italian|French|German|English|Dutch)\b", r"\bprima edizione italiana\b"],
    "extreme_rarity": [r"\bextremely rare\b", r"\bonly one copy\b", r"\bonly one known copy\b", r"\bseule copie\b"],
    "rarity": [r"\bvery rare\b", r"\bvery scarce\b", r"\brare first edition\b", r"\bscarce copy\b"],
    "provenance": [r"\bex-?libris\b", r"\bbookplate\b", r"\bprovenance of the present copy\b", r"\bprovenance:.*?copy\b", r"\bfrom the collection of\b"],
    "association": [r"\bpresentation copy\b", r"\bassociation copy\b", r"\bcopy presented to\b", r"\binscribed in the present copy\b"],
    "illustration": [r"\bwoodcut\b", r"\bengraved\b", r"\bwoodcuts\b", r"\bfull-page illustration\b", r"\billustrations\b"],
    "censorship": [r"\bcensored\b", r"\bexpurged\b", r"\bplaced on the Index\b", r"\bIndex of Forbidden Books\b"],
    "manuscript_annotation": [r"\bmanuscript notes\b", r"\bmarginalia\b", r"\bunderlinings\b", r"\bannotated\b"],
    "binding": [r"\bcontemporary half calf\b", r"\bcontemporary vellum\b", r"\bparchment\b", r"\boriginal binding\b", r"\bclasp preserved\b"],
    "incunable": [r"\bincunable\b", r"\bincunabula\b"],
    "sammelband": [r"\bsammelband\b", r"\bsammleband\b", r"\bbound together\b", r"\bmiscellany\b"],
}


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def lower(text: str) -> str:
    return norm(text).lower()


def load(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def name_of(item):
    if isinstance(item, dict):
        return norm(item.get("name", ""))
    return norm(item)


def names(record, field):
    value = record.get(field, [])
    if not isinstance(value, list):
        return []
    return [name_of(v) for v in value if name_of(v)]


def year_value(record):
    value = record.get("publication_year")
    try:
        return int(value)
    except Exception:
        match = re.search(r"\b(1[0-9]{3}|20[0-9]{2})\b", str(value or ""))
        return int(match.group(1)) if match else None


def period(year):
    if year is None:
        return "unknown"
    if year < 1501:
        return "incunable"
    if year < 1601:
        return "sixteenth century"
    if year < 1701:
        return "seventeenth century"
    if year < 1801:
        return "eighteenth century"
    if year < 1901:
        return "nineteenth century"
    return "twentieth century or later"


def concept_hits(text):
    t = lower(text)
    hits = []
    for concept, terms in CONCEPTS.items():
        if any(re.search(rf"\b{re.escape(term.lower())}\b", t) for term in terms):
            hits.append(concept)
    return hits


def collector_signals(text, bib_item):
    """Extract collector-facing signals from the CURRENT RECORD only.

    Author biographies and general bibliography are deliberately excluded:
    an author being early, pioneering, controversial, etc. is not the same
    thing as the physical copy being offered being rare or a first edition.
    """
    t = str(text or "")
    signals = []

    historical_phrases = [
        r"\balso issued\b",
        r"\balso printed\b",
        r"\balso published\b",
        r"\bsecond issue\b",
        r"\bsecond edition\b",
        r"\blater edition\b",
        r"\banother edition\b",
        r"\blatin edition\b",
        r"\bfrench edition\b",
        r"\bitalian edition\b",
        r"\bgerman edition\b",
        r"\bSpanish edition\b",
        r"\bin \d{4} .*? issued\b",
        r"\bin \d{4} .*? published\b",
    ]

    def clean_sentence(sentence):
        text_local = norm(sentence)
        cut=[]
        for pattern in historical_phrases:
            m=re.search(pattern,text_local,flags=re.IGNORECASE)
            if m:
                cut.append(m.start())
        if cut:
            text_local=norm(text_local[:min(cut)])
        return text_local

    for kind, patterns in COLLECTOR_PATTERNS.items():
        matches = []
        for pattern in patterns:
            for m in re.finditer(pattern, t, flags=re.IGNORECASE):
                start = max(0, t.rfind('.', 0, m.start()) + 1)
                end = t.find('.', m.end())
                if end == -1:
                    end = min(len(t), m.end() + 260)
                sentence = clean_sentence(t[start:end])
                if not sentence:
                    continue
                if any(re.search(pat, sentence, flags=re.IGNORECASE) for pat in historical_phrases):
                    continue
                required = {
                    "first_edition": [r"\bfirst edition\b", r"\bprima edizione\b"],
                    "first_issue": [r"\bfirst issue\b", r"\bfirst state\b"],
                    "first_in_language": [r"\bfirst edition in\b", r"\bprima edizione\b"],
                    "extreme_rarity": [r"\bextremely rare\b", r"\bonly one copy\b", r"\bonly one known copy\b"],
                    "rarity": [r"\brare\b", r"\bscarce\b", r"\bvery rare\b"],
                }
                if kind in required and not any(re.search(pat, sentence, flags=re.IGNORECASE) for pat in required[kind]):
                    continue
                if sentence not in matches:
                    matches.append(sentence)
        if matches:
            signals.append({"type": kind, "evidence": matches[:3]})

    if bib_item:
        for scope, typed in bib_item.get("signals", {}).items():
            if scope != "edition_or_copy":
                continue
            for kind, entries in typed.items():
                valid = []
                for entry in entries:
                    if entry.get("current_copy") is True and entry.get("historical_edition") is not True:
                        sent = clean_sentence(entry.get("sentence", ""))
                        if sent and not any(re.search(pat, sent, flags=re.IGNORECASE) for pat in historical_phrases):
                            valid.append(sent)
                if valid:
                    signals.append({"type": kind, "evidence": valid[:3], "source": "bibliographic_analysis"})

    # Deduplicate same type/sentence.
    output=[]
    seen=set()
    for signal in signals:
        cleaned=[]
        for ev in signal.get("evidence",[]):
            key=(signal["type"], ev.lower())
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(ev)
        if cleaned:
            item={"type":signal["type"],"evidence":cleaned}
            if signal.get("source"):
                item["source"]=signal["source"]
            output.append(item)
    return output


def record_text(record):
    parts = [
        record.get("title", ""),
        record.get("academic_description", ""),
        record.get("bibliography", ""),
        record.get("publisher_data", ""),
        record.get("publication_place", ""),
        " ".join(record.get("topics", []) or []),
        " ".join(names(record, "related_names")),
    ]
    for author in record.get("authors", []) or []:
        if isinstance(author, dict):
            parts += [author.get("name", ""), author.get("biography", ""), author.get("biographical_data", ""), author.get("bibliography", "")]
    for publisher in record.get("publishers", []) or []:
        if isinstance(publisher, dict):
            parts += [publisher.get("name", ""), publisher.get("biography", ""), publisher.get("biographical_data", ""), publisher.get("bibliography", "")]
        else:
            parts.append(str(publisher))
    return " ".join(norm(p) for p in parts if norm(p))


def author_profile(record):
    out = []
    for a in record.get("authors", []) or []:
        if not isinstance(a, dict):
            continue
        name = norm(a.get("name", ""))
        if not name:
            continue
        out.append({
            "name": name,
            "biography": norm(a.get("biography", "")),
            "biographical_data": norm(a.get("biographical_data", "")),
            "bibliography": norm(a.get("bibliography", "")),
        })
    return out


def build():
    dataset = load(DATASET_PATH)
    bib = load(BIBLIO_PATH) if BIBLIO_PATH.exists() else {"items": []}
    bib_map = {x.get("book_id"): x for x in bib.get("items", [])}

    records = {}
    authors = {}
    publishers = {}
    related = {}
    topics = {}
    edges = defaultdict(lambda: defaultdict(set))

    for r in dataset:
        rid = str(r.get("id"))
        year = year_value(r)
        record_content = " ".join([
            norm(r.get("title", "")),
            norm(r.get("academic_description", "")),
            " ".join(norm(x) for x in (r.get("topics", []) or [])),
            " ".join(norm(x) for x in names(r, "related_names")),
        ])
        concepts = sorted(set(concept_hits(record_content)))
        signals = collector_signals(
            " ".join([
                norm(r.get("title", "")),
                norm(r.get("academic_description", "")),
                norm(r.get("highlight", "")),
            ]),
            bib_map.get(rid),
        )
        author_list = author_profile(r)
        pub_names = names(r, "publishers")
        rel_names = names(r, "related_names")
        topic_list = [norm(x) for x in (r.get("topics", []) or []) if norm(x)]

        records[rid] = {
            "id": rid,
            "title": norm(r.get("title", "")),
            "authors": [a["name"] for a in author_list],
            "publishers": pub_names,
            "related_names": rel_names,
            "topics": topic_list,
            "publication_year": year,
            "publication_place": norm(r.get("publication_place", "")),
            "period": period(year),
            "concepts": concepts,
            "collector_signals": signals,
        }

        for a in author_list:
            existing = authors.setdefault(a["name"], {"name": a["name"], "biographies": [], "records": set(), "topics": set(), "related_names": set(), "publishers": set(), "periods": set(), "concepts": set()})
            for k in ("biography", "biographical_data", "bibliography"):
                if a.get(k):
                    existing["biographies"].append(a[k])
            existing["records"].add(rid)
            existing["topics"].update(topic_list)
            existing["related_names"].update(rel_names)
            existing["publishers"].update(pub_names)
            existing["periods"].add(period(year))
            existing["concepts"].update(concepts)
            for b in author_list:
                if b["name"] != a["name"]:
                    edges["author_author"][(a["name"], b["name"])].add(rid)

        for p in pub_names:
            obj = publishers.setdefault(p, {"name": p, "records": set(), "authors": set(), "topics": set(), "periods": set()})
            obj["records"].add(rid); obj["authors"].update(x["name"] for x in author_list); obj["topics"].update(topic_list); obj["periods"].add(period(year))
            for a in author_list:
                edges["author_publisher"][(a["name"], p)].add(rid)

        for n in rel_names:
            obj = related.setdefault(n, {"name": n, "records": set(), "authors": set(), "topics": set(), "periods": set()})
            obj["records"].add(rid); obj["authors"].update(x["name"] for x in author_list); obj["topics"].update(topic_list); obj["periods"].add(period(year))
            for a in author_list:
                edges["author_related"][(a["name"], n)].add(rid)

        for t in topic_list:
            obj = topics.setdefault(t, {"name": t, "records": set(), "authors": set(), "periods": set()})
            obj["records"].add(rid); obj["authors"].update(x["name"] for x in author_list); obj["periods"].add(period(year))
            for a in author_list:
                edges["author_topic"][(a["name"], t)].add(rid)

    def freeze(obj):
        out = dict(obj)
        for key, val in list(out.items()):
            if isinstance(val, set):
                out[key] = sorted(val)
        return out

    frozen_edges = {}
    for edge_type, pair_map in edges.items():
        frozen_edges[edge_type] = [
            {"from": a, "to": b, "records": sorted(rs)}
            for (a, b), rs in sorted(pair_map.items())
        ]

    author_list = [freeze(v) for v in authors.values()]
    publisher_list = [freeze(v) for v in publishers.values()]
    related_list = [freeze(v) for v in related.values()]
    topic_list = [freeze(v) for v in topics.values()]

    # Stable hashes permit incremental scholar rebuilds later.
    for a in author_list:
        payload = json.dumps(a, ensure_ascii=False, sort_keys=True).encode()
        a["hash"] = hashlib.sha256(payload).hexdigest()

    output = {
        "source": DATASET_PATH.name,
        "records": records,
        "authors": {a["name"]: a for a in author_list},
        "publishers": {x["name"]: x for x in publisher_list},
        "related_names": {x["name"]: x for x in related_list},
        "topics": {x["name"]: x for x in topic_list},
        "edges": frozen_edges,
        "methodology": {
            "authority": "master JSON records",
            "bibliographic_signals": "rag/bibliographic_analysis.json when current_copy=true and historical_edition=false",
            "derived_only": True,
            "notes": [
                "Scholar relationships are derived from co-occurrence inside individual catalog records.",
                "Collector signals are evidence-backed and are not authoritative outside the source record.",
                "No inferred relationship is treated as a factual attribution unless supported by a source record.",
            ],
        },
    }

    OUTPUT_JSON.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print("="*80)
    print("GOVI SCHOLAR INDEX")
    print("="*80)
    print(f"Records: {len(records)}")
    print(f"Unique authors: {len(author_list)}")
    print(f"Unique publishers: {len(publisher_list)}")
    print(f"Unique related names: {len(related_list)}")
    print(f"Unique topics: {len(topic_list)}")
    print(f"Author-author links: {len(frozen_edges.get('author_author', []))}")
    print(f"Author-topic links: {len(frozen_edges.get('author_topic', []))}")
    print(f"Author-publisher links: {len(frozen_edges.get('author_publisher', []))}")
    print(f"Author-related-name links: {len(frozen_edges.get('author_related', []))}")
    print(f"Output: {OUTPUT_JSON}")

if __name__ == "__main__":
    build()
