#!/usr/bin/env python3

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from sentence_transformers import CrossEncoder
from scholar_engine import ScholarEngine


ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = ROOT / "govi-rare-books-academic-dataset.json"
BIBLIOGRAPHIC_PATH = ROOT / "rag" / "bibliographic_analysis.json"

CROSS_ENCODER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

FINAL_RESULTS = 5
EVIDENCE_RESULTS = 4

MIN_EVIDENCE_WORDS = 25
MAX_EVIDENCE_WORDS = 150


CONCEPT_FAMILIES = {
    "divination": {
        "divination",
        "divinatory",
        "prophecy",
        "prophetic",
        "oracle",
        "oracles",
        "prognostication",
        "prognostications",
        "practica",
        "prediction",
        "predictions",
        "fortune",
        "fortunes",
    },
    "celestial": {
        "celestial",
        "heavenly",
        "heavens",
        "comet",
        "comets",
        "meteor",
        "meteors",
        "planet",
        "planets",
        "eclipse",
        "eclipses",
        "conjunction",
        "conjunctions",
        "astronomy",
        "astrological",
        "astrology",
        "zodiac",
        "star",
        "stars",
        "sky",
    },
    "omens": {
        "omen",
        "omens",
        "portent",
        "portents",
        "sign",
        "signs",
        "foreshadow",
        "foreshadowing",
        "foretell",
        "foretelling",
        "foreseen",
        "future",
        "calamity",
        "calamities",
        "prognostication",
        "prognostications",
        "prediction",
        "predictions",
    },
}


HARD_BOUNDARY_PATTERNS = [
    r"\bVD16,\s*[A-Z0-9\-]+",
    r"\bVD17,\s*[A-Z0-9\-]+",
    r"\bOCLC,\s*[0-9]+",
    r"\bISTC,\s*[A-Z0-9\-]+",
]


HISTORICAL_EDITION_PATTERNS = [
    r"\bsecond issue\b",
    r"\bsecond edition\b",
    r"\blater edition\b",
    r"\banother edition\b",
    r"\bother edition\b",
    r"\bsubsequent edition\b",
    r"\balso issued\b",
    r"\balso printed\b",
    r"\balso published\b",
    r"\bissued in \d{4}\b",
    r"\bissued by\b",
    r"\bprinted in \d{4}\b",
    r"\bpublished in \d{4}\b",
    r"\blatin edition\b",
    r"\bfrench edition\b",
    r"\bitalian edition\b",
    r"\bgerman edition\b",
    r"\bspanish edition\b",
    r"\bfirst issue\b",
]


DATABASE_NOISE_PATTERNS = [
    r"\bVD16\b",
    r"\bVD17\b",
    r"\bOCLC\b",
    r"\bISTC\b",
    r"\bCf\.\s",
    r"\bsee also\b",
]


COMPOSITE_PATTERNS = [
    r"\bsammle?band\b",
    r"\bsammelband\b",
    r"\bmiscellany\b",
    r"\bbound together\b",
    r"\bbound-with\b",
    r"\bbound with\b",
    r"\bcontains six works\b",
    r"\bcontains five works\b",
    r"\bcontains four works\b",
    r"\bcontains three works\b",
]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def word_count(text: str) -> int:
    return len(
        re.findall(
            r"\b[\wÀ-ÿ'’\-]+\b",
            text or "",
            flags=re.UNICODE,
        )
    )


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_dataset():
    return load_json(DATASET_PATH)


def load_bibliographic_analysis():
    if not BIBLIOGRAPHIC_PATH.exists():
        return {"items": []}
    return load_json(BIBLIOGRAPHIC_PATH)


def bibliographic_map(data):
    return {
        item.get("book_id"): item
        for item in data.get("items", [])
        if item.get("book_id")
    }


def get_discovery_output(query: str):
    command = [
        sys.executable,
        str(ROOT / "rag" / "discovery_engine.py"),
        query,
    ]

    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    return result.stdout


def parse_discovery_ids(output: str):
    ids = []

    for line in output.splitlines():
        match = re.match(
            r"^\s*ID:\s*([A-Z0-9]+)\s*$",
            line.strip(),
        )

        if match:
            book_id = match.group(1)

            if book_id not in ids:
                ids.append(book_id)

    return ids


def extract_discovery_relationships(discovery_output: str):
    output = {}
    current_id = None

    for line in discovery_output.splitlines():
        stripped = line.strip()

        id_match = re.match(
            r"^ID:\s*([A-Z0-9]+)",
            stripped,
        )

        if id_match:
            current_id = id_match.group(1)
            continue

        for label, key in [
            ("Final", "final_score"),
            ("Core topic", "core_topic_score"),
            ("Related", "related_score"),
        ]:
            match = re.match(
                rf"{re.escape(label)}:\s*([0-9.\-]+)",
                stripped,
            )

            if current_id and match:
                output.setdefault(current_id, {})[key] = float(
                    match.group(1)
                )

    return output


def split_sentences(text: str):
    return [
        normalize(item)
        for item in re.split(
            r'(?<=[.!?])\s+(?=[A-ZÀ-ÖØ-Þ0-9"“‘])',
            normalize(text),
        )
        if normalize(item)
    ]


def split_catalog_records(text: str):
    text = text or ""
    positions = []

    for pattern in HARD_BOUNDARY_PATTERNS:
        for match in re.finditer(
            pattern,
            text,
            flags=re.IGNORECASE,
        ):
            positions.append(match.start())

    if not positions:
        return [normalize(text)] if normalize(text) else []

    positions = sorted(set(positions))

    records = []
    start = 0

    for position in positions:
        chunk = normalize(text[start:position])

        if chunk:
            records.append(chunk)

        start = position

    tail = normalize(text[start:])

    if tail:
        records.append(tail)

    return records


def build_evidence_units(text: str):
    catalog_records = split_catalog_records(text)
    units = []

    for record in catalog_records:
        sentences = split_sentences(record)

        if not sentences:
            continue

        current = []

        for sentence in sentences:
            candidate = normalize(" ".join(current + [sentence]))

            if word_count(candidate) <= MAX_EVIDENCE_WORDS:
                current.append(sentence)
            else:
                if current:
                    units.append(normalize(" ".join(current)))
                current = [sentence]

        if current:
            units.append(normalize(" ".join(current)))

    return [
        unit
        for unit in units
        if word_count(unit) >= MIN_EVIDENCE_WORDS
    ]


def concept_families_in_text(text: str):
    lower = text.lower()
    concepts = set()

    for family, terms in CONCEPT_FAMILIES.items():
        if any(
            re.search(
                rf"\b{re.escape(term)}\b",
                lower,
            )
            for term in terms
        ):
            concepts.add(family)

    return concepts


def query_concepts(query: str):
    return concept_families_in_text(query)


def has_pattern(text: str, patterns) -> bool:
    lower = text.lower()

    return any(
        re.search(pattern, lower)
        for pattern in patterns
    )


def evidence_has_query_overlap(query: str, text: str) -> bool:
    return bool(
        query_concepts(query).intersection(
            concept_families_in_text(text)
        )
    )


def is_historical_only(query: str, text: str) -> bool:
    if not has_pattern(
        text,
        HISTORICAL_EDITION_PATTERNS,
    ):
        return False

    return not evidence_has_query_overlap(
        query,
        text,
    )


def evidence_quality(query: str, text: str) -> float:
    qconcepts = query_concepts(query)
    econcepts = concept_families_in_text(text)

    overlap = qconcepts.intersection(econcepts)

    score = 1.75 * len(overlap)

    if not overlap:
        score -= 10.0

    if is_historical_only(query, text):
        score -= 15.0
    elif has_pattern(
        text,
        HISTORICAL_EDITION_PATTERNS,
    ):
        score -= 2.5

    if has_pattern(
        text,
        DATABASE_NOISE_PATTERNS,
    ):
        score -= 1.5

    wc = word_count(text)

    if wc > 125:
        score -= (wc - 125) / 70.0

    return score


def find_evidence(
    model: CrossEncoder,
    query: str,
    description: str,
):
    units = build_evidence_units(description)

    if not units:
        return []

    overlapping_units = [
        unit
        for unit in units
        if evidence_has_query_overlap(
            query,
            unit,
        )
    ]

    if overlapping_units:
        units = overlapping_units

    pairs = [
        [query, unit]
        for unit in units
    ]

    scores = model.predict(
        pairs,
        show_progress_bar=False,
    )

    ranked = []

    for unit, cross_score in zip(
        units,
        scores,
    ):
        cross_score = float(cross_score)

        quality = evidence_quality(
            query,
            unit,
        )

        ranking_score = (
            cross_score
            + quality * 0.35
        )

        ranked.append(
            {
                "text": unit,
                "cross_encoder_score": cross_score,
                "quality_score": quality,
                "ranking_score": ranking_score,
                "words": word_count(unit),
            }
        )

    ranked.sort(
        key=lambda item: item["ranking_score"],
        reverse=True,
    )

    results = []
    seen = set()

    for item in ranked:
        key = normalize(item["text"].lower())

        if key in seen:
            continue

        seen.add(key)
        results.append(item)

        if len(results) >= EVIDENCE_RESULTS:
            break

    return results


def topic_overlap_score(query: str, topics):
    qconcepts = query_concepts(query)
    score = 0

    for topic in topics:
        topic_concepts = concept_families_in_text(topic)

        if topic_concepts.intersection(qconcepts):
            score += 1

    return score


def direct_semantic_match(query: str, evidence) -> bool:
    qconcepts = query_concepts(query)

    if not evidence:
        return False

    if qconcepts == {"divination"}:
        return any(
            "divination"
            in concept_families_in_text(item["text"])
            for item in evidence
        )

    needs_celestial = "celestial" in qconcepts

    needs_divination_or_omens = (
        "divination" in qconcepts
        or "omens" in qconcepts
    )

    for item in evidence:
        econcepts = concept_families_in_text(item["text"])

        celestial_ok = (
            not needs_celestial
            or "celestial" in econcepts
        )

        divination_ok = (
            not needs_divination_or_omens
            or (
                "divination" in econcepts
                or "omens" in econcepts
            )
        )

        if celestial_ok and divination_ok:
            return True

    return False


def classify_match(
    query: str,
    record: dict[str, Any],
    evidence,
    discovery_relation,
):
    if direct_semantic_match(query, evidence):
        return "direct_match"

    qconcepts = query_concepts(query)

    evidence_concepts = set()

    for item in evidence:
        evidence_concepts.update(
            concept_families_in_text(item["text"])
        )

    topic_overlap = topic_overlap_score(
        query,
        record.get("topics", []),
    )

    core_score = discovery_relation.get(
        "core_topic_score",
        0.0,
    )

    if evidence_concepts.intersection(qconcepts):
        return "conceptual_match"

    if topic_overlap >= 1 or core_score >= 0.40:
        return "conceptual_match"

    return "contextual_match"


def is_composite_record(record: dict[str, Any]) -> bool:
    blob = " ".join(
        [
            str(record.get("title", "")),
            str(record.get("academic_description", "")),
            " ".join(
                map(
                    str,
                    record.get("topics", []),
                )
            ),
        ]
    )

    return has_pattern(
        blob,
        COMPOSITE_PATTERNS,
    )


def clean_current_copy_sentence(sentence: str) -> str:
    """
    Mantiene la parte che descrive l'esemplare corrente
    e tronca la frase quando inizia la discussione
    di un'altra issue/edizione.

    Esempio:

        Incredibly rare first edition ... only one copy
        recorded ... A second issue ... was published
        in 1515 ...

    diventa:

        Incredibly rare first edition ... only one copy
        recorded ...

    Questo evita di attribuire caratteristiche di altre
    edizioni all'esemplare corrente.
    """

    text = normalize(sentence)

    cut_patterns = [
        r'\bA second issue\b',
        r'\bThe second issue\b',
        r'\bA second edition\b',
        r'\bThe second edition\b',
        r'\bA later edition\b',
        r'\bAnother edition\b',
        r'\bThe later edition\b',
        r'\bAlso issued\b',
        r'\bAlso printed\b',
        r'\bAlso published\b',
        r'\bIn \d{4} .*? issued\b',
        r'\bIn \d{4} .*? published\b',
        r'\bIn \d{4} .*? printed\b',
        r'\bA Latin edition\b',
        r'\bThe Latin edition\b',
        r'\bA French edition\b',
        r'\bThe French edition\b',
        r'\bAn Italian edition\b',
        r'\bThe Italian edition\b',
    ]

    positions = []

    for pattern in cut_patterns:
        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if match:
            positions.append(match.start())

    if positions:
        text = normalize(
            text[:min(positions)]
        )

    return text


def historical_sentence_safe(sentence: str) -> bool:
    lower = sentence.lower()

    forbidden = [
        "also issued",
        "also printed",
        "also published",
        "latin edition",
        "french edition",
        "italian edition",
        "german edition",
        "spanish edition",
        "second issue",
        "second edition",
        "later edition",
        "another edition",
        "other edition",
        "subsequent edition",
    ]

    return not any(
        phrase in lower
        for phrase in forbidden
    )


def extract_bibliographic_highlights(
    record,
    bib_item,
):
    if not bib_item:
        return []

    signals = bib_item.get(
        "signals",
        {},
    )

    composite = is_composite_record(record)

    suppressed_for_composite = {
        "first_edition",
        "first_issue",
        "first_in_language",
        "very_rare",
        "rare",
        "copies_recorded",
    }

    highlights = []

    for signal_type, entries in signals.get(
        "edition_or_copy",
        {},
    ).items():

        if (
            composite
            and signal_type in suppressed_for_composite
        ):
            continue

        for entry in entries:

            if entry.get(
                "current_copy",
                False,
            ) is not True:
                continue

            if entry.get(
                "historical_edition",
                False,
            ) is True:
                continue

            sentence = normalize(
                entry.get(
                    "sentence",
                    "",
                )
            )

            if not sentence:
                continue

            # Prima puliamo la frase per mantenere soltanto
            # la parte relativa all'esemplare corrente.
            sentence = clean_current_copy_sentence(
                sentence
            )

            if not sentence:
                continue

            if not historical_sentence_safe(sentence):
                continue

            highlights.append(
                {
                    "type": signal_type,
                    "match": entry.get("match"),
                    "sentence": sentence,
                    "evidence": entry.get(
                        "evidence",
                        "",
                    ),
                }
            )

    output = []
    seen = set()

    for item in highlights:
        key = (
            item["type"],
            item["sentence"].lower(),
        )

        if key in seen:
            continue

        seen.add(key)
        output.append(item)

    return output


def summarize_bibliographic_signals(highlights):
    labels = {
        "first_edition": "prima edizione",
        "first_issue": "first issue",
        "first_in_language": "prima edizione in lingua",
        "very_rare": "esemplare/edizione molto rara",
        "rare": "edizione rara",
        "copies_recorded": "numero limitato di copie registrate",
        "important_provenance": "provenienza significativa",
        "important_association": "associazione significativa",
    }

    output = []
    seen = set()

    for item in highlights:
        label = labels.get(item["type"])

        if not label or label in seen:
            continue

        seen.add(label)
        output.append(label)

    return output


def recommendation_reason(
    title: str,
    match_type: str,
):
    if match_type == "direct_match":
        return (
            f"«{title}» è una corrispondenza diretta "
            f"alla ricerca: la descrizione tratta "
            f"esplicitamente il nucleo tematico richiesto."
        )

    if match_type == "conceptual_match":
        return (
            f"«{title}» presenta un collegamento "
            f"concettuale forte con la ricerca, pur "
            f"non coincidendo letteralmente con tutto "
            f"il nucleo della richiesta."
        )

    return (
        f"«{title}» presenta un collegamento "
        f"contestuale con la ricerca."
    )



def build_scholar_context(scholar_engine, record_id):
    """Return concise scholar-layer context for a recommendation."""
    try:
        profile = scholar_engine.collector_profile(record_id)
        _, similar = scholar_engine.book_similarity(record_id, limit=4)
    except Exception:
        return {
            "period": None,
            "concepts": [],
            "collector_signals": [],
            "hidden_connections": [],
        }

    connections = []

    for item in similar:
        reasons = []
        if item.get("shared_concepts"):
            reasons.append("concetti condivisi: " + ", ".join(item["shared_concepts"][:4]))
        if item.get("conceptual_links") and not item.get("shared_concepts"):
            links = item["conceptual_links"][:3]
            rendered = [f"{x['from']} → {x['to']}" for x in links]
            reasons.append("ponte concettuale: " + ", ".join(rendered))
        if item.get("shared_authors"):
            reasons.append("autore condiviso: " + ", ".join(item["shared_authors"][:2]))
        if item.get("shared_related_names"):
            reasons.append("nomi correlati: " + ", ".join(item["shared_related_names"][:2]))
        if item.get("shared_topics"):
            reasons.append("temi condivisi: " + ", ".join(item["shared_topics"][:3]))
        if item.get("period") and item.get("period") == profile.get("period"):
            reasons.append("stesso periodo storico")

        if not reasons:
            continue

        connections.append({
            "book_id": item["id"],
            "title": item["title"],
            "score": item["score"],
            "why_connected": reasons,
        })

        if len(connections) >= 2:
            break

    return {
        "period": profile.get("period"),
        "concepts": [c for c in profile.get("concepts", []) if c not in {"rarity", "printing", "religion", "humanism", "medicine", "history_of_science", "illustration", "censorship"}],
        "collector_signals": profile.get("collector_signals", []),
        "hidden_connections": connections,
    }


def build_result(
    query,
    record,
    evidence,
    bib_item,
    discovery_relation,
    scholar_context,
):
    bibliographic_highlights = (
        extract_bibliographic_highlights(
            record,
            bib_item,
        )
    )

    match_type = classify_match(
        query,
        record,
        evidence,
        discovery_relation,
    )

    return {
        "id": record.get("id"),
        "title": record.get("title"),
        "topics": record.get("topics", []),
        "match_type": match_type,
        "why_recommended": recommendation_reason(
            record.get("title", "Questo libro"),
            match_type,
        ),
        "evidence": evidence,
        "bibliographic_signals": (
            summarize_bibliographic_signals(
                bibliographic_highlights
            )
        ),
        "bibliographic_highlights": (
            bibliographic_highlights
        ),
        "highlight": record.get("highlight", ""),
        "scholar_context": scholar_context,
    }


def print_result(
    result,
    position,
):
    print()
    print("=" * 80)
    print(
        f"RACCOMANDAZIONE #{position}"
    )
    print("=" * 80)
    print()

    print("TITOLO:")
    print(result["title"])

    print()

    print("TIPO DI CORRISPONDENZA:")
    print(result["match_type"])

    print()

    print("PERCHÉ TE LO SUGGERIAMO:")
    print(result["why_recommended"])

    scholar = result.get("scholar_context", {})
    concepts = scholar.get("concepts", [])
    period = scholar.get("period")

    if period or concepts:
        print()
        print("LETTURA SCHOLAR:")
        if period:
            print(f"- periodo: {period}")
        if concepts:
            print("- nucleo concettuale: " + ", ".join(concepts[:8]))

    connections = scholar.get("hidden_connections", [])
    if connections:
        print()
        print("CONNESSIONI SCHOLAR:")
        for connection in connections:
            print(f"- {connection['title']}")
            for why in connection["why_connected"]:
                print(f"  {why}")

    print()

    print(
        "PASSAGGI DELLA DESCRIZIONE "
        "CHE HANNO GUIDATO LA SCELTA:"
    )

    for index, item in enumerate(
        result.get("evidence", []),
        start=1,
    ):
        print()

        print(
            f"[{index}] "
            f"Cross-Encoder: "
            f"{item['cross_encoder_score']:.4f} | "
            f"Quality: "
            f"{item['quality_score']:.2f} | "
            f"Ranking: "
            f"{item['ranking_score']:.2f}"
        )

        print(item["text"])

    if result.get("bibliographic_signals"):
        print()
        print(
            "PERCHÉ QUESTO ESEMPLARE È SPECIALE:"
        )

        for signal in result["bibliographic_signals"]:
            print(
                f"- {signal}"
            )

        print()
        print("EVIDENZA BIBLIOGRAFICA:")

        for item in result.get(
            "bibliographic_highlights",
            [],
        )[:5]:
            print()
            print(
                f"[{item['type']}]"
            )
            print(
                item["sentence"]
            )


def run(query: str):
    dataset = load_dataset()

    records = {
        record.get("id"): record
        for record in dataset
        if record.get("id")
    }

    bibliography = bibliographic_map(
        load_bibliographic_analysis()
    )

    discovery_output = get_discovery_output(query)

    candidate_ids = parse_discovery_ids(
        discovery_output
    )[:FINAL_RESULTS]

    discovery_relations = (
        extract_discovery_relationships(
            discovery_output
        )
    )

    print("=" * 80)
    print(
        "GOVI RARE BOOKS — AI RECOMMENDATION ENGINE"
    )
    print("=" * 80)
    print()

    print("QUERY:")
    print(query)

    print()

    print("CANDIDATI DISCOVERY:")
    print(len(candidate_ids))

    model = CrossEncoder(
        CROSS_ENCODER_MODEL
    )

    scholar_engine = ScholarEngine()

    results = []

    for book_id in candidate_ids:
        record = records.get(book_id)

        if not record:
            continue

        description = normalize(
            record.get(
                "academic_description",
                "",
            )
        )

        evidence = find_evidence(
            model,
            query,
            description,
        )

        results.append(
            build_result(
                query,
                record,
                evidence,
                bibliography.get(book_id),
                discovery_relations.get(
                    book_id,
                    {},
                ),
                build_scholar_context(
                    scholar_engine,
                    book_id,
                ),
            )
        )

    results.sort(
        key=lambda result: (
            discovery_relations.get(
                result["id"],
                {},
            ).get(
                "final_score",
                0.0,
            ),
        ),
        reverse=True,
    )

    results = results[:FINAL_RESULTS]

    for position, result in enumerate(
        results,
        start=1,
    ):
        print_result(
            result,
            position,
        )

    print()
    print("=" * 80)
    print("MACHINE-READABLE OUTPUT")
    print("=" * 80)

    print(
        json.dumps(
            {
                "query": query,
                "candidate_ids": candidate_ids,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main():
    if len(sys.argv) < 2:
        print(
            'Uso: python rag/recommendation_engine.py '
            '"richiesta del cliente"'
        )
        raise SystemExit(1)

    run(
        " ".join(sys.argv[1:])
    )


if __name__ == "__main__":
    main()
