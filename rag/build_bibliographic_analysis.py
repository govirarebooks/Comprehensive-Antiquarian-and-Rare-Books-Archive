#!/usr/bin/env python3

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = ROOT / "govi-rare-books-academic-dataset.json"
OUTPUT_PATH = ROOT / "rag" / "bibliographic_analysis.json"


PATTERNS = {
    "first_edition": [
        r"\bfirst edition\b",
        r"\bprima edizione\b",
        r"\bpremière édition\b",
        r"\berste ausgabe\b",
        r"\berste auflage\b",
    ],
    "first_issue": [
        r"\bfirst issue\b",
        r"\bfirst state\b",
        r"\bprima emissione\b",
        r"\bfirst state of the edition\b",
        r"\bfirst issue of the edition\b",
    ],
    "first_in_language": [
        r"\bfirst edition in (?:Italian|French|German|English|Dutch)\b",
        r"\bfirst (?:edition|book) in (?:Italian|French|German|English|Dutch)\b",
        r"\bprima edizione italiana\b",
        r"\bprima edizione in italiano\b",
        r"\bfirst edition in German\b",
        r"\bfirst edition in French\b",
    ],
    "very_rare": [
        r"\bextremely rare\b",
        r"\bvery rare\b",
        r"\bvery scarce\b",
        r"\bextremely scarce\b",
        r"\bonly one copy\b",
        r"\bonly one known copy\b",
        r"\bseul exemplaire\b",
        r"\bseule copie\b",
    ],
    "rare": [
        r"\brare edition\b",
        r"\brare copy\b",
        r"\brare first edition\b",
        r"\bscarce edition\b",
        r"\bscarce copy\b",
    ],
    "copies_recorded": [
        r"\bonly (\d+) cop(?:y|ies)\b",
        r"\b(\d+) cop(?:y|ies) (?:are|is) recorded\b",
        r"\b(\d+) cop(?:y|ies) (?:are|is) known\b",
        r"\bonly (\d+) known\b",
        r"\b(\d+) exemplaires?\b",
        r"\b(\d+) exemplaires? connus\b",
    ],
    "important_provenance": [
        r"\bprovenance\b",
        r"\bprovenienza\b",
        r"\bex-?libris\b",
        r"\bbookplate\b",
        r"\bformerly in the library of\b",
        r"\bfrom the library of\b",
        r"\bcollection of\b",
        r"\bfrom the collection of\b",
        r"\bprovenant de la bibliothèque\b",
        r"\bbelonged to\b",
        r"\bbelonged formerly to\b",
    ],
    "important_association": [
        r"\bassociation copy\b",
        r"\bpresentation copy\b",
        r"\bcopy presented to\b",
        r"\binscribed by\b",
        r"\bannotated by\b",
        r"\bowned by\b",
        r"\bbelonged to\b",
    ],
    "historical_statement": [
        r"\bfirst appearance\b",
        r"\bone of the earliest\b",
        r"\bearliest writers\b",
        r"\bfirst time\b",
        r"\bfirst known\b",
        r"\bfirst printed\b",
        r"\bearliest printed\b",
        r"\bseminal\b",
        r"\bpioneering\b",
        r"\bgroundbreaking\b",
    ],
}


SECONDARY_EDITION_PATTERNS = [
    r"\balso printed\b",
    r"\balso published\b",
    r"\banother edition\b",
    r"\banother issue\b",
    r"\banother printing\b",
    r"\bsecond edition\b",
    r"\bsecond issue\b",
    r"\bthird edition\b",
    r"\bfourth edition\b",
    r"\bsubsequent edition\b",
    r"\blater edition\b",
    r"\blater printing\b",
    r"\bpublished again\b",
    r"\bprinted again\b",
    r"\bedition .* printed before\b",
    r"\bedition .* published before\b",
    r"\bVenice edition\b",
    r"\bLatin edition\b",
    r"\bFrench edition\b",
    r"\bItalian edition\b",
    r"\bGerman edition\b",
    r"\bSpanish edition\b",
    r"\bEnglish edition\b",
    r"\bof the text\b",
    r"\btext was printed\b",
    r"\bwas printed in \d{4}\b",
    r"\bwas published in \d{4}\b",
    r"\bprinted in \d{4}\b",
    r"\bpublished in \d{4}\b",
    r"\bissued in \d{4}\b",
]


# Espressioni che identificano quasi certamente
# una discussione storica su un'altra edizione.
HISTORICAL_EDITION_PATTERNS = [
    r"\balso printed in \d{4}\b",
    r"\balso published in \d{4}\b",
    r"\bprinted in \d{4}\b",
    r"\bpublished in \d{4}\b",
    r"\bissued in \d{4}\b",
    r"\bthe first edition in (?:Italian|French|German|English|Dutch) of the text\b",
    r"\bthe .* edition in (?:Italian|French|German|English|Dutch)\b",
    r"\bVenice edition\b",
    r"\bLatin edition\b",
    r"\bFrench edition\b",
    r"\bItalian edition\b",
    r"\bGerman edition\b",
    r"\bSpanish edition\b",
    r"\bEnglish edition\b",
    r"\bthe edition was\b",
    r"\bthe edition could\b",
    r"\bthe edition may\b",
    r"\bthe edition might\b",
    r"\bprinted before\b",
    r"\bpublished before\b",
    r"\bpublished again\b",
]


EDITION_CONTEXT_PATTERNS = [
    r"\b8vo\b",
    r"\b4to\b",
    r"\bfolio\b",
    r"\bpp\.\b",
    r"\bleaves\b",
    r"\bcollation\b",
    r"\btitle page\b",
    r"\btitle-page\b",
    r"\bwoodcut\b",
    r"\bengraved\b",
    r"\bbound\b",
    r"\bbinding\b",
    r"\bboards\b",
    r"\bparchment\b",
    r"\bvelum\b",
    r"\bvellum\b",
    r"\bhalf calf\b",
    r"\bcontemporary\b",
    r"\bcopy\b",
    r"\bissue\b",
    r"\bedition\b",
    r"\bprinted by\b",
    r"\brare\b",
    r"\bscarce\b",
    r"\bfirst edition\b",
]


COMPOSITE_RECORD_PATTERNS = [
    r"\bsammelband\b",
    r"\bsammleband\b",
    r"\bvolume contains\b",
    r"\bvolumes? contains\b",
    r"\bcontaining \d+ works?\b",
    r"\bcontaining several works\b",
    r"\bcontaining censored\b",
    r"\bbound together\b",
    r"\bbound with\b",
    r"\bfollowed by:\b",
    r"\bfollowed by\]\b",
    r"\bfour works in one volume\b",
    r"\bthree works in one volume\b",
    r"\btwo works in one volume\b",
]


CURRENT_COPY_PATTERNS = [
    r"^\s*(?:[ivx]+\.\s*)?first edition(?:[\s\.,:;\-]|$)",
    r"\bfirst edition of\b",
    r"\bfirst edition of this\b",
    r"\bfirst edition of the present\b",
    r"\bfirst edition of the work\b",
    r"\bthis is the first edition\b",
    r"\brare first edition\b",
    r"\bfirst and only edition\b",
    r"^\s*(?:[ivx]+\.\s*)?first issue(?:[\s\.,:;\-]|$)",
    r"\bfirst issue of\b",
    r"\bfirst state(?:[\s\.,:;\-]|$)",
    r"\bfirst edition in (?:German|Italian|French|English|Dutch)\b",
    r"\bfirst book in (?:German|Italian|French|English|Dutch)\b",
    r"\bprima edizione italiana\b",
    r"\bprima edizione in italiano\b",
    r"\bextremely rare first\b",
    r"\bvery rare first\b",
    r"\bextremely rare .* edition\b",
    r"\bvery rare .* edition\b",
    r"\bvery scarce .* edition\b",
    r"\brare first edition\b",
    r"\bscarce first edition\b",
    r"\bfirst and only edition\b",
]


def normalize(text: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        text or "",
    ).strip()


def load_dataset():
    with DATASET_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)



def clean_current_copy_sentence(text: str) -> str:
    """Keep only the current-copy statement and cut later-edition history."""
    text = normalize(text)
    cut_patterns = [
        r"\bA second issue\b",
        r"\bThe second issue\b",
        r"\bA second edition\b",
        r"\bThe second edition\b",
        r"\bA later edition\b",
        r"\bAnother edition\b",
        r"\bAlso issued\b",
        r"\bAlso printed\b",
        r"\bAlso published\b",
        r"\bIn \d{4} .*? issued\b",
        r"\bIn \d{4} .*? published\b",
        r"\bIn \d{4} .*? printed\b",
        r"\bA Latin edition\b",
        r"\bThe Latin edition\b",
        r"\bA French edition\b",
        r"\bThe French edition\b",
        r"\bAn Italian edition\b",
        r"\bThe Italian edition\b",
    ]
    positions=[]
    for pattern in cut_patterns:
        m=re.search(pattern,text,flags=re.IGNORECASE)
        if m:
            positions.append(m.start())
    if positions:
        text=normalize(text[:min(positions)])
    return text

def sentence_for_match(
    text: str,
    match,
):
    start = text.rfind(
        ".",
        0,
        match.start(),
    )

    if start == -1:
        start = 0
    else:
        start += 1

    end = text.find(
        ".",
        match.end(),
    )

    if end == -1:
        end = len(text)

    return normalize(
        text[start:end]
    )


def context_for_match(
    text: str,
    match,
    window: int = 420,
):
    start = max(
        0,
        match.start() - window,
    )

    end = min(
        len(text),
        match.end() + window,
    )

    return normalize(
        text[start:end]
    )


def immediate_context(
    text: str,
    match,
    radius: int = 180,
):
    start = max(
        0,
        match.start() - radius,
    )

    end = min(
        len(text),
        match.end() + radius,
    )

    return normalize(
        text[start:end]
    )


def context_strength(
    evidence: str,
) -> int:

    lower = evidence.lower()

    return sum(
        bool(
            re.search(
                pattern,
                lower,
            )
        )
        for pattern in EDITION_CONTEXT_PATTERNS
    )


def looks_like_secondary_edition(
    evidence: str,
) -> bool:

    lower = evidence.lower()

    return any(
        re.search(
            pattern,
            lower,
        )
        for pattern in SECONDARY_EDITION_PATTERNS
    )


def looks_like_historical_edition(
    sentence: str,
) -> bool:

    lower = sentence.lower()

    return any(
        re.search(
            pattern,
            lower,
        )
        for pattern in HISTORICAL_EDITION_PATTERNS
    )


def looks_like_composite_record(
    text: str,
) -> bool:

    lower = text.lower()

    return any(
        re.search(
            pattern,
            lower,
        )
        for pattern in COMPOSITE_RECORD_PATTERNS
    )


def matches_current_copy_pattern(
    sentence: str,
    nearby: str,
) -> bool:

    combined = normalize(
        sentence
        + " "
        + nearby
    ).lower()

    return any(
        re.search(
            pattern,
            combined,
        )
        for pattern in CURRENT_COPY_PATTERNS
    )


def looks_like_current_copy(
    full_text: str,
    match,
    signal_type: str,
) -> bool:

    sentence = sentence_for_match(
        full_text,
        match,
    )

    nearby = immediate_context(
        full_text,
        match,
    )

    # Provenienza/associazione:
    if signal_type in {
        "important_provenance",
        "important_association",
    }:

        patterns = [
            r"\bthis copy\b",
            r"\bthe present copy\b",
            r"\bbookplate\b",
            r"\bex-?libris\b",
            r"\bprovenance\b",
            r"\bowned by\b",
            r"\bbelonged to\b",
            r"\bfrom the collection of\b",
        ]

        combined = (
            sentence
            + " "
            + nearby
        ).lower()

        return any(
            re.search(
                pattern,
                combined,
            )
            for pattern in patterns
        )

    # Se stiamo parlando esplicitamente di un'altra edizione,
    # non consideriamola la copia corrente.
    if looks_like_historical_edition(
        sentence
    ):
        return False

    # Formula catalografica esplicita.
    if matches_current_copy_pattern(
        sentence,
        nearby,
    ):
        return True

    # Segnali bibliografici con contesto fisico.
    if signal_type in {
        "first_edition",
        "first_issue",
        "first_in_language",
        "very_rare",
        "rare",
        "copies_recorded",
    }:

        nearby_lower = nearby.lower()

        physical_patterns = [
            r"\bcopy\b",
            r"\bcontemporary\b",
            r"\bbinding\b",
            r"\bcollation\b",
            r"\btitle page\b",
            r"\bboards\b",
            r"\bparchment\b",
            r"\bvellum\b",
            r"\bhalf calf\b",
            r"\bfolio\b",
            r"\b4to\b",
            r"\b8vo\b",
            r"\bleaves\b",
            r"\bpages\b",
        ]

        if any(
            re.search(
                pattern,
                nearby_lower,
            )
            for pattern in physical_patterns
        ):
            return True

    return False


def extract_signals(
    text: str,
):

    text = normalize(text)

    composite_record = (
        looks_like_composite_record(text)
    )

    signals = []

    for signal_type, patterns in PATTERNS.items():

        for pattern in patterns:

            for match in re.finditer(
                pattern,
                text,
                flags=re.IGNORECASE,
            ):

                sentence = sentence_for_match(
                    text,
                    match,
                )

                sentence = clean_current_copy_sentence(
                    sentence
                )

                evidence = context_for_match(
                    text,
                    match,
                )

                signals.append(
                    {
                        "type": signal_type,
                        "match": match.group(0),
                        "sentence": sentence,
                        "evidence": evidence,
                        "context_strength": context_strength(
                            evidence
                        ),
                        "secondary_edition": looks_like_secondary_edition(
                            evidence
                        ),
                        "historical_edition": looks_like_historical_edition(
                            sentence
                        ),
                        "current_copy": looks_like_current_copy(
                            text,
                            match,
                            signal_type,
                        ),
                        "composite_record": composite_record,
                    }
                )

    return signals


def deduplicate(signals):

    output = []
    seen = set()

    signals = sorted(
        signals,
        key=lambda item: (
            item["current_copy"],
            item["context_strength"],
        ),
        reverse=True,
    )

    for item in signals:

        key = (
            item["type"],
            item["match"].lower(),
            item["sentence"].lower(),
        )

        if key in seen:
            continue

        seen.add(key)
        output.append(item)

    return output


def classify_signal(signal):

    signal_type = signal["type"]

    if signal_type == "historical_statement":
        return "historical_context"

    # La presenza di un'indicazione esplicita che si tratta
    # di un'altra edizione prevale.
    if signal["historical_edition"]:
        return "contextual"

    if signal["current_copy"]:

        if signal_type in {
            "first_edition",
            "first_issue",
            "first_in_language",
            "very_rare",
            "rare",
            "copies_recorded",
            "important_provenance",
            "important_association",
        }:
            return "edition_or_copy"

    # Nei record compositi, niente fallback automatico.
    if signal["composite_record"]:

        if signal_type in {
            "first_edition",
            "first_issue",
            "first_in_language",
            "very_rare",
            "rare",
            "copies_recorded",
        }:
            return "contextual"

    if signal["secondary_edition"]:
        return "contextual"

    strength = signal["context_strength"]

    if signal_type in {
        "very_rare",
        "rare",
        "copies_recorded",
    } and strength >= 3:

        return "edition_or_copy"

    if signal_type in {
        "first_edition",
        "first_in_language",
    } and strength >= 5:

        return "edition_or_copy"

    return "contextual"


def summarize(signals):

    result = {
        "edition_or_copy": {},
        "historical_context": {},
        "contextual": {},
    }

    for signal in signals:

        scope = classify_signal(
            signal
        )

        clean = {
            "match": signal["match"],
            "sentence": signal["sentence"],
            "evidence": signal["evidence"],
            "context_strength": signal["context_strength"],
            "current_copy": signal["current_copy"],
            "secondary_edition": signal["secondary_edition"],
            "historical_edition": signal["historical_edition"],
            "composite_record": signal["composite_record"],
        }

        result.setdefault(
            scope,
            {},
        ).setdefault(
            signal["type"],
            [],
        ).append(clean)

    return {
        scope: values
        for scope, values in result.items()
        if values
    }


def main():

    dataset = load_dataset()

    output = {
        "source": DATASET_PATH.name,
        "records": len(dataset),
        "items": [],
    }

    total_signals = 0
    edition_signals = 0
    historical_signals = 0
    contextual_signals = 0

    for record in dataset:

        description = normalize(
            record.get(
                "academic_description",
                "",
            )
        )

        highlight = normalize(
            record.get(
                "highlight",
                "",
            )
        )

        combined = (
            description
            + " "
            + highlight
        ).strip()

        signals = deduplicate(
            extract_signals(
                combined
            )
        )

        total_signals += len(
            signals
        )

        for signal in signals:

            scope = classify_signal(
                signal
            )

            if scope == "edition_or_copy":
                edition_signals += 1

            elif scope == "historical_context":
                historical_signals += 1

            elif scope == "contextual":
                contextual_signals += 1

        output["items"].append(
            {
                "book_id": record.get("id"),
                "title": record.get("title"),
                "signals": summarize(
                    signals
                ),
            }
        )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    non_empty = [
        item
        for item in output["items"]
        if item["signals"]
    ]

    print("=" * 80)
    print("GOVI BIBLIOGRAPHIC ANALYSIS")
    print("=" * 80)
    print()

    print(
        "Records:",
        len(dataset),
    )

    print(
        "Signals extracted:",
        total_signals,
    )

    print(
        "Edition/copy signals:",
        edition_signals,
    )

    print(
        "Historical-context signals:",
        historical_signals,
    )

    print(
        "Contextual signals:",
        contextual_signals,
    )

    print()

    print(
        "Output:",
        OUTPUT_PATH,
    )

    print(
        "Records with signals:",
        len(non_empty),
    )


if __name__ == "__main__":
    main()
