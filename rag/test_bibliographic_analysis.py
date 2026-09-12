#!/usr/bin/env python3

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = ROOT / "govi-rare-books-academic-dataset.json"
ANALYSIS_PATH = ROOT / "rag" / "bibliographic_analysis.json"


SECONDARY_PATTERNS = [
    r"\balso printed\b",
    r"\balso published\b",
    r"\banother edition\b",
    r"\banother issue\b",
    r"\banother printing\b",
    r"\bsecond edition\b",
    r"\bsecond issue\b",
    r"\bthird edition\b",
    r"\blater edition\b",
    r"\blater printing\b",
    r"\bpublished again\b",
    r"\bprinted again\b",
    r"\bVenice edition\b",
    r"\bLatin edition\b",
    r"\bFrench edition\b",
    r"\bItalian edition\b",
    r"\bGerman edition\b",
    r"\bSpanish edition\b",
    r"\bEnglish edition\b",
    r"\bprinted in \d{4}\b",
    r"\bpublished in \d{4}\b",
    r"\bissued in \d{4}\b",
]


def load_json(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def contains_secondary_edition(text: str) -> bool:
    lower = text.lower()

    return any(
        re.search(
            pattern,
            lower,
        )
        for pattern in SECONDARY_PATTERNS
    )


def main():
    dataset = load_json(
        DATASET_PATH
    )

    analysis = load_json(
        ANALYSIS_PATH
    )

    dataset_titles = {
        item.get("id"): item.get("title")
        for item in dataset
    }

    total_items = 0
    items_with_signals = 0

    suspicious_edition = []
    ambiguous_historical = []

    for item in analysis.get(
        "items",
        [],
    ):

        total_items += 1

        book_id = item.get(
            "book_id"
        )

        title = (
            item.get("title")
            or dataset_titles.get(book_id)
            or book_id
        )

        signals = item.get(
            "signals",
            {}
        )

        if signals:
            items_with_signals += 1

        edition_or_copy = signals.get(
            "edition_or_copy",
            {}
        )

        historical_context = signals.get(
            "historical_context",
            {}
        )

        # ---------------------------------------------------------
        # TEST 1
        #
        # Un vero problema esiste quando un segnale finito in
        # edition_or_copy viene attribuito a un contesto che NON
        # identifica la copia corrente.
        #
        # La presenza di "second issue", "later edition", ecc.
        # NON è sufficiente per generare un warning se
        # current_copy=True.
        # ---------------------------------------------------------

        for signal_type, entries in edition_or_copy.items():

            for entry in entries:

                current_copy = entry.get(
                    "current_copy",
                    False
                )

                secondary_edition = entry.get(
                    "secondary_edition",
                    False
                )

                evidence = entry.get(
                    "evidence",
                    ""
                )

                # Questa è una vera anomalia:
                # il segnale è stato classificato come
                # caratteristica della copia ma il contesto
                # non identifica chiaramente la copia corrente.
                if not current_copy:

                    suspicious_edition.append(
                        {
                            "book_id": book_id,
                            "title": title,
                            "signal_type": signal_type,
                            "match": entry.get("match"),
                            "current_copy": current_copy,
                            "secondary_edition": secondary_edition,
                            "evidence": evidence,
                        }
                    )

        # ---------------------------------------------------------
        # TEST 2
        #
        # historical_context è corretto anche quando
        # current_copy=True.
        #
        # current_copy=True significa:
        # "la frase parla dell'opera corrente".
        #
        # Non significa:
        # "questa frase è una caratteristica fisica
        # dell'esemplare".
        #
        # Pertanto NON consideriamo più questo un errore.
        # Facciamo soltanto un report informativo sui segnali storici.
        # ---------------------------------------------------------

        for signal_type, entries in historical_context.items():

            for entry in entries:

                evidence = entry.get(
                    "evidence",
                    ""
                )

                current_copy = entry.get(
                    "current_copy",
                    False
                )

                # Registriamo solo i casi potenzialmente
                # interessanti per una revisione semantica:
                # una frase storica molto forte che non è
                # esplicitamente collegata all'opera corrente.
                if not current_copy:

                    ambiguous_historical.append(
                        {
                            "book_id": book_id,
                            "title": title,
                            "signal_type": signal_type,
                            "match": entry.get("match"),
                            "evidence": evidence,
                        }
                    )

    print("=" * 80)
    print("GOVI BIBLIOGRAPHIC ANALYSIS - QUALITY GATE")
    print("=" * 80)
    print()

    print(
        "Dataset records:",
        len(dataset)
    )

    print(
        "Analysis records:",
        total_items
    )

    print(
        "Records with signals:",
        items_with_signals
    )

    print()

    print(
        "TRUE EDITION/COPY ANOMALIES:",
        len(suspicious_edition)
    )

    print(
        "HISTORICAL SIGNALS WITHOUT CURRENT-WORK LINK:",
        len(ambiguous_historical)
    )

    print()

    # -------------------------------------------------------------
    # MOSTRA LE ANOMALIE VERE
    # -------------------------------------------------------------

    if suspicious_edition:

        print("-" * 80)
        print(
            "TRUE EDITION/COPY ANOMALIES"
        )
        print("-" * 80)

        for index, item in enumerate(
            suspicious_edition[:20],
            start=1,
        ):

            print()
            print(
                f"[{index}] {item['title']}"
            )

            print(
                "ID:",
                item["book_id"]
            )

            print(
                "Signal:",
                item["signal_type"]
            )

            print(
                "Match:",
                item["match"]
            )

            print(
                "current_copy:",
                item["current_copy"]
            )

            print(
                "secondary_edition:",
                item["secondary_edition"]
            )

            print(
                "Evidence:",
                item["evidence"][:800]
            )

    # -------------------------------------------------------------
    # REPORT STORICO
    # -------------------------------------------------------------

    if ambiguous_historical:

        print()
        print("-" * 80)
        print(
            "HISTORICAL SIGNALS WITHOUT CURRENT-WORK LINK"
        )
        print("-" * 80)

        for index, item in enumerate(
            ambiguous_historical[:20],
            start=1,
        ):

            print()
            print(
                f"[{index}] {item['title']}"
            )

            print(
                "ID:",
                item["book_id"]
            )

            print(
                "Signal:",
                item["signal_type"]
            )

            print(
                "Match:",
                item["match"]
            )

            print(
                "Evidence:",
                item["evidence"][:800]
            )

    print()

    # -------------------------------------------------------------
    # RISULTATO FINALE
    # -------------------------------------------------------------

    if not suspicious_edition:

        print(
            "QUALITY GATE: PASS"
        )

        print(
            "Nessuna anomalia reale trovata "
            "nei segnali edition_or_copy."
        )

    else:

        print(
            "QUALITY GATE: REVIEW NEEDED"
        )

        print(
            "Sono presenti segnali classificati "
            "come edition_or_copy senza un "
            "collegamento esplicito alla copia corrente."
        )


if __name__ == "__main__":
    main()
