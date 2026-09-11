#!/usr/bin/env python3

import json
import sys
from pathlib import Path
from urllib.parse import urlparse


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_FILE = REPO_ROOT / "govi-rare-books-academic-dataset.json"


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def is_url(value):
    """Return True if value looks like a valid HTTP(S) URL."""
    if not isinstance(value, str) or not value.strip():
        return False

    try:
        parsed = urlparse(value)
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except Exception:
        return False


def add_error(errors, message):
    errors.append(message)


def add_warning(warnings, message):
    warnings.append(message)


# ---------------------------------------------------------
# Main validation
# ---------------------------------------------------------

def validate_dataset():
    errors = []
    warnings = []

    print("=" * 60)
    print("GOVI RARE BOOKS DATASET VALIDATION")
    print("=" * 60)
    print()

    # -----------------------------------------------------
    # 1. Dataset exists
    # -----------------------------------------------------

    if not DATASET_FILE.exists():
        print(f"ERROR: dataset not found: {DATASET_FILE}")
        return 1

    print(f"Dataset: {DATASET_FILE.name}")

    # -----------------------------------------------------
    # 2. JSON syntax
    # -----------------------------------------------------

    try:
        with open(DATASET_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        print()
        print("ERROR: invalid JSON")
        print(f"       {exc}")
        return 1

    print("JSON syntax: OK")

    # -----------------------------------------------------
    # 3. Top-level structure
    # -----------------------------------------------------

    if not isinstance(data, list):
        add_error(
            errors,
            "Top-level JSON object must be a list of records."
        )
        records = []
    else:
        records = data

    print(f"Records found: {len(records)}")

    # -----------------------------------------------------
    # 4. Required fields
    # -----------------------------------------------------

    required_fields = [
        "title",
        "authors",
        "publication_year",
        "publication_place",
        "publisher_data",
        "topics",
        "academic_description",
        "bibliography",
        "source_url",
        "institution",
        "linked_open_data",
    ]

    # -----------------------------------------------------
    # 5. Record-by-record validation
    # -----------------------------------------------------

    for index, record in enumerate(records, start=1):

        prefix = f"Record #{index}"

        if not isinstance(record, dict):
            add_error(
                errors,
                f"{prefix}: record is not an object."
            )
            continue

        # Required fields
        for field in required_fields:
            if field not in record:
                add_error(
                    errors,
                    f"{prefix}: missing field '{field}'."
                )

        # -------------------------------------------------
        # Title
        # -------------------------------------------------

        if "title" in record:
            if not isinstance(record["title"], str):
                add_error(
                    errors,
                    f"{prefix}: 'title' must be a string."
                )
            elif not record["title"].strip():
                add_warning(
                    warnings,
                    f"{prefix}: empty title."
                )

        # -------------------------------------------------
        # Authors
        # -------------------------------------------------

        if "authors" in record:
            if not isinstance(record["authors"], list):
                add_error(
                    errors,
                    f"{prefix}: 'authors' must be a list."
                )
            elif len(record["authors"]) == 0:
                add_warning(
                    warnings,
                    f"{prefix}: no authors."
                )

        # -------------------------------------------------
        # Publication year
        # -------------------------------------------------

        if "publication_year" in record:
            year = record["publication_year"]

            if not isinstance(year, int):
                add_error(
                    errors,
                    f"{prefix}: 'publication_year' must be an integer."
                )
            else:
                if year < 1000 or year > 2100:
                    add_warning(
                        warnings,
                        f"{prefix}: suspicious publication year: {year}."
                    )

        # -------------------------------------------------
        # Topics
        # -------------------------------------------------

        if "topics" in record:
            if not isinstance(record["topics"], list):
                add_error(
                    errors,
                    f"{prefix}: 'topics' must be a list."
                )

        # -------------------------------------------------
        # Source URL
        # -------------------------------------------------

        if "source_url" in record:
            url = record["source_url"]

            if not is_url(url):
                add_error(
                    errors,
                    f"{prefix}: invalid source_url: {url!r}"
                )

        # -------------------------------------------------
        # Linked open data
        # -------------------------------------------------

        if "linked_open_data" in record:
            lod = record["linked_open_data"]

            if not isinstance(lod, dict):
                add_error(
                    errors,
                    f"{prefix}: 'linked_open_data' must be an object."
                )

        # -------------------------------------------------
        # Academic description
        # -------------------------------------------------

        if "academic_description" in record:
            description = record["academic_description"]

            if not isinstance(description, str):
                add_error(
                    errors,
                    f"{prefix}: 'academic_description' must be a string."
                )
            elif not description.strip():
                add_warning(
                    warnings,
                    f"{prefix}: empty academic_description."
                )

    # -----------------------------------------------------
    # 6. Summary
    # -----------------------------------------------------

    print()
    print("-" * 60)
    print("VALIDATION SUMMARY")
    print("-" * 60)

    print(f"Records checked : {len(records)}")
    print(f"Errors          : {len(errors)}")
    print(f"Warnings        : {len(warnings)}")

    # -----------------------------------------------------
    # 7. Errors
    # -----------------------------------------------------

    if errors:
        print()
        print("ERRORS")
        print("-" * 60)

        for error in errors:
            print(f"✗ {error}")

    # -----------------------------------------------------
    # 8. Warnings
    # -----------------------------------------------------

    if warnings:
        print()
        print("WARNINGS")
        print("-" * 60)

        for warning in warnings:
            print(f"! {warning}")

    # -----------------------------------------------------
    # 9. Final result
    # -----------------------------------------------------

    print()
    print("=" * 60)

    if errors:
        print("RESULT: FAILED")
        print("=" * 60)
        return 1

    print("RESULT: PASSED")
    print("=" * 60)

    return 0


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------

if __name__ == "__main__":
    sys.exit(validate_dataset())
