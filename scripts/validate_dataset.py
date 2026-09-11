from pathlib import Path
import json
import re
import sys
from urllib.parse import urlparse
from collections import Counter


# ============================================================
# Govi Rare Books Archive
# Dataset validator
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
DATASET_FILE = ROOT / "govi-rare-books-academic-dataset.json"

REQUIRED_FIELDS = [
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

AUTHOR_REQUIRED_FIELDS = [
    "name",
    "biography",
    "links",
]

LINK_FIELDS = [
    "wikipedia",
    "treccani",
    "sep",
    "viaf",
]

MIN_YEAR = 1000
MAX_YEAR = 2100

errors = []
warnings = []


def error(record_no, message):
    errors.append(f"[ERROR] Record {record_no}: {message}")


def warning(record_no, message):
    warnings.append(f"[WARNING] Record {record_no}: {message}")


def is_valid_url(value):
    if not isinstance(value, str) or not value.strip():
        return False

    try:
        parsed = urlparse(value)
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except Exception:
        return False


def check_string(record_no, field, value, required=True):
    if value is None:
        if required:
            error(record_no, f"'{field}' is missing")
        return

    if not isinstance(value, str):
        error(record_no, f"'{field}' must be a string")
        return

    if required and not value.strip():
        warning(record_no, f"'{field}' is empty")


def check_list(record_no, field, value, required=True):
    if value is None:
        if required:
            error(record_no, f"'{field}' is missing")
        return

    if not isinstance(value, list):
        error(record_no, f"'{field}' must be a list")
        return

    if required and len(value) == 0:
        warning(record_no, f"'{field}' is empty")


def check_publication_year(record_no, value):
    if value is None:
        error(record_no, "'publication_year' is missing")
        return

    if not isinstance(value, int):
        error(record_no, "'publication_year' must be an integer")
        return

    if value < MIN_YEAR or value > MAX_YEAR:
        warning(
            record_no,
            f"publication year '{value}' is outside expected range "
            f"{MIN_YEAR}-{MAX_YEAR}",
        )


def check_authors(record_no, authors):
    if not isinstance(authors, list):
        error(record_no, "'authors' must be a list")
        return

    if len(authors) == 0:
        warning(record_no, "'authors' is empty")
        return

    for author_no, author in enumerate(authors, start=1):

        if not isinstance(author, dict):
            error(
                record_no,
                f"author #{author_no} must be an object",
            )
            continue

        for field in AUTHOR_REQUIRED_FIELDS:
            if field not in author:
                error(
                    record_no,
                    f"author #{author_no} missing '{field}'",
                )

        # ----------------------------------------------------
        # Author name
        # ----------------------------------------------------

        name = author.get("name")

        if not isinstance(name, str):
            error(
                record_no,
                f"author #{author_no} 'name' must be a string",
            )
        elif not name.strip():
            warning(
                record_no,
                f"author #{author_no} has an empty name",
            )

        # ----------------------------------------------------
        # Biography
        # ----------------------------------------------------

        biography = author.get("biography")

        if biography is None:
            warning(
                record_no,
                f"author #{author_no} biography is missing",
            )
        elif not isinstance(biography, str):
            error(
                record_no,
                f"author #{author_no} biography must be a string",
            )
        elif not biography.strip():
            warning(
                record_no,
                f"author #{author_no} biography is empty",
            )
        else:
            check_suspicious_biography_dates(
                record_no,
                author_no,
                biography,
            )

        # ----------------------------------------------------
        # Author links
        # ----------------------------------------------------

        links = author.get("links")

        if links is None:
            error(
                record_no,
                f"author #{author_no} 'links' is missing",
            )
            continue

        if not isinstance(links, dict):
            error(
                record_no,
                f"author #{author_no} 'links' must be an object",
            )
            continue

        for link_field in LINK_FIELDS:

            if link_field not in links:
                error(
                    record_no,
                    f"author #{author_no} missing link field "
                    f"'{link_field}'",
                )
                continue

            value = links.get(link_field)

            # None is allowed: not every authority exists
            if value is None:
                continue

            if not isinstance(value, str):
                error(
                    record_no,
                    f"author #{author_no} link '{link_field}' "
                    f"must be a string or null",
                )
                continue

            if not is_valid_url(value):
                warning(
                    record_no,
                    f"author #{author_no} link '{link_field}' "
                    f"does not appear to be a valid URL",
                )


def check_linked_open_data(record_no, value):
    if value is None:
        error(record_no, "'linked_open_data' is missing")
        return

    if not isinstance(value, dict):
        error(record_no, "'linked_open_data' must be an object")
        return

    for authority, url in value.items():

        if url is None:
            continue

        if not isinstance(url, str):
            error(
                record_no,
                f"linked_open_data '{authority}' must be "
                f"a string or null",
            )
            continue

        if not is_valid_url(url):
            warning(
                record_no,
                f"linked_open_data '{authority}' does not "
                f"appear to be a valid URL",
            )


def check_suspicious_biography_dates(record_no, author_no, biography):
    """
    Looks for simple date patterns such as:
        1521–1583
        1521-1583
        5 December 1495 – 5 November 1492

    This is intentionally a WARNING-only test.
    Historical dates require human/editorial judgement.
    """

    patterns = re.findall(
        r"\b(\d{3,4})\s*[–-]\s*(\d{3,4})\b",
        biography,
    )

    for birth, death in patterns:

        birth_year = int(birth)
        death_year = int(death)

        if birth_year < MIN_YEAR or death_year < MIN_YEAR:
            continue

        if birth_year > MAX_YEAR or death_year > MAX_YEAR:
            continue

        if death_year < birth_year:
            warning(
                record_no,
                f"author #{author_no} biography contains a "
                f"suspicious date range: {birth}-{death}",
            )


def check_record(record_no, record):

    if not isinstance(record, dict):
        error(record_no, "record must be an object")
        return

    # --------------------------------------------------------
    # Required fields
    # --------------------------------------------------------

    for field in REQUIRED_FIELDS:
        if field not in record:
            error(record_no, f"missing required field '{field}'")

    # --------------------------------------------------------
    # Strings
    # --------------------------------------------------------

    check_string(record_no, "title", record.get("title"))
    check_string(
        record_no,
        "publication_place",
        record.get("publication_place"),
    )
    check_string(
        record_no,
        "publisher_data",
        record.get("publisher_data"),
    )
    check_string(
        record_no,
        "academic_description",
        record.get("academic_description"),
    )
    check_string(
        record_no,
        "bibliography",
        record.get("bibliography"),
    )
    check_string(
        record_no,
        "institution",
        record.get("institution"),
    )

    # --------------------------------------------------------
    # Lists
    # --------------------------------------------------------

    check_list(record_no, "authors", record.get("authors"))
    check_list(record_no, "topics", record.get("topics"))

    # --------------------------------------------------------
    # Year
    # --------------------------------------------------------

    check_publication_year(
        record_no,
        record.get("publication_year"),
    )

    # --------------------------------------------------------
    # Source URL
    # --------------------------------------------------------

    source_url = record.get("source_url")

    if source_url is None:
        error(record_no, "'source_url' is missing")
    elif not is_valid_url(source_url):
        warning(
            record_no,
            "'source_url' does not appear to be a valid URL",
        )

    # --------------------------------------------------------
    # Linked Open Data
    # --------------------------------------------------------

    check_linked_open_data(
        record_no,
        record.get("linked_open_data"),
    )

    # --------------------------------------------------------
    # Authors
    # --------------------------------------------------------

    check_authors(
        record_no,
        record.get("authors"),
    )


def check_duplicates(dataset):

    titles = []
    normalized_titles = {}

    for index, record in enumerate(dataset, start=1):

        if not isinstance(record, dict):
            continue

        title = record.get("title")

        if not isinstance(title, str):
            continue

        normalized = " ".join(title.lower().split())

        titles.append(normalized)

        normalized_titles.setdefault(
            normalized,
            [],
        ).append(index)

    for title, records in normalized_titles.items():

        if len(records) > 1:
            warning(
                "GLOBAL",
                f"duplicate title detected in records {records}: "
                f"{title}",
            )


def main():

    print("=" * 70)
    print("GOVI RARE BOOKS ARCHIVE — DATASET VALIDATOR")
    print("=" * 70)
    print()

    # --------------------------------------------------------
    # File existence
    # --------------------------------------------------------

    if not DATASET_FILE.exists():
        print("[ERROR] Dataset file not found:")
        print(DATASET_FILE)
        sys.exit(1)

    print(f"Dataset: {DATASET_FILE}")
    print()

    # --------------------------------------------------------
    # JSON loading
    # --------------------------------------------------------

    try:
        with DATASET_FILE.open(
            "r",
            encoding="utf-8",
        ) as f:
            dataset = json.load(f)

    except json.JSONDecodeError as exc:
        print("[ERROR] Invalid JSON")
        print(exc)
        sys.exit(1)

    except Exception as exc:
        print("[ERROR] Could not read dataset")
        print(exc)
        sys.exit(1)

    # --------------------------------------------------------
    # Top-level structure
    # --------------------------------------------------------

    if not isinstance(dataset, list):
        print("[ERROR] Dataset root must be a list.")
        sys.exit(1)

    print(f"Records found: {len(dataset)}")
    print()

    # --------------------------------------------------------
    # Validate records
    # --------------------------------------------------------

    for record_no, record in enumerate(
        dataset,
        start=1,
    ):
        check_record(
            record_no,
            record,
        )

    # --------------------------------------------------------
    # Global checks
    # --------------------------------------------------------

    check_duplicates(dataset)

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    author_count = 0
    unique_authors = set()

    for record in dataset:

        if not isinstance(record, dict):
            continue

        authors = record.get("authors", [])

        if not isinstance(authors, list):
            continue

        author_count += len(authors)

        for author in authors:

            if isinstance(author, dict):

                name = author.get("name")

                if isinstance(name, str):
                    unique_authors.add(
                        " ".join(name.lower().split())
                    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print("-" * 70)
    print("SUMMARY")
    print("-" * 70)

    print(f"Records:             {len(dataset)}")
    print(f"Author occurrences:  {author_count}")
    print(f"Unique authors:      {len(unique_authors)}")
    print(f"Errors:              {len(errors)}")
    print(f"Warnings:            {len(warnings)}")
    print()

    if errors:

        print("-" * 70)
        print("ERRORS")
        print("-" * 70)

        for message in errors:
            print(message)

        print()

    if warnings:

        print("-" * 70)
        print("WARNINGS")
        print("-" * 70)

        for message in warnings:
            print(message)

        print()

    if not errors and not warnings:

        print("✓ Dataset passed all validation checks.")
        print()

    elif not errors:

        print(
            "✓ No structural errors found."
        )
        print(
            "⚠ Review the warnings above."
        )
        print()

    else:

        print(
            "✗ Dataset contains errors that should be fixed."
        )
        print()

    print("=" * 70)

    # --------------------------------------------------------
    # Exit code
    # --------------------------------------------------------

    if errors:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
