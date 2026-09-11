from pathlib import Path
import json
import re
import sys
from urllib.parse import urlparse
from collections import Counter


ROOT = Path(__file__).resolve().parent.parent
DATASET_FILE = ROOT / "govi-rare-books-academic-dataset.json"


# ============================================================
# REQUIRED DATASET STRUCTURE
# ============================================================

REQUIRED_FIELDS = [
    "title",
    "authors",
    "publishers",
    "related_names",
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


PERSON_REQUIRED_FIELDS = [
    "name",
    "biography",
    "biographical_data",
    "bibliography",
    "links",
]


LINK_FIELDS = [
    "wikipedia",
    "treccani",
    "sep",
    "viaf",
]


OPEN_DATA_FIELDS = [
    "sbn",
    "oclc",
    "edit16",
    "ustc",
    "wikidata",
]


MIN_YEAR = 1000
MAX_YEAR = 2100


errors = []
warnings = []


# ============================================================
# MESSAGES
# ============================================================

def error(record_no, message):
    errors.append(
        f"[ERROR] Record {record_no}: {message}"
    )


def warning(record_no, message):
    warnings.append(
        f"[WARNING] Record {record_no}: {message}"
    )


# ============================================================
# GENERIC VALIDATION
# ============================================================

def is_valid_url(value):
    if not isinstance(value, str) or not value.strip():
        return False

    try:
        parsed = urlparse(value)

        return (
            parsed.scheme in ("http", "https")
            and bool(parsed.netloc)
        )

    except Exception:
        return False


def check_string(
    record_no,
    field,
    value,
    required=True
):
    if value is None:

        if required:
            error(
                record_no,
                f"'{field}' is missing"
            )

        return

    if not isinstance(value, str):

        error(
            record_no,
            f"'{field}' must be a string"
        )

        return

    if required and not value.strip():

        warning(
            record_no,
            f"'{field}' is empty"
        )


def check_list(
    record_no,
    field,
    value,
    required=True
):
    if value is None:

        if required:
            error(
                record_no,
                f"'{field}' is missing"
            )

        return

    if not isinstance(value, list):

        error(
            record_no,
            f"'{field}' must be a list"
        )

        return

    # An empty list is valid.
    #
    # For example:
    #   publishers: []
    #   related_names: []
    #
    # These are legitimate states in the dataset and
    # therefore are not reported as warnings.


# ============================================================
# PUBLICATION YEAR
# ============================================================

def check_publication_year(
    record_no,
    value
):
    if value is None:

        error(
            record_no,
            "'publication_year' is missing"
        )

        return

    if isinstance(value, int):

        if (
            value < MIN_YEAR
            or value > MAX_YEAR
        ):

            warning(
                record_no,
                f"publication year '{value}' "
                f"is outside expected range "
                f"{MIN_YEAR}-{MAX_YEAR}"
            )

        return

    if isinstance(value, str):

        if value.strip().upper() == "S.D.":

            return

        warning(
            record_no,
            f"'publication_year' is a string: "
            f"'{value}'"
        )

        return

    error(
        record_no,
        "'publication_year' must be "
        "an integer or 'S.D.'"
    )


# ============================================================
# PERSON / NAME VALIDATION
# ============================================================

def check_person_list(
    record_no,
    field,
    people
):
    if not isinstance(people, list):

        error(
            record_no,
            f"'{field}' must be a list"
        )

        return

    for person_no, person in enumerate(
        people,
        start=1
    ):

        if not isinstance(person, dict):

            error(
                record_no,
                f"'{field}' entry #{person_no} "
                "must be an object"
            )

            continue

        # ----------------------------------------------------
        # Required person fields
        # ----------------------------------------------------

        for required_field in PERSON_REQUIRED_FIELDS:

            if required_field not in person:

                error(
                    record_no,
                    f"'{field}' entry #{person_no} "
                    f"missing '{required_field}'"
                )

        # ----------------------------------------------------
        # Name
        # ----------------------------------------------------

        name = person.get("name")

        if not isinstance(name, str):

            error(
                record_no,
                f"'{field}' entry #{person_no} "
                "'name' must be a string"
            )

        elif not name.strip():

            warning(
                record_no,
                f"'{field}' entry #{person_no} "
                "has an empty name"
            )

        # ----------------------------------------------------
        # Biography
        # ----------------------------------------------------

        biography = person.get("biography")

        if biography is None:

            warning(
                record_no,
                f"'{field}' entry #{person_no} "
                "biography is missing"
            )

        elif not isinstance(biography, str):

            error(
                record_no,
                f"'{field}' entry #{person_no} "
                "biography must be a string"
            )

        elif biography.strip():

            check_suspicious_biography_dates(
                record_no,
                field,
                person_no,
                biography
            )

        # ----------------------------------------------------
        # Biographical data
        # ----------------------------------------------------

        biographical_data = person.get(
            "biographical_data"
        )

        if biographical_data is None:

            warning(
                record_no,
                f"'{field}' entry #{person_no} "
                "biographical_data is missing"
            )

        elif not isinstance(
            biographical_data,
            str
        ):

            error(
                record_no,
                f"'{field}' entry #{person_no} "
                "biographical_data must be a string"
            )

        # ----------------------------------------------------
        # Bibliography
        # ----------------------------------------------------

        bibliography = person.get(
            "bibliography"
        )

        if bibliography is None:

            warning(
                record_no,
                f"'{field}' entry #{person_no} "
                "bibliography is missing"
            )

        elif not isinstance(
            bibliography,
            str
        ):

            error(
                record_no,
                f"'{field}' entry #{person_no} "
                "bibliography must be a string"
            )

        # ----------------------------------------------------
        # Links
        # ----------------------------------------------------

        links = person.get("links")

        if links is None:

            error(
                record_no,
                f"'{field}' entry #{person_no} "
                "'links' is missing"
            )

            continue

        if not isinstance(links, dict):

            error(
                record_no,
                f"'{field}' entry #{person_no} "
                "'links' must be an object"
            )

            continue

        for link_field in LINK_FIELDS:

            if link_field not in links:

                error(
                    record_no,
                    f"'{field}' entry #{person_no} "
                    f"missing link field "
                    f"'{link_field}'"
                )

                continue

            value = links.get(link_field)

            if value is None:
                continue

            if not isinstance(value, str):

                error(
                    record_no,
                    f"'{field}' entry #{person_no} "
                    f"link '{link_field}' must be "
                    "a string or null"
                )

                continue

            if not is_valid_url(value):

                warning(
                    record_no,
                    f"'{field}' entry #{person_no} "
                    f"link '{link_field}' does not "
                    "appear to be a valid URL"
                )


# ============================================================
# LINKED OPEN DATA
# ============================================================

def check_linked_open_data(
    record_no,
    value
):
    if value is None:

        error(
            record_no,
            "'linked_open_data' is missing"
        )

        return

    if not isinstance(value, dict):

        error(
            record_no,
            "'linked_open_data' must be an object"
        )

        return

    for field in OPEN_DATA_FIELDS:

        if field not in value:

            error(
                record_no,
                f"linked_open_data missing "
                f"'{field}'"
            )

            continue

        url = value.get(field)

        if url is None:
            continue

        if not isinstance(url, str):

            error(
                record_no,
                f"linked_open_data '{field}' "
                "must be a string or null"
            )

            continue

        if not is_valid_url(url):

            warning(
                record_no,
                f"linked_open_data '{field}' "
                "does not appear to be "
                "a valid URL"
            )


# ============================================================
# TOPICS
# ============================================================

def check_topics(
    record_no,
    topics
):
    if not isinstance(topics, list):

        error(
            record_no,
            "'topics' must be a list"
        )

        return

    for topic_no, topic in enumerate(
        topics,
        start=1
    ):

        if not isinstance(topic, str):

            error(
                record_no,
                f"topic #{topic_no} "
                "must be a string"
            )

            continue

        if not topic.strip():

            warning(
                record_no,
                f"topic #{topic_no} is empty"
            )


# ============================================================
# BIOGRAPHY DATE CHECK
# ============================================================

def check_suspicious_biography_dates(
    record_no,
    field,
    person_no,
    biography
):
    patterns = re.findall(
        r"\b(\d{3,4})\s*[–-]\s*(\d{3,4})\b",
        biography
    )

    for birth, death in patterns:

        birth_year = int(birth)
        death_year = int(death)

        if (
            birth_year < MIN_YEAR
            or death_year < MIN_YEAR
        ):
            continue

        if (
            birth_year > MAX_YEAR
            or death_year > MAX_YEAR
        ):
            continue

        if death_year < birth_year:

            warning(
                record_no,
                f"{field} entry #{person_no} "
                "biography contains a suspicious "
                f"date range: {birth}-{death}"
            )


# ============================================================
# RECORD VALIDATION
# ============================================================

def check_record(
    record_no,
    record
):
    if not isinstance(record, dict):

        error(
            record_no,
            "record must be an object"
        )

        return

    # --------------------------------------------------------
    # Required top-level fields
    # --------------------------------------------------------

    for field in REQUIRED_FIELDS:

        if field not in record:

            error(
                record_no,
                f"missing required field "
                f"'{field}'"
            )

    # --------------------------------------------------------
    # Bibliographic fields
    # --------------------------------------------------------

    check_string(
        record_no,
        "title",
        record.get("title")
    )

    check_string(
        record_no,
        "publication_place",
        record.get("publication_place")
    )

    check_string(
        record_no,
        "publisher_data",
        record.get("publisher_data")
    )

    check_string(
        record_no,
        "academic_description",
        record.get("academic_description")
    )

    # Bibliography may legitimately be empty.
    check_string(
        record_no,
        "bibliography",
        record.get("bibliography"),
        required=False
    )

    check_string(
        record_no,
        "institution",
        record.get("institution")
    )

    # --------------------------------------------------------
    # Collections
    # --------------------------------------------------------

    check_list(
        record_no,
        "authors",
        record.get("authors")
    )

    check_list(
        record_no,
        "publishers",
        record.get("publishers")
    )

    check_list(
        record_no,
        "related_names",
        record.get("related_names")
    )

    check_topics(
        record_no,
        record.get("topics")
    )

    # --------------------------------------------------------
    # Publication year
    # --------------------------------------------------------

    check_publication_year(
        record_no,
        record.get("publication_year")
    )

    # --------------------------------------------------------
    # Source URL
    # --------------------------------------------------------

    source_url = record.get(
        "source_url"
    )

    if source_url is None:

        error(
            record_no,
            "'source_url' is missing"
        )

    elif not is_valid_url(source_url):

        warning(
            record_no,
            "'source_url' does not appear "
            "to be a valid URL"
        )

    # --------------------------------------------------------
    # Linked Open Data
    # --------------------------------------------------------

    check_linked_open_data(
        record_no,
        record.get("linked_open_data")
    )

    # --------------------------------------------------------
    # People
    # --------------------------------------------------------

    check_person_list(
        record_no,
        "authors",
        record.get("authors")
    )

    check_person_list(
        record_no,
        "publishers",
        record.get("publishers")
    )

    check_person_list(
        record_no,
        "related_names",
        record.get("related_names")
    )


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(value):
    return " ".join(
        value.lower().split()
    )


# ============================================================
# DUPLICATE TITLES
# ============================================================

def check_duplicates(dataset):

    normalized_titles = {}

    for index, record in enumerate(
        dataset,
        start=1
    ):

        if not isinstance(record, dict):
            continue

        title = record.get("title")

        if not isinstance(title, str):
            continue

        normalized = normalize_text(title)

        normalized_titles.setdefault(
            normalized,
            []
        ).append(index)

    for title, records in normalized_titles.items():

        if len(records) > 1:

            warning(
                "GLOBAL",
                f"duplicate title detected "
                f"in records {records}: {title}"
            )


# ============================================================
# TOPIC ANALYSIS
# ============================================================

def analyze_topics(dataset):

    topic_counter = Counter()

    for record in dataset:

        if not isinstance(record, dict):
            continue

        topics = record.get(
            "topics",
            []
        )

        if not isinstance(topics, list):
            continue

        for topic in topics:

            if (
                isinstance(topic, str)
                and topic.strip()
            ):

                topic_counter[
                    normalize_text(topic)
                ] += 1

    return topic_counter


# ============================================================
# PEOPLE ANALYSIS
# ============================================================

def analyze_people(
    dataset,
    field
):
    names = Counter()

    for record in dataset:

        if not isinstance(record, dict):
            continue

        people = record.get(
            field,
            []
        )

        if not isinstance(people, list):
            continue

        for person in people:

            if not isinstance(person, dict):
                continue

            name = person.get("name")

            if (
                isinstance(name, str)
                and name.strip()
            ):

                names[
                    normalize_text(name)
                ] += 1

    return names


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "GOVI RARE BOOKS ARCHIVE — DATASET VALIDATOR"
    )
    print("=" * 70)
    print()

    # --------------------------------------------------------
    # Dataset existence
    # --------------------------------------------------------

    if not DATASET_FILE.exists():

        print(
            "[ERROR] Dataset file not found:"
        )

        print(
            DATASET_FILE
        )

        sys.exit(1)

    print(
        f"Dataset: {DATASET_FILE}"
    )

    print()

    # --------------------------------------------------------
    # Load JSON
    # --------------------------------------------------------

    try:

        with DATASET_FILE.open(
            "r",
            encoding="utf-8"
        ) as f:

            dataset = json.load(f)

    except json.JSONDecodeError as exc:

        print(
            "[ERROR] Invalid JSON"
        )

        print(exc)

        sys.exit(1)

    except Exception as exc:

        print(
            "[ERROR] Could not read dataset"
        )

        print(exc)

        sys.exit(1)

    # --------------------------------------------------------
    # Root structure
    # --------------------------------------------------------

    if not isinstance(dataset, list):

        print(
            "[ERROR] Dataset root must be a list."
        )

        sys.exit(1)

    print(
        f"Records found: {len(dataset)}"
    )

    print()

    # --------------------------------------------------------
    # Validate records
    # --------------------------------------------------------

    for record_no, record in enumerate(
        dataset,
        start=1
    ):

        check_record(
            record_no,
            record
        )

    # --------------------------------------------------------
    # Global checks
    # --------------------------------------------------------

    check_duplicates(
        dataset
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    topic_counter = analyze_topics(
        dataset
    )

    author_counter = analyze_people(
        dataset,
        "authors"
    )

    publisher_counter = analyze_people(
        dataset,
        "publishers"
    )

    related_counter = analyze_people(
        dataset,
        "related_names"
    )

    author_count = sum(
        author_counter.values()
    )

    publisher_count = sum(
        publisher_counter.values()
    )

    related_count = sum(
        related_counter.values()
    )

    topic_assignment_count = sum(
        topic_counter.values()
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("-" * 70)
    print("SUMMARY")
    print("-" * 70)

    print(
        f"Records:                       "
        f"{len(dataset)}"
    )

    print(
        f"Author occurrences:            "
        f"{author_count}"
    )

    print(
        f"Unique authors:                "
        f"{len(author_counter)}"
    )

    print(
        f"Publisher occurrences:         "
        f"{publisher_count}"
    )

    print(
        f"Unique publishers:             "
        f"{len(publisher_counter)}"
    )

    print(
        f"Related-name occurrences:      "
        f"{related_count}"
    )

    print(
        f"Unique related names:          "
        f"{len(related_counter)}"
    )

    print(
        f"Topic assignments:             "
        f"{topic_assignment_count}"
    )

    print(
        f"Unique topics:                 "
        f"{len(topic_counter)}"
    )

    if len(dataset) > 0:

        average_topics = (
            topic_assignment_count
            / len(dataset)
        )

        print(
            f"Average topics per record:     "
            f"{average_topics:.2f}"
        )

    print(
        f"Errors:                        "
        f"{len(errors)}"
    )

    print(
        f"Warnings:                      "
        f"{len(warnings)}"
    )

    print()

    # ========================================================
    # TOP TOPICS
    # ========================================================

    print("-" * 70)
    print("TOP TOPICS")
    print("-" * 70)

    for topic, count in topic_counter.most_common(
        15
    ):

        print(
            f"{count:>4}  {topic}"
        )

    print()

    # ========================================================
    # ERRORS
    # ========================================================

    if errors:

        print("-" * 70)
        print("ERRORS")
        print("-" * 70)

        for message in errors:

            print(message)

        print()

    # ========================================================
    # WARNINGS
    # ========================================================

    if warnings:

        print("-" * 70)
        print("WARNINGS")
        print("-" * 70)

        for message in warnings:

            print(message)

        print()

    # ========================================================
    # FINAL RESULT
    # ========================================================

    if not errors and not warnings:

        print(
            "✓ Dataset passed all validation checks."
        )

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
            "✗ Dataset contains errors "
            "that should be fixed."
        )

        print()

    print("=" * 70)

    # GitHub Actions fails only on real errors.
    if errors:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
