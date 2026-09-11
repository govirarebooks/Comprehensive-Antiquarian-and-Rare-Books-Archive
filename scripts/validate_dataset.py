import json
import re
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "govi-rare-books-academic-dataset.json"


# ULID:
# - 26 caratteri
# - Crockford Base32
# - primo carattere 0-7
ULID_PATTERN = re.compile(r"^[0-7][0-9A-HJKMNP-TV-Z]{25}$")


REQUIRED_TOP_LEVEL_FIELDS = [
    "id",
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


PERSON_FIELDS = [
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


def error(message):
    errors.append(message)


def warning(message):
    warnings.append(message)


def validate_string(value, field_name, required=True):
    if not isinstance(value, str):
        error(
            f"{field_name}: expected string, "
            f"got {type(value).__name__}"
        )
        return False

    if required and not value.strip():
        error(f"{field_name}: empty string")

    return True


def validate_id(record, index):
    """
    Validate stable stock.id.

    Requirements:
    - present
    - string
    - non-empty
    - valid ULID format
    """

    record_number = index + 1
    record_id = record.get("id")

    if record_id is None:
        error(
            f"Record {record_number}: missing 'id'"
        )
        return None

    if not isinstance(record_id, str):
        error(
            f"Record {record_number}: 'id' must be a string"
        )
        return None

    record_id = record_id.strip()

    if not record_id:
        error(
            f"Record {record_number}: 'id' is empty"
        )
        return None

    if not ULID_PATTERN.fullmatch(record_id):
        error(
            f"Record {record_number}: invalid ULID "
            f"format for id '{record_id}'"
        )

    return record_id


def validate_links(links, context):
    if not isinstance(links, dict):
        error(
            f"{context}.links: expected object, "
            f"got {type(links).__name__}"
        )
        return

    for field in LINK_FIELDS:
        if field not in links:
            error(
                f"{context}.links: missing '{field}'"
            )
            continue

        value = links[field]

        if value is not None and not isinstance(value, str):
            error(
                f"{context}.links.{field}: expected "
                f"string or null"
            )


def validate_person(person, context):
    if not isinstance(person, dict):
        error(
            f"{context}: expected object, "
            f"got {type(person).__name__}"
        )
        return

    for field in PERSON_FIELDS:
        if field not in person:
            error(
                f"{context}: missing '{field}'"
            )

    if "name" in person:
        validate_string(
            person["name"],
            f"{context}.name"
        )

    for field in [
        "biography",
        "biographical_data",
        "bibliography",
    ]:
        if field in person and person[field] is not None:
            if not isinstance(person[field], str):
                error(
                    f"{context}.{field}: expected string "
                    f"or null"
                )

    if "links" in person:
        validate_links(
            person["links"],
            context
        )


def validate_open_data(value, context):
    if not isinstance(value, dict):
        error(
            f"{context}: expected object, "
            f"got {type(value).__name__}"
        )
        return

    for field in OPEN_DATA_FIELDS:
        if field not in value:
            error(
                f"{context}: missing '{field}'"
            )
            continue

        field_value = value[field]

        if field_value is not None and not isinstance(
            field_value, str
        ):
            error(
                f"{context}.{field}: expected "
                f"string or null"
            )


def validate_year(value, context):
    if value is None:
        error(
            f"{context}: publication year is null"
        )
        return

    if isinstance(value, int):
        return

    if isinstance(value, str):
        if value.strip() == "S.D.":
            return

    error(
        f"{context}: expected integer or 'S.D.', "
        f"got {value!r}"
    )


def validate_record(record, index):
    record_number = index + 1
    context = f"Record {record_number}"

    if not isinstance(record, dict):
        error(
            f"{context}: expected object, "
            f"got {type(record).__name__}"
        )
        return

    # ---------------------------------------------------------
    # Top-level schema
    # ---------------------------------------------------------

    for field in REQUIRED_TOP_LEVEL_FIELDS:
        if field not in record:
            error(
                f"{context}: missing top-level field '{field}'"
            )

    # ---------------------------------------------------------
    # Stable ID
    # ---------------------------------------------------------

    validate_id(record, index)

    # ---------------------------------------------------------
    # Basic fields
    # ---------------------------------------------------------

    for field in [
        "title",
        "publication_place",
        "publisher_data",
        "academic_description",
        "source_url",
        "institution",
    ]:
        if field in record:
            validate_string(
                record[field],
                f"{context}.{field}"
            )

    # ---------------------------------------------------------
    # Publication year
    # ---------------------------------------------------------

    if "publication_year" in record:
        validate_year(
            record["publication_year"],
            f"{context}.publication_year"
        )

    # ---------------------------------------------------------
    # Authors
    # ---------------------------------------------------------

    authors = record.get("authors")

    if not isinstance(authors, list):
        error(
            f"{context}.authors: expected list"
        )
    else:
        for person_index, author in enumerate(authors):
            validate_person(
                author,
                f"{context}.authors[{person_index}]"
            )

    # ---------------------------------------------------------
    # Publishers
    # ---------------------------------------------------------

    publishers = record.get("publishers")

    if not isinstance(publishers, list):
        error(
            f"{context}.publishers: expected list"
        )
    else:
        for person_index, publisher in enumerate(
            publishers
        ):
            validate_person(
                publisher,
                f"{context}.publishers[{person_index}]"
            )

    # ---------------------------------------------------------
    # Related names
    # ---------------------------------------------------------

    related_names = record.get("related_names")

    if not isinstance(related_names, list):
        error(
            f"{context}.related_names: expected list"
        )
    else:
        for person_index, person in enumerate(
            related_names
        ):
            validate_person(
                person,
                f"{context}.related_names[{person_index}]"
            )

    # ---------------------------------------------------------
    # Topics
    # ---------------------------------------------------------

    topics = record.get("topics")

    if not isinstance(topics, list):
        error(
            f"{context}.topics: expected list"
        )
    else:
        for topic_index, topic in enumerate(topics):
            if not isinstance(topic, str):
                error(
                    f"{context}.topics[{topic_index}]: "
                    f"expected string"
                )
            elif not topic.strip():
                error(
                    f"{context}.topics[{topic_index}]: "
                    f"empty topic"
                )

    # ---------------------------------------------------------
    # Record bibliography
    # ---------------------------------------------------------

    if "bibliography" in record:
        bibliography = record["bibliography"]

        if bibliography is not None and not isinstance(
            bibliography,
            str
        ):
            error(
                f"{context}.bibliography: expected "
                f"string or null"
            )

    # ---------------------------------------------------------
    # Linked open data
    # ---------------------------------------------------------

    if "linked_open_data" in record:
        validate_open_data(
            record["linked_open_data"],
            f"{context}.linked_open_data"
        )


def check_duplicate_titles(dataset):
    titles = []

    for record in dataset:
        title = record.get("title")

        if isinstance(title, str) and title.strip():
            titles.append(title.strip())

    counts = Counter(titles)

    for title, count in counts.items():
        if count > 1:
            warning(
                f"Duplicate title ({count} occurrences): "
                f"{title}"
            )


def check_duplicate_ids(dataset):
    ids = []

    for record in dataset:
        record_id = record.get("id")

        if isinstance(record_id, str):
            record_id = record_id.strip()

            if record_id:
                ids.append(record_id)

    counts = Counter(ids)

    duplicates = {
        record_id: count
        for record_id, count in counts.items()
        if count > 1
    }

    for record_id, count in sorted(
        duplicates.items()
    ):
        error(
            f"Duplicate stable ID: "
            f"{record_id} ({count} occurrences)"
        )

    return duplicates


def check_biography_date_ranges(dataset):
    """
    Detect obviously impossible date ranges in biographies.

    This is intentionally conservative.
    It reports warnings rather than errors.
    """

    date_pattern = re.compile(
        r"\b(1[0-9]{3}|20[0-9]{2})\b"
    )

    for index, record in enumerate(dataset):
        people_groups = [
            ("authors", record.get("authors", [])),
            ("publishers", record.get("publishers", [])),
            (
                "related_names",
                record.get("related_names", [])
            ),
        ]

        for group_name, people in people_groups:
            if not isinstance(people, list):
                continue

            for person_index, person in enumerate(people):
                if not isinstance(person, dict):
                    continue

                biography = person.get("biography")

                if not isinstance(biography, str):
                    continue

                years = [
                    int(year)
                    for year in date_pattern.findall(
                        biography
                    )
                ]

                if len(years) < 2:
                    continue

                first_year = years[0]
                second_year = years[1]

                if (
                    first_year >= 1000
                    and second_year >= 1000
                    and second_year < first_year
                ):
                    warning(
                        "Suspicious biography date range: "
                        f"record {index + 1}, "
                        f"{group_name}[{person_index}], "
                        f"{person.get('name', '')}: "
                        f"{first_year}–{second_year}"
                    )


def calculate_statistics(dataset):
    author_occurrences = 0
    publisher_occurrences = 0
    related_name_occurrences = 0
    topic_assignments = 0

    authors = set()
    publishers = set()
    related_names = set()
    topics = set()

    ids = set()

    for record in dataset:
        record_id = record.get("id")

        if isinstance(record_id, str):
            record_id = record_id.strip()

            if record_id:
                ids.add(record_id)

        for author in record.get("authors", []):
            author_occurrences += 1

            if isinstance(author, dict):
                name = author.get("name")

                if isinstance(name, str):
                    name = name.strip()

                    if name:
                        authors.add(name)

        for publisher in record.get("publishers", []):
            publisher_occurrences += 1

            if isinstance(publisher, dict):
                name = publisher.get("name")

                if isinstance(name, str):
                    name = name.strip()

                    if name:
                        publishers.add(name)

        for person in record.get("related_names", []):
            related_name_occurrences += 1

            if isinstance(person, dict):
                name = person.get("name")

                if isinstance(name, str):
                    name = name.strip()

                    if name:
                        related_names.add(name)

        for topic in record.get("topics", []):
            topic_assignments += 1

            if isinstance(topic, str):
                topic = topic.strip()

                if topic:
                    topics.add(topic)

    return {
        "records": len(dataset),
        "ids_present": len(ids),
        "unique_ids": len(ids),
        "author_occurrences": author_occurrences,
        "unique_authors": len(authors),
        "publisher_occurrences": publisher_occurrences,
        "unique_publishers": len(publishers),
        "related_name_occurrences": related_name_occurrences,
        "unique_related_names": len(related_names),
        "topic_assignments": topic_assignments,
        "unique_topics": len(topics),
    }


def main():
    global errors
    global warnings

    errors = []
    warnings = []

    # ---------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------

    if not DATASET_PATH.exists():
        error(
            f"Dataset not found: {DATASET_PATH}"
        )

        print_validation_result(
            statistics=None
        )

        sys.exit(1)

    try:
        with DATASET_PATH.open(
            "r",
            encoding="utf-8"
        ) as file:
            dataset = json.load(file)

    except json.JSONDecodeError as exc:
        error(
            f"Invalid JSON: {exc}"
        )

        print_validation_result(
            statistics=None
        )

        sys.exit(1)

    except Exception as exc:
        error(
            f"Unable to read dataset: {exc}"
        )

        print_validation_result(
            statistics=None
        )

        sys.exit(1)

    # ---------------------------------------------------------
    # Dataset must be a list
    # ---------------------------------------------------------

    if not isinstance(dataset, list):
        error(
            "Top-level dataset must be a list"
        )

        print_validation_result(
            statistics=None
        )

        sys.exit(1)

    # ---------------------------------------------------------
    # Validate every record
    # ---------------------------------------------------------

    for index, record in enumerate(dataset):
        validate_record(
            record,
            index
        )

    # ---------------------------------------------------------
    # Global checks
    # ---------------------------------------------------------

    check_duplicate_titles(dataset)

    duplicate_ids = check_duplicate_ids(dataset)

    check_biography_date_ranges(dataset)

    statistics = calculate_statistics(
        dataset
    )

    # ---------------------------------------------------------
    # Additional ID integrity checks
    # ---------------------------------------------------------

    if statistics["records"] != 146:
        warning(
            "Expected 146 records, found "
            f"{statistics['records']}"
        )

    if statistics["ids_present"] != statistics["records"]:
        error(
            "Not every record has a valid non-empty ID."
        )

    if statistics["unique_ids"] != statistics["ids_present"]:
        error(
            "Stable IDs are not unique."
        )

    if duplicate_ids:
        error(
            f"Found {len(duplicate_ids)} duplicated "
            f"stable ID(s)."
        )

    # ---------------------------------------------------------
    # Print result
    # ---------------------------------------------------------

    print_validation_result(
        statistics=statistics
    )

    if errors:
        sys.exit(1)


def print_validation_result(statistics):
    print()
    print("=" * 60)
    print("GOVI RARE BOOKS DATASET VALIDATION")
    print("=" * 60)
    print()

    if statistics is not None:
        print(
            f"Records: {statistics['records']}"
        )

        print(
            f"IDs present: "
            f"{statistics['ids_present']}"
        )

        print(
            f"Unique IDs: "
            f"{statistics['unique_ids']}"
        )

        print(
            f"Author occurrences: "
            f"{statistics['author_occurrences']}"
        )

        print(
            f"Unique authors: "
            f"{statistics['unique_authors']}"
        )

        print(
            f"Publisher occurrences: "
            f"{statistics['publisher_occurrences']}"
        )

        print(
            f"Unique publishers: "
            f"{statistics['unique_publishers']}"
        )

        print(
            f"Related-name occurrences: "
            f"{statistics['related_name_occurrences']}"
        )

        print(
            f"Unique related names: "
            f"{statistics['unique_related_names']}"
        )

        print(
            f"Topic assignments: "
            f"{statistics['topic_assignments']}"
        )

        print(
            f"Unique topics: "
            f"{statistics['unique_topics']}"
        )

        if statistics["records"] > 0:
            average_topics = (
                statistics["topic_assignments"]
                / statistics["records"]
            )

            print(
                f"Average topics per record: "
                f"{average_topics:.2f}"
            )

    print()

    print(
        f"Errors: {len(errors)}"
    )

    print(
        f"Warnings: {len(warnings)}"
    )

    if errors:
        print()
        print("ERRORS:")

        for message in errors:
            print(
                f"  ✗ {message}"
            )

    if warnings:
        print()
        print("WARNINGS:")

        for message in warnings:
            print(
                f"  ⚠ {message}"
            )

    print()

    if errors:
        print(
            "✗ Dataset FAILED validation."
        )
    else:
        print(
            "✓ Dataset passed all validation checks."
        )

    print(
        "=" * 60
    )
    print()


if __name__ == "__main__":
    main()
