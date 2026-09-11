import json
import re
import sys
from pathlib import Path
from collections import Counter


ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "govi-rare-books-academic-dataset.json"

EXPECTED_RECORDS = 146

# Stable IDs exported by the Govi management system.
# Current IDs are ULIDs.
ULID_PATTERN = re.compile(r"^[0-7][0-9A-HJKMNP-TV-Z]{25}$")


REQUIRED_RECORD_FIELDS = [
    "id",
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
    "publishers",
    "related_names",
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


def add_error(errors, message):
    errors.append(message)


def add_warning(warnings, message):
    warnings.append(message)


def validate_person(person, path, errors):
    if not isinstance(person, dict):
        add_error(errors, f"{path}: expected object")
        return

    for field in PERSON_FIELDS:
        if field not in person:
            add_error(errors, f"{path}: missing field '{field}'")

    # name
    if "name" in person and not isinstance(person["name"], str):
        add_error(errors, f"{path}.name: expected string")

    # biography
    if "biography" in person and not isinstance(person["biography"], str):
        add_error(errors, f"{path}.biography: expected string")

    # biographical_data
    #
    # This is a structured editorial field, but the source system contains
    # several legitimate formats, for example:
    #
    #   1602-1676
    #   (Roma, 1521 – Venezia, 28 settembre 1583)
    #   ca. 1520-1591
    #   fl. 1503-1544
    #   b. 1574
    #   43 BC - 17/18 AD
    #   Salerno, 1700 - Napoli, 1740
    #
    # Therefore we deliberately validate only the type here and do not
    # impose a single date syntax.
    if "biographical_data" in person:
        if not isinstance(person["biographical_data"], str):
            add_error(errors, f"{path}.biographical_data: expected string")

    # bibliography
    if "bibliography" in person and not isinstance(person["bibliography"], str):
        add_error(errors, f"{path}.bibliography: expected string")

    # links
    if "links" in person:
        links = person["links"]

        if not isinstance(links, dict):
            add_error(errors, f"{path}.links: expected object")
        else:
            for field in LINK_FIELDS:
                if field not in links:
                    add_error(errors, f"{path}.links: missing field '{field}'")

            for field in LINK_FIELDS:
                if field in links and not isinstance(links[field], str):
                    add_error(
                        errors,
                        f"{path}.links.{field}: expected string",
                    )


def validate_record(record, index, errors, warnings):
    path = f"Record {index}"

    if not isinstance(record, dict):
        add_error(errors, f"{path}: expected object")
        return

    # ------------------------------------------------------------------
    # Required fields
    # ------------------------------------------------------------------

    for field in REQUIRED_RECORD_FIELDS:
        if field not in record:
            add_error(errors, f"{path}: missing field '{field}'")

    # ------------------------------------------------------------------
    # Stable ID
    # ------------------------------------------------------------------

    if "id" in record:
        record_id = record["id"]

        if not isinstance(record_id, str):
            add_error(errors, f"{path}.id: expected string")

        elif not record_id.strip():
            add_error(errors, f"{path}.id: ID is empty")

        elif not ULID_PATTERN.fullmatch(record_id):
            add_error(
                errors,
                f"{path}.id: invalid ULID format: '{record_id}'",
            )

    # ------------------------------------------------------------------
    # Basic record fields
    # ------------------------------------------------------------------

    string_fields = [
        "title",
        "publication_place",
        "publisher_data",
        "academic_description",
        "bibliography",
        "source_url",
        "institution",
    ]

    for field in string_fields:
        if field in record and not isinstance(record[field], str):
            add_error(errors, f"{path}.{field}: expected string")

    # ------------------------------------------------------------------
    # Publication year
    # ------------------------------------------------------------------

    if "publication_year" in record:
        year = record["publication_year"]

        if year is not None and not isinstance(year, (int, str)):
            add_error(
                errors,
                f"{path}.publication_year: expected integer, string, or null",
            )

    # ------------------------------------------------------------------
    # Authors
    # ------------------------------------------------------------------

    if "authors" in record:
        authors = record["authors"]

        if not isinstance(authors, list):
            add_error(errors, f"{path}.authors: expected list")
        else:
            for author_index, author in enumerate(authors):
                validate_person(
                    author,
                    f"{path}.authors[{author_index}]",
                    errors,
                )

    # ------------------------------------------------------------------
    # Publishers
    # ------------------------------------------------------------------

    if "publishers" in record:
        publishers = record["publishers"]

        if not isinstance(publishers, list):
            add_error(errors, f"{path}.publishers: expected list")
        else:
            for publisher_index, publisher in enumerate(publishers):
                validate_person(
                    publisher,
                    f"{path}.publishers[{publisher_index}]",
                    errors,
                )

    # ------------------------------------------------------------------
    # Related names
    # ------------------------------------------------------------------

    if "related_names" in record:
        related_names = record["related_names"]

        if not isinstance(related_names, list):
            add_error(errors, f"{path}.related_names: expected list")
        else:
            for related_index, related_name in enumerate(related_names):
                validate_person(
                    related_name,
                    f"{path}.related_names[{related_index}]",
                    errors,
                )

    # ------------------------------------------------------------------
    # Topics
    # ------------------------------------------------------------------

    if "topics" in record:
        topics = record["topics"]

        if not isinstance(topics, list):
            add_error(errors, f"{path}.topics: expected list")
        else:
            for topic_index, topic in enumerate(topics):
                if not isinstance(topic, str):
                    add_error(
                        errors,
                        f"{path}.topics[{topic_index}]: expected string",
                    )

    # ------------------------------------------------------------------
    # Linked open data
    # ------------------------------------------------------------------

    if "linked_open_data" in record:
        linked_data = record["linked_open_data"]

        if not isinstance(linked_data, dict):
            add_error(
                errors,
                f"{path}.linked_open_data: expected object",
            )
        else:
            for field in OPEN_DATA_FIELDS:
                if field not in linked_data:
                    add_error(
                        errors,
                        f"{path}.linked_open_data: missing field '{field}'",
                    )

            for field in OPEN_DATA_FIELDS:
                if field in linked_data and not isinstance(
                    linked_data[field], str
                ):
                    add_error(
                        errors,
                        f"{path}.linked_open_data.{field}: expected string",
                    )

    # ------------------------------------------------------------------
    # Warnings
    # ------------------------------------------------------------------

    # Empty title is a warning rather than a hard schema error.
    if isinstance(record.get("title"), str):
        if not record["title"].strip():
            add_warning(
                warnings,
                f"{path}.title: empty title",
            )


def main():
    errors = []
    warnings = []

    print("=" * 60)
    print("GOVI RARE BOOKS DATASET VALIDATION")
    print("=" * 60)
    print()

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------

    if not DATASET_PATH.exists():
        print(f"ERROR: Dataset not found: {DATASET_PATH}")
        sys.exit(1)

    try:
        with DATASET_PATH.open("r", encoding="utf-8") as file:
            dataset = json.load(file)
    except json.JSONDecodeError as exc:
        print(f"ERROR: Invalid JSON: {exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"ERROR: Could not read dataset: {exc}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Top-level structure
    # ------------------------------------------------------------------

    if not isinstance(dataset, list):
        print("ERROR: Dataset root must be a JSON list.")
        sys.exit(1)

    record_count = len(dataset)

    if record_count != EXPECTED_RECORDS:
        add_warning(
            warnings,
            f"Expected {EXPECTED_RECORDS} records, found {record_count}",
        )

    # ------------------------------------------------------------------
    # Counters
    # ------------------------------------------------------------------

    ids = []

    author_occurrences = 0
    publisher_occurrences = 0
    related_name_occurrences = 0
    topic_assignments = 0

    authors = []
    publishers = []
    related_names = []
    topics = []

    # ------------------------------------------------------------------
    # Validate records
    # ------------------------------------------------------------------

    for index, record in enumerate(dataset, start=1):
        validate_record(
            record,
            index,
            errors,
            warnings,
        )

        if not isinstance(record, dict):
            continue

        # IDs
        record_id = record.get("id")

        if isinstance(record_id, str) and record_id.strip():
            ids.append(record_id)

        # Authors
        record_authors = record.get("authors", [])

        if isinstance(record_authors, list):
            author_occurrences += len(record_authors)

            for author in record_authors:
                if isinstance(author, dict):
                    name = author.get("name")

                    if isinstance(name, str) and name.strip():
                        authors.append(name.strip())

        # Publishers
        record_publishers = record.get("publishers", [])

        if isinstance(record_publishers, list):
            publisher_occurrences += len(record_publishers)

            for publisher in record_publishers:
                if isinstance(publisher, dict):
                    name = publisher.get("name")

                    if isinstance(name, str) and name.strip():
                        publishers.append(name.strip())

        # Related names
        record_related_names = record.get("related_names", [])

        if isinstance(record_related_names, list):
            related_name_occurrences += len(record_related_names)

            for related_name in record_related_names:
                if isinstance(related_name, dict):
                    name = related_name.get("name")

                    if isinstance(name, str) and name.strip():
                        related_names.append(name.strip())

        # Topics
        record_topics = record.get("topics", [])

        if isinstance(record_topics, list):
            topic_assignments += len(record_topics)

            for topic in record_topics:
                if isinstance(topic, str) and topic.strip():
                    topics.append(topic.strip())

    # ------------------------------------------------------------------
    # Duplicate IDs
    # ------------------------------------------------------------------

    id_counts = Counter(ids)

    duplicate_ids = {
        record_id: count
        for record_id, count in id_counts.items()
        if count > 1
    }

    for record_id, count in duplicate_ids.items():
        add_error(
            errors,
            f"Duplicate stable ID '{record_id}' appears {count} times",
        )

    # ------------------------------------------------------------------
    # Duplicate titles
    # ------------------------------------------------------------------

    titles = []

    for record in dataset:
        if isinstance(record, dict):
            title = record.get("title")

            if isinstance(title, str) and title.strip():
                titles.append(title.strip())

    title_counts = Counter(titles)

    duplicate_titles = {
        title: count
        for title, count in title_counts.items()
        if count > 1
    }

    for title, count in duplicate_titles.items():
        add_warning(
            warnings,
            f"Duplicate title appears {count} times: {title}",
        )

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    unique_ids = len(set(ids))
    unique_authors = len(set(authors))
    unique_publishers = len(set(publishers))
    unique_related_names = len(set(related_names))
    unique_topics = len(set(topics))

    average_topics = (
        topic_assignments / record_count
        if record_count
        else 0
    )

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    print(f"Records: {record_count}")
    print(f"IDs present: {len(ids)}")
    print(f"Unique IDs: {unique_ids}")
    print(f"Author occurrences: {author_occurrences}")
    print(f"Unique authors: {unique_authors}")
    print(f"Publisher occurrences: {publisher_occurrences}")
    print(f"Unique publishers: {unique_publishers}")
    print(f"Related-name occurrences: {related_name_occurrences}")
    print(f"Unique related names: {unique_related_names}")
    print(f"Topic assignments: {topic_assignments}")
    print(f"Unique topics: {unique_topics}")
    print(f"Average topics per record: {average_topics:.2f}")
    print(f"Errors: {len(errors)}")
    print(f"Warnings: {len(warnings)}")
    print()

    # ------------------------------------------------------------------
    # Errors
    # ------------------------------------------------------------------

    if errors:
        print("ERRORS:")
        for error in errors:
            print(f"  ✗ {error}")
        print()

    # ------------------------------------------------------------------
    # Warnings
    # ------------------------------------------------------------------

    if warnings:
        print("WARNINGS:")
        for warning in warnings:
            print(f"  ⚠ {warning}")
        print()

    # ------------------------------------------------------------------
    # Final result
    # ------------------------------------------------------------------

    if errors:
        print("✗ Dataset validation failed.")
        sys.exit(1)

    print("✓ Dataset passed all validation checks.")


if __name__ == "__main__":
    main()
