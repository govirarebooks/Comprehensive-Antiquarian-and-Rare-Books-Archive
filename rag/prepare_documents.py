import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "govi-rare-books-academic-dataset.json"


def person_to_text(person):
    parts = []

    if person.get("name"):
        parts.append(f"Name: {person['name']}")

    if person.get("biographical_data"):
        parts.append(
            f"Biographical data: {person['biographical_data']}"
        )

    if person.get("biography"):
        parts.append(
            f"Biography: {person['biography']}"
        )

    if person.get("bibliography"):
        parts.append(
            f"Bibliography: {person['bibliography']}"
        )

    return "\n".join(parts)


def build_work_document(record):
    sections = []

    sections.append(
        f"TITLE\n{record.get('title', '')}"
    )

    authors = record.get("authors", [])
    if authors:
        sections.append(
            "AUTHORS\n" +
            "\n\n".join(
                person_to_text(author)
                for author in authors
            )
        )

    publishers = record.get("publishers", [])
    if publishers:
        sections.append(
            "PUBLISHERS\n" +
            "\n\n".join(
                person_to_text(publisher)
                for publisher in publishers
            )
        )

    related_names = record.get("related_names", [])
    if related_names:
        sections.append(
            "RELATED NAMES\n" +
            "\n\n".join(
                person_to_text(person)
                for person in related_names
            )
        )

    if record.get("publication_year") is not None:
        sections.append(
            f"PUBLICATION YEAR\n"
            f"{record['publication_year']}"
        )

    if record.get("publication_place"):
        sections.append(
            f"PUBLICATION PLACE\n"
            f"{record['publication_place']}"
        )

    if record.get("publisher_data"):
        sections.append(
            f"PUBLISHER DATA\n"
            f"{record['publisher_data']}"
        )

    topics = record.get("topics", [])
    if topics:
        sections.append(
            "TOPICS\n" +
            "\n".join(
                f"- {topic}"
                for topic in topics
            )
        )

    if record.get("academic_description"):
        sections.append(
            "ACADEMIC DESCRIPTION\n" +
            record["academic_description"]
        )

    if record.get("bibliography"):
        sections.append(
            "BIBLIOGRAPHY\n" +
            record["bibliography"]
        )

    if record.get("source_url"):
        sections.append(
            f"SOURCE\n{record['source_url']}"
        )

    return "\n\n".join(sections)


def main():
    with DATASET_PATH.open(
        "r",
        encoding="utf-8"
    ) as file:
        dataset = json.load(file)

    documents = []

    for index, record in enumerate(dataset):

        # The stable identifier must come from the
        # original stock.id in the management system.
        record_id = record.get("id")

        if record_id is None:
            raise ValueError(
                f"Record {index + 1} has no stable 'id' field."
            )

        document_id = f"work-{record_id}"

        text = build_work_document(record)

        # Stable hash of the document content.
        # Used later by the vector index to determine
        # whether an embedding must be recalculated.
        content_hash = hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest()

        documents.append({
            "id": document_id,

            "type": "work",

            "title":
                record.get("title", ""),

            "text":
                text,

            "content_hash":
                content_hash,

            "metadata": {
                "source_id":
                    record_id,

                "publication_year":
                    record.get("publication_year"),

                "publication_place":
                    record.get("publication_place"),

                "topics":
                    record.get("topics", []),

                "source_url":
                    record.get("source_url")
            }
        })

    output_path = (
        ROOT /
        "rag" /
        "documents.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            documents,
            file,
            ensure_ascii=False,
            indent=2
        )

    print(
        f"Prepared {len(documents)} RAG documents."
    )

    print(
        f"Output: {output_path}"
    )


if __name__ == "__main__":
    main()
