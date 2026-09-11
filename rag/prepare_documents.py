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
        documents.append({
            "id": f"work-{index + 1:04d}",
            "type": "work",
            "title": record.get("title", ""),
            "text": build_work_document(record),
            "metadata": {
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
