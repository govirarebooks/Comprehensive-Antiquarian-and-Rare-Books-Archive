import sys

from discovery_engine import (
    VECTOR_MODEL_NAME,
    SentenceTransformer,
    cosine_similarity,
    get_record_id,
    get_topics,
    load_ai_index,
    load_dataset,
    normalize,
)


def get_people(record):
    people = set()

    for field in (
        "authors",
        "publishers",
        "related_names",
    ):
        value = record.get(
            field,
            [],
        )

        if not isinstance(value, list):
            continue

        for item in value:

            if isinstance(item, dict):

                name = item.get(
                    "name"
                )

                if name:
                    people.add(
                        normalize(name)
                    )

            elif isinstance(item, str):

                people.add(
                    normalize(item)
                )

    return {
        person
        for person in people
        if person
    }


def shared_topics(a, b):
    topics_a = {
        normalize(topic)
        for topic in get_topics(a)
    }

    topics_b = {
        normalize(topic)
        for topic in get_topics(b)
    }

    return (
        topics_a
        & topics_b
    )


def shared_people(a, b):
    return (
        get_people(a)
        & get_people(b)
    )


def build_reason(
    source,
    candidate,
    full_score,
    intellectual_score,
    bibliographic_score,
):
    reasons = []

    topics = shared_topics(
        source,
        candidate,
    )

    if topics:
        reasons.append(
            "shared topics: "
            + ", ".join(
                sorted(topics)
            )
        )

    people = shared_people(
        source,
        candidate,
    )

    if people:
        reasons.append(
            "shared people: "
            + ", ".join(
                sorted(people)
            )
        )

    if intellectual_score >= 0.60:
        reasons.append(
            "very strong intellectual-content similarity"
        )

    elif intellectual_score >= 0.48:
        reasons.append(
            "strong intellectual-content similarity"
        )

    elif intellectual_score >= 0.38:
        reasons.append(
            "related intellectual content"
        )

    if bibliographic_score >= 0.55:
        reasons.append(
            "strong bibliographic similarity"
        )

    if full_score >= 0.55:
        reasons.append(
            "strong overall semantic similarity"
        )

    return reasons


def main():
    if len(sys.argv) < 2:

        print(
            "Usage: python "
            "rag/similar_books.py BOOK_ID"
        )

        raise SystemExit(1)

    target_id = sys.argv[1]

    records = load_dataset()

    (
        ids,
        full_embeddings,
        intellectual_embeddings,
        bibliographic_embeddings,
    ) = load_ai_index()

    records_by_id = {
        get_record_id(record): record
        for record in records
    }

    source = records_by_id.get(
        target_id
    )

    if source is None:

        print(
            f"Book not found: {target_id}"
        )

        raise SystemExit(1)

    target_index = None

    for index, value in enumerate(ids):

        if str(value) == target_id:
            target_index = index
            break

    if target_index is None:

        print(
            "No AI embedding found for "
            "this book."
        )

        raise SystemExit(1)

    source_full = full_embeddings[
        target_index
    ]

    source_intellectual = (
        intellectual_embeddings[
            target_index
        ]
    )

    source_bibliographic = (
        bibliographic_embeddings[
            target_index
        ]
    )

    full_scores = cosine_similarity(
        full_embeddings,
        source_full,
    )

    intellectual_scores = cosine_similarity(
        intellectual_embeddings,
        source_intellectual,
    )

    bibliographic_scores = cosine_similarity(
        bibliographic_embeddings,
        source_bibliographic,
    )

    ranked = []

    for index, value in enumerate(ids):

        candidate_id = str(value)

        if candidate_id == target_id:
            continue

        candidate = records_by_id.get(
            candidate_id
        )

        if candidate is None:
            continue

        full_score = float(
            full_scores[index]
        )

        intellectual_score = float(
            intellectual_scores[index]
        )

        bibliographic_score = float(
            bibliographic_scores[index]
        )

        topics = shared_topics(
            source,
            candidate,
        )

        people = shared_people(
            source,
            candidate,
        )

        topic_bonus = min(
            0.15,
            len(topics) * 0.05,
        )

        people_bonus = min(
            0.10,
            len(people) * 0.05,
        )

        final_score = (
            full_score * 0.25
            + intellectual_score * 0.55
            + bibliographic_score * 0.20
            + topic_bonus
            + people_bonus
        )

        ranked.append(
            {
                "record": candidate,
                "final_score": final_score,
                "full_score": full_score,
                "intellectual_score":
                    intellectual_score,
                "bibliographic_score":
                    bibliographic_score,
                "topics": topics,
                "people": people,
            }
        )

    ranked.sort(
        key=lambda item: (
            item["final_score"],
            item["intellectual_score"],
            item["full_score"],
        ),
        reverse=True,
    )

    print()
    print("=" * 80)
    print("GOVI AI — SIMILAR BOOKS")
    print("=" * 80)

    print()
    print(
        f"Source book: "
        f"{source.get('title', '')}"
    )

    print(
        f"ID: {target_id}"
    )

    print()
    print(
        "Similar books:"
    )

    for number, item in enumerate(
        ranked[:10],
        start=1,
    ):

        candidate = item["record"]

        reasons = build_reason(
            source,
            candidate,
            item["full_score"],
            item["intellectual_score"],
            item["bibliographic_score"],
        )

        print()
        print(
            f"{number}. "
            f"{candidate.get('title', '')}"
        )

        print(
            "   Final score: "
            f"{item['final_score']:.4f}"
        )

        print(
            "   Full: "
            f"{item['full_score']:.4f}"
        )

        print(
            "   Intellectual: "
            f"{item['intellectual_score']:.4f}"
        )

        print(
            "   Bibliographic: "
            f"{item['bibliographic_score']:.4f}"
        )

        if reasons:
            print(
                "   Why: "
                + "; ".join(reasons)
            )

        print(
            "   Topics: "
            + ", ".join(
                get_topics(candidate)
            )
        )

        print(
            "   ID: "
            f"{candidate.get('id')}"
        )


if __name__ == "__main__":
    main()
