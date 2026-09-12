import json
import sys
import re
import unicodedata
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = ROOT / "govi-rare-books-academic-dataset.json"
CHUNK_INDEX_PATH = ROOT / "rag" / "chunk_index.npz"
CHUNK_METADATA_PATH = ROOT / "rag" / "chunk_metadata.json"

MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

CHUNK_CANDIDATES = 20
FINAL_RESULTS = 10


def normalize(text):
    if text is None:
        return ""

    text = str(text).lower()

    text = unicodedata.normalize(
        "NFKD",
        text,
    )

    text = "".join(
        char
        for char in text
        if not unicodedata.combining(char)
    )

    text = text.replace("’", "'")

    text = re.sub(
        r"[^\w\s]",
        " ",
        text,
        flags=re.UNICODE,
    )

    return " ".join(
        text.split()
    )


def tokenize(text):
    return {
        token
        for token in normalize(text).split()
        if len(token) >= 3
    }


def load_dataset():
    with DATASET_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(
            "Master dataset must be a JSON list."
        )

    return data


def load_chunks():
    data = np.load(
        CHUNK_INDEX_PATH,
        allow_pickle=True,
    )

    embeddings = data[
        "embeddings"
    ]

    with CHUNK_METADATA_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        metadata = json.load(file)

    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True,
    )

    norms[norms == 0] = 1.0

    embeddings = (
        embeddings / norms
    )

    return embeddings, metadata


def get_topics(record):
    topics = record.get(
        "topics",
        [],
    )

    if not isinstance(topics, list):
        return []

    return [
        str(topic)
        for topic in topics
        if topic
    ]


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


def record_text(record):
    fields = []

    for field in (
        "title",
        "academic_description",
        "bibliography",
        "publication_place",
        "publisher_data",
    ):

        value = record.get(field)

        if value:
            fields.append(
                str(value)
            )

    return " ".join(fields)


def lexical_score(query, record):
    query_tokens = tokenize(query)

    if not query_tokens:
        return 0.0

    record_tokens = tokenize(
        record_text(record)
    )

    overlap = (
        query_tokens
        & record_tokens
    )

    return (
        len(overlap)
        / len(query_tokens)
    )


def topic_score(record, query_topics):
    if not query_topics:
        return 0.0

    record_topics = {
        normalize(topic)
        for topic in get_topics(record)
    }

    requested = {
        normalize(topic)
        for topic in query_topics
    }

    if not requested:
        return 0.0

    return (
        len(
            record_topics
            & requested
        )
        / len(requested)
    )


def people_score(
    record,
    related_people,
):
    if not related_people:
        return 0.0

    overlap = (
        get_people(record)
        & {
            normalize(name)
            for name in related_people
        }
    )

    return min(
        1.0,
        len(overlap) / 2.0,
    )


def retrieve_chunks(
    query,
    model,
    chunk_embeddings,
    chunk_metadata,
):
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )[0]

    similarities = (
        chunk_embeddings
        @ query_embedding
    )

    indices = np.argsort(
        similarities
    )[::-1][
        :CHUNK_CANDIDATES
    ]

    return [
        {
            "score": float(
                similarities[index]
            ),
            "metadata":
                chunk_metadata[index],
        }
        for index in indices
    ]


def discover_query_signals(
    query,
    records,
    model,
):
    """
    Use the dataset's own topic vocabulary as a semantic
    discovery layer. This deliberately remains independent
    from any fixed keyword dictionary.
    """

    topics = sorted(
        {
            topic
            for record in records
            for topic in get_topics(record)
        }
    )

    if not topics:
        return [], []

    embeddings = model.encode(
        [
            f"Bibliographic subject: {topic}"
            for topic in topics
        ],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )[0]

    scores = (
        embeddings
        @ query_embedding
    )

    ranked = np.argsort(
        scores
    )[::-1]

    discovered = [
        {
            "topic": topics[index],
            "score": float(
                scores[index]
            ),
        }
        for index in ranked[:8]
    ]

    core = [
        item["topic"]
        for item in discovered
        if item["score"] >= max(
            0.55,
            discovered[0]["score"] - 0.08,
        )
    ][:3]

    related = [
        item["topic"]
        for item in discovered
        if (
            item["score"] >= max(
                0.44,
                discovered[0]["score"] - 0.18,
            )
            and item["topic"] not in core
        )
    ][:4]

    return core, related


def rerank(
    query,
    records,
    model,
):
    (
        chunk_embeddings,
        chunk_metadata,
    ) = load_chunks()

    core_topics, related_topics = (
        discover_query_signals(
            query,
            records,
            model,
        )
    )

    query_topics = (
        core_topics
        + related_topics
    )

    chunk_results = retrieve_chunks(
        query,
        model,
        chunk_embeddings,
        chunk_metadata,
    )

    records_by_id = {
        str(record.get("id")): record
        for record in records
        if record.get("id") is not None
    }

    grouped = {}

    for item in chunk_results:

        chunk = item["metadata"]

        book_id = str(
            chunk["book_id"]
        )

        grouped.setdefault(
            book_id,
            [],
        ).append(
            {
                "score": item["score"],
                "text": chunk["text"],
            }
        )

    ranked = []

    for book_id, matches in grouped.items():

        record = records_by_id.get(
            book_id
        )

        if record is None:
            continue

        matches.sort(
            key=lambda item:
                item["score"],
            reverse=True,
        )

        best_chunk = matches[0][
            "score"
        ]

        second_chunk = (
            matches[1]["score"]
            if len(matches) > 1
            else best_chunk
        )

        # Evidence from more than one passage is stronger.
        chunk_evidence = (
            best_chunk * 0.70
            + second_chunk * 0.30
        )

        core_score = topic_score(
            record,
            core_topics,
        )

        related_score = topic_score(
            record,
            related_topics,
        )

        lexical = lexical_score(
            query,
            record,
        )

        # Strong evidence from the description.
        final_score = (
            chunk_evidence * 0.50
            + core_score * 0.25
            + related_score * 0.10
            + lexical * 0.05
        )

        # ----------------------------------------------------
        # BIBLIOGRAPHIC GATES
        # ----------------------------------------------------

        if core_topics:

            if core_score == 0:

                if related_score == 0:
                    # Only admit a topic-free result if the
                    # description itself is exceptionally strong.
                    if chunk_evidence < 0.52:
                        continue

                else:

                    if chunk_evidence < 0.42:
                        continue

        ranked.append(
            {
                "record": record,
                "score": final_score,
                "chunk_evidence":
                    chunk_evidence,
                "core_topic_score":
                    core_score,
                "related_topic_score":
                    related_score,
                "lexical_score":
                    lexical,
                "matching_chunks":
                    matches[:3],
            }
        )

    ranked.sort(
        key=lambda item: (
            item["score"],
            item["chunk_evidence"],
            item["core_topic_score"],
        ),
        reverse=True,
    )

    return {
        "core_topics": core_topics,
        "related_topics": related_topics,
        "results": ranked[
            :FINAL_RESULTS
        ],
    }


def main():
    if len(sys.argv) < 2:

        print(
            'Usage: python '
            'rag/bibliographic_reranker.py '
            '"query"'
        )

        raise SystemExit(1)

    query = " ".join(
        sys.argv[1:]
    ).strip()

    records = load_dataset()

    model = SentenceTransformer(
        MODEL_NAME
    )

    response = rerank(
        query,
        records,
        model,
    )

    print()
    print("=" * 80)
    print(
        "GOVI AI — BIBLIOGRAPHIC "
        "RERANKER"
    )
    print("=" * 80)

    print()
    print(
        f'Query: "{query}"'
    )

    print()
    print("Core topics:")

    for topic in response[
        "core_topics"
    ]:
        print(
            f"  - {topic}"
        )

    print()
    print("Related topics:")

    for topic in response[
        "related_topics"
    ]:
        print(
            f"  - {topic}"
        )

    print()
    print(
        f"Results: "
        f"{len(response['results'])}"
    )

    for index, item in enumerate(
        response["results"],
        start=1,
    ):

        record = item[
            "record"
        ]

        print()
        print(
            f"{index}. "
            f"{record.get('title', '')}"
        )

        print(
            "   Final: "
            f"{item['score']:.4f}"
        )

        print(
            "   Chunk evidence: "
            f"{item['chunk_evidence']:.4f}"
        )

        print(
            "   Core topic: "
            f"{item['core_topic_score']:.4f}"
        )

        print(
            "   Related topic: "
            f"{item['related_topic_score']:.4f}"
        )

        print(
            "   Lexical: "
            f"{item['lexical_score']:.4f}"
        )

        print(
            "   ID: "
            f"{record.get('id')}"
        )

        for chunk in item[
            "matching_chunks"
        ]:

            print()
            print(
                "   Evidence:"
            )
            print(
                "   "
                + chunk["text"][:800]
                .replace("\n", " ")
            )


if __name__ == "__main__":
    main()
