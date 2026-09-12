import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = ROOT / "govi-rare-books-academic-dataset.json"
CHUNK_INDEX_PATH = ROOT / "rag" / "chunk_index.npz"
CHUNK_METADATA_PATH = ROOT / "rag" / "chunk_metadata.json"
AI_INDEX_PATH = ROOT / "rag" / "ai_index.npz"

MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

CANDIDATE_LIMIT = 30
FINAL_LIMIT = 10
TOP_CHUNKS_PER_BOOK = 3


def normalize(text):
    if text is None:
        return ""

    return " ".join(
        str(text).lower().split()
    )


def load_dataset():
    with DATASET_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(
            "Dataset must be a JSON list."
        )

    return data


def load_ai_index():
    data = np.load(
        AI_INDEX_PATH,
        allow_pickle=True,
    )

    return {
        "ids": data["ids"],
        "full": data["embedding_full"],
        "intellectual": data["embedding_intellectual"],
        "bibliographic": data["embedding_bibliographic"],
    }


def load_chunks():
    data = np.load(
        CHUNK_INDEX_PATH,
        allow_pickle=True,
    )

    with CHUNK_METADATA_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    embeddings = data["embeddings"]

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
    value = record.get(
        "topics",
        [],
    )

    if not isinstance(value, list):
        return []

    return [
        str(item)
        for item in value
        if item
    ]


def topic_keys(record):
    return {
        normalize(item)
        for item in get_topics(record)
    }


def build_topic_embeddings(
    model,
    records,
):
    topics = sorted(
        {
            topic
            for record in records
            for topic in get_topics(record)
        }
    )

    if not topics:
        return [], None

    embeddings = model.encode(
        [
            f"Bibliographic subject: {topic}"
            for topic in topics
        ],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    return topics, embeddings


def semantic_score(matrix, query):
    return matrix @ query


def main():
    if len(sys.argv) < 2:
        print(
            'Usage: python rag/candidate_discovery.py "query"'
        )
        raise SystemExit(1)

    query = " ".join(
        sys.argv[1:]
    ).strip()

    records = load_dataset()

    record_by_id = {
        str(record["id"]): record
        for record in records
        if record.get("id") is not None
    }

    ai = load_ai_index()

    (
        chunk_embeddings,
        chunk_metadata,
    ) = load_chunks()

    model = SentenceTransformer(
        MODEL_NAME
    )

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )[0]

    # --------------------------------------------------------
    # 1. QUERY -> TOPICS
    # --------------------------------------------------------

    topics, topic_embeddings = (
        build_topic_embeddings(
            model,
            records,
        )
    )

    if topics:

        topic_scores = (
            topic_embeddings
            @ query_embedding
        )

        ranked_topics = np.argsort(
            topic_scores
        )[::-1]

        discovered_topics = [
            {
                "topic": topics[index],
                "score": float(
                    topic_scores[index]
                ),
            }
            for index in ranked_topics[:8]
        ]

    else:
        discovered_topics = []

    core_topics = [
        item["topic"]
        for item in discovered_topics
        if item["score"] >= max(
            0.55,
            discovered_topics[0]["score"] - 0.08,
        )
    ][:3]

    related_topics = [
        item["topic"]
        for item in discovered_topics
        if (
            item["score"] >= max(
                0.44,
                discovered_topics[0]["score"] - 0.18,
            )
            and item["topic"] not in core_topics
        )
    ][:4]

    core_keys = {
        normalize(topic)
        for topic in core_topics
    }

    # Related topics must remain semantically close to the
    # actual query; a merely adjacent catalog concept such
    # as "Heretical Books" must not automatically become a
    # relevance bridge.
    related_keys = {
        normalize(topic)
        for topic in related_topics
        if next(
            (
                item["score"]
                for item in discovered_topics
                if item["topic"] == topic
            ),
            0.0,
        ) >= 0.48
    }

    # --------------------------------------------------------
    # 2. WHOLE-BOOK SEMANTIC SIGNAL
    # --------------------------------------------------------

    full_scores = (
        ai["full"]
        @ query_embedding
    )

    intellectual_scores = (
        ai["intellectual"]
        @ query_embedding
    )

    bibliographic_scores = (
        ai["bibliographic"]
        @ query_embedding
    )

    # --------------------------------------------------------
    # 3. BUILD CANDIDATE POOL
    # --------------------------------------------------------

    candidates = {}

    for index, value in enumerate(
        ai["ids"]
    ):

        record_id = str(value)
        record = record_by_id.get(
            record_id
        )

        if record is None:
            continue

        topics_for_book = topic_keys(
            record
        )

        core_overlap = len(
            topics_for_book
            & core_keys
        )

        related_overlap = len(
            topics_for_book
            & related_keys
        )

        whole_score = float(
            full_scores[index]
        )

        intellectual_score = float(
            intellectual_scores[index]
        )

        bibliographic_score = float(
            bibliographic_scores[index]
        )

        # Candidate score is deliberately broad.
        candidate_score = (
            whole_score * 0.45
            + intellectual_score * 0.30
            + bibliographic_score * 0.10
            + min(
                1.0,
                core_overlap / 2.0,
            ) * 0.10
            + min(
                1.0,
                related_overlap / 2.0,
            ) * 0.05
        )

        # Never discard a bibliographically relevant topic match
        # at the candidate-pool stage.
        #
        # A related topic such as "Occult Sciences" may be
        # exactly the bridge needed to discover a book whose
        # explicit core topic is different, e.g. Chiromancy.
        if (
            core_overlap > 0
            or related_overlap > 0
            or candidate_score >= 0.34
        ):
            candidates[record_id] = {
                "record": record,
                "candidate_score":
                    candidate_score,
                "core_overlap":
                    core_overlap,
                "related_overlap":
                    related_overlap,
                "whole_score":
                    whole_score,
                "intellectual_score":
                    intellectual_score,
                "bibliographic_score":
                    bibliographic_score,
            }

    candidate_list = sorted(
        candidates.values(),
        key=lambda item: (
            item["candidate_score"],
            item["core_overlap"],
            item["intellectual_score"],
        ),
        reverse=True,
    )[:CANDIDATE_LIMIT]

    # --------------------------------------------------------
    # 4. QUERY -> BEST CHUNKS WITHIN EACH CANDIDATE
    # --------------------------------------------------------

    chunks_by_book = {}

    for index, item in enumerate(
        chunk_metadata
    ):

        book_id = str(
            item["book_id"]
        )

        if book_id not in candidates:
            continue

        score = float(
            chunk_embeddings[index]
            @ query_embedding
        )

        chunks_by_book.setdefault(
            book_id,
            [],
        ).append(
            {
                "score": score,
                "text": item["text"],
                "chunk_index":
                    item["chunk_index"],
            }
        )

    # --------------------------------------------------------
    # 5. FINAL BOOK RERANKING
    # --------------------------------------------------------

    final_results = []

    for candidate in candidate_list:

        record = candidate[
            "record"
        ]

        record_id = str(
            record["id"]
        )

        matches = sorted(
            chunks_by_book.get(
                record_id,
                [],
            ),
            key=lambda item:
                item["score"],
            reverse=True,
        )[
            :TOP_CHUNKS_PER_BOOK
        ]

        if matches:

            best_chunk = matches[0][
                "score"
            ]

            support = (
                sum(
                    item["score"]
                    for item in matches
                )
                / len(matches)
            )

        else:

            best_chunk = 0.0
            support = 0.0

        # Evidence inside this specific book.
        chunk_evidence = (
            best_chunk * 0.70
            + support * 0.30
        )

        core_bonus = min(
            1.0,
            candidate["core_overlap"]
            / 2.0,
        )

        related_bonus = min(
            1.0,
            candidate["related_overlap"]
            / 2.0,
        )

        final_score = (
            chunk_evidence * 0.45
            + candidate["intellectual_score"] * 0.20
            + candidate["whole_score"] * 0.10
            + candidate["bibliographic_score"] * 0.05
            + core_bonus * 0.15
            + related_bonus * 0.05
        )

        # Strong relevance gates.
        if core_topics:

            if (
                candidate["core_overlap"] == 0
                and candidate["related_overlap"] == 0
                and chunk_evidence < 0.54
            ):
                continue

            if (
                candidate["core_overlap"] == 0
                and candidate["related_overlap"] > 0
                and chunk_evidence < 0.30
            ):
                continue

        final_results.append(
            {
                **candidate,
                "final_score":
                    final_score,
                "chunk_evidence":
                    chunk_evidence,
                "matching_chunks":
                    matches,
            }
        )

    final_results.sort(
        key=lambda item: (
            item["final_score"],
            item["chunk_evidence"],
            item["core_overlap"],
        ),
        reverse=True,
    )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "GOVI — CANDIDATE POOL "
        "BIBLIOGRAPHIC DISCOVERY"
    )
    print("=" * 80)

    print()
    print(
        f'Query: "{query}"'
    )

    print()
    print("Core topics:")

    for topic in core_topics:
        print(
            f"  - {topic}"
        )

    print()
    print("Related topics:")

    for topic in related_topics:
        print(
            f"  - {topic}"
        )

    print()
    print(
        f"Candidate pool: "
        f"{len(candidate_list)}"
    )

    print(
        f"Final results: "
        f"{len(final_results[:FINAL_LIMIT])}"
    )

    for number, item in enumerate(
        final_results[:FINAL_LIMIT],
        start=1,
    ):

        record = item[
            "record"
        ]

        print()
        print(
            f"{number}. "
            f"{record.get('title', '')}"
        )

        print(
            f"   Final: "
            f"{item['final_score']:.4f}"
        )

        print(
            f"   Chunk evidence: "
            f"{item['chunk_evidence']:.4f}"
        )

        print(
            f"   Whole semantic: "
            f"{item['whole_score']:.4f}"
        )

        print(
            f"   Intellectual: "
            f"{item['intellectual_score']:.4f}"
        )

        print(
            f"   Bibliographic: "
            f"{item['bibliographic_score']:.4f}"
        )

        print(
            f"   Core topic matches: "
            f"{item['core_overlap']}"
        )

        print(
            f"   Related topic matches: "
            f"{item['related_overlap']}"
        )

        print(
            f"   ID: "
            f"{record.get('id')}"
        )

        for chunk in item[
            "matching_chunks"
        ]:

            print()
            print(
                f"   Evidence "
                f"(chunk {chunk['chunk_index']}, "
                f"{chunk['score']:.4f}):"
            )

            print(
                "   "
                + chunk["text"][:900]
                .replace("\n", " ")
            )


if __name__ == "__main__":
    main()
