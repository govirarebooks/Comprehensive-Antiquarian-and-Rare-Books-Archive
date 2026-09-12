import json
import re
import unicodedata
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = ROOT / "govi-rare-books-academic-dataset.json"
AI_INDEX_PATH = ROOT / "rag" / "ai_index.npz"

VECTOR_MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)


# ============================================================
# NORMALIZATION
# ============================================================

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

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


def tokenize(text):
    return {
        token
        for token in normalize(text).split()
        if len(token) >= 3
    }


# ============================================================
# FULL RECORD TEXT
# ============================================================

def field_label(path):
    if not path:
        return "record"

    return " > ".join(
        str(item)
        for item in path
    )


def flatten_record(value, path=None):
    """
    Convert the complete JSON record into structured text.

    Nothing important is discarded:
    nested dictionaries, lists, biographies,
    bibliographies, authority fields, related names,
    publishers and metadata are all traversed.
    """

    if path is None:
        path = []

    parts = []

    if value is None:
        return parts

    if isinstance(value, dict):

        for key, child in value.items():

            child_path = path + [
                str(key)
            ]

            child_parts = flatten_record(
                child,
                child_path,
            )

            parts.extend(
                child_parts
            )

        return parts

    if isinstance(value, list):

        for index, child in enumerate(
            value
        ):

            child_path = path + [
                str(index)
            ]

            child_parts = flatten_record(
                child,
                child_path,
            )

            parts.extend(
                child_parts
            )

        return parts

    text = str(value).strip()

    if not text:
        return parts

    parts.append(
        f"{field_label(path)}: {text}"
    )

    return parts


def build_record_text(record):
    parts = flatten_record(
        record
    )

    return "\n".join(
        parts
    )


# ============================================================
# DATASET
# ============================================================

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

    return [
        record
        for record in data
        if isinstance(record, dict)
    ]


def get_record_id(record):
    return str(
        record.get("id")
    )


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


# ============================================================
# VECTOR INDEX
# ============================================================

def load_ai_index():
    data = np.load(
        AI_INDEX_PATH,
        allow_pickle=True,
    )

    return (
        data["ids"],
        data["embedding_full"],
        data["embedding_intellectual"],
        data["embedding_bibliographic"],
    )


# ============================================================
# TOPIC DISCOVERY
# ============================================================

def build_topic_index(records):
    topics = {}

    for record in records:

        for topic in get_topics(
            record
        ):

            key = normalize(topic)

            if key:
                topics[key] = topic

    return list(
        topics.values()
    )


def discover_topics(
    model,
    query,
    topics,
):
    if not topics:
        return []

    topic_texts = [
        f"Bibliographic subject: {topic}"
        for topic in topics
    ]

    topic_embeddings = model.encode(
        topic_texts,
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

    similarities = (
        topic_embeddings
        @ query_embedding
    )

    ranked = np.argsort(
        similarities
    )[::-1]

    return [
        {
            "topic": topics[index],
            "score": float(
                similarities[index]
            ),
        }
        for index in ranked
    ]


def select_topic_profile(
    discovered_topics,
):
    if not discovered_topics:
        return {
            "core": [],
            "related": [],
        }

    top = discovered_topics[0][
        "score"
    ]

    # Dynamic rather than fixed:
    # core topics must be close to the strongest
    # interpretation of the query.
    core_floor = max(
        0.56,
        top - 0.07,
    )

    core = [
        item
        for item in discovered_topics
        if item["score"] >= core_floor
    ][:3]

    related_floor = max(
        0.43,
        top - 0.17,
    )

    related = [
        item
        for item in discovered_topics
        if (
            item["score"] >= related_floor
            and item not in core
        )
    ][:4]

    return {
        "core": core,
        "related": related,
    }


# ============================================================
# SEMANTIC SIMILARITY
# ============================================================

def cosine_similarity(
    matrix,
    vector,
):
    matrix = np.asarray(
        matrix,
        dtype=np.float32,
    )

    vector = np.asarray(
        vector,
        dtype=np.float32,
    )

    matrix_norms = np.linalg.norm(
        matrix,
        axis=1,
        keepdims=True,
    )

    matrix_norms[
        matrix_norms == 0
    ] = 1.0

    matrix = (
        matrix / matrix_norms
    )

    vector_norm = np.linalg.norm(
        vector
    )

    if vector_norm == 0:
        return np.zeros(
            len(matrix),
            dtype=np.float32,
        )

    vector = (
        vector / vector_norm
    )

    return matrix @ vector


# ============================================================
# BIBLIOGRAPHIC SIGNALS
# ============================================================

def topic_overlap_score(
    record,
    core_topics,
):
    if not core_topics:
        return 0.0

    record_topics = {
        normalize(topic)
        for topic in get_topics(
            record
        )
    }

    requested = {
        normalize(item["topic"])
        for item in core_topics
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


def text_overlap_score(
    query,
    record,
):
    query_tokens = tokenize(
        query
    )

    if not query_tokens:
        return 0.0

    record_tokens = tokenize(
        build_record_text(record)
    )

    overlap = (
        query_tokens
        & record_tokens
    )

    return (
        len(overlap)
        / len(query_tokens)
    )


def get_people(record):
    people = []

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
                    people.append(
                        str(name)
                    )

            elif isinstance(item, str):

                people.append(item)

    return people


# ============================================================
# FULL-CORPUS RERANKING
# ============================================================

def rerank_records(
    query,
    records,
    ids,
    full_embeddings,
    intellectual_embeddings,
    bibliographic_embeddings,
    core_topics,
    related_topics,
    model,
    limit=10,
):
    query_embeddings = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )[0]

    full_similarity = cosine_similarity(
        full_embeddings,
        query_embeddings,
    )

    intellectual_similarity = cosine_similarity(
        intellectual_embeddings,
        query_embeddings,
    )

    bibliographic_similarity = cosine_similarity(
        bibliographic_embeddings,
        query_embeddings,
    )

    records_by_id = {
        get_record_id(record): record
        for record in records
    }

    related_topic_names = {
        normalize(item["topic"])
        for item in related_topics
    }

    ranked = []

    for index, document_id in enumerate(ids):

        record = records_by_id.get(
            str(document_id)
        )

        if record is None:
            continue

        full_semantic = float(
            full_similarity[index]
        )

        intellectual_semantic = float(
            intellectual_similarity[index]
        )

        bibliographic_semantic = float(
            bibliographic_similarity[index]
        )

        semantic = (
            full_semantic * 0.30
            + intellectual_semantic * 0.50
            + bibliographic_semantic * 0.20
        )

        core_score = (
            topic_overlap_score(
                record,
                core_topics,
            )
        )

        record_topic_keys = {
            normalize(topic)
            for topic in get_topics(record)
        }

        related_score = 0.0

        if related_topic_names:

            related_score = (
                len(
                    record_topic_keys
                    & related_topic_names
                )
                / len(
                    related_topic_names
                )
            )

        text_score = (
            text_overlap_score(
                query,
                record,
            )
        )

        # ----------------------------------------------------
        # CANDIDATE POLICY
        #
        # Core-topic matches are strongest.
        # Related-topic matches can help.
        # Strong intellectual similarity can rescue a
        # record even when the catalog topic taxonomy does
        # not explicitly contain the queried concept.
        # ----------------------------------------------------

        if core_topics:

            has_core = (
                core_score > 0
            )

            has_related = (
                related_score > 0
            )

            strong_intellectual_match = (
                intellectual_semantic >= 0.40
                and full_semantic >= 0.36
                and (
                    related_score >= 0.25
                    or text_score >= 0.65
                )
            )

            if not has_core:

                if has_related:

                    # A related-topic match is admitted only
                    # when the full semantic/intellectual evidence
                    # is strong enough.
                    if semantic < 0.355:
                        continue

                elif not strong_intellectual_match:

                    continue

        final_score = (
            semantic * 0.50
            + core_score * 0.30
            + related_score * 0.10
            + text_score * 0.10
        )

        if core_topics and core_score == 0:

            if related_score == 0:

                final_score *= 0.72

            else:

                final_score *= 0.88

        ranked.append(
            {
                "record": record,
                "semantic_score": semantic,
                "full_semantic_score": full_semantic,
                "intellectual_semantic_score":
                    intellectual_semantic,
                "bibliographic_semantic_score":
                    bibliographic_semantic,
                "core_topic_score": core_score,
                "related_topic_score": related_score,
                "text_score": text_score,
                "final_score": final_score,
            }
        )

    ranked.sort(
        key=lambda item: (
            item["final_score"],
            item["intellectual_semantic_score"],
            item["core_topic_score"],
            item["semantic_score"],
        ),
        reverse=True,
    )

    return ranked[:limit]

def search(
    query,
    limit=10,
):
    records = load_dataset()

    (
        ids,
        full_embeddings,
        intellectual_embeddings,
        bibliographic_embeddings,
    ) = load_ai_index()

    model = SentenceTransformer(
        VECTOR_MODEL_NAME
    )

    topics = build_topic_index(
        records
    )

    discovered = discover_topics(
        model,
        query,
        topics,
    )

    profile = select_topic_profile(
        discovered
    )

    results = rerank_records(
        query=query,
        records=records,
        ids=ids,
        full_embeddings=full_embeddings,
        intellectual_embeddings=intellectual_embeddings,
        bibliographic_embeddings=bibliographic_embeddings,
        core_topics=profile["core"],
        related_topics=profile["related"],
        model=model,
        limit=limit,
    )

    return {
        "query": query,
        "discovered_topics": discovered,
        "core_topics": profile["core"],
        "related_topics": profile["related"],
        "results": results,
    }


# ============================================================
# CLI
# ============================================================

def main():
    import sys

    if len(sys.argv) < 2:

        print(
            'Usage: python rag/discovery_engine.py '
            '"your query"'
        )

        raise SystemExit(1)

    query = " ".join(
        sys.argv[1:]
    ).strip()

    response = search(
        query
    )

    print()
    print("=" * 80)
    print(
        "GOVI AI BIBLIOGRAPHIC "
        "DISCOVERY"
    )
    print("=" * 80)

    print()
    print(
        f'Query: "{query}"'
    )

    print()
    print("Core topics:")

    if response[
        "core_topics"
    ]:

        for item in response[
            "core_topics"
        ]:

            print(
                f"  - {item['topic']}: "
                f"{item['score']:.4f}"
            )

    else:
        print("  - none")

    print()
    print("Related topics:")

    if response[
        "related_topics"
    ]:

        for item in response[
            "related_topics"
        ]:

            print(
                f"  - {item['topic']}: "
                f"{item['score']:.4f}"
            )

    else:
        print("  - none")

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

        print(
            f"\n{index}. "
            f"{record.get('title', '')}"
        )

        print(
            "   Final: "
            f"{item['final_score']:.4f}"
        )

        print(
            "   Semantic: "
            f"{item['semantic_score']:.4f}"
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
            "   Text: "
            f"{item['text_score']:.4f}"
        )

        print(
            "   Topics: "
            + ", ".join(
                get_topics(record)
            )
        )

        print(
            "   ID: "
            f"{record.get('id')}"
        )


if __name__ == "__main__":
    main()
