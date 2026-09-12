import json
import re
import sys
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

FINAL_LIMIT = 10
CANDIDATE_LIMIT = 30
TOP_CHUNKS_PER_BOOK = 3


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

    return " ".join(
        text.split()
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
            "Dataset must be a JSON list."
        )

    return data


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


def topic_overlap(a, b):
    return (
        {
            normalize(x)
            for x in get_topics(a)
        }
        & {
            normalize(x)
            for x in get_topics(b)
        }
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

        if not isinstance(
            value,
            list,
        ):
            continue

        for item in value:

            if isinstance(
                item,
                dict,
            ):
                name = item.get(
                    "name"
                )

                if name:
                    people.add(
                        normalize(name)
                    )

            elif isinstance(
                item,
                str,
            ):
                people.add(
                    normalize(item)
                )

    return {
        person
        for person in people
        if person
    }


# ============================================================
# SUBJECT MATTER EXTRACTION
# ============================================================

SUBJECT_KEYS = {
    "academic_description",
    "description",
    "descriptions",
    "summary",
    "content",
    "topics",
    "subject",
    "subjects",
    "keywords",
    "themes",
    "notes",
    "provenance",
}


CONTEXT_KEYS = {
    "title",
    "authors",
    "publishers",
    "related_names",
    "publication_year",
    "publication_place",
    "publisher_data",
    "bibliography",
    "linked_open_data",
    "institution",
}


def collect_text(
    value,
    allowed_keys=None,
):
    parts = []

    if value is None:
        return parts

    if isinstance(value, dict):

        for key, child in value.items():

            key_normalized = (
                str(key)
                .lower()
                .replace("-", "_")
                .replace(" ", "_")
            )

            if (
                allowed_keys is not None
                and key_normalized
                not in allowed_keys
            ):
                continue

            parts.extend(
                collect_text(
                    child,
                    allowed_keys,
                )
            )

        return parts

    if isinstance(value, list):

        for child in value:
            parts.extend(
                collect_text(
                    child,
                    allowed_keys,
                )
            )

        return parts

    text = str(value).strip()

    if text:
        parts.append(text)

    return parts


def subject_text(record):
    return " ".join(
        collect_text(
            record,
            SUBJECT_KEYS,
        )
    )


def context_text(record):
    return " ".join(
        collect_text(
            record,
            CONTEXT_KEYS,
        )
    )


# ============================================================
# VECTOR INDEX
# ============================================================

def load_chunk_index():
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

    norms[
        norms == 0
    ] = 1.0

    embeddings = (
        embeddings / norms
    )

    return embeddings, metadata


def cosine_rows(
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

    norms = np.linalg.norm(
        matrix,
        axis=1,
        keepdims=True,
    )

    norms[
        norms == 0
    ] = 1.0

    vector_norm = np.linalg.norm(
        vector
    )

    if vector_norm == 0:
        return np.zeros(
            len(matrix)
        )

    return (
        matrix / norms
    ) @ (
        vector / vector_norm
    )


# ============================================================
# CHUNK SIMILARITY BETWEEN BOOKS
# ============================================================

def best_chunk_similarity(
    source_chunks,
    candidate_chunks,
):
    if (
        len(source_chunks) == 0
        or len(candidate_chunks) == 0
    ):
        return 0.0, []

    source_norms = np.linalg.norm(
        source_chunks,
        axis=1,
        keepdims=True,
    )

    source_norms[
        source_norms == 0
    ] = 1.0

    source_chunks = (
        source_chunks
        / source_norms
    )

    candidate_norms = np.linalg.norm(
        candidate_chunks,
        axis=1,
        keepdims=True,
    )

    candidate_norms[
        candidate_norms == 0
    ] = 1.0

    candidate_chunks = (
        candidate_chunks
        / candidate_norms
    )

    matrix = (
        candidate_chunks
        @ source_chunks.T
    )

    best_for_candidate = matrix.max(
        axis=1
    )

    best_indices = np.argsort(
        best_for_candidate
    )[::-1]

    evidence = []

    for candidate_index in best_indices[
        :TOP_CHUNKS_PER_BOOK
    ]:
        source_index = int(
            matrix[
                candidate_index
            ].argmax()
        )

        evidence.append(
            {
                "candidate_index":
                    int(candidate_index),
                "source_chunk_index":
                    source_index,
                "score":
                    float(
                        best_for_candidate[
                            candidate_index
                        ]
                    ),
            }
        )

    if not evidence:
        return 0.0, []

    best = evidence[0]["score"]

    support = (
        sum(
            item["score"]
            for item in evidence
        )
        / len(evidence)
    )

    score = (
        best * 0.70
        + support * 0.30
    )

    return score, evidence


# ============================================================
# MAIN
# ============================================================

def main():
    if len(sys.argv) != 2:
        print(
            "Usage: python "
            "rag/similar_books_final.py BOOK_ID"
        )
        raise SystemExit(1)

    target_id = sys.argv[1]

    records = load_dataset()

    records_by_id = {
        str(record.get("id")):
            record
        for record in records
        if record.get("id") is not None
    }

    source = records_by_id.get(
        target_id
    )

    if source is None:
        print(
            f"Book not found: {target_id}"
        )
        raise SystemExit(1)

    chunk_embeddings, chunk_metadata = (
        load_chunk_index()
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    # --------------------------------------------------------
    # Build source subject representation.
    # --------------------------------------------------------

    source_subject = subject_text(
        source
    )

    source_context = context_text(
        source
    )

    source_subject_embedding = model.encode(
        [source_subject],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )[0]

    source_context_embedding = model.encode(
        [source_context],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )[0]

    # --------------------------------------------------------
    # Prepare chunks by book.
    # --------------------------------------------------------

    chunk_indices_by_book = {}

    for index, metadata in enumerate(
        chunk_metadata
    ):

        book_id = str(
            metadata.get("book_id")
        )

        chunk_indices_by_book.setdefault(
            book_id,
            [],
        ).append(index)

    source_chunk_indices = (
        chunk_indices_by_book.get(
            target_id,
            [],
        )
    )

    source_chunks = (
        chunk_embeddings[
            source_chunk_indices
        ]
        if source_chunk_indices
        else np.empty(
            (
                0,
                chunk_embeddings.shape[1],
            ),
            dtype=np.float32,
        )
    )

    # --------------------------------------------------------
    # Candidate generation.
    # --------------------------------------------------------

    candidates = []

    for candidate_id, candidate in (
        records_by_id.items()
    ):

        if candidate_id == target_id:
            continue

        candidate_subject = subject_text(
            candidate
        )

        candidate_context = context_text(
            candidate
        )

        if not candidate_subject:
            candidate_subject = candidate_context

        candidate_subject_embedding = (
            model.encode(
                [candidate_subject],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )[0]
        )

        candidate_context_embedding = (
            model.encode(
                [candidate_context],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )[0]
        )

        subject_similarity = float(
            candidate_subject_embedding
            @ source_subject_embedding
        )

        context_similarity = float(
            candidate_context_embedding
            @ source_context_embedding
        )

        shared_topics = topic_overlap(
            source,
            candidate,
        )

        shared_people = (
            get_people(source)
            & get_people(candidate)
        )

        # Do NOT give a large bonus just because
        # two books share the same person.
        people_bonus = min(
            0.05,
            len(shared_people) * 0.025,
        )

        topic_bonus = min(
            0.15,
            len(shared_topics) * 0.05,
        )

        candidate_score = (
            subject_similarity * 0.60
            + context_similarity * 0.10
            + topic_bonus
            + people_bonus
        )

        candidates.append(
            {
                "id":
                    candidate_id,
                "record":
                    candidate,
                "subject_similarity":
                    subject_similarity,
                "context_similarity":
                    context_similarity,
                "shared_topics":
                    shared_topics,
                "shared_people":
                    shared_people,
                "candidate_score":
                    candidate_score,
            }
        )

    candidates.sort(
        key=lambda item: (
            item["candidate_score"],
            item["subject_similarity"],
        ),
        reverse=True,
    )

    candidates = candidates[
        :CANDIDATE_LIMIT
    ]

    # --------------------------------------------------------
    # Chunk evidence.
    # --------------------------------------------------------

    results = []

    for item in candidates:

        candidate_id = item[
            "id"
        ]

        candidate_chunk_indices = (
            chunk_indices_by_book.get(
                candidate_id,
                [],
            )
        )

        candidate_chunks = (
            chunk_embeddings[
                candidate_chunk_indices
            ]
            if candidate_chunk_indices
            else np.empty(
                (
                    0,
                    chunk_embeddings.shape[1],
                ),
                dtype=np.float32,
            )
        )

        chunk_score, evidence = (
            best_chunk_similarity(
                source_chunks,
                candidate_chunks,
            )
        )

        # ----------------------------------------------------
        # Final similarity
        #
        # Subject matter dominates.
        # Generic context is deliberately weak.
        # Topics are supporting evidence.
        # People are only a tiny contextual bonus.
        # ----------------------------------------------------

        final_score = (
            chunk_score * 0.45
            + item["subject_similarity"] * 0.40
            + item["context_similarity"] * 0.05
            + min(
                0.10,
                len(
                    item["shared_topics"]
                ) * 0.025,
            )
        )

        # If subject similarity is weak, shared people
        # alone must never make the book look highly similar.
        if (
            item["subject_similarity"] < 0.35
            and not item["shared_topics"]
        ):
            final_score *= 0.60

        results.append(
            {
                **item,
                "chunk_score":
                    chunk_score,
                "evidence":
                    evidence,
                "final":
                    final_score,
            }
        )

    results.sort(
        key=lambda item: (
            item["final"],
            item["chunk_score"],
            item["subject_similarity"],
        ),
        reverse=True,
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "GOVI AI — SUBJECT-AWARE "
        "SIMILAR BOOKS"
    )
    print("=" * 80)

    print()
    print(
        "Source:"
    )

    print(
        source.get(
            "title",
            "",
        )
    )

    print(
        f"ID: {target_id}"
    )

    print()

    for number, item in enumerate(
        results[:FINAL_LIMIT],
        start=1,
    ):

        candidate = item[
            "record"
        ]

        print(
            f"{number}. "
            + candidate.get(
                "title",
                "",
            )
        )

        print(
            "   Final: "
            f"{item['final']:.4f}"
        )

        print(
            "   Subject similarity: "
            f"{item['subject_similarity']:.4f}"
        )

        print(
            "   Description evidence: "
            f"{item['chunk_score']:.4f}"
        )

        print(
            "   Context similarity: "
            f"{item['context_similarity']:.4f}"
        )

        if item[
            "shared_topics"
        ]:

            print(
                "   Shared topics: "
                + ", ".join(
                    sorted(
                        item[
                            "shared_topics"
                        ]
                    )
                )
            )

        if item[
            "shared_people"
        ]:

            print(
                "   Shared people: "
                + ", ".join(
                    sorted(
                        item[
                            "shared_people"
                        ]
                    )
                )
            )

        for evidence in item[
            "evidence"
        ][:2]:

            source_chunk_index = (
                source_chunk_indices[
                    evidence[
                        "source_chunk_index"
                    ]
                ]
            )

            candidate_chunk_index = (
                chunk_indices_by_book[
                    item["id"]
                ][
                    evidence[
                        "candidate_index"
                    ]
                ]
            )

            source_text = chunk_metadata[
                source_chunk_index
            ]["text"]

            candidate_text = chunk_metadata[
                candidate_chunk_index
            ]["text"]

            print()
            print(
                "   Evidence similarity: "
                f"{evidence['score']:.4f}"
            )

            print(
                "   Source passage: "
                + source_text[
                    :500
                ].replace(
                    "\n",
                    " ",
                )
            )

            print(
                "   Related passage: "
                + candidate_text[
                    :500
                ].replace(
                    "\n",
                    " ",
                )
            )

        print(
            "   ID: "
            + str(
                candidate.get(
                    "id"
                )
            )
        )

        print()


if __name__ == "__main__":
    main()
