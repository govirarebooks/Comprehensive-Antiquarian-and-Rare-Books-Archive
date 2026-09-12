import json
import sys
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

TOP_CHUNKS = 12
TOP_BOOKS = 10


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


def load_chunk_index():
    data = np.load(
        CHUNK_INDEX_PATH,
        allow_pickle=True,
    )

    embeddings = data["embeddings"]
    chunk_ids = data["chunk_ids"]

    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True,
    )

    norms[norms == 0] = 1.0

    embeddings = (
        embeddings / norms
    )

    with CHUNK_METADATA_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    return (
        embeddings,
        chunk_ids,
        metadata,
    )


def main():
    if len(sys.argv) < 2:
        print(
            'Usage: python rag/chunk_discovery.py "query"'
        )
        raise SystemExit(1)

    query = " ".join(
        sys.argv[1:]
    ).strip()

    records = load_dataset()

    (
        chunk_embeddings,
        chunk_ids,
        chunk_metadata,
    ) = load_chunk_index()

    model = SentenceTransformer(
        MODEL_NAME
    )

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

    ranked_indices = np.argsort(
        similarities
    )[::-1]

    records_by_id = {
        str(record.get("id")): record
        for record in records
        if record.get("id") is not None
    }

    # --------------------------------------------------------
    # TOP SEMANTIC CHUNKS
    # --------------------------------------------------------

    top_chunks = []

    for index in ranked_indices[
        :TOP_CHUNKS
    ]:

        item = chunk_metadata[index]

        book_id = str(
            item["book_id"]
        )

        score = float(
            similarities[index]
        )

        top_chunks.append(
            {
                "book_id": book_id,
                "chunk_index": item[
                    "chunk_index"
                ],
                "score": score,
                "text": item["text"],
            }
        )

    # --------------------------------------------------------
    # AGGREGATE BY BOOK
    # --------------------------------------------------------

    grouped = {}

    for item in top_chunks:

        book_id = item[
            "book_id"
        ]

        grouped.setdefault(
            book_id,
            [],
        ).append(item)

    book_results = []

    for book_id, chunks in grouped.items():

        record = records_by_id.get(
            book_id
        )

        if record is None:
            continue

        scores = [
            item["score"]
            for item in chunks
        ]

        # Best chunk matters most.
        best_score = max(
            scores
        )

        # Additional matching chunks
        # increase confidence.
        support_score = sum(
            sorted(
                scores,
                reverse=True,
            )[:3]
        ) / min(
            3,
            len(scores),
        )

        final_score = (
            best_score * 0.70
            + support_score * 0.30
        )

        book_results.append(
            {
                "record": record,
                "final_score": final_score,
                "best_chunk_score":
                    best_score,
                "support_score":
                    support_score,
                "chunks": sorted(
                    chunks,
                    key=lambda item:
                        item["score"],
                    reverse=True,
                ),
            }
        )

    book_results.sort(
        key=lambda item: (
            item["final_score"],
            item["best_chunk_score"],
        ),
        reverse=True,
    )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "GOVI — CHUNK-LEVEL "
        "BIBLIOGRAPHIC DISCOVERY"
    )
    print("=" * 80)

    print()
    print(
        f'Query: "{query}"'
    )

    print()
    print(
        f"Top semantic chunks: "
        f"{len(top_chunks)}"
    )

    print(
        f"Books discovered: "
        f"{len(book_results)}"
    )

    print()

    for number, item in enumerate(
        book_results[:TOP_BOOKS],
        start=1,
    ):

        record = item[
            "record"
        ]

        print(
            f"{number}. "
            f"{record.get('title', '')}"
        )

        print(
            "   Book score: "
            f"{item['final_score']:.4f}"
        )

        print(
            "   Best chunk: "
            f"{item['best_chunk_score']:.4f}"
        )

        print(
            "   Support: "
            f"{item['support_score']:.4f}"
        )

        print(
            "   ID: "
            f"{record.get('id')}"
        )

        for chunk in item[
            "chunks"
        ][:2]:

            print()
            print(
                "   Matching passage "
                f"(score {chunk['score']:.4f}):"
            )

            print(
                "   "
                + chunk["text"][:1200]
                .replace("\n", " ")
            )

        print()
        print("-" * 80)


if __name__ == "__main__":
    main()
