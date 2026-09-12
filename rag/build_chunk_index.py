import hashlib
import json
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = ROOT / "govi-rare-books-academic-dataset.json"
CHUNK_INDEX_PATH = ROOT / "rag" / "chunk_index.npz"
CHUNK_METADATA_PATH = ROOT / "rag" / "chunk_metadata.json"

VECTOR_MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

WORDS_PER_CHUNK = 500


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


def split_into_chunks(text):
    words = re.findall(
        r"\S+",
        str(text),
    )

    chunks = []

    for start in range(
        0,
        len(words),
        WORDS_PER_CHUNK,
    ):

        chunk = " ".join(
            words[
                start:start + WORDS_PER_CHUNK
            ]
        ).strip()

        if chunk:
            chunks.append(chunk)

    return chunks


def content_hash(text):
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def main():
    records = load_dataset()

    chunks = []
    metadata = []

    for record in records:

        record_id = str(
            record.get("id")
        )

        description = str(
            record.get(
                "academic_description",
                "",
            )
        ).strip()

        if not description:
            continue

        description_hash = content_hash(
            description
        )

        record_chunks = split_into_chunks(
            description
        )

        for chunk_index, chunk in enumerate(
            record_chunks
        ):

            chunks.append(chunk)

            metadata.append(
                {
                    "chunk_id": (
                        f"{record_id}:"
                        f"{chunk_index}"
                    ),
                    "book_id": record_id,
                    "chunk_index": chunk_index,
                    "description_hash":
                        description_hash,
                    "text": chunk,
                }
            )

    print(
        f"Records: {len(records)}"
    )

    print(
        f"Description chunks: "
        f"{len(chunks)}"
    )

    model = SentenceTransformer(
        VECTOR_MODEL_NAME
    )

    embeddings = model.encode(
        chunks,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(
        np.float32
    )

    chunk_ids = np.array(
        [
            item["chunk_id"]
            for item in metadata
        ],
        dtype=str,
    )

    np.savez_compressed(
        CHUNK_INDEX_PATH,
        embeddings=embeddings,
        chunk_ids=chunk_ids,
    )

    with CHUNK_METADATA_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(
        "Chunk semantic index built successfully."
    )

    print(
        f"Chunks: {len(chunks)}"
    )

    print(
        f"Embedding dimensions: "
        f"{embeddings.shape[1]}"
    )

    print(
        f"Index: {CHUNK_INDEX_PATH}"
    )

    print(
        f"Metadata: "
        f"{CHUNK_METADATA_PATH}"
    )


if __name__ == "__main__":
    main()
