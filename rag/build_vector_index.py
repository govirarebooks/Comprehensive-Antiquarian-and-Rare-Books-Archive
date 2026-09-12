import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parent.parent

DOCUMENTS_PATH = ROOT / "rag" / "documents.json"
VECTOR_INDEX_PATH = ROOT / "rag" / "vector_index.npz"
VECTOR_METADATA_PATH = ROOT / "rag" / "vector_metadata.json"

VECTOR_MODEL_NAME = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


def load_documents():
    with DOCUMENTS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_previous_index():
    if not VECTOR_INDEX_PATH.exists() or not VECTOR_METADATA_PATH.exists():
        return {}

    data = np.load(VECTOR_INDEX_PATH)

    embeddings = data["embeddings"]

    with VECTOR_METADATA_PATH.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    previous = {}

    for index, item in enumerate(metadata):
        document_id = item["id"]

        previous[document_id] = {
            "content_hash": item.get("content_hash"),
            "embedding": embeddings[index],
            "metadata": item,
        }

    return previous


def main():
    documents = load_documents()

    previous_index = load_previous_index()

    current_ids = {
        document["id"]
        for document in documents
    }

    model = None

    embeddings_by_id = {}

    new_count = 0
    updated_count = 0
    reused_count = 0

    for document in documents:
        document_id = document["id"]
        content_hash = document.get("content_hash")

        previous = previous_index.get(document_id)

        # Reuse the existing embedding when the document
        # has not changed.
        if (
            previous is not None
            and previous.get("content_hash") == content_hash
        ):
            embeddings_by_id[document_id] = previous["embedding"]
            reused_count += 1
            continue

        # Load the model only when an embedding is actually needed.
        if model is None:
            print(
                f"Loading embedding model: {VECTOR_MODEL_NAME}"
            )

            model = SentenceTransformer(
                VECTOR_MODEL_NAME
            )

        text = document.get("text", "")

        embedding = model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        embeddings_by_id[document_id] = embedding

        if previous is None:
            new_count += 1
        else:
            updated_count += 1

    removed_count = len(
        set(previous_index.keys()) - current_ids
    )

    if not embeddings_by_id:
        print("No documents to index.")
        return

    # Keep exactly the same document ordering for IDs,
    # embeddings and metadata.
    ordered_ids = [
        document["id"]
        for document in documents
    ]

    embeddings = np.vstack(
        [
            embeddings_by_id[document_id]
            for document_id in ordered_ids
        ]
    ).astype(np.float32)

    ids = np.array(
        ordered_ids,
        dtype=str,
    )

    metadata = []

    for document in documents:
        metadata.append(
            {
                "id": document["id"],
                "type": document.get("type"),
                "title": document.get("title"),
                "content_hash": document.get("content_hash"),
                "metadata": document.get(
                    "metadata",
                    {},
                ),
            }
        )

    # Save both embeddings AND IDs inside the NPZ.
    np.savez_compressed(
        VECTOR_INDEX_PATH,
        embeddings=embeddings,
        ids=ids,
    )

    with VECTOR_METADATA_PATH.open(
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
    print("Vector index built successfully.")
    print(f"Documents: {len(documents)}")
    print(f"New: {new_count}")
    print(f"Updated: {updated_count}")
    print(f"Reused: {reused_count}")
    print(f"Removed: {removed_count}")
    print(f"Vector index: {VECTOR_INDEX_PATH}")
    print(f"Metadata: {VECTOR_METADATA_PATH}")


if __name__ == "__main__":
    main()
