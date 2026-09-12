import json
import hashlib
import numpy as np
from pathlib import Path

from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    ROOT /
    "govi-rare-books-academic-dataset.json"
)

VECTOR_INDEX_PATH = (
    ROOT /
    "rag" /
    "vector_index.npz"
)

VECTOR_METADATA_PATH = (
    ROOT /
    "rag" /
    "vector_metadata.json"
)


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)


# ============================================================
# HELPERS
# ============================================================


def document_hash(document):
    """
    Creates a stable hash of the complete document.

    Any change in:
    - title
    - text
    - author
    - bibliography
    - biography
    - authority
    - metadata

    forces embedding regeneration.
    """

    content = json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
    )

    return hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()



def extract_search_text(document):
    """
    Builds the text representation used by AI.

    Includes current and future authority fields.
    """

    parts = []


    def add(value):

        if value is None:
            return

        if isinstance(value, list):

            for item in value:
                add(item)

        elif isinstance(value, dict):

            for key, value2 in value.items():

                add(key)
                add(value2)

        else:

            parts.append(
                str(value)
            )


    add(
        document.get(
            "title",
            ""
        )
    )

    add(
        document.get(
            "text",
            ""
        )
    )


    metadata = document.get(
        "metadata",
        {}
    )

    add(metadata)


    # future authority blocks
    add(
        document.get(
            "authority",
            {}
        )
    )

    add(
        document.get(
            "authors",
            []
        )
    )

    add(
        document.get(
            "publishers",
            []
        )
    )


    return "\n".join(parts)



# ============================================================
# LOAD DATASET
# ============================================================


if not DATASET_PATH.exists():

    raise FileNotFoundError(
        f"Dataset not found: {DATASET_PATH}"
    )


with DATASET_PATH.open(
    "r",
    encoding="utf-8",
) as file:

    documents = json.load(
        file
    )


if not isinstance(
    documents,
    list,
):

    raise ValueError(
        "Dataset must contain a JSON list."
    )



print(
    f"Documents loaded: {len(documents)}"
)



# ============================================================
# LOAD OLD INDEX
# ============================================================


old_metadata = {}

old_embeddings = {}

if VECTOR_METADATA_PATH.exists():

    with VECTOR_METADATA_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        old_metadata = json.load(
            file
        )


if isinstance(
    old_metadata,
    list,
):

    old_metadata = {
        item["id"]: item
        for item in old_metadata
        if "id" in item
    }



if VECTOR_INDEX_PATH.exists():

    data = np.load(
        VECTOR_INDEX_PATH,
        allow_pickle=True,
    )

    old_vectors = data[
        "embeddings"
    ]

    old_ids = data[
        "ids"
    ]

    old_embeddings = {
        str(doc_id): vector
        for doc_id, vector in zip(
            old_ids,
            old_vectors
        )
    }



# ============================================================
# MODEL
# ============================================================


print()

print(
    "Loading embedding model:"
)

print(
    MODEL_NAME
)


model = SentenceTransformer(
    MODEL_NAME
)



# ============================================================
# BUILD INDEX
# ============================================================


new_embeddings = []

new_ids = []

new_metadata = []


created = 0
updated = 0
reused = 0


for document in documents:

    document_id = str(
        document.get(
            "id"
        )
    )


    if document_id == "None":

        continue


    current_hash = document_hash(
        document
    )


    previous = old_metadata.get(
        document_id,
        {}
    )


    previous_hash = previous.get(
        "hash"
    )


    if (
        previous_hash == current_hash
        and document_id in old_embeddings
    ):

        vector = old_embeddings[
            document_id
        ]

        reused += 1


    else:

        text = extract_search_text(
            document
        )


        vector = model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )


        if previous_hash:

            updated += 1

        else:

            created += 1



    new_ids.append(
        document_id
    )

    new_embeddings.append(
        vector
    )


    new_metadata.append(
        {
            "id": document_id,
            "hash": current_hash,
        }
    )



# ============================================================
# SAVE
# ============================================================


embeddings = np.array(
    new_embeddings,
    dtype=np.float32,
)


np.savez_compressed(
    VECTOR_INDEX_PATH,
    embeddings=embeddings,
    ids=np.array(
        new_ids
    ),
)



with VECTOR_METADATA_PATH.open(
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        new_metadata,
        file,
        ensure_ascii=False,
        indent=2,
    )



print()

print(
    "Vector index built successfully."
)

print(
    f"Documents: {len(new_ids)}"
)

print(
    f"New: {created}"
)

print(
    f"Updated: {updated}"
)

print(
    f"Reused: {reused}"
)

print(
    "Removed: 0"
)

print()

print(
    f"Vector index: {VECTOR_INDEX_PATH}"
)

print(
    f"Metadata: {VECTOR_METADATA_PATH}"
)
