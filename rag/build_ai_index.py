import hashlib
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = ROOT / "govi-rare-books-academic-dataset.json"
AI_INDEX_PATH = ROOT / "rag" / "ai_index.npz"
AI_METADATA_PATH = ROOT / "rag" / "ai_metadata.json"

VECTOR_MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)


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


def record_id(record):
    return str(record.get("id"))


def normalize_space(text):
    return " ".join(
        str(text).split()
    ).strip()


def collect_all_text(value):
    parts = []

    if value is None:
        return parts

    if isinstance(value, dict):
        for key, child in value.items():

            child_parts = collect_all_text(
                child
            )

            for child_text in child_parts:
                if child_text:
                    parts.append(
                        f"{key}: {child_text}"
                    )

        return parts

    if isinstance(value, list):
        for child in value:
            parts.extend(
                collect_all_text(child)
            )

        return parts

    text = normalize_space(value)

    if text:
        parts.append(text)

    return parts

def collect_key_text(
    value,
    wanted_keys,
    path=(),
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

            child_path = path + (
                key_normalized,
            )

            if key_normalized in wanted_keys:

                child_text = " ".join(
                    collect_all_text(child)
                )

                if child_text:
                    parts.append(
                        f"{key}: {child_text}"
                    )

            parts.extend(
                collect_key_text(
                    child,
                    wanted_keys,
                    child_path,
                )
            )

        return parts

    if isinstance(value, list):

        for child in value:
            parts.extend(
                collect_key_text(
                    child,
                    wanted_keys,
                    path,
                )
            )

    return parts


def build_profiles(record):
    full_text = "\n".join(
        collect_all_text(record)
    )

    intellectual_keys = {
        "academic_description",
        "description",
        "descriptions",
        "biography",
        "biographical_data",
        "bibliography",
        "bibliographies",
        "content",
        "summary",
        "notes",
        "provenance",
    }

    bibliographic_keys = {
        "title",
        "authors",
        "publishers",
        "related_names",
        "topics",
        "publication_year",
        "publication_place",
        "publisher_data",
        "bibliography",
        "source_url",
        "institution",
        "linked_open_data",
    }

    intellectual = "\n".join(
        collect_key_text(
            record,
            intellectual_keys,
        )
    )

    bibliographic = "\n".join(
        collect_key_text(
            record,
            bibliographic_keys,
        )
    )

    # Always preserve the complete record as a fallback
    # so no information is lost.
    if not intellectual:
        intellectual = full_text

    if not bibliographic:
        bibliographic = full_text

    return {
        "full": full_text,
        "intellectual": intellectual,
        "bibliographic": bibliographic,
    }


def content_hash(text):
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def load_previous():
    if (
        not AI_INDEX_PATH.exists()
        or not AI_METADATA_PATH.exists()
    ):
        return None, {}

    data = np.load(
        AI_INDEX_PATH,
        allow_pickle=True,
    )

    with AI_METADATA_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        metadata = json.load(file)

    return data, {
        str(item["id"]): item
        for item in metadata
        if isinstance(item, dict)
        and item.get("id") is not None
    }


def encode_texts(model, texts):
    return model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32)


def main():
    records = load_dataset()

    previous_data, previous_metadata = (
        load_previous()
    )

    embeddings = {
        "full": {},
        "intellectual": {},
        "bibliographic": {},
    }

    new_count = 0
    updated_count = 0
    reused_count = 0
    removed_count = 0

    model = None

    pending = {
        "full": [],
        "intellectual": [],
        "bibliographic": [],
    }

    pending_ids = {
        "full": [],
        "intellectual": [],
        "bibliographic": [],
    }

    metadata_output = []

    current_ids = set()

    for record in records:

        identifier = record_id(record)
        current_ids.add(identifier)

        profiles = build_profiles(
            record
        )

        hashes = {
            profile: content_hash(
                text
            )
            for profile, text
            in profiles.items()
        }

        previous = previous_metadata.get(
            identifier
        )

        unchanged = (
            previous is not None
            and previous.get(
                "hashes"
            ) == hashes
        )

        if unchanged:

            reused_count += 1

            for profile in embeddings:

                embeddings[
                    profile
                ][identifier] = previous_data[
                    f"embedding_{profile}"
                ][
                    previous_data[
                        "ids"
                    ].tolist().index(
                        identifier
                    )
                ]

        else:

            if previous is None:
                new_count += 1
            else:
                updated_count += 1

            for profile, text in profiles.items():

                pending[
                    profile
                ].append(text)

                pending_ids[
                    profile
                ].append(identifier)

        metadata_output.append(
            {
                "id": identifier,
                "hashes": hashes,
            }
        )

    if any(
        pending[profile]
        for profile in pending
    ):

        print()
        print(
            "Loading embedding model:"
        )
        print(
            VECTOR_MODEL_NAME
        )

        model = SentenceTransformer(
            VECTOR_MODEL_NAME
        )

        for profile in (
            "full",
            "intellectual",
            "bibliographic",
        ):

            if not pending[profile]:
                continue

            print()
            print(
                f"Encoding {profile} profiles:"
            )

            encoded = encode_texts(
                model,
                pending[profile],
            )

            for index, identifier in enumerate(
                pending_ids[profile]
            ):

                embeddings[
                    profile
                ][identifier] = encoded[index]

    removed_count = len(
        set(previous_metadata.keys())
        - current_ids
    )

    if not records:
        raise ValueError(
            "No records to index."
        )

    ordered_ids = [
        record_id(record)
        for record in records
    ]

    arrays = {}

    for profile in (
        "full",
        "intellectual",
        "bibliographic",
    ):

        arrays[
            f"embedding_{profile}"
        ] = np.vstack(
            [
                embeddings[
                    profile
                ][identifier]
                for identifier in ordered_ids
            ]
        ).astype(np.float32)

    arrays["ids"] = np.array(
        ordered_ids,
        dtype=str,
    )

    np.savez_compressed(
        AI_INDEX_PATH,
        **arrays,
    )

    with AI_METADATA_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata_output,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(
        "AI multi-profile index built successfully."
    )
    print(
        f"Documents: {len(records)}"
    )
    print(
        f"New: {new_count}"
    )
    print(
        f"Updated: {updated_count}"
    )
    print(
        f"Reused: {reused_count}"
    )
    print(
        f"Removed: {removed_count}"
    )
    print()
    print(
        f"AI index: {AI_INDEX_PATH}"
    )
    print(
        f"Metadata: {AI_METADATA_PATH}"
    )


if __name__ == "__main__":
    main()
