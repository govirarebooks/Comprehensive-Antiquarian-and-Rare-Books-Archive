import re
import unicodedata


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
    return set(
        normalize(text).split()
    )


def build_topic_text(topic):
    return normalize(topic)


def rank_topics_by_semantic_similarity(
    query_embedding,
    topic_embeddings,
    topics,
    limit=8,
):
    import numpy as np

    if not topics or len(topic_embeddings) == 0:
        return []

    matrix = np.asarray(
        topic_embeddings,
        dtype=np.float32,
    )

    query = np.asarray(
        query_embedding,
        dtype=np.float32,
    )

    query_norm = np.linalg.norm(query)

    if query_norm == 0:
        return []

    matrix_norms = np.linalg.norm(
        matrix,
        axis=1,
        keepdims=True,
    )

    matrix_norms[
        matrix_norms == 0
    ] = 1.0

    normalized_matrix = (
        matrix / matrix_norms
    )

    normalized_query = (
        query / query_norm
    )

    similarities = (
        normalized_matrix
        @ normalized_query
    )

    ranked = np.argsort(
        similarities
    )[::-1]

    results = []

    for index in ranked[:limit]:

        results.append(
            {
                "topic": topics[index],
                "score": float(
                    similarities[index]
                ),
            }
        )

    return results


def build_query_profile(
    query,
    semantic_topics=None,
):
    """
    Build a generic query profile.

    The profile intentionally avoids a manually maintained
    dictionary of individual phrases. Topics are discovered
    from the actual dataset taxonomy and can therefore evolve
    with the master JSON.
    """

    normalized = normalize(query)

    profile = {
        "query": query,
        "normalized_query": normalized,
        "tokens": sorted(
            tokenize(query)
        ),
        "candidate_topics": [],
    }

    if semantic_topics:
        profile["candidate_topics"] = [
            item["topic"]
            for item in semantic_topics
            if item.get("score", 0.0) >= 0.30
        ]

    return profile


def topic_overlap_score(
    document_topics,
    requested_topics,
):
    if not requested_topics:
        return 0.0

    document_keys = {
        normalize(topic)
        for topic in document_topics
    }

    requested_keys = {
        normalize(topic)
        for topic in requested_topics
    }

    if not requested_keys:
        return 0.0

    matches = (
        document_keys
        & requested_keys
    )

    return (
        len(matches)
        / len(requested_keys)
    )
