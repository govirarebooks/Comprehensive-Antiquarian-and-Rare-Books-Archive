import json
import re
import sys
import unicodedata
from pathlib import Path


# ============================================================
# CONFIGURATION
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

DEFAULT_LIMIT = 10

# Number of semantic candidates considered before
# hybrid ranking.
SEMANTIC_TOP_K = 50

# Reciprocal Rank Fusion constant.
RRF_K = 60

# Must match the embedding model used when building
# the vector index.
VECTOR_MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)


# ============================================================
# GENERIC STOPWORDS
# ============================================================

STOPWORDS = {
    # English
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
    "without",

    # Italian
    "a",
    "ad",
    "al",
    "alla",
    "alle",
    "allo",
    "ai",
    "agli",
    "da",
    "dal",
    "dalla",
    "dalle",
    "dello",
    "dei",
    "degli",
    "di",
    "e",
    "ed",
    "in",
    "nel",
    "nella",
    "nelle",
    "nello",
    "nei",
    "negli",
    "o",
    "per",
    "su",
    "sul",
    "sulla",
    "sulle",
    "sullo",
    "con",
    "tra",
    "fra",
    "il",
    "lo",
    "la",
    "i",
    "gli",
    "le",
    "un",
    "uno",
    "una",

    # French
    "le",
    "la",
    "les",
    "un",
    "une",
    "des",
    "du",
    "de",
    "et",
    "en",
    "dans",
    "sur",
    "pour",
    "avec",
    "sans",

    # Spanish
    "el",
    "la",
    "los",
    "las",
    "un",
    "una",
    "unos",
    "unas",
    "de",
    "del",
    "y",
    "en",
    "para",
    "con",
    "sin",

    # German
    "der",
    "die",
    "das",
    "den",
    "dem",
    "des",
    "ein",
    "eine",
    "und",
    "von",
    "zu",
    "mit",
    "für",
    "im",
    "in",
    "auf",

    # Generic bibliographic words
    "book",
    "books",
    "libro",
    "libri",
    "livre",
    "livres",
    "libros",
    "buch",
    "bücher",

    # Generic period words
    "century",
    "centuries",
    "secolo",
    "secoli",
    "siècle",
    "siècles",
    "siglo",
    "siglos",
    "jahrhundert",
    "jahrhunderte",
}


# ============================================================
# MULTILINGUAL ALIASES
# ============================================================

MULTILINGUAL_ALIASES = {

    "astrology": "Astrology",
    "astrologia": "Astrology",
    "astrologie": "Astrology",
    "astrología": "Astrology",

    "astronomy": "Astronomy",
    "astronomia": "Astronomy",
    "astronomie": "Astronomy",
    "astronomía": "Astronomy",

    "medicine": "Medicine",
    "medicina": "Medicine",
    "médecine": "Medicine",
    "medizin": "Medicine",

    "philosophy": "Philosophy",
    "filosofia": "Philosophy",
    "philosophie": "Philosophy",
    "filosofía": "Philosophy",

    "theology": "Theology",
    "teologia": "Theology",
    "théologie": "Theology",
    "teología": "Theology",
    "theologie": "Theology",

    "history": "History",
    "storia": "History",
    "histoire": "History",
    "historia": "History",
    "geschichte": "History",

    "literature": "Literature",
    "letteratura": "Literature",
    "littérature": "Literature",
    "literatura": "Literature",
    "literatur": "Literature",

    "poetry": "Poetry",
    "poesia": "Poetry",
    "poésie": "Poetry",
    "poesía": "Poetry",
    "lyrik": "Poetry",

    "philology": "Philology",
    "filologia": "Philology",
    "philologie": "Philology",
    "filología": "Philology",

    "mathematics": "Mathematics",
    "matematica": "Mathematics",
    "mathématiques": "Mathematics",
    "matemáticas": "Mathematics",
    "mathematik": "Mathematics",

    "geometry": "Geometry",
    "geometria": "Geometry",
    "géométrie": "Geometry",
    "geometría": "Geometry",
    "geometrie": "Geometry",

    "algebra": "Algebra",
    "algèbre": "Algebra",
    "álgebra": "Algebra",

    "physics": "Physics",
    "fisica": "Physics",
    "physique": "Physics",
    "física": "Physics",
    "physik": "Physics",

    "chemistry": "Chemistry",
    "chimica": "Chemistry",
    "chimie": "Chemistry",
    "química": "Chemistry",
    "chemie": "Chemistry",

    "botany": "Botany",
    "botanica": "Botany",
    "botanique": "Botany",
    "botánica": "Botany",
    "botanik": "Botany",

    "zoology": "Zoology",
    "zoologia": "Zoology",
    "zoologie": "Zoology",
    "zoología": "Zoology",

    "agriculture": "Agriculture",
    "agricoltura": "Agriculture",
    "agricultura": "Agriculture",
    "landwirtschaft": "Agriculture",

    "architecture": "Architecture",
    "architettura": "Architecture",
    "arquitectura": "Architecture",
    "architektur": "Architecture",

    "art": "Art",
    "arte": "Art",
    "kunst": "Art",

    "music": "Music",
    "musica": "Music",
    "musique": "Music",
    "música": "Music",
    "musik": "Music",

    "law": "Law",
    "diritto": "Law",
    "droit": "Law",
    "derecho": "Law",
    "recht": "Law",

    "politics": "Politics",
    "politica": "Politics",
    "politique": "Politics",
    "política": "Politics",
    "politik": "Politics",

    "religion": "Religion",
    "religione": "Religion",
    "religión": "Religion",

    "grammar": "Grammar",
    "grammatica": "Grammar",
    "grammaire": "Grammar",
    "gramática": "Grammar",
    "grammatik": "Grammar",

    "linguistics": "Linguistics",
    "linguistica": "Linguistics",
    "linguistique": "Linguistics",
    "lingüística": "Linguistics",
    "linguistik": "Linguistics",

    "bibliography": "Bibliography",
    "bibliografia": "Bibliography",
    "bibliographie": "Bibliography",
    "bibliografía": "Bibliography",
    "bibliografie": "Bibliography",

    "printing": "Printing",
    "stampa": "Printing",
    "impression": "Printing",
    "imprenta": "Printing",
    "druck": "Printing",

    "typography": "Typography",
    "tipografia": "Typography",
    "typographie": "Typography",
    "tipografía": "Typography",
    "typografie": "Typography",

    "alchemy": "Alchemy",
    "alchimia": "Alchemy",
    "alchimie": "Alchemy",
    "alquimia": "Alchemy",
    "alchemie": "Alchemy",

    "witchcraft": "Witchcraft",
    "stregoneria": "Witchcraft",
    "sorcellerie": "Witchcraft",
    "brujería": "Witchcraft",
    "hexerei": "Witchcraft",

    "esoterica": "Esoterica Books",
    "esoterismo": "Esoterica Books",
    "ésotérisme": "Esoterica Books",
    "esoterik": "Esoterica Books",

    "humanism": "Humanism",
    "umanesimo": "Humanism",
    "humanisme": "Humanism",
    "humanismo": "Humanism",
    "humanismus": "Humanism",

    "reformation": "Reformation",
    "riforma": "Reformation",
    "réforme": "Reformation",
    "reforma": "Reformation",
}


# ============================================================
# HISTORICAL PERIOD ALIASES
# ============================================================

PERIOD_ALIASES = {

    "incunabula": "Incunabula",
    "incunable": "Incunabula",
    "incunables": "Incunabula",
    "incunaboli": "Incunabula",
    "incunabolo": "Incunabula",

    "fifteenth century": "Fifteenth Century",
    "15th century": "Fifteenth Century",
    "15th c": "Fifteenth Century",
    "xv century": "Fifteenth Century",
    "xv secolo": "Fifteenth Century",
    "quattrocento": "Fifteenth Century",
    "quinzième siècle": "Fifteenth Century",
    "siglo xv": "Fifteenth Century",

    "sixteenth century": "Sixteenth Century",
    "16th century": "Sixteenth Century",
    "16th c": "Sixteenth Century",
    "xvi century": "Sixteenth Century",
    "xvi secolo": "Sixteenth Century",
    "cinquecento": "Sixteenth Century",
    "seizième siècle": "Sixteenth Century",
    "siglo xvi": "Sixteenth Century",
    "sechzehntes jahrhundert": "Sixteenth Century",

    "seventeenth century": "Seventeenth Century",
    "17th century": "Seventeenth Century",
    "17th c": "Seventeenth Century",
    "xvii century": "Seventeenth Century",
    "xvii secolo": "Seventeenth Century",
    "seicento": "Seventeenth Century",
    "dix-septième siècle": "Seventeenth Century",
    "siglo xvii": "Seventeenth Century",
    "siebzehntes jahrhundert": "Seventeenth Century",

    "eighteenth century": "Eighteenth Century",
    "18th century": "Eighteenth Century",
    "18th c": "Eighteenth Century",
    "xviii century": "Eighteenth Century",
    "xviii secolo": "Eighteenth Century",
    "settecento": "Eighteenth Century",
    "dix-huitième siècle": "Eighteenth Century",
    "siglo xviii": "Eighteenth Century",
    "achtzehntes jahrhundert": "Eighteenth Century",

    "nineteenth century": "Nineteenth Century",
    "19th century": "Nineteenth Century",
    "19th c": "Nineteenth Century",
    "xix century": "Nineteenth Century",
    "xix secolo": "Nineteenth Century",
    "ottocento": "Nineteenth Century",
    "dix-neuvième siècle": "Nineteenth Century",
    "siglo xix": "Nineteenth Century",
    "neunzehntes jahrhundert": "Nineteenth Century",

    "twentieth century": "Twentieth Century",
    "20th century": "Twentieth Century",
    "20th c": "Twentieth Century",
    "xx century": "Twentieth Century",
    "xx secolo": "Twentieth Century",
    "novecento": "Twentieth Century",
    "vingtième siècle": "Twentieth Century",
    "siglo xx": "Twentieth Century",
    "zwanzigstes jahrhundert": "Twentieth Century",
}


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

    text = text.replace(
        "’",
        "'",
    )

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


# ============================================================
# DATASET TOPIC TAXONOMY
# ============================================================

def build_topic_index(documents):

    topics = {}

    for document in documents:

        metadata = document.get(
            "metadata",
            {},
        )

        for topic in metadata.get(
            "topics",
            [],
        ):

            if not topic:
                continue

            key = normalize(topic)

            if key:
                topics[key] = topic

    return topics


# ============================================================
# TOPIC LOOKUP
# ============================================================

def find_canonical_topic(
    candidate,
    official_topics,
):

    candidate_key = normalize(
        candidate
    )

    if not candidate_key:
        return None

    if candidate_key in official_topics:
        return official_topics[
            candidate_key
        ]

    candidate_tokens = set(
        candidate_key.split()
    )

    possible = []

    for (
        official_key,
        official_topic,
    ) in official_topics.items():

        official_tokens = set(
            official_key.split()
        )

        if not candidate_tokens:
            continue

        if candidate_tokens.issubset(
            official_tokens
        ):

            possible.append(
                (
                    len(official_tokens),
                    official_topic,
                )
            )

    if possible:

        possible.sort(
            key=lambda item: item[0]
        )

        return possible[0][1]

    return None


# ============================================================
# DETECT HISTORICAL PERIODS
# ============================================================

def detect_period_topics(
    query,
    official_topics,
):

    normalized_query = normalize(
        query
    )

    detected = []

    aliases = sorted(
        PERIOD_ALIASES.items(),
        key=lambda item: len(
            normalize(item[0])
        ),
        reverse=True,
    )

    for alias, canonical in aliases:

        normalized_alias = normalize(
            alias
        )

        if normalized_alias in normalized_query:

            topic = find_canonical_topic(
                canonical,
                official_topics,
            )

            if (
                topic
                and topic not in detected
            ):

                detected.append(topic)

    return detected


# ============================================================
# DETECT SUBJECT TOPICS
# ============================================================

def detect_subject_topics(
    query,
    official_topics,
):

    normalized_query = normalize(
        query
    )

    query_tokens = tokenize(
        query
    )

    detected = []

    for alias, canonical in (
        MULTILINGUAL_ALIASES.items()
    ):

        alias_normalized = normalize(
            alias
        )

        if " " in alias_normalized:

            if (
                alias_normalized
                not in normalized_query
            ):
                continue

        else:

            if (
                alias_normalized
                not in query_tokens
            ):
                continue

        topic = find_canonical_topic(
            canonical,
            official_topics,
        )

        if (
            topic
            and topic not in detected
        ):

            detected.append(topic)

    for (
        official_key,
        official_topic,
    ) in official_topics.items():

        if official_topic in detected:
            continue

        if official_key in normalized_query:
            detected.append(
                official_topic
            )

    meaningful_query_tokens = {
        token
        for token in query_tokens
        if token not in STOPWORDS
        and len(token) >= 4
    }

    for (
        official_key,
        official_topic,
    ) in official_topics.items():

        if official_topic in detected:
            continue

        topic_tokens = {
            token
            for token in official_key.split()
            if token not in STOPWORDS
            and len(token) >= 4
        }

        if not topic_tokens:
            continue

        if len(topic_tokens) == 1:

            topic_token = next(
                iter(topic_tokens)
            )

            if (
                topic_token
                in meaningful_query_tokens
            ):

                detected.append(
                    official_topic
                )

    return detected


# ============================================================
# DETECT ALL TOPICS
# ============================================================

def detect_topics(
    query,
    official_topics,
):

    detected = []

    for topic in detect_period_topics(
        query,
        official_topics,
    ):

        if topic not in detected:
            detected.append(topic)

    for topic in detect_subject_topics(
        query,
        official_topics,
    ):

        if topic not in detected:
            detected.append(topic)

    return detected


# ============================================================
# REMOVE RECOGNIZED TOPICS
# ============================================================

def get_free_query_tokens(
    query,
    recognized_topics,
):

    normalized_query = normalize(
        query
    )

    aliases = list(
        PERIOD_ALIASES.keys()
    )

    aliases.extend(
        MULTILINGUAL_ALIASES.keys()
    )

    aliases = sorted(
        aliases,
        key=lambda value: len(
            normalize(value)
        ),
        reverse=True,
    )

    for alias in aliases:

        normalized_alias = normalize(
            alias
        )

        if normalized_alias:

            normalized_query = (
                normalized_query.replace(
                    normalized_alias,
                    " ",
                )
            )

    for topic in recognized_topics:

        normalized_topic = normalize(
            topic
        )

        if normalized_topic:

            normalized_query = (
                normalized_query.replace(
                    normalized_topic,
                    " ",
                )
            )

    tokens = [
        token
        for token in normalized_query.split()
        if token
        and token not in STOPWORDS
        and len(token) >= 3
    ]

    return tokens


# ============================================================
# DOCUMENT TOPICS
# ============================================================

def get_document_topics(
    document
):

    return document.get(
        "metadata",
        {},
    ).get(
        "topics",
        [],
    )


def document_has_topic(
    document,
    requested_topic,
):

    requested_key = normalize(
        requested_topic
    )

    for topic in get_document_topics(
        document
    ):

        if (
            normalize(topic)
            == requested_key
        ):

            return True

    return False


# ============================================================
# TEXT EXTRACTION
# ============================================================

def get_document_text(
    document
):

    return normalize(
        document.get(
            "text",
            "",
        )
    )


def get_document_title(
    document
):

    return normalize(
        document.get(
            "title",
            "",
        )
    )


# ============================================================
# KEYWORD SCORING
# ============================================================

def score_document(
    document,
    requested_topics,
    free_tokens,
):

    score = 0.0

    reasons = []

    matched_topics = []

    missing_topics = []

    # --------------------------------------------------------
    # TOPIC MATCH
    # --------------------------------------------------------

    for requested_topic in requested_topics:

        if document_has_topic(
            document,
            requested_topic,
        ):

            score += 20.0

            matched_topics.append(
                requested_topic
            )

            reasons.append(
                f"topic: {requested_topic}"
            )

        else:

            missing_topics.append(
                requested_topic
            )

    if requested_topics:

        if len(matched_topics) == len(
            requested_topics
        ):

            score += 30.0

            reasons.append(
                "all requested topics matched"
            )

    # --------------------------------------------------------
    # TITLE MATCH
    # --------------------------------------------------------

    title = get_document_title(
        document
    )

    for token in free_tokens:

        if token in title:

            score += 3.0

            reasons.append(
                f"title: {token}"
            )

    # --------------------------------------------------------
    # FULL TEXT MATCH
    # --------------------------------------------------------

    text = get_document_text(
        document
    )

    for token in free_tokens:

        if token in text:
            score += 1.0

    # --------------------------------------------------------
    # MISSING TOPIC PENALTY
    # --------------------------------------------------------

    if requested_topics:

        score -= (
            len(missing_topics)
            * 15.0
        )

    return (
        score,
        reasons,
        matched_topics,
        missing_topics,
    )


# ============================================================
# KEYWORD SEARCH
# ============================================================

def keyword_search(
    documents,
    query,
    limit=DEFAULT_LIMIT,
):

    official_topics = (
        build_topic_index(
            documents
        )
    )

    recognized_topics = detect_topics(
        query,
        official_topics,
    )

    free_tokens = (
        get_free_query_tokens(
            query,
            recognized_topics,
        )
    )

    candidates = []

    for document in documents:

        (
            score,
            reasons,
            matched_topics,
            missing_topics,
        ) = score_document(
            document,
            recognized_topics,
            free_tokens,
        )

        if recognized_topics:

            if (
                not matched_topics
                and not free_tokens
            ):

                continue

        else:

            if not free_tokens:
                continue

            title = get_document_title(
                document
            )

            text = get_document_text(
                document
            )

            if not any(
                token in title
                or token in text
                for token in free_tokens
            ):

                continue

        candidates.append(
            {
                "document": document,
                "score": score,
                "reasons": reasons,
                "matched_topics":
                    matched_topics,
                "missing_topics":
                    missing_topics,
            }
        )

    # --------------------------------------------------------
    # Preserve existing conjunctive behavior.
    # --------------------------------------------------------

    if recognized_topics:

        complete = [
            result
            for result in candidates
            if len(
                result["matched_topics"]
            ) == len(
                recognized_topics
            )
        ]

        if complete:
            candidates = complete

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    def sort_key(result):

        document = result[
            "document"
        ]

        year = (
            document
            .get("metadata", {})
            .get("publication_year")
        )

        if year is None:
            year = 9999

        return (
            result["score"],
            len(
                result[
                    "matched_topics"
                ]
            ),
            -year,
        )

    candidates.sort(
        key=sort_key,
        reverse=True,
    )

    total_candidates = len(
        candidates
    )

    return (
        candidates[:limit],
        total_candidates,
        recognized_topics,
        free_tokens,
    )


# ============================================================
# VECTOR INDEX
# ============================================================

def vector_index_available():

    return (
        VECTOR_INDEX_PATH.exists()
        and VECTOR_METADATA_PATH.exists()
    )


def semantic_search(
    documents,
    query,
    limit=SEMANTIC_TOP_K,
):

    if not vector_index_available():

        return []

    try:

        import numpy as np

        from sentence_transformers import (
            SentenceTransformer,
        )

    except ImportError:

        print(
            "WARNING: semantic search "
            "dependencies are not installed."
        )

        print(
            "Falling back to keyword search."
        )

        return []

    # --------------------------------------------------------
    # Load vector index.
    # --------------------------------------------------------

    data = np.load(
        VECTOR_INDEX_PATH,
        allow_pickle=True,
    )

    embeddings = data[
        "embeddings"
    ]

    document_ids = data[
        "ids"
    ]

    # --------------------------------------------------------
    # Load metadata.
    # --------------------------------------------------------

    with VECTOR_METADATA_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        vector_metadata = json.load(
            file
        )

    # --------------------------------------------------------
    # Normalize metadata format.
    #
    # The current vector builder writes metadata as a list:
    #
    # [
    #     {"id": "work-123", ...},
    #     {"id": "work-456", ...}
    # ]
    #
    # Convert it to a dictionary indexed by document ID.
    #
    # Dictionary-formatted metadata is also accepted for
    # backwards compatibility.
    # --------------------------------------------------------

    if isinstance(
        vector_metadata,
        list,
    ):

        vector_metadata = {
            str(item.get("id")): item
            for item in vector_metadata
            if isinstance(item, dict)
            and item.get("id") is not None
        }

    elif isinstance(
        vector_metadata,
        dict,
    ):

        vector_metadata = {
            str(key): value
            for key, value in vector_metadata.items()
        }

    else:

        print(
            "WARNING: invalid vector metadata format."
        )

        vector_metadata = {}

    documents_by_id = {
        str(document.get("id")):
            document
        for document in documents
        if document.get("id") is not None
    }

    # --------------------------------------------------------
    # Load embedding model.
    # --------------------------------------------------------

    model = SentenceTransformer(
        VECTOR_MODEL_NAME
    )

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]

    # --------------------------------------------------------
    # Ensure document vectors are normalized.
    # --------------------------------------------------------

    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True,
    )

    norms[
        norms == 0
    ] = 1.0

    normalized_embeddings = (
        embeddings / norms
    )

    # --------------------------------------------------------
    # Cosine similarity.
    #
    # Because both vectors are normalized,
    # dot product = cosine similarity.
    # --------------------------------------------------------

    similarities = (
        normalized_embeddings
        @ query_embedding
    )

    # --------------------------------------------------------
    # Rank.
    # --------------------------------------------------------

    ranked_indices = np.argsort(
        similarities
    )[::-1]

    results = []

    for index in ranked_indices:

        document_id = str(
            document_ids[index]
        )

        document = documents_by_id.get(
            document_id
        )

        if document is None:
            continue

        similarity = float(
            similarities[index]
        )

        metadata = vector_metadata.get(
            document_id,
            {},
        )

        results.append(
            {
                "document": document,
                "semantic_score":
                    similarity,
                "vector_metadata":
                    metadata,
            }
        )

        if len(results) >= limit:
            break

    return results


# ============================================================
# RECIPROCAL RANK FUSION
# ============================================================

def hybrid_search(
    documents,
    query,
    limit=DEFAULT_LIMIT,
):

    # --------------------------------------------------------
    # Existing deterministic search.
    # --------------------------------------------------------

    keyword_results = keyword_search(
        documents,
        query,
        limit=max(
            limit,
            SEMANTIC_TOP_K,
        ),
    )[0]

    # --------------------------------------------------------
    # Semantic search.
    # --------------------------------------------------------

    semantic_results = semantic_search(
        documents,
        query,
        limit=SEMANTIC_TOP_K,
    )

    # --------------------------------------------------------
    # If vector index does not exist yet,
    # return exactly the existing search behavior.
    # --------------------------------------------------------

    if not semantic_results:

        return keyword_results

    # --------------------------------------------------------
    # RRF scores.
    # --------------------------------------------------------

    fused = {}

    for rank, result in enumerate(
        keyword_results,
        start=1,
    ):

        document = result[
            "document"
        ]

        document_id = str(
            document.get("id")
        )

        fused.setdefault(
            document_id,
            {
                "document": document,
                "hybrid_score": 0.0,
                "keyword_rank": None,
                "semantic_rank": None,
                "keyword_score": None,
                "semantic_score": None,
                "reasons": [],
                "matched_topics": [],
                "missing_topics": [],
            },
        )

        fused[
            document_id
        ][
            "keyword_rank"
        ] = rank

        fused[
            document_id
        ][
            "keyword_score"
        ] = result[
            "score"
        ]

        fused[
            document_id
        ][
            "matched_topics"
        ] = result[
            "matched_topics"
        ]

        fused[
            document_id
        ][
            "missing_topics"
        ] = result[
            "missing_topics"
        ]

        fused[
            document_id
        ][
            "reasons"
        ] = result[
            "reasons"
        ]

        fused[
            document_id
        ][
            "hybrid_score"
        ] += (
            1.0 /
            (
                RRF_K
                + rank
            )
        )

    # --------------------------------------------------------
    # Semantic rankings.
    # --------------------------------------------------------

    for rank, result in enumerate(
        semantic_results,
        start=1,
    ):

        document = result[
            "document"
        ]

        document_id = str(
            document.get("id")
        )

        if document_id not in fused:

            fused[
                document_id
            ] = {
                "document": document,
                "hybrid_score": 0.0,
                "keyword_rank": None,
                "semantic_rank": None,
                "keyword_score": None,
                "semantic_score": None,
                "reasons": [],
                "matched_topics": [],
                "missing_topics": [],
            }

        fused[
            document_id
        ][
            "semantic_rank"
        ] = rank

        fused[
            document_id
        ][
            "semantic_score"
        ] = result[
            "semantic_score"
        ]

        fused[
            document_id
        ][
            "hybrid_score"
        ] += (
            1.0 /
            (
                RRF_K
                + rank
            )
        )

    # --------------------------------------------------------
    # Sort hybrid results.
    # --------------------------------------------------------

    results = list(
        fused.values()
    )

    results.sort(
        key=lambda result: (
            result["hybrid_score"],
            result["semantic_score"]
                if result["semantic_score"]
                is not None
                else -1.0,
            result["keyword_score"]
                if result["keyword_score"]
                is not None
                else -9999.0,
        ),
        reverse=True,
    )

    return results[:limit]


# ============================================================
# OUTPUT
# ============================================================

def print_results(
    query,
    results,
    total_candidates,
    recognized_topics,
    free_tokens,
):

    print()
    print("=" * 80)
    print(
        "GOVI RARE BOOKS — "
        "HYBRID INTERNATIONAL SEARCH"
    )
    print("=" * 80)

    print()
    print(
        f'Query: "{query}"'
    )

    print()
    print("Recognized topics:")

    if recognized_topics:

        for topic in recognized_topics:

            print(
                f"  - {topic}"
            )

    else:

        print("  - none")

    print()
    print("Free-text terms:")

    if free_tokens:

        print(
            "  - "
            + ", ".join(
                free_tokens
            )
        )

    else:

        print("  - none")

    print()
    print(
        f"Candidates: {total_candidates}"
    )

    print(
        f"Results displayed: "
        f"{len(results)}"
    )

    print()

    if not results:

        print(
            "No results found."
        )

        print()

        return

    for index, result in enumerate(
        results,
        start=1,
    ):

        document = result[
            "document"
        ]

        metadata = document.get(
            "metadata",
            {},
        )

        title = document.get(
            "title",
            "",
        )

        year = metadata.get(
            "publication_year"
        )

        place = metadata.get(
            "publication_place"
        )

        topics = metadata.get(
            "topics",
            [],
        )

        source_id = metadata.get(
            "source_id"
        )

        source_url = metadata.get(
            "source_url"
        )

        print(
            f"{index}. {title}"
        )

        # ----------------------------------------------------
        # Hybrid score.
        # ----------------------------------------------------

        if "hybrid_score" in result:

            print(
                "   Hybrid score: "
                f"{result['hybrid_score']:.6f}"
            )

        # ----------------------------------------------------
        # Keyword score.
        # ----------------------------------------------------

        if result.get(
            "keyword_score"
        ) is not None:

            print(
                "   Keyword score: "
                f"{result['keyword_score']:.2f}"
            )

        # ----------------------------------------------------
        # Semantic score.
        # ----------------------------------------------------

        if result.get(
            "semantic_score"
        ) is not None:

            print(
                "   Semantic score: "
                f"{result['semantic_score']:.4f}"
            )

        # ----------------------------------------------------
        # Ranks.
        # ----------------------------------------------------

        if result.get(
            "keyword_rank"
        ) is not None:

            print(
                "   Keyword rank: "
                f"{result['keyword_rank']}"
            )

        if result.get(
            "semantic_rank"
        ) is not None:

            print(
                "   Semantic rank: "
                f"{result['semantic_rank']}"
            )

        # ----------------------------------------------------
        # Existing match information.
        # ----------------------------------------------------

        if result.get(
            "matched_topics"
        ):

            print(
                "   Matched topics: "
                + ", ".join(
                    result[
                        "matched_topics"
                    ]
                )
            )

        if result.get(
            "missing_topics"
        ):

            print(
                "   Missing topics: "
                + ", ".join(
                    result[
                        "missing_topics"
                    ]
                )
            )

        if result.get(
            "reasons"
        ):

            print("   Match:")

            for reason in result[
                "reasons"
            ]:

                print(
                    f"     - {reason}"
                )

        print(
            f"   Year: {year}"
        )

        print(
            f"   Place: {place}"
        )

        if topics:

            print(
                "   Topics: "
                + ", ".join(
                    topics
                )
            )

        print(
            f"   ID: {source_id}"
        )

        print(
            f"   Source: {source_url}"
        )

        print()


# ============================================================
# MAIN
# ============================================================

def main():

    if len(sys.argv) < 2:

        print(
            'Usage: python rag/search.py '
            '"your query"'
        )

        print()

        print("Examples:")

        print(
            '  python rag/search.py '
            '"astrology"'
        )

        print(
            '  python rag/search.py '
            '"astrology sixteenth century"'
        )

        print(
            '  python rag/search.py '
            '"medicine sixteenth century"'
        )

        print(
            '  python rag/search.py '
            '"astronomy seventeenth century"'
        )

        print(
            '  python rag/search.py '
            '"incunabula"'
        )

        print(
            '  python rag/search.py '
            '"witchcraft"'
        )

        print()

        sys.exit(1)

    query = " ".join(
        sys.argv[1:]
    ).strip()

    if not query:

        print(
            "Query cannot be empty."
        )

        sys.exit(1)

    # --------------------------------------------------------
    # LOAD MASTER DATASET
    #
    # The public master JSON is the source of truth.
    # Search works directly from it; rag/documents.json is
    # not required.
    # --------------------------------------------------------

    if not DATASET_PATH.exists():

        print(
            f"ERROR: "
            f"{DATASET_PATH} not found."
        )

        sys.exit(1)

    with DATASET_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        raw_records = json.load(
            file
        )

    if not isinstance(raw_records, list):

        print(
            "ERROR: master dataset must contain a JSON list."
        )

        sys.exit(1)

    def flatten_value(value):

        if value is None:
            return ""

        if isinstance(value, dict):

            parts = []

            for key, item in value.items():

                if key in {
                    "id",
                    "source_id",
                }:
                    continue

                flattened = flatten_value(item)

                if flattened:
                    parts.append(
                        f"{key}: {flattened}"
                    )

            return " | ".join(parts)

        if isinstance(value, list):

            parts = []

            for item in value:

                flattened = flatten_value(item)

                if flattened:
                    parts.append(flattened)

            return " | ".join(parts)

        return str(value)

    documents = []

    for record in raw_records:

        if not isinstance(record, dict):
            continue

        metadata = dict(
            record.get("metadata", {})
            if isinstance(
                record.get("metadata", {}),
                dict,
            )
            else {}
        )

        for field in (
            "publication_year",
            "publication_place",
            "source_url",
            "institution",
        ):

            if field in record:
                metadata[field] = record[field]

        if "topics" in record:
            metadata["topics"] = record.get(
                "topics",
                [],
            )

        if "source_id" in record:
            metadata["source_id"] = record["source_id"]

        elif "id" in record:
            metadata["source_id"] = record["id"]

        documents.append(
            {
                "id": record.get("id"),
                "title": record.get(
                    "title",
                    "",
                ),
                "text": flatten_value(record),
                "metadata": metadata,
                "dataset_record": record,
            }
        )

    # --------------------------------------------------------
    # SEARCH
    # --------------------------------------------------------

    official_topics = (
        build_topic_index(
            documents
        )
    )

    recognized_topics = detect_topics(
        query,
        official_topics,
    )

    free_tokens = (
        get_free_query_tokens(
            query,
            recognized_topics,
        )
    )

    results = hybrid_search(
        documents,
        query,
        limit=DEFAULT_LIMIT,
    )

    # --------------------------------------------------------
    # Candidate count.
    #
    # Keep the deterministic keyword candidate count
    # for transparency.
    # --------------------------------------------------------

    (
        _keyword_results,
        total_candidates,
        _,
        _,
    ) = keyword_search(
        documents,
        query,
        limit=max(
            DEFAULT_LIMIT,
            SEMANTIC_TOP_K,
        ),
    )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    print_results(
        query,
        results,
        total_candidates,
        recognized_topics,
        free_tokens,
    )


if __name__ == "__main__":
    main()
