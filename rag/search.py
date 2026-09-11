import json
import re
import sys
import unicodedata
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS_PATH = ROOT / "rag" / "documents.json"

DEFAULT_LIMIT = 10


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
    "libro",
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
#
# Canonical vocabulary = English dataset taxonomy.
#
# Queries may arrive in several languages and are mapped
# toward the English taxonomy.
# ============================================================

MULTILINGUAL_ALIASES = {

    # --------------------------------------------------------
    # ASTROLOGY
    # --------------------------------------------------------

    "astrology": "Astrology",
    "astrologia": "Astrology",
    "astrologie": "Astrology",
    "astrología": "Astrology",
    "astrologie": "Astrology",

    # --------------------------------------------------------
    # ASTRONOMY
    # --------------------------------------------------------

    "astronomy": "Astronomy",
    "astronomia": "Astronomy",
    "astronomie": "Astronomy",
    "astronomía": "Astronomy",
    "astronomie": "Astronomy",

    # --------------------------------------------------------
    # MEDICINE
    # --------------------------------------------------------

    "medicine": "Medicine",
    "medicina": "Medicine",
    "médecine": "Medicine",
    "medicina": "Medicine",
    "medizin": "Medicine",

    # --------------------------------------------------------
    # PHILOSOPHY
    # --------------------------------------------------------

    "philosophy": "Philosophy",
    "filosofia": "Philosophy",
    "philosophie": "Philosophy",
    "filosofía": "Philosophy",
    "philosophie": "Philosophy",

    # --------------------------------------------------------
    # THEOLOGY
    # --------------------------------------------------------

    "theology": "Theology",
    "teologia": "Theology",
    "théologie": "Theology",
    "teología": "Theology",
    "theologie": "Theology",

    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    "history": "History",
    "storia": "History",
    "histoire": "History",
    "historia": "History",
    "geschichte": "History",

    # --------------------------------------------------------
    # LITERATURE
    # --------------------------------------------------------

    "literature": "Literature",
    "letteratura": "Literature",
    "littérature": "Literature",
    "literatura": "Literature",
    "literatur": "Literature",

    # --------------------------------------------------------
    # POETRY
    # --------------------------------------------------------

    "poetry": "Poetry",
    "poesia": "Poetry",
    "poésie": "Poetry",
    "poesía": "Poetry",
    "lyrik": "Poetry",

    # --------------------------------------------------------
    # PHILOLOGY
    # --------------------------------------------------------

    "philology": "Philology",
    "filologia": "Philology",
    "philologie": "Philology",
    "filología": "Philology",
    "philologie": "Philology",

    # --------------------------------------------------------
    # MATHEMATICS
    # --------------------------------------------------------

    "mathematics": "Mathematics",
    "matematica": "Mathematics",
    "mathématiques": "Mathematics",
    "matemáticas": "Mathematics",
    "mathematik": "Mathematics",

    # --------------------------------------------------------
    # GEOMETRY
    # --------------------------------------------------------

    "geometry": "Geometry",
    "geometria": "Geometry",
    "géométrie": "Geometry",
    "geometría": "Geometry",
    "geometrie": "Geometry",

    # --------------------------------------------------------
    # ALGEBRA
    # --------------------------------------------------------

    "algebra": "Algebra",
    "algèbre": "Algebra",
    "álgebra": "Algebra",

    # --------------------------------------------------------
    # PHYSICS
    # --------------------------------------------------------

    "physics": "Physics",
    "fisica": "Physics",
    "physique": "Physics",
    "física": "Physics",
    "physik": "Physics",

    # --------------------------------------------------------
    # CHEMISTRY
    # --------------------------------------------------------

    "chemistry": "Chemistry",
    "chimica": "Chemistry",
    "chimie": "Chemistry",
    "química": "Chemistry",
    "chemie": "Chemistry",

    # --------------------------------------------------------
    # BOTANY
    # --------------------------------------------------------

    "botany": "Botany",
    "botanica": "Botany",
    "botanique": "Botany",
    "botánica": "Botany",
    "botanik": "Botany",

    # --------------------------------------------------------
    # ZOOLOGY
    # --------------------------------------------------------

    "zoology": "Zoology",
    "zoologia": "Zoology",
    "zoologie": "Zoology",
    "zoología": "Zoology",
    "zoologie": "Zoology",

    # --------------------------------------------------------
    # AGRICULTURE
    # --------------------------------------------------------

    "agriculture": "Agriculture",
    "agricoltura": "Agriculture",
    "agriculture": "Agriculture",
    "agricultura": "Agriculture",
    "landwirtschaft": "Agriculture",

    # --------------------------------------------------------
    # ARCHITECTURE
    # --------------------------------------------------------

    "architecture": "Architecture",
    "architettura": "Architecture",
    "architecture": "Architecture",
    "arquitectura": "Architecture",
    "architektur": "Architecture",

    # --------------------------------------------------------
    # ART
    # --------------------------------------------------------

    "art": "Art",
    "arte": "Art",
    "art": "Art",
    "arte": "Art",
    "kunst": "Art",

    # --------------------------------------------------------
    # MUSIC
    # --------------------------------------------------------

    "music": "Music",
    "musica": "Music",
    "musique": "Music",
    "música": "Music",
    "musik": "Music",

    # --------------------------------------------------------
    # LAW
    # --------------------------------------------------------

    "law": "Law",
    "diritto": "Law",
    "droit": "Law",
    "derecho": "Law",
    "recht": "Law",

    # --------------------------------------------------------
    # POLITICS
    # --------------------------------------------------------

    "politics": "Politics",
    "politica": "Politics",
    "politique": "Politics",
    "política": "Politics",
    "politik": "Politics",

    # --------------------------------------------------------
    # RELIGION
    # --------------------------------------------------------

    "religion": "Religion",
    "religione": "Religion",
    "religión": "Religion",
    "religion": "Religion",

    # --------------------------------------------------------
    # GRAMMAR
    # --------------------------------------------------------

    "grammar": "Grammar",
    "grammatica": "Grammar",
    "grammaire": "Grammar",
    "gramática": "Grammar",
    "grammatik": "Grammar",

    # --------------------------------------------------------
    # LINGUISTICS
    # --------------------------------------------------------

    "linguistics": "Linguistics",
    "linguistica": "Linguistics",
    "linguistique": "Linguistics",
    "lingüística": "Linguistics",
    "linguistik": "Linguistics",

    # --------------------------------------------------------
    # BIBLIOGRAPHY
    # --------------------------------------------------------

    "bibliography": "Bibliography",
    "bibliografia": "Bibliography",
    "bibliographie": "Bibliography",
    "bibliografía": "Bibliography",
    "bibliografie": "Bibliography",

    # --------------------------------------------------------
    # PRINTING
    # --------------------------------------------------------

    "printing": "Printing",
    "stampa": "Printing",
    "impression": "Printing",
    "imprenta": "Printing",
    "druck": "Printing",

    # --------------------------------------------------------
    # TYPOGRAPHY
    # --------------------------------------------------------

    "typography": "Typography",
    "tipografia": "Typography",
    "typographie": "Typography",
    "tipografía": "Typography",
    "typografie": "Typography",

    # --------------------------------------------------------
    # ALCHEMY
    # --------------------------------------------------------

    "alchemy": "Alchemy",
    "alchimia": "Alchemy",
    "alchimie": "Alchemy",
    "alquimia": "Alchemy",
    "alchemie": "Alchemy",

    # --------------------------------------------------------
    # WITCHCRAFT
    # --------------------------------------------------------

    "witchcraft": "Witchcraft",
    "stregoneria": "Witchcraft",
    "sorcellerie": "Witchcraft",
    "brujería": "Witchcraft",
    "hexerei": "Witchcraft",

    # --------------------------------------------------------
    # ESOTERICA
    # --------------------------------------------------------

    "esoterica": "Esoterica Books",
    "esoterismo": "Esoterica Books",
    "ésotérisme": "Esoterica Books",
    "esoterismo": "Esoterica Books",
    "esoterik": "Esoterica Books",

    # --------------------------------------------------------
    # HUMANISM
    # --------------------------------------------------------

    "humanism": "Humanism",
    "umanesimo": "Humanism",
    "humanisme": "Humanism",
    "humanismo": "Humanism",
    "humanismus": "Humanism",

    # --------------------------------------------------------
    # REFORMATION
    # --------------------------------------------------------

    "reformation": "Reformation",
    "riforma": "Reformation",
    "réforme": "Reformation",
    "reforma": "Reformation",
    "reformation": "Reformation",
}


# ============================================================
# HISTORICAL PERIOD ALIASES
# ============================================================

PERIOD_ALIASES = {

    # Incunabula
    "incunabula": "Incunabula",
    "incunable": "Incunabula",
    "incunables": "Incunabula",
    "incunaboli": "Incunabula",
    "incunabolo": "Incunabula",
    "incunables": "Incunabula",
    "incunables": "Incunabula",

    # Fifteenth century
    "fifteenth century": "Fifteenth Century",
    "15th century": "Fifteenth Century",
    "15th c": "Fifteenth Century",
    "xv century": "Fifteenth Century",
    "xv secolo": "Fifteenth Century",
    "quattrocento": "Fifteenth Century",
    "quinzième siècle": "Fifteenth Century",
    "siglo xv": "Fifteenth Century",

    # Sixteenth century
    "sixteenth century": "Sixteenth Century",
    "16th century": "Sixteenth Century",
    "16th c": "Sixteenth Century",
    "xvi century": "Sixteenth Century",
    "xvi secolo": "Sixteenth Century",
    "cinquecento": "Sixteenth Century",
    "seizième siècle": "Sixteenth Century",
    "siglo xvi": "Sixteenth Century",
    "sechzehntes jahrhundert": "Sixteenth Century",

    # Seventeenth century
    "seventeenth century": "Seventeenth Century",
    "17th century": "Seventeenth Century",
    "17th c": "Seventeenth Century",
    "xvii century": "Seventeenth Century",
    "xvii secolo": "Seventeenth Century",
    "seicento": "Seventeenth Century",
    "dix-septième siècle": "Seventeenth Century",
    "siglo xvii": "Seventeenth Century",
    "siebzehntes jahrhundert": "Seventeenth Century",

    # Eighteenth century
    "eighteenth century": "Eighteenth Century",
    "18th century": "Eighteenth Century",
    "18th c": "Eighteenth Century",
    "xviii century": "Eighteenth Century",
    "xviii secolo": "Eighteenth Century",
    "settecento": "Eighteenth Century",
    "dix-huitième siècle": "Eighteenth Century",
    "siglo xviii": "Eighteenth Century",
    "achtzehntes jahrhundert": "Eighteenth Century",

    # Nineteenth century
    "nineteenth century": "Nineteenth Century",
    "19th century": "Nineteenth Century",
    "19th c": "Nineteenth Century",
    "xix century": "Nineteenth Century",
    "xix secolo": "Nineteenth Century",
    "ottocento": "Nineteenth Century",
    "dix-neuvième siècle": "Nineteenth Century",
    "siglo xix": "Nineteenth Century",
    "neunzehntes jahrhundert": "Nineteenth Century",

    # Twentieth century
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
    """
    Lowercase, remove accents, punctuation and duplicate spaces.
    """

    if text is None:
        return ""

    text = str(text).lower()

    # Normalize Unicode accents.
    text = unicodedata.normalize("NFKD", text)
    text = "".join(
        char
        for char in text
        if not unicodedata.combining(char)
    )

    # Normalize apostrophes.
    text = text.replace("’", "'")

    # Keep letters/numbers/spaces.
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)

    # Collapse whitespace.
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize(text):
    """
    Return normalized tokens.
    """

    return set(normalize(text).split())


# ============================================================
# DATASET TOPIC TAXONOMY
# ============================================================

def build_topic_index(documents):
    """
    Extract the canonical topic taxonomy directly from the
    RAG documents.

    Returns:
        {
            normalized_topic: original_topic
        }
    """

    topics = {}

    for document in documents:

        metadata = document.get("metadata", {})

        for topic in metadata.get("topics", []):

            if not topic:
                continue

            key = normalize(topic)

            if key:
                topics[key] = topic

    return topics


# ============================================================
# TOPIC LOOKUP
# ============================================================

def find_canonical_topic(candidate, official_topics):
    """
    Find a canonical dataset topic.

    First tries exact matching, then a conservative partial
    match for cases such as:

        Medicine -> Medicine Books
        Philology -> Philology Books
    """

    candidate_key = normalize(candidate)

    if not candidate_key:
        return None

    # Exact match.
    if candidate_key in official_topics:
        return official_topics[candidate_key]

    candidate_tokens = set(candidate_key.split())

    # Conservative partial match.
    possible = []

    for official_key, official_topic in official_topics.items():

        official_tokens = set(official_key.split())

        if not candidate_tokens:
            continue

        if candidate_tokens.issubset(official_tokens):

            # Prefer the shortest/closest topic.
            possible.append(
                (
                    len(official_tokens),
                    official_topic,
                )
            )

    if possible:

        possible.sort(key=lambda item: item[0])

        return possible[0][1]

    return None


# ============================================================
# DETECT HISTORICAL PERIODS
# ============================================================

def detect_period_topics(query, official_topics):
    """
    Detect historical periods in multilingual queries.
    """

    normalized_query = normalize(query)

    detected = []

    aliases = sorted(
        PERIOD_ALIASES.items(),
        key=lambda item: len(normalize(item[0])),
        reverse=True,
    )

    for alias, canonical in aliases:

        normalized_alias = normalize(alias)

        if normalized_alias in normalized_query:

            topic = find_canonical_topic(
                canonical,
                official_topics,
            )

            if topic and topic not in detected:
                detected.append(topic)

    return detected


# ============================================================
# DETECT SUBJECT TOPICS
# ============================================================

def detect_subject_topics(query, official_topics):
    """
    Detect subject topics from multilingual queries.

    The dataset taxonomy remains authoritative.
    """

    normalized_query = normalize(query)
    query_tokens = tokenize(query)

    detected = []

    # --------------------------------------------------------
    # Explicit multilingual aliases.
    # --------------------------------------------------------

    for alias, canonical in MULTILINGUAL_ALIASES.items():

        alias_normalized = normalize(alias)

        if " " in alias_normalized:

            if alias_normalized not in normalized_query:
                continue

        else:

            if alias_normalized not in query_tokens:
                continue

        topic = find_canonical_topic(
            canonical,
            official_topics,
        )

        if topic and topic not in detected:
            detected.append(topic)

    # --------------------------------------------------------
    # Direct match against official English taxonomy.
    # --------------------------------------------------------

    for official_key, official_topic in official_topics.items():

        if official_topic in detected:
            continue

        if official_key in normalized_query:
            detected.append(official_topic)

    # --------------------------------------------------------
    # Conservative single-token matching.
    #
    # Example:
    #     "astronomy" -> Astronomy
    #
    # But generic words such as "books" are excluded.
    # --------------------------------------------------------

    meaningful_query_tokens = {
        token
        for token in query_tokens
        if token not in STOPWORDS
        and len(token) >= 4
    }

    for official_key, official_topic in official_topics.items():

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

        # Single-word canonical topics.
        if len(topic_tokens) == 1:

            topic_token = next(iter(topic_tokens))

            if topic_token in meaningful_query_tokens:
                detected.append(official_topic)

    return detected


# ============================================================
# DETECT ALL TOPICS
# ============================================================

def detect_topics(query, official_topics):

    detected = []

    # Historical periods first.
    for topic in detect_period_topics(
        query,
        official_topics,
    ):
        if topic not in detected:
            detected.append(topic)

    # Subjects.
    for topic in detect_subject_topics(
        query,
        official_topics,
    ):
        if topic not in detected:
            detected.append(topic)

    return detected


# ============================================================
# REMOVE RECOGNIZED TOPICS FROM FREE QUERY
# ============================================================

def get_free_query_tokens(query, recognized_topics):

    normalized_query = normalize(query)

    # Remove period aliases.
    aliases = list(PERIOD_ALIASES.keys())

    # Remove subject aliases.
    aliases.extend(MULTILINGUAL_ALIASES.keys())

    # Longest first prevents partial leftovers.
    aliases = sorted(
        aliases,
        key=lambda value: len(normalize(value)),
        reverse=True,
    )

    for alias in aliases:

        normalized_alias = normalize(alias)

        if normalized_alias:
            normalized_query = normalized_query.replace(
                normalized_alias,
                " ",
            )

    # Remove canonical topic names too.
    for topic in recognized_topics:

        normalized_topic = normalize(topic)

        if normalized_topic:
            normalized_query = normalized_query.replace(
                normalized_topic,
                " ",
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

def get_document_topics(document):

    return document.get(
        "metadata",
        {},
    ).get(
        "topics",
        [],
    )


def document_has_topic(document, requested_topic):

    requested_key = normalize(requested_topic)

    for topic in get_document_topics(document):

        if normalize(topic) == requested_key:
            return True

    return False


# ============================================================
# TEXT EXTRACTION
# ============================================================

def get_document_text(document):

    return normalize(
        document.get("text", "")
    )


def get_document_title(document):

    return normalize(
        document.get("title", "")
    )


# ============================================================
# SCORING
# ============================================================

def score_document(
    document,
    requested_topics,
    free_tokens,
):
    """
    Score a document.

    Topic matches are the strongest signal.
    Text/title matches are secondary.
    """

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

    # Strong conjunction bonus.
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

    title = get_document_title(document)

    for token in free_tokens:

        if token in title:

            score += 3.0

            reasons.append(
                f"title: {token}"
            )

    # --------------------------------------------------------
    # FULL TEXT MATCH
    # --------------------------------------------------------

    text = get_document_text(document)

    for token in free_tokens:

        if token in text:

            score += 1.0

    # --------------------------------------------------------
    # PENALTY FOR MISSING TOPICS
    # --------------------------------------------------------

    if requested_topics:

        score -= (
            len(missing_topics) * 15.0
        )

    return (
        score,
        reasons,
        matched_topics,
        missing_topics,
    )


# ============================================================
# SEARCH
# ============================================================

def search(
    documents,
    query,
    limit=DEFAULT_LIMIT,
):

    official_topics = build_topic_index(
        documents
    )

    recognized_topics = detect_topics(
        query,
        official_topics,
    )

    free_tokens = get_free_query_tokens(
        query,
        recognized_topics,
    )

    candidates = []

    # --------------------------------------------------------
    # SCORE DOCUMENTS
    # --------------------------------------------------------

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

        # --------------------------------------------
        # Topic search
        # --------------------------------------------

        if recognized_topics:

            # If the document has no requested topic
            # and there are no free terms, ignore it.
            if (
                not matched_topics
                and not free_tokens
            ):
                continue

        # --------------------------------------------
        # Free-text search
        # --------------------------------------------

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
                "matched_topics": matched_topics,
                "missing_topics": missing_topics,
            }
        )

    # --------------------------------------------------------
    # CONJUNCTIVE SEARCH
    #
    # If documents exist that match ALL requested topics,
    # discard partial matches.
    # --------------------------------------------------------

    if recognized_topics:

        complete = [
            result
            for result in candidates
            if len(
                result["matched_topics"]
            ) == len(recognized_topics)
        ]

        if complete:
            candidates = complete

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    def sort_key(result):

        document = result["document"]

        year = (
            document
            .get("metadata", {})
            .get("publication_year")
        )

        if year is None:
            year = 9999

        return (
            result["score"],
            len(result["matched_topics"]),
            -year,
        )

    candidates.sort(
        key=sort_key,
        reverse=True,
    )

    total_candidates = len(candidates)

    return (
        candidates[:limit],
        total_candidates,
        recognized_topics,
        free_tokens,
    )


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
    print("GOVI RARE BOOKS — INTERNATIONAL SEARCH")
    print("=" * 80)

    print()
    print(f'Query: "{query}"')

    print()
    print("Recognized topics:")

    if recognized_topics:

        for topic in recognized_topics:
            print(f"  - {topic}")

    else:

        print("  - none")

    print()
    print("Free-text terms:")

    if free_tokens:

        print(
            "  - "
            + ", ".join(free_tokens)
        )

    else:

        print("  - none")

    print()
    print(
        f"Candidates: {total_candidates}"
    )

    print(
        f"Results displayed: {len(results)}"
    )

    print()

    if not results:

        print("No results found.")
        print()

        return

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    for index, result in enumerate(
        results,
        start=1,
    ):

        document = result["document"]

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

        print(
            f"   Score: {result['score']:.2f}"
        )

        if result["matched_topics"]:

            print(
                "   Matched topics: "
                + ", ".join(
                    result["matched_topics"]
                )
            )

        if result["missing_topics"]:

            print(
                "   Missing topics: "
                + ", ".join(
                    result["missing_topics"]
                )
            )

        if result["reasons"]:

            print("   Match:")

            for reason in result["reasons"]:

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
                + ", ".join(topics)
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
            'Usage: python rag/search.py "your query"'
        )

        print()

        print("Examples:")

        print(
            '  python rag/search.py "astrology"'
        )

        print(
            '  python rag/search.py "astrology sixteenth century"'
        )

        print(
            '  python rag/search.py "medicine sixteenth century"'
        )

        print(
            '  python rag/search.py "astronomy seventeenth century"'
        )

        print(
            '  python rag/search.py "incunabula"'
        )

        print(
            '  python rag/search.py "witchcraft"'
        )

        print()

        sys.exit(1)

    query = " ".join(
        sys.argv[1:]
    ).strip()

    if not query:

        print("Query cannot be empty.")
        sys.exit(1)

    # --------------------------------------------------------
    # LOAD RAG DOCUMENTS
    # --------------------------------------------------------

    if not DOCUMENTS_PATH.exists():

        print(
            f"ERROR: {DOCUMENTS_PATH} not found."
        )

        print(
            "Run rag/prepare_documents.py first."
        )

        sys.exit(1)

    with DOCUMENTS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        documents = json.load(file)

    # --------------------------------------------------------
    # SEARCH
    # --------------------------------------------------------

    (
        results,
        total_candidates,
        recognized_topics,
        free_tokens,
    ) = search(
        documents,
        query,
        limit=DEFAULT_LIMIT,
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
