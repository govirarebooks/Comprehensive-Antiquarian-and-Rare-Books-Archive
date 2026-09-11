import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS_PATH = ROOT / "rag" / "documents.json"


# Natural-language aliases -> official Govi topics
TOPIC_ALIASES = {
    "astrologia": "Astrology",
    "astrology": "Astrology",
    "astronomia": "Astronomy",
    "astronomy": "Astronomy",
    "letteratura italiana": "Italian Literature",
    "italian literature": "Italian Literature",
    "letteratura francese": "French Literature",
    "french literature": "French Literature",
    "letteratura tedesca": "German Literature",
    "german literature": "German Literature",
    "letteratura inglese": "English Literature",
    "english literature": "English Literature",
}


PERIOD_ALIASES = {
    "incunaboli": "Incunabula",
    "incunabolo": "Incunabula",
    "incunabula": "Incunabula",
    "incunaboli": "Incunabula",

    "quindicesimo secolo": "Fifteenth century",
    "xv secolo": "Fifteenth century",
    "15 secolo": "Fifteenth century",
    "15th century": "Fifteenth century",
    "fifteenth century": "Fifteenth century",
    "quattrocento": "Fifteenth century",

    "sedicesimo secolo": "Sixteenth century",
    "xvi secolo": "Sixteenth century",
    "16 secolo": "Sixteenth century",
    "16th century": "Sixteenth century",
    "sixteenth century": "Sixteenth century",
    "cinquecento": "Sixteenth century",

    "diciassettesimo secolo": "Seventeenth century",
    "xvii secolo": "Seventeenth century",
    "17 secolo": "Seventeenth century",
    "17th century": "Seventeenth century",
    "seventeenth century": "Seventeenth century",
    "seicento": "Seventeenth century",

    "diciottesimo secolo": "Eighteenth century",
    "xviii secolo": "Eighteenth century",
    "18 secolo": "Eighteenth century",
    "18th century": "Eighteenth century",
    "eighteenth century": "Eighteenth century",
    "settecento": "Eighteenth century",

    "diciannovesimo secolo": "Nineteenth century",
    "xix secolo": "Nineteenth century",
    "19 secolo": "Nineteenth century",
    "19th century": "Nineteenth century",
    "nineteenth century": "Nineteenth century",
    "ottocento": "Nineteenth century",

    "ventesimo secolo": "Twentieth century",
    "xx secolo": "Twentieth century",
    "20 secolo": "Twentieth century",
    "20th century": "Twentieth century",
    "twentieth century": "Twentieth century",
    "novecento": "Twentieth century",
}


def normalize(text):
    text = unicodedata.normalize("NFKD", text)
    text = "".join(
        char for char in text
        if not unicodedata.combining(char)
    )
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text):
    return [
        token
        for token in normalize(text).split()
        if len(token) > 2
    ]


def load_documents():
    with DOCUMENTS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def detect_topics(query, documents):
    """
    Detect official Govi topics from natural-language query.

    First checks known aliases, then checks whether an exact
    topic name occurs in the query.
    """
    normalized_query = normalize(query)
    detected = set()

    # Known aliases
    for alias, official_topic in {
        **TOPIC_ALIASES,
        **PERIOD_ALIASES,
    }.items():
        if normalize(alias) in normalized_query:
            detected.add(official_topic)

    # Exact official topics from the actual dataset
    all_topics = set()

    for document in documents:
        for topic in document.get("metadata", {}).get("topics", []):
            all_topics.add(topic)

    for topic in all_topics:
        if normalize(topic) in normalized_query:
            detected.add(topic)

    return sorted(detected)


def remove_detected_topic_words(query, detected_topics):
    """
    Keep the original query available for full-text search.
    This function only identifies the remaining free-text terms.
    """
    normalized = normalize(query)

    aliases_to_remove = []

    for alias, official_topic in {
        **TOPIC_ALIASES,
        **PERIOD_ALIASES,
    }.items():
        if official_topic in detected_topics:
            aliases_to_remove.append(normalize(alias))

    for alias in sorted(aliases_to_remove, key=len, reverse=True):
        normalized = normalized.replace(alias, " ")

    return normalized


def score_document(query, query_topics, document):
    metadata = document.get("metadata", {})
    topics = metadata.get("topics", [])

    normalized_topics = {
        normalize(topic): topic
        for topic in topics
    }

    score = 0.0
    reasons = []

    # ---------------------------------------------------------
    # 1. Strong topic matching
    # ---------------------------------------------------------
    for requested_topic in query_topics:
        if normalize(requested_topic) in normalized_topics:
            score += 10
            reasons.append(f"topic: {requested_topic}")

    # ---------------------------------------------------------
    # 2. Publication period matching through official topic
    # ---------------------------------------------------------
    # Already handled above because periods are official topics.
    # This section is intentionally kept separate conceptually.
    # It makes the ranking logic easier to extend later.
    # ---------------------------------------------------------

    # ---------------------------------------------------------
    # 3. Full-text matching
    # ---------------------------------------------------------
    free_text = remove_detected_topic_words(query, query_topics)
    query_tokens = tokenize(free_text)

    if query_tokens:
        text_tokens = tokenize(document.get("text", ""))
        counts = Counter(text_tokens)

        title_tokens = set(tokenize(document.get("title", "")))

        for token in query_tokens:
            frequency = counts.get(token, 0)

            if frequency:
                score += 1
                score += min(frequency - 1, 3) * 0.25

            if token in title_tokens:
                score += 3
                reasons.append(f"title: {token}")

    # ---------------------------------------------------------
    # 4. Bonus for documents matching ALL requested topics
    # ---------------------------------------------------------
    if query_topics:
        matched_topics = [
            topic
            for topic in query_topics
            if normalize(topic) in normalized_topics
        ]

        if len(matched_topics) == len(query_topics):
            score += 8
            reasons.append("all requested topics matched")

    return score, reasons


def search(query, documents, limit=10):
    query_topics = detect_topics(query, documents)

    results = []

    for document in documents:
        score, reasons = score_document(
            query,
            query_topics,
            document,
        )

        if score > 0:
            results.append(
                {
                    "score": score,
                    "document": document,
                    "reasons": reasons,
                }
            )

    results.sort(
        key=lambda item: (
            -item["score"],
            item["document"].get("title", ""),
        )
    )

    return query_topics, results[:limit]


def main():
    if len(sys.argv) < 2:
        print("Uso:")
        print('  python rag/search.py "astrologia XVI secolo"')
        sys.exit(1)

    query = " ".join(sys.argv[1:])

    documents = load_documents()

    query_topics, results = search(
        query,
        documents,
    )

    print()
    print(f'Query: "{query}"')

    if query_topics:
        print("Topic riconosciuti:")
        for topic in query_topics:
            print(f"  - {topic}")
    else:
        print("Topic riconosciuti: nessuno")

    print()
    print(f"Risultati: {len(results)}")
    print()

    if not results:
        print("Nessun risultato.")
        return

    for rank, result in enumerate(results, start=1):
        document = result["document"]
        metadata = document.get("metadata", {})

        print(f"{rank}. {document.get('title', '')}")
        print(f"   Score: {result['score']:.2f}")

        if result["reasons"]:
            print("   Match:")
            for reason in result["reasons"]:
                print(f"     - {reason}")

        print(f"   ID: {document.get('id', '')}")
        print(f"   Source ID: {metadata.get('source_id', '')}")
        print(f"   Anno: {metadata.get('publication_year', '')}")
        print(f"   Luogo: {metadata.get('publication_place', '')}")
        print(f"   Topics: {', '.join(metadata.get('topics', []))}")
        print(f"   Fonte: {metadata.get('source_url', '')}")
        print()


if __name__ == "__main__":
    main()
