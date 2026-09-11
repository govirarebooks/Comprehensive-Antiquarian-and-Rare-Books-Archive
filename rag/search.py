import json
import re
import sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS_PATH = ROOT / "rag" / "documents.json"


def normalize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return text


def tokenize(text):
    return [
        token
        for token in normalize(text).split()
        if len(token) > 2
    ]


def load_documents():
    with DOCUMENTS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def score_document(query_tokens, document):
    text_tokens = tokenize(document["text"])

    if not text_tokens:
        return 0

    counts = Counter(text_tokens)
    score = 0

    for token in query_tokens:
        frequency = counts.get(token, 0)

        if frequency:
            # Presenza della parola
            score += 1

            # Piccolo bonus per le occorrenze multiple
            score += min(frequency - 1, 3) * 0.25

    # Bonus se il termine compare nel titolo
    title_tokens = set(tokenize(document.get("title", "")))

    for token in query_tokens:
        if token in title_tokens:
            score += 3

    return score


def search(query, documents, limit=10):
    query_tokens = tokenize(query)

    if not query_tokens:
        return []

    results = []

    for document in documents:
        score = score_document(query_tokens, document)

        if score > 0:
            results.append((score, document))

    results.sort(
        key=lambda item: (-item[0], item[1].get("title", ""))
    )

    return results[:limit]


def main():
    if len(sys.argv) < 2:
        print("Uso:")
        print('  python rag/search.py "astrologia XVI secolo"')
        sys.exit(1)

    query = " ".join(sys.argv[1:])

    documents = load_documents()
    results = search(query, documents)

    print()
    print(f'Query: "{query}"')
    print(f"Risultati: {len(results)}")
    print()

    if not results:
        print("Nessun risultato.")
        return

    for rank, (score, document) in enumerate(results, start=1):
        metadata = document.get("metadata", {})

        print(f"{rank}. {document.get('title', '')}")
        print(f"   Score: {score:.2f}")
        print(f"   ID: {document.get('id', '')}")
        print(f"   Source ID: {metadata.get('source_id', '')}")
        print(f"   Anno: {metadata.get('publication_year', '')}")
        print(f"   Luogo: {metadata.get('publication_place', '')}")
        print(f"   Fonte: {metadata.get('source_url', '')}")
        print()


if __name__ == "__main__":
    main()
