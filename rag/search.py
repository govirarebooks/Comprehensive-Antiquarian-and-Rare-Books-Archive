import json
import re
import sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS_PATH = ROOT / "rag" / "documents.json"

# Parole troppo generiche per essere utili nel ranking.
STOPWORDS = {
    "a", "ad", "al", "alla", "alle", "allo", "ai", "agli",
    "da", "dal", "dalla", "dalle", "dallo", "dai", "dagli",
    "di", "del", "della", "delle", "dello", "dei", "degli",
    "e", "ed", "in", "nel", "nella", "nelle", "nello",
    "nei", "negli", "su", "sul", "sulla", "sulle", "sullo",
    "per", "con", "tra", "fra",
    "il", "lo", "la", "i", "gli", "le",
    "un", "uno", "una",
    "the", "a", "an", "of", "and", "in", "on", "for", "with",
    "from", "to", "by",
    "libro", "libri", "book", "books",
    "secolo", "secoli", "century", "centuries",
}

# Traduzioni/varianti italiane delle categorie bibliografiche.
# Le categorie ufficiali vengono comunque ricavate direttamente
# dai documenti: questa mappa serve solo per capire la query italiana.
ITALIAN_ALIASES = {
    "astrologia": "astrology",
    "astronomia": "astronomy",
    "medicina": "medicine",
    "farmacia": "pharmacy",
    "chirurgia": "surgery",
    "anatomia": "anatomy",
    "filosofia": "philosophy",
    "teologia": "theology",
    "religione": "religion",
    "letteratura": "literature",
    "poesia": "poetry",
    "poetica": "poetics",
    "storia": "history",
    "geografia": "geography",
    "matematica": "mathematics",
    "algebra": "algebra",
    "geometria": "geometry",
    "fisica": "physics",
    "chimica": "chemistry",
    "botanica": "botany",
    "zoologia": "zoology",
    "agricoltura": "agriculture",
    "diritto": "law",
    "giurisprudenza": "law",
    "politica": "politics",
    "arte": "art",
    "architettura": "architecture",
    "musica": "music",
    "linguistica": "linguistics",
    "grammatica": "grammar",
    "filologia": "philology",
    "bibliografia": "bibliography",
    "stampa": "printing",
    "tipografia": "typography",
    "esoterismo": "esoterica",
    "stregoneria": "witchcraft",
}

# Periodi: vengono trasformati nel nome della categoria ufficiale.
PERIOD_ALIASES = {
    "incunaboli": "Incunabula",
    "incunabolo": "Incunabula",
    "incunabula": "Incunabula",

    "cinquecento": "Sixteenth Century",
    "xvi secolo": "Sixteenth Century",
    "16 secolo": "Sixteenth Century",
    "16° secolo": "Sixteenth Century",
    "sedicesimo secolo": "Sixteenth Century",
    "sixteenth century": "Sixteenth Century",

    "seicento": "Seventeenth Century",
    "xvii secolo": "Seventeenth Century",
    "17 secolo": "Seventeenth Century",
    "17° secolo": "Seventeenth Century",
    "diciassettesimo secolo": "Seventeenth Century",
    "seventeenth century": "Seventeenth Century",

    "settecento": "Eighteenth Century",
    "xviii secolo": "Eighteenth Century",
    "18 secolo": "Eighteenth Century",
    "18° secolo": "Eighteenth Century",
    "diciottesimo secolo": "Eighteenth Century",
    "eighteenth century": "Eighteenth Century",

    "ottocento": "Nineteenth Century",
    "xix secolo": "Nineteenth Century",
    "19 secolo": "Nineteenth Century",
    "19° secolo": "Nineteenth Century",
    "diciannovesimo secolo": "Nineteenth Century",
    "nineteenth century": "Nineteenth Century",

    "novecento": "Twentieth Century",
    "xx secolo": "Twentieth Century",
    "20 secolo": "Twentieth Century",
    "20° secolo": "Twentieth Century",
    "ventesimo secolo": "Twentieth Century",
    "twentieth century": "Twentieth Century",
}


def normalize(text):
    """Normalizza il testo per confronti robusti."""
    text = str(text or "").lower()
    text = text.replace("’", "'")
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text):
    return set(normalize(text).split())


def topic_key(topic):
    return normalize(topic)


def build_topic_index(documents):
    """
    Ricava la tassonomia direttamente dai documenti RAG.

    Questo è importante: search.py non contiene una lista chiusa
    dei 97 topic. Se la tassonomia cambia, il motore la vede
    automaticamente.
    """
    topics = {}

    for doc in documents:
        for topic in doc.get("metadata", {}).get("topics", []):
            key = topic_key(topic)
            if key:
                topics[key] = topic

    return topics


def detect_periods(query):
    """
    Riconosce i periodi storici prima del normale token matching.
    """
    normalized = normalize(query)
    detected = []

    # Prima le espressioni più lunghe.
    aliases = sorted(
        PERIOD_ALIASES.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    )

    for alias, official_topic in aliases:
        if normalize(alias) in normalized:
            if official_topic not in detected:
                detected.append(official_topic)

    return detected


def detect_topics(query, official_topics):
    """
    Individua i topic richiesti dall'utente.

    Strategia:
    1. periodi storici;
    2. alias italiani;
    3. match diretto con i topic ufficiali;
    4. match per parole significative dei topic.
    """
    normalized_query = normalize(query)
    query_tokens = tokenize(query)

    detected = []

    # ---------------------------------------------------------
    # 1. PERIODI
    # ---------------------------------------------------------
    for topic in detect_periods(query):
        # Accetta il topic solo se esiste realmente nella tassonomia.
        key = topic_key(topic)
        if key in official_topics:
            detected.append(official_topics[key])
        else:
            # fallback case-insensitive
            for official_key, official_topic in official_topics.items():
                if official_key == key:
                    detected.append(official_topic)
                    break

    # ---------------------------------------------------------
    # 2. ALIAS ITALIANI
    # ---------------------------------------------------------
    for italian, english in ITALIAN_ALIASES.items():
        if italian in query_tokens:
            english_key = normalize(english)

            # Cerca il topic ufficiale più pertinente.
            exact = official_topics.get(english_key)
            if exact and exact not in detected:
                detected.append(exact)
                continue

            # Esempio:
            # medicina -> "Medicine Books"
            english_tokens = set(english_key.split())

            for official_key, official_topic in official_topics.items():
                official_tokens = set(official_key.split())

                if english_tokens and english_tokens.issubset(official_tokens):
                    if official_topic not in detected:
                        detected.append(official_topic)
                    break

    # ---------------------------------------------------------
    # 3. MATCH DIRETTO DI UN TOPIC UFFICIALE
    # ---------------------------------------------------------
    # Prima proviamo frasi intere.
    for official_key, official_topic in official_topics.items():
        if official_topic in detected:
            continue

        if official_key in normalized_query:
            detected.append(official_topic)

    # ---------------------------------------------------------
    # 4. MATCH PER PAROLE SIGNIFICATIVE
    # ---------------------------------------------------------
    # Evitiamo che "libri", "secolo", ecc. generino falsi topic.
    meaningful_query_tokens = {
        token
        for token in query_tokens
        if token not in STOPWORDS and len(token) >= 4
    }

    for official_key, official_topic in official_topics.items():
        if official_topic in detected:
            continue

        topic_tokens = {
            token
            for token in official_key.split()
            if token not in STOPWORDS and len(token) >= 4
        }

        if not topic_tokens:
            continue

        # Match forte: tutte le parole significative del topic
        # sono presenti nella query.
        if topic_tokens.issubset(meaningful_query_tokens):
            detected.append(official_topic)
            continue

        # Match per singola parola importante, ma solo se il topic
        # non è composto da una parola troppo generica.
        if len(topic_tokens) == 1 and topic_tokens & meaningful_query_tokens:
            detected.append(official_topic)

    # Mantieni l'ordine e rimuovi duplicati.
    result = []
    seen = set()

    for topic in detected:
        key = topic_key(topic)
        if key not in seen:
            result.append(topic)
            seen.add(key)

    return result


def meaningful_query_tokens(query, detected_topics):
    """
    Restituisce le parole della query che NON sono già state
    interpretate come topic/periodi.
    """
    tokens = [
        token
        for token in tokenize(query)
        if token not in STOPWORDS and len(token) >= 3
    ]

    topic_tokens = set()

    for topic in detected_topics:
        topic_tokens.update(
            token
            for token in tokenize(topic)
            if token not in STOPWORDS
        )

    return [
        token
        for token in tokens
        if token not in topic_tokens
    ]


def get_doc_topics(doc):
    return doc.get("metadata", {}).get("topics", [])


def topic_matches(doc_topics, requested_topic):
    """
    Confronto case-insensitive con la tassonomia ufficiale.
    """
    requested_key = topic_key(requested_topic)

    return any(
        topic_key(topic) == requested_key
        for topic in doc_topics
    )


def score_document(doc, requested_topics, free_tokens):
    """
    Ranking bibliografico.

    I topic sono il segnale principale.
    Il testo libero serve come segnale secondario.
    """
    title = doc.get("title", "")
    text = doc.get("text", "")
    doc_topics = get_doc_topics(doc)

    score = 0.0
    reasons = []
    matched_topics = []

    # ---------------------------------------------------------
    # TOPIC MATCH
    # ---------------------------------------------------------
    for requested_topic in requested_topics:
        if topic_matches(doc_topics, requested_topic):
            score += 15.0
            matched_topics.append(requested_topic)
            reasons.append(f"topic: {requested_topic}")

    # Bonus forte se il documento contiene TUTTI i topic richiesti.
    if requested_topics and len(matched_topics) == len(requested_topics):
        score += 25.0
        reasons.append("all requested topics matched")

    # ---------------------------------------------------------
    # TITLE
    # ---------------------------------------------------------
    title_tokens = tokenize(title)

    for token in free_tokens:
        if token in title_tokens:
            score += 4.0
            reasons.append(f"title: {token}")

    # ---------------------------------------------------------
    # FULL TEXT
    # ---------------------------------------------------------
    normalized_text = normalize(text)

    for token in free_tokens:
        if token in normalized_text:
            score += 1.0

    # ---------------------------------------------------------
    # PENALTY PER TOPIC MANCANTE
    # ---------------------------------------------------------
    missing_topics = [
        topic
        for topic in requested_topics
        if topic not in matched_topics
    ]

    if requested_topics and missing_topics:
        score -= 12.0 * len(missing_topics)

    return score, reasons, matched_topics, missing_topics


def search(documents, query, limit=10):
    official_topics = build_topic_index(documents)

    requested_topics = detect_topics(query, official_topics)
    free_tokens = meaningful_query_tokens(query, requested_topics)

    results = []

    for doc in documents:
        score, reasons, matched_topics, missing_topics = score_document(
            doc,
            requested_topics,
            free_tokens,
        )

        # Se ci sono topic richiesti, il documento deve avere almeno
        # un segnale pertinente per entrare nella classifica.
        if requested_topics:
            if not matched_topics and not free_tokens:
                continue

        else:
            # Ricerca libera: almeno una parola deve comparire.
            if not free_tokens:
                continue

            normalized_text = normalize(doc.get("text", ""))
            normalized_title = normalize(doc.get("title", ""))

            if not any(
                token in normalized_text or token in normalized_title
                for token in free_tokens
            ):
                continue

        results.append({
            "doc": doc,
            "score": score,
            "reasons": reasons,
            "matched_topics": matched_topics,
            "missing_topics": missing_topics,
        })

    # ---------------------------------------------------------
    # SE ESISTONO RISULTATI CHE MATCHANO TUTTI I TOPIC,
    # NON INQUINARE LA LISTA CON RISULTATI PARZIALI.
    # ---------------------------------------------------------
    if requested_topics:
        complete_results = [
            result
            for result in results
            if len(result["matched_topics"]) == len(requested_topics)
        ]

        if complete_results:
            results = complete_results

    # Ordinamento:
    # 1. score
    # 2. numero di topic corrispondenti
    # 3. anno più antico prima, come tie-break bibliografico
    results.sort(
        key=lambda result: (
            result["score"],
            len(result["matched_topics"]),
            -(result["doc"].get("metadata", {}).get("publication_year") or 9999),
        ),
        reverse=True,
    )

    return results[:limit], requested_topics, free_tokens


def print_results(query, results, requested_topics, free_tokens):
    print()
    print("=" * 80)
    print(f"QUERY: {query}")
    print("=" * 80)

    print()
    print("Topic riconosciuti:")

    if requested_topics:
        for topic in requested_topics:
            print(f"  - {topic}")
    else:
        print("  - nessuno")

    print()
    print("Termini liberi:")
    if free_tokens:
        print("  - " + ", ".join(free_tokens))
    else:
        print("  - nessuno")

    print()
    print(f"Risultati: {len(results)}")
    print()

    if not results:
        print("Nessun risultato.")
        return

    for index, result in enumerate(results, start=1):
        doc = result["doc"]
        metadata = doc.get("metadata", {})

        print(f"{index}. {doc.get('title', '')}")
        print(f"   Score: {result['score']:.2f}")

        if result["matched_topics"]:
            print(
                "   Topic matched: "
                + ", ".join(result["matched_topics"])
            )

        if result["missing_topics"]:
            print(
                "   Topic mancanti: "
                + ", ".join(result["missing_topics"])
            )

        if result["reasons"]:
            print("   Match:")
            for reason in result["reasons"]:
                print(f"     - {reason}")

        print(
            f"   Anno: {metadata.get('publication_year')}"
        )
        print(
            f"   Luogo: {metadata.get('publication_place')}"
        )

        topics = metadata.get("topics", [])
        if topics:
            print("   Topics: " + ", ".join(topics))

        print(
            f"   ID: {metadata.get('source_id')}"
        )
        print(
            f"   Source: {metadata.get('source_url')}"
        )
        print()


def main():
    if len(sys.argv) < 2:
        print(
            'Uso: python rag/search.py "testo della ricerca"'
        )
        sys.exit(1)

    query = " ".join(sys.argv[1:])

    with DOCUMENTS_PATH.open("r", encoding="utf-8") as file:
        documents = json.load(file)

    results, requested_topics, free_tokens = search(
        documents,
        query,
        limit=10,
    )

    print_results(
        query,
        results,
        requested_topics,
        free_tokens,
    )


if __name__ == "__main__":
    main()
