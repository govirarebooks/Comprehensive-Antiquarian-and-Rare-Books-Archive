from __future__ import annotations

import re

GENERIC_CONCEPTS = {
    "rarity", "printing", "religion", "humanism", "medicine",
    "history_of_science", "illustration", "censorship"
}

CONCEPT_TERMS = {
    "astrology": {"astrology", "astrological", "astrologer", "planet", "zodiac", "eclipse", "conjunction"},
    "astronomy": {"astronomy", "astronomical", "astronomer", "celestial", "heavens", "stars"},
    "divination": {"divination", "divinatory", "prophecy", "oracle", "prognostication", "prediction", "practica"},
    "omens": {"omen", "omens", "portent", "signs", "foreshadow", "future calamities"},
    "chiromancy": {"chiromancy", "palmistry", "palm reading", "palmists", "hand reading", "lines of the hand"},
    "physiognomy": {"physiognomy", "physiognomic", "physiognomist"},
    "natural_magic": {"natural magic", "occult sciences", "magic", "magical"},
    "witchcraft": {"witchcraft", "witches", "witch trials", "sorcery", "witch craze"},
    "medicine": {"medicine", "medical", "physician", "surgery", "anatomy", "humours", "galenic"},
    "natural_philosophy": {"natural philosophy", "natural philosopher"},
    "printing": {"printer", "printing", "press", "typography", "bookshop", "bookseller"},
    "humanism": {"humanist", "humanism", "renaissance", "classical", "philology"},
    "religion": {"church", "theology", "religious", "reformation", "catholic", "lutheran", "protestant"},
    "censorship": {"index", "censorship", "censored", "forbidden books", "prohibited books"},
    "physiology": {"physiology", "bodily", "body", "temperament", "complexion"},
}

CONCEPT_RELATIONS = {
    "astrology": {"astrology": 1.0, "divination": .95, "omens": .90, "astronomy": .86, "natural_magic": .82, "history_of_science": .72},
    "astronomy": {"astronomy": 1.0, "astrology": .86, "history_of_science": .90, "natural_philosophy": .82},
    "divination": {"divination": 1.0, "astrology": .95, "omens": .92, "natural_magic": .84, "witchcraft": .68},
    "omens": {"omens": 1.0, "divination": .92, "astrology": .90, "natural_magic": .78, "witchcraft": .66},
    "chiromancy": {"chiromancy": 1.0, "physiognomy": .90, "divination": .86, "natural_magic": .74, "medicine": .65},
    "physiognomy": {"physiognomy": 1.0, "chiromancy": .90, "divination": .84, "medicine": .62, "natural_magic": .70, "physiology": .80},
    "natural_magic": {"natural_magic": 1.0, "astrology": .82, "divination": .84, "omens": .78, "witchcraft": .72, "chiromancy": .74, "medicine": .68, "history_of_science": .58},
    "witchcraft": {"witchcraft": 1.0, "natural_magic": .72, "divination": .68, "omens": .66, "religion": .60},
    "medicine": {"medicine": 1.0, "chiromancy": .65, "physiognomy": .62, "natural_magic": .68, "history_of_science": .78, "physiology": .84},
    "history_of_science": {"history_of_science": 1.0, "medicine": .78, "astronomy": .90, "astrology": .72, "natural_philosophy": .92, "natural_magic": .58},
    "natural_philosophy": {"natural_philosophy": 1.0, "history_of_science": .92, "astronomy": .82, "medicine": .70, "natural_magic": .28},
    "physiology": {"physiology": 1.0, "medicine": .84, "physiognomy": .80},
    "printing": {"printing": 1.0, "humanism": .42},
    "humanism": {"humanism": 1.0, "printing": .42, "religion": .25},
    "religion": {"religion": 1.0, "witchcraft": .60, "censorship": .55, "humanism": .25},
    "censorship": {"censorship": 1.0, "religion": .55, "witchcraft": .48},
}


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def concepts_from_text(text: str) -> list[str]:
    hay = norm(text)
    found = []
    for concept, terms in CONCEPT_TERMS.items():
        if any(term in hay for term in terms):
            found.append(concept)
    return sorted(set(found))


def conceptual_affinity(source_concepts, target_concepts):
    source = [c for c in source_concepts if c in CONCEPT_RELATIONS]
    target = [c for c in target_concepts if c in CONCEPT_RELATIONS]
    if not source or not target:
        return 0.0, []
    best_pairs = []
    best_score = 0.0
    for left in source:
        for right in target:
            score = CONCEPT_RELATIONS.get(left, {}).get(right, 0.0)
            if score <= 0:
                continue
            pair = {"from": left, "to": right, "strength": round(float(score), 2)}
            if score > best_score + 1e-9:
                best_score = score
                best_pairs = [pair]
            elif abs(score - best_score) < 1e-9:
                best_pairs.append(pair)
    exact = set(source) & set(target)
    exact_score = min(1.0, len(exact) / 2.0)
    if exact_score >= best_score:
        best_score = exact_score
        best_pairs = [{"from": c, "to": c, "strength": 1.0} for c in sorted(exact)]
    return float(best_score), best_pairs[:4]
