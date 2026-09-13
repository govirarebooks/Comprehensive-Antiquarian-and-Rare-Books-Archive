#!/usr/bin/env python3
from scholar_taxonomy import concepts_from_text, conceptual_affinity

text = "The author practiced palmistry for medicine and surgery and was associated with astrology and divination."
found = set(concepts_from_text(text))
assert {"chiromancy", "medicine", "astrology", "divination"}.issubset(found)
assert "natural_magic" not in concepts_from_text("A treatise on natural philosophy and astronomy")
score, links = conceptual_affinity(["chiromancy"], ["physiognomy"])
assert score >= 0.9 and links
print("GOVI EXTERNAL SCHOLAR CONCEPT TEST")
print("PASS")
