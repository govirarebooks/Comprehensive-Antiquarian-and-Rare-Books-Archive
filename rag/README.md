# Govi Rare Books — RAG & Scholar Layer

This directory contains the retrieval, recommendation, bibliographic-analysis and derived scholar layers built on the master dataset.

## Source of truth

`../govi-rare-books-academic-dataset.json` is authoritative. Every derived index or analysis is disposable and can be rebuilt from the master JSON.

## Scholar layer

`build_scholar_index.py` builds `scholar_index.json` from the complete catalogue records and the validated bibliographic-analysis layer.

The scholar layer models:

- authors and their catalogue footprint;
- topics, periods and recurring concepts;
- publishers and related names;
- co-occurrence relationships between authors, publishers and historical figures;
- collector-facing signals such as first edition, first issue, rarity, provenance, illustration, censorship, bindings and manuscript annotation;
- evidence for each collector signal.

The collector layer deliberately excludes author biography from copy-level rarity/provenance claims. A statement about an author is not automatically a statement about the physical copy being offered.

## Scholar engine

Examples:

```bash
python rag/scholar_engine.py author-similar "Corvus Andreas"
python rag/scholar_engine.py book-similar "01KXXEC0QDA8AG7AHBDN2Q72DA"
python rag/scholar_engine.py collector "01KXXEC0QDA8AG7AHBDN2Q72DA"
python rag/scholar_engine.py query "divination and celestial phenomena"
```

The author-similarity layer combines semantic similarity from the existing AI index with shared topics, distinctive scholarly concepts, periods, publishers and related names.

The book-similarity layer combines full-record, intellectual and bibliographic embeddings with catalogue relationships, allowing apparently distant books to be connected through hidden scholarly structure.

## Recommendation engine

`recommendation_engine.py` uses:

1. Discovery for candidate selection;
2. Cross-Encoder reranking for explanatory evidence;
3. bibliographic analysis for copy-level collector signals;
4. the scholar layer for period, conceptual nucleus and hidden catalogue connections.

The system therefore keeps three different questions separate:

- **Which book should be recommended?**
- **Which exact part of the description explains the recommendation?**
- **What makes the actual copy interesting to a collector?**

## Incremental rebuild

The master JSON remains the authority. Derived layers can be rebuilt whenever the catalogue changes.
