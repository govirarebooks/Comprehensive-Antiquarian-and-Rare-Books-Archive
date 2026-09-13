# Govi External Scholar Knowledge

This directory contains derived context fetched from external public knowledge sources.

## Authority boundary

`govi-rare-books-academic-dataset.json` remains the authoritative source for facts about books and copies in the Govi catalogue.

This layer adds contextual knowledge about entities that occur in the catalogue. It must never be used to overwrite:

- edition or issue facts
- rarity or copy counts
- provenance of the present copy
- catalogue descriptions
- Govi ownership facts

## Sources

- **Wikidata**: structured entity identity and relationships.
- **Wikipedia / Wikimedia**: narrative context and summaries.
- **DBpedia**: secondary entity resolution/context.

The collector uses the Wikimedia/MediaWiki APIs and DBpedia Lookup; all retrieved items retain source URLs and entity identifiers where available.

## Incremental behavior

The cache is keyed by the exact catalogue entity name. Existing entries are reused unless `--refresh` is supplied.

Typical run:

```bash
python rag/build_external_knowledge.py
```

To include publishers too:

```bash
python rag/build_external_knowledge.py --scope authors,related_names,publishers
```

To rebuild everything from external sources:

```bash
python rag/build_external_knowledge.py --scope authors,related_names,publishers --refresh
```
