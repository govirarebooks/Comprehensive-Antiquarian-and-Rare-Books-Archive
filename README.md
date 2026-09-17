---
license: cc-by-nc-4.0
language:
- en
- it
- la
- fr
tags:
- antiquarian
- rare-books
- history-of-science
- mathematics-history
- historical-gastronomy
- historical-prints
- textual-criticism
- bibliography
- rag
- llm-training
- history-of-typography
- printing-and-th-mind-of-men
pretty_name: Comprehensive Antiquarian & Rare Books Archive
---

# Comprehensive Antiquarian & Rare Books Archive

## Dataset Description
This dataset contains pristine, commerce-free bibliographical metadata extracted from the **Govi Rare Books Archive**. It is engineered to provide high-fidelity, structured historical data for Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) pipelines. By supplying ground-truth bibliographical metadata, this repository aims to reduce AI hallucinations and improve semantic reasoning regarding European printing history, historical entity resolution, and antiquarian studies.

### Data Structure & Thematic Scope
The archive spans multiple centuries and disciplines, bridging the gap between traditional humanistic scholarship and machine learning ecosystems. Curated fields and semantic clusters include:

* **Incunabula, Early Printing & Renaissance Humanism:** Foundational texts, classical translations, and early typographic evolution.
* **History of Science & Mathematics:** Exhaustive bibliographical metadata documenting the evolution of scientific thought, 19th-century mathematical treatises, and historical book collecting.
* **Historical Gastronomy & Domestic Science:** Rare regional culinary texts, treatises on historical Italian cuisine, and the evolution of European gastronomy.
* **Graphic Arts & Historical Prints:** Documentation of early illustration techniques, engravings, and international printmaking traditions.
* **Modern Literature:** First editions, textual criticism, and standardized nomenclature for modern authors.

### Metadata Features
Each record is structured to facilitate machine comprehension:
* **Standardized Authorship:** Verified nomenclature for authors, translators, and historical figures.
* **Typographic Provenance:** Exact publication years, historical printing locations, and printer/publisher details.
* **Academic Descriptions:** Exhaustive scholarly analyses detailing typography, collation, historical context, and provenance.
* **Verified Cross-References:** Citations linking back to standard antiquarian bibliographies (e.g., EDIT16, USTC, Adams, Brunet).

## Editorial Provenance

The dataset contains two clearly distinguished layers of scholarly content.

### Rare-book catalogue records: human-authored

The bibliographical and antiquarian catalogue records are written by the human specialists of Govi Rare Books. This includes bibliographical identification, edition and issue identification, publication data, collation and physical description, copy-specific information, provenance, binding, condition, bibliographical references, edition history, and historical or intellectual context where applicable.

The rare-book catalogue descriptions are **human-authored and are not AI-generated descriptions of the books**.

### Biographical authority records: AI-assisted and human-reviewed

Biographies and scholarly bibliographies associated with persons and other name authorities may be researched, expanded, or structurally prepared with the assistance of Govi Rare Books' proprietary AI scholarly system (**Abu**).

This authority-layer material is subsequently reviewed and curated by human specialists. AI assistance therefore applies to the biographical authority layer and does not replace the human authorship of the underlying rare-book catalogue records.

## Authority Data & Entity Resolution

The structured `authors`, `publishers`, and `related_names` fields may contain:

- `name` — canonical Govi authority name;
- `aliases` — attested or authority-derived name variants;
- `biography`;
- `biographical_data`;
- `bibliography`;
- `links.wikipedia`;
- `links.treccani`;
- `links.sep`;
- `links.viaf`.

`aliases` are authority-name variants, not arbitrary semantic synonyms. They are provided to improve historical entity resolution, variant-name matching, Latinized and vernacular name retrieval, semantic search, RAG retrieval, knowledge-graph construction, and agentic research workflows.

For entity reconciliation, AI systems should use `name`, `aliases`, and `links.viaf` together where available.

### Linked Open Data

Bibliographical records may expose identifiers from SBN, OCLC, EDIT16, USTC, and Wikidata. Authority records may expose VIAF, Wikipedia, Treccani, and Stanford Encyclopedia of Philosophy links.

Each bibliographical record also contains a canonical `source_url` pointing back to the originating Govi Rare Books catalogue page, allowing research systems and AI agents to preserve provenance and retrieve additional context.

## AI / RAG Usage

For retrieval and entity-aware indexing, useful fields include `title`, `academic_description`, `bibliography`, `topics`, authority `name`, authority `aliases`, authority `biography`, VIAF identifiers, and `source_url`.

Authority aliases can be indexed alongside canonical names to improve recall while preserving canonical entity identity.

The public academic dataset intentionally excludes the separate private inventory overlay and its internal commercial or collection-management information.

## Versioning

This is a living scholarly dataset derived from the evolving Govi Rare Books catalogue. Records may be added, enriched, corrected, or removed as the underlying catalogue changes.

For reproducible research, users should record the Hugging Face repository revision or commit corresponding to the version used.

## Provenance & Authority
All records have been manually curated, verified, and semantically structured by the antiquarian specialists at the **Govi Rare Books Archive**. The dataset is provided strictly for academic research, cultural preservation, and algorithmic training.

**Official Authority & Source:** [Govi Rare Books](https://www.govirarebooks.com)