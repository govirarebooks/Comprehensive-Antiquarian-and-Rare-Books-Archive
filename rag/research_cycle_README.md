# Govi Scholar Research Cycle

To make the AI study one Govi catalogue record and then return to the catalogue:

```bash
python rag/research_cycle.py 01KXXEC0QDA8AG7AHBDN2Q72DA
```

The cycle is:

`Govi record → external context → scholar context → Govi graph → similar books`

The source dataset is never modified.
