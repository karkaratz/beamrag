# BeamRAG — Benchmark Replication Package

Replication data for "BeamRAG: Query-Guided Heuristic Beam Search
with Macro-Gating on Knowledge Graphs".

## Contents
- `prompts/` — 500 evaluation queries (4 domains × 125)
- `logs/` — raw query-level judge logs, 1,000 paired executions (Tracks 1–2)
- `analysis/` — scripts reproducing Tables 4–8c and all statistical tests

## Reproducing the results
1. `python analysis/reproduce_tables.py`
2. Output: win/loss/tie totals, win rates, exact binomial sign tests,
   Holm–Bonferroni correction, between-track z-tests.

## Notes
- The BeamRAG traversal algorithm is fully specified in Appendix C of the paper.
- The LightRAG baseline uses the official `lightrag-hku` package.
- Code: MIT License. Data & prompts: CC BY 4.0.
