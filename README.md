# BeamRAG: Query-Guided Heuristic Beam Search with Macro-Gating on Knowledge Graphs

Official replication package and evaluation benchmark logs for the paper:
> **BeamRAG: Query-Guided Heuristic Beam Search with Macro-Gating on Knowledge Graphs**  
> *Vincenzo Forciniti* (Independent Researcher)

---

## Repository Structure

```
beamrag/
├── README.md
├── LICENSE                      # MIT License
├── prompts/                     # 500 Curated benchmark queries across 4 domains
│   ├── computer_science.json    # 125 queries (CS domain)
│   ├── agriculture.json         # 125 queries (Agricultural science)
│   ├── mix.json                 # 125 queries (Heterogeneous multi-domain)
│   └── legal.json               # 125 queries (Contractual & administrative law)
├── logs/                        # 1,000 Paired evaluation logs across bidirectional tracks
│   ├── track1/                  # Track 1: Gemma 26B Generator -> Qwen 27B Judge
│   │   ├── computer_science.json
│   │   ├── agriculture.json
│   │   ├── mix.json
│   │   └── legal.json
│   └── track2/                  # Track 2: Qwen 27B Generator -> Gemma 26B Judge
│       ├── computer_science.json
│       ├── agriculture.json
│       ├── mix.json
│       └── legal.json
└── analysis/
    └── reproduce_tables.py      # Standalone deterministic replication script
```

---

## Replication Instructions

The replication script requires only standard Python 3 (no third-party dependencies required):

```bash
python3 analysis/reproduce_tables.py
```

This deterministically recalculates and formats all paper tables directly from the query-level judge logs:
- **Table 3b**: Soft-Tie Sensitivity Spectrum ($\epsilon \in [0.00, 0.20]$, $N=1,000$).
- **Table 4 & 5**: Computer Science domain win rates and continuous score shares ($N=125$).
- **Table 6 & 7**: Agricultural Science domain win rates and continuous score shares ($N=125$).
- **Table 8c**: Cross-Track Comparative Consistency across inverted tracks ($N=1,000$).

---

## Benchmark & Experimental Details

- **Corpora**: 4 diverse domains spanning Computer Science, Agricultural Science, Heterogeneous Multi-Domain, and Legal contracts.
- **Host & Application Node**: Apple MacBook Pro (M1 Pro, 16 GB RAM), hosting the isolated Docker microservice network (`aia-backend`, `aia-lightrag`, PostgreSQL 16 with `pgvector` v0.7.0).
- **Inference Server**: Dedicated Neural Inference Server (ASUS Ascent GX10, 128 GB RAM) running LM Studio local inference endpoints (`google/gemma-4-26b-a4b` and `qwen/qwen3.8-27b`).
- **Database Engine**: PostgreSQL 16 with pgvector (HNSW vector indices) and relational property-graph tables, traversed via application-tier Python engine (Listing C.1).
- **Evaluated LightRAG Baseline**: `lightrag-hku==1.5.7` (commit `28ff1b0`), retrieval configuration `hybrid`, file-backed storage (`nano-vectordb` + `NetworkX`).

---

## Citation

If you use this benchmark or replication package in your research, please cite:

```bibtex
@article{forciniti2026beamrag,
  title={BeamRAG: Query-Guided Heuristic Beam Search with Macro-Gating on Knowledge Graphs},
  author={Forciniti, Vincenzo},
  journal={arXiv preprint},
  year={2026}
}
```

## License

This replication package is released under the MIT License.
