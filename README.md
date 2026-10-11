# BeamRAG: Query-Guided Heuristic Beam Search with Macro-Gating on Knowledge Graphs

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23294593.svg)](https://doi.org/10.5281/zenodo.23294593)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official replication package and evaluation benchmark logs for the paper:
> **BeamRAG: Query-Guided Heuristic Beam Search with Macro-Gating on Knowledge Graphs**  
> *Vincenzo Forciniti* (Independent Researcher)

---

## Overview: What is BeamRAG?

**BeamRAG** is a hybrid Retrieval-Augmented Generation (RAG) architecture engineered to resolve the **multi-hop cross-document reasoning gap** while actively suppressing **semantic drift** across heterogeneous knowledge graphs.

Standard dense retrieval often fails when critical supporting evidence is scattered across distinct, interconnected documents. Conversely, unconstrained graph traversal algorithms easily drift into irrelevant corpora when crossing polysemous entities or generic connector nodes.

BeamRAG resolves this trade-off through three core mechanisms unified within a **PostgreSQL** substrate:

1. **Query-Guided Heuristic Beam Search ($N$-Hop Pathfinding)**:
   - Traverses multi-relational property graphs via bounded beam search ($B = 6$, depth horizon $L \le 4$).
   - Explicitly navigates semantic cross-document bridges (`SAME_AS`, $w_{\text{rel}} = 1.00$) and temporal revisions (`SUPERSEDES`, $w_{\text{rel}} = 0.90$).
   - Attenuates distraction from high-degree topological hubs ("Document", "System") via dynamic logarithmic suppression:
     $$\frac{1}{\log_2(2 + \deg(v))}$$

2. **Decoupled Hierarchical Thresholding (DHT & Macro-Gating)**:
   - **Micro-admissibility ($\tau_{\text{node}} = 0.40$)**: Enforces query semantic relevance on every visited candidate entity, pruning irrelevant branches early.
   - **Macro-document coherence gating ($\tau_{\text{coherence}} = 0.65$)**: Checks whether the document provenance of an edge satisfies query relevance ($\max_{c \in \mathcal{C}(D)} \cos(\mathbf{e}_c, \mathbf{e}_q) \ge 0.65$), pruning cross-corpus contamination before it bleeds into unrelated documents.

3. **Asymmetric Dual-Channel Retrieval & Graph–Text Alignment Reranking**:
   - Decouples textual passage retrieval (fusing dense similarity, lexical matching, and graph entity density via Reciprocal Rank Fusion) from graph traversal.
   - Re-ranks harvested relational triples using a joint objective ($\beta = 0.60$) that scores traversal confidence against surface-entity mention alignment in the top retrieved passages before LLM prompt injection.

---

## Repository Structure

```
beamrag/
├── README.md
├── LICENSE                      # MIT License
├── kg_exports/                  # Exported knowledge graphs & text chunks (entities, relations, chunks)
│   ├── custom_kg_cs.json.gz     # Computer Science domain
│   ├── custom_kg_agriculture.json.gz # Agricultural Science domain
│   ├── custom_kg_mix.json.gz    # Heterogeneous Multi-Domain (MIX)
│   └── custom_kg_legal.json.gz  # Contractual & Administrative Law domain
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

### Evaluation Tracks: Bidirectional Role-Reversal Protocol
The evaluation benchmark comprises 1,000 query-level evaluations across two inverted tracks ($N=500$ queries per track, totaling 4,000 candidate answers and 3,000 pairwise judge comparisons across 3 head-to-head baselines):
- **Track 1**: Generator = Google Gemma 4 26B (`google/gemma-4-26b-a4b`) $\to$ Judge = Qwen 3.8 27B (`qwen/qwen3.8-27b`).
- **Track 2**: Generator = Qwen 3.8 27B (`qwen/qwen3.8-27b`) $\to$ Judge = Google Gemma 4 26B (`google/gemma-4-26b-a4b`).

Swapping the generator and judge models across tracks explicitly validates cross-model robustness and controls for evaluator preference bias (e.g., self-preference artifacts).

---

## Knowledge Graph Exports (`kg_exports/`)

The `kg_exports/` directory provides the exact pre-extracted knowledge graphs and textual chunks used in the benchmark across all four domains. Each `.json.gz` file contains three primary keys:
- `chunks`: Corpus text passages with unique `source_id` document provenance.
- `entities`: Extracted entity vertices, types, descriptions, and source documents.
- `relationships`: Directed multi-relational edges with typed predicates (`SAME_AS`, `SUPERSEDES`, `DEFINES`, etc.), ontology weights, descriptions, and provenance.

### Quick Load / Ingestion Snippet
The exported structure is directly compatible with standard graph libraries (NetworkX, igraph), PostgreSQL schemas, as well as LightRAG's native custom ingestion interface (`LightRAG.ainsert_custom_kg`):

```python
import gzip
import json

# Load exported knowledge graph for Computer Science
with gzip.open("kg_exports/custom_kg_cs.json.gz", "rt", encoding="utf-8") as f:
    kg_data = json.load(f)

print(f"Chunks:        {len(kg_data['chunks'])}")
print(f"Entities:      {len(kg_data['entities'])}")
print(f"Relationships: {len(kg_data['relationships'])}")

# Ingest directly into LightRAG:
# await lightrag_instance.ainsert_custom_kg(kg_data)
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

- **Corpora**: 4 corpora drawn from the public UltraDomain benchmark (Qian et al.), adopting the same domain selections evaluated in the LightRAG benchmark (Guo et al.): Computer Science, Agricultural Science, Heterogeneous Multi-Domain (MIX), and Legal contracts.
- **Host & Application Node**: Apple MacBook Pro (M1 Pro, 16 GB RAM), hosting the isolated Docker microservice network (`aia-backend`, `aia-lightrag`, PostgreSQL 16 with `pgvector` v0.7.0).
- **Inference Server**: Dedicated Neural Inference Server (ASUS Ascent GX10, 128 GB RAM) running LM Studio local inference endpoints (`google/gemma-4-26b-a4b` and `qwen/qwen3.8-27b`).
- **Database Engine**: PostgreSQL 16 with pgvector (HNSW vector indices) and relational property-graph tables, traversed via application-tier Python engine (Listing C.1).
- **Evaluated LightRAG Baseline**: `lightrag-hku==1.5.7` (commit `28ff1b0`), retrieval configuration `hybrid`, file-backed storage (`nano-vectordb` + `NetworkX`).

---

## Citation

If you use this benchmark or replication package in your research, please cite the permanent artifact registered on Zenodo:

```bibtex
@software{forciniti2026beamrag,
  author    = {Forciniti, Vincenzo},
  title     = {{BeamRAG: Query-Guided Heuristic Beam Search with Macro-Gating on Knowledge Graphs}},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.23294593},
  url       = {https://doi.org/10.5281/zenodo.23294593}
}
```

## License

This replication package is released under the MIT License.
