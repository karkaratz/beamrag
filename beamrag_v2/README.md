# BeamRAG v2: Evaluation Logs & Deterministic Reproduction Suite

This directory contains the official evaluation logs, benchmark queries, and reproduction script for the research manuscript:
> **BeamRAG v2: Alias-Aware Adaptive Beam Traversal with Thresholded Hub Regularization for Multi-Hop GraphRAG**  
> *Vincenzo Forciniti* (Independent Researcher)

---

## Directory Contents

### 1. Generation Transcripts & LLM-as-a-Judge Evaluation Logs
- `v2_llm_eval_track1_results.json`: Track 1 full evaluation logs (500 generation transcripts with Gemma 26B and 1,000 pairwise evaluations judged by Qwen 27B against Baseline BFS and BeamRAG v1).
- `v2_vs_lightrag_track1_results.json`: Track 1 pairwise judge evaluations (500 comparisons) against LightRAG.
- `v2_llm_eval_track2_results.json`: Track 2 full evaluation logs (500 generation transcripts with Qwen 27B and 1,000 pairwise evaluations judged by Gemma 26B against Baseline BFS and BeamRAG v1).
- `v2_vs_lightrag_track2_results.json`: Track 2 pairwise judge evaluations (500 comparisons) against LightRAG.

Total evaluation records: **1,000 generation transcripts** (500 Track 1 + 500 Track 2) and **3,000 pairwise judging records** across all domains.

### 2. Retrieval & Component-Level Ablation Logs
- `v2_benchmark_results.json`: Full 500-query primary retrieval benchmark logs (Table 1).
- `exact_ablation_results.json`: Per-query retrieval logs across all 5 configurations evaluated in the controlled single-factor component ablation (Table C.2).
- `exact_ablation_summary.json`: Summary metrics for the component ablation across all 500 queries and by domain.

### 3. Benchmark Queries
The benchmark suite evaluates 500 queries partitioned equally across four domains (125 queries each):
- Computer Science (CS)
- Agricultural Science (Agri)
- Contractual Legal (Legal)
- Cross-Domain Heterogeneous (MIX)

The query definitions are located in `prompts/`:
- `prompts/computer_science.json`
- `prompts/agriculture.json`
- `prompts/legal.json`
- `prompts/mix.json`

---

## Deterministic Reproduction Instructions

To deterministically recompute and verify all manuscript tables directly from the evaluation logs, run:

```bash
python3 reproduce_v2_tables.py
```

This standalone script uses standard Python 3 (with optional `scipy` for exact binomial sign tests) and outputs:
- **Table 3**: Track 1 Multi-Domain Benchmark Totals ($N=500$, Gemma 26B Exec $\to$ Qwen 27B Judge) with exact binomial sign tests and step-down Holm--Bonferroni adjustments.
- **Table 4**: Track 2 Multi-Domain Benchmark Totals ($N=500$, Qwen 27B Exec $\to$ Gemma 26B Judge) with Holm--Bonferroni adjustments.
- **Table 5**: Cross-Track Comparative Consistency across inverted tracks ($N=1,000$ pooled runs) with exact McNemar discordance tests.
- **Table 5b**: Paired Cross-Track Consistency ($N=500$ paired queries) with exact McNemar tests and Holm correction.
- **Tables B.1 & B.2**: Disaggregated multi-criteria panels (*Comprehensiveness*, *Diversity*, *Empowerment*) across both tracks.
- **Table C.2**: Component-level retrieval ablation across all 500 benchmark queries (`v2_full`, `v2_no_alias`, `v2_beam_4`, `v2_beam_8`, `v2_reg_v1`).
