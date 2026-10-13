#!/usr/bin/env python3
"""
BeamRAG Retrieval-Level Ablation Study (Zero LLM Calls)
Evaluates all 500 benchmark queries across 7 configurations + Baseline BFS:
  C0_Full_BeamRAG (Production Reference)
  C1_No_MacroGating
  C2_No_HubPenalty
  C3_No_RelationWeights
  C4_No_Reranker
  C5_No_HopDecay
  C6_No_LeafDampening
  Baseline_BFS (Lower Reference Bound)

Computes:
  1. Top-25 Jaccard Overlap (%)
  2. Top-10 Exact Retention in Top-25 (%) & RBO (p=0.90)
  3. Cross-Document Drift Contamination Rate (%)
  4. Super-Hub Intrusion Rate (%)
  5. Winning Answer Grounding Recall (%) & Delta Loss with 95% Bootstrap CI (B=10000, seed=42) & Wilcoxon p-value
"""

import os
import sys
import json
import time
import math
import re
import urllib.request
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor

import psycopg2
from psycopg2.extras import RealDictCursor
import networkx as nx
import numpy as np
from scipy import stats

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))
from ablation_engine import (
    adaptive_beam_search_ablation,
    traverse_subgraph_bfs,
    rerank_chunks,
    rerank_relations,
    get_node_documents,
    is_structural_artifact,
)

BASE_DIR = Path(__file__).resolve().parent.parent
PROMPTS_DIR = BASE_DIR / "prompts"
LOGS_TRACK1_DIR = BASE_DIR / "logs" / "track1"
OUTPUT_JSON_PATH = BASE_DIR / "ablation" / "ablation_results.json"
CACHE_EMBEDDINGS_PATH = BASE_DIR / "ablation" / "query_embeddings_cache.json"

DB_CONFIG = {
    "host": "localhost",
    "port": 5433,
    "dbname": "aia_db",
    "user": "aia_user",
    "password": "aia_password"
}

EMBED_URL = "http://192.168.1.165:1234/v1/embeddings"
EMBED_MODEL = "text-embedding-nomic-embed-text-v1.5"

DOMAINS = ["computer_science", "agriculture", "legal", "mix"]


def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)


def rbo_score(l1: List[str], l2: List[str], p: float = 0.90, k: int = 25) -> float:
    """Calculates normalized Rank-Biased Overlap (RBO) between two ranked lists up to depth k."""
    if not l1 and not l2:
        return 1.0
    if not l1 or not l2:
        return 0.0

    l1_k = l1[:k]
    l2_k = l2[:k]
    depth = max(len(l1_k), len(l2_k))
    if depth == 0:
        return 1.0

    s1: Set[str] = set()
    s2: Set[str] = set()
    num = 0.0
    den = 0.0

    for d in range(1, depth + 1):
        if d - 1 < len(l1_k):
            s1.add(l1_k[d - 1])
        if d - 1 < len(l2_k):
            s2.add(l2_k[d - 1])

        agreement = len(s1 & s2) / d
        weight = p ** (d - 1)
        num += agreement * weight
        den += weight

    return min(1.0, num / den) if den > 0.0 else 1.0


def load_knowledge_graph() -> Tuple[nx.DiGraph, Dict[str, List[float]]]:
    """Loads knowledge graph and node embeddings from PostgreSQL into in-memory DiGraph."""
    print("Loading Knowledge Graph and node embeddings from PostgreSQL...")
    conn = get_db_connection()
    graph = nx.DiGraph()
    node_embeddings: Dict[str, List[float]] = {}

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT id, label, entity_type, summary, document_ids, document_id, is_latest,
                   embedding::text as emb_str
            FROM public.knowledge_graph_nodes;
        """)
        for r in cur.fetchall():
            node_id = str(r["id"])
            doc_ids = set(r.get("document_ids") or [])
            if r.get("document_id"):
                doc_ids.add(str(r["document_id"]))

            emb_list = []
            if r.get("emb_str"):
                try:
                    raw = r["emb_str"].strip("[] \t\n")
                    if raw:
                        emb_list = [float(x) for x in raw.split(",") if x.strip()]
                except Exception:
                    pass

            if emb_list:
                node_embeddings[node_id] = emb_list

            graph.add_node(
                node_id,
                label=r["label"],
                entity_type=r["entity_type"],
                summary=r["summary"] or "",
                document_id=str(r["document_id"]) if r.get("document_id") else "",
                document_ids=doc_ids,
                is_latest=r.get("is_latest", True)
            )

        cur.execute("""
            SELECT source, target, relation_type, context, document_id
            FROM public.knowledge_graph_edges;
        """)
        for e in cur.fetchall():
            graph.add_edge(
                e["source"],
                e["target"],
                relation_type=e["relation_type"],
                context=e["context"] or "",
                document_id=str(e["document_id"]) if e.get("document_id") else ""
            )

    conn.close()
    print(f"Graph loaded successfully: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges, {len(node_embeddings)} embeddings.")
    return graph, node_embeddings


def load_benchmark_queries() -> List[Dict[str, Any]]:
    """Loads all 500 benchmark queries along with their track1 logs."""
    all_queries = []
    for domain in DOMAINS:
        prompt_file = PROMPTS_DIR / f"{domain}.json"
        log_file = LOGS_TRACK1_DIR / f"{domain}.json"

        with open(prompt_file, "r") as f:
            prompt_data = json.load(f)
        with open(log_file, "r") as f:
            log_data = json.load(f)

        records = log_data.get("records", [])
        questions = prompt_data.get("questions", [])

        for idx, q_text in enumerate(questions):
            rec = records[idx] if idx < len(records) else {}
            # Verify winning answer vs BFS: score_beam > 0.525
            judge = rec.get("judge_vs_bfs", {})
            s2_sum = 0.0
            for crit in ["Comprehensiveness", "Diversity", "Empowerment"]:
                s2_sum += float(judge.get(crit, {}).get("score_2", 0.5))
            score_beam = s2_sum / 3.0
            is_winner = score_beam > 0.525

            all_queries.append({
                "id": f"{domain}_{idx+1:03d}",
                "domain": domain,
                "is_holdout": domain in ["computer_science", "agriculture", "legal"],
                "index": idx,
                "query": q_text,
                "answer_beam": rec.get("answer_beam", ""),
                "score_beam_vs_bfs": score_beam,
                "is_winning_beam": is_winner
            })

    print(f"Loaded {len(all_queries)} benchmark queries across 4 domains.")
    winning_count = sum(1 for q in all_queries if q["is_winning_beam"])
    print(f"Winning queries vs BFS (score_beam > 0.525): {winning_count} / {len(all_queries)}.")
    return all_queries


def get_query_embeddings(queries: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    """Fetches or loads cached embeddings for all queries."""
    cache: Dict[str, List[float]] = {}
    if CACHE_EMBEDDINGS_PATH.exists():
        try:
            with open(CACHE_EMBEDDINGS_PATH, "r") as f:
                cache = json.load(f)
            print(f"Loaded {len(cache)} embeddings from cache.")
        except Exception as e:
            print(f"Could not load cache: {e}")

    missing_queries = [q for q in queries if q["query"] not in cache]
    if missing_queries:
        print(f"Generating embeddings for {len(missing_queries)} queries via {EMBED_URL}...")
        batch_size = 50
        for i in range(0, len(missing_queries), batch_size):
            batch = missing_queries[i:i + batch_size]
            texts = [q["query"] for q in batch]
            req_data = {
                "input": texts,
                "model": EMBED_MODEL
            }
            req = urllib.request.Request(
                EMBED_URL,
                data=json.dumps(req_data).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                res = json.loads(resp.read().decode("utf-8"))
            for q, item in zip(batch, res["data"]):
                cache[q["query"]] = item["embedding"]

        # Save cache
        with open(CACHE_EMBEDDINGS_PATH, "w") as f:
            json.dump(cache, f)
        print(f"Saved {len(cache)} embeddings to cache.")

    return cache


def fetch_query_context_data(q_text: str, q_emb: List[float], graph_node_names: List[str]) -> Dict[str, Any]:
    """Fetches candidate chunks, doc coherence map, and candidate seeds from PostgreSQL for a query."""
    conn = get_db_connection()
    emb_str = f"[{','.join(map(str, q_emb))}]"

    candidate_chunks = []
    doc_coherence = {}
    candidate_seeds = []

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        # 1. Candidate chunks (top 20)
        cur.execute("""
            SELECT c.id, c.user_id, c.version_id, c.chunk_index, c.content, c.metadata,
                   1 - (c.embedding <=> %s::vector) as similarity,
                   v.version_num, d.title as doc_title, d.id as document_id
            FROM public.document_chunks c
            JOIN public.document_versions v ON c.version_id = v.id
            JOIN public.documents d ON v.document_id = d.id
            WHERE v.is_latest = TRUE
            ORDER BY c.embedding <=> %s::vector
            LIMIT 20;
        """, (emb_str, emb_str))
        for r in cur.fetchall():
            candidate_chunks.append({
                "id": str(r["id"]),
                "content": r["content"] or "",
                "similarity": float(r["similarity"] or 0.0),
                "doc_title": r["doc_title"] or "",
                "document_id": str(r["document_id"] or "")
            })

        # 2. Document coherence map
        cur.execute("""
            SELECT d.id as document_id, d.title as doc_title, 
                   MAX(1 - (c.embedding <=> %s::vector)) as max_similarity
            FROM public.document_chunks c
            JOIN public.document_versions v ON c.version_id = v.id
            JOIN public.documents d ON v.document_id = d.id
            WHERE c.embedding IS NOT NULL AND v.is_latest = TRUE
            GROUP BY d.id, d.title;
        """, (emb_str,))
        for r in cur.fetchall():
            sim = float(r.get("max_similarity") or 0.0)
            if not math.isnan(sim):
                d_id = str(r.get("document_id") or "")
                d_title = str(r.get("doc_title") or "")
                if d_id:
                    doc_coherence[d_id] = max(doc_coherence.get(d_id, 0.0), sim)
                if d_title:
                    doc_coherence[d_title] = max(doc_coherence.get(d_title, 0.0), sim)

        # Candidate chunks integration for complete doc_coherence map (same as graphrag.py lines 210-218)
        for c in candidate_chunks:
            sim = float(c.get("similarity", 0.0))
            d_id = str(c.get("document_id") or "")
            if d_id:
                doc_coherence[d_id] = max(doc_coherence.get(d_id, 0.0), sim)
            title = c.get("doc_title") or ""
            if title:
                doc_coherence[title] = max(doc_coherence.get(title, 0.0), sim)

        # 3. Candidate seeds from pgvector (top 50)
        cur.execute("""
            SELECT id, (1.0 - (embedding <=> %s::vector)) AS sim
            FROM public.knowledge_graph_nodes
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector ASC
            LIMIT 50;
        """, (emb_str, emb_str))
        for r in cur.fetchall():
            candidate_seeds.append((str(r["id"]), float(r.get("sim", 0.0))))

    conn.close()

    # 4. Matched entities from question text and candidate chunk titles
    matched_entities = set()
    q_low = q_text.lower()
    for name in graph_node_names:
        if len(name) >= 3 and not is_structural_artifact(name):
            pattern = r'\b' + re.escape(name.strip().lower()) + r'\b'
            if re.search(pattern, q_low):
                matched_entities.add(name)

    for chunk in candidate_chunks:
        title = chunk.get("doc_title", "")
        if title and not is_structural_artifact(title):
            matched_entities.add(title)

    return {
        "candidate_chunks": candidate_chunks,
        "doc_coherence": doc_coherence,
        "candidate_seeds": candidate_seeds,
        "matched_entities": list(matched_entities)
    }


def execute_ablation_pipeline(
    graph: nx.DiGraph,
    node_embeddings: Dict[str, List[float]],
    queries: List[Dict[str, Any]],
    query_embeddings: Dict[str, List[float]]
) -> Dict[str, Any]:
    """Runs all 7 configurations + BFS on all 500 queries."""
    graph_node_names = list(graph.nodes)

    print("Prefetching query database contexts (candidate chunks, doc coherence, pgvector seeds)...")
    t0 = time.time()

    def fetch_for_query(q):
        q_emb = query_embeddings[q["query"]]
        ctx = fetch_query_context_data(q["query"], q_emb, graph_node_names)
        return q["id"], ctx

    query_contexts: Dict[str, Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for q_id, ctx in pool.map(fetch_for_query, queries):
            query_contexts[q_id] = ctx

    t1 = time.time()
    print(f"Contexts prefetched in {t1 - t0:.2f} seconds.")

    # Configurations definition
    CONFIGS = {
        "C0_Full_BeamRAG": {
            "alpha": 0.60,
            "tau_coherence": 0.65,
            "use_hub_penalty": True,
            "use_leaf_dampening": True,
            "use_hop_decay": True,
            "gamma": 0.85,
            "beta": 0.60
        },
        "C1_No_MacroGating": {
            "alpha": 0.60,
            "tau_coherence": 0.00,
            "use_hub_penalty": True,
            "use_leaf_dampening": True,
            "use_hop_decay": True,
            "gamma": 0.85,
            "beta": 0.60
        },
        "C2_No_HubPenalty": {
            "alpha": 0.60,
            "tau_coherence": 0.65,
            "use_hub_penalty": False,
            "use_leaf_dampening": True,
            "use_hop_decay": True,
            "gamma": 0.85,
            "beta": 0.60
        },
        "C3_No_RelationWeights": {
            "alpha": 1.00,
            "tau_coherence": 0.65,
            "use_hub_penalty": True,
            "use_leaf_dampening": True,
            "use_hop_decay": True,
            "gamma": 0.85,
            "beta": 0.60
        },
        "C4_No_Reranker": {
            "alpha": 0.60,
            "tau_coherence": 0.65,
            "use_hub_penalty": True,
            "use_leaf_dampening": True,
            "use_hop_decay": True,
            "gamma": 0.85,
            "beta": 1.00
        },
        "C5_No_HopDecay": {
            "alpha": 0.60,
            "tau_coherence": 0.65,
            "use_hub_penalty": True,
            "use_leaf_dampening": True,
            "use_hop_decay": False,
            "gamma": 1.00,
            "beta": 0.60
        },
        "C6_No_LeafDampening": {
            "alpha": 0.60,
            "tau_coherence": 0.65,
            "use_hub_penalty": True,
            "use_leaf_dampening": False,
            "use_hop_decay": True,
            "gamma": 0.85,
            "beta": 0.60
        }
    }

    print("Executing graph traversal & reranking across all configurations...")
    t_start_traversal = time.time()

    # Pre-calculate degree lookup
    degrees = dict(graph.degree())

    query_results: List[Dict[str, Any]] = []

    for q_idx, q in enumerate(queries):
        q_id = q["id"]
        q_text = q["query"]
        q_emb = query_embeddings[q_text]
        ctx = query_contexts[q_id]

        c_chunks = ctx["candidate_chunks"]
        doc_coh = ctx["doc_coherence"]
        pg_seeds = ctx["candidate_seeds"]
        matched_ent = ctx["matched_entities"]

        # Top-3 RRF chunks
        reranked_chunks_top3 = rerank_chunks(
            query=q_text,
            chunks=c_chunks,
            graph_entities=graph_node_names,
            top_k=3
        )

        per_config_output: Dict[str, Any] = {}

        # Run C0..C6
        for cfg_name, cfg in CONFIGS.items():
            res_traversal = adaptive_beam_search_ablation(
                graph=graph,
                node_embeddings=node_embeddings,
                query=q_text,
                query_embedding=q_emb,
                candidate_seeds_pgvector=pg_seeds,
                matched_entities=matched_ent,
                doc_coherence=doc_coh if cfg["tau_coherence"] > 0.0 else None,
                max_hops=4,
                beam_width=6,
                gamma=cfg["gamma"],
                alpha=cfg["alpha"],
                tau_node=0.40,
                tau_coherence=cfg["tau_coherence"],
                use_hub_penalty=cfg["use_hub_penalty"],
                use_leaf_dampening=cfg["use_leaf_dampening"],
                use_hop_decay=cfg["use_hop_decay"]
            )
            top_relations = rerank_relations(
                relations=res_traversal["relations"],
                top_chunks=reranked_chunks_top3,
                beta=cfg["beta"],
                max_relations=25
            )
            triples_keys = [f"{r['from']}->{r['to']}#{r['relation']}" for r in top_relations]
            per_config_output[cfg_name] = {
                "relations": top_relations,
                "triple_keys": triples_keys
            }

        # Run Baseline BFS
        bfs_traversal = traverse_subgraph_bfs(
            graph=graph,
            matched_entities=matched_ent,
            max_hops=2
        )
        bfs_top_relations = rerank_relations(
            relations=bfs_traversal["relations"],
            top_chunks=reranked_chunks_top3,
            beta=0.60,
            max_relations=25
        )
        bfs_triple_keys = [f"{r['from']}->{r['to']}#{r['relation']}" for r in bfs_top_relations]
        per_config_output["Baseline_BFS"] = {
            "relations": bfs_top_relations,
            "triple_keys": bfs_triple_keys
        }

        # Extract U_q for Grounding Recall if winning query
        u_q: Set[str] = set()
        ans_beam = q.get("answer_beam", "")
        ans_beam_low = ans_beam.lower() if ans_beam else ""
        c0_top25_relations = per_config_output["C0_Full_BeamRAG"]["relations"]

        if q["is_winning_beam"] and ans_beam_low:
            for r in c0_top25_relations:
                f_n = (r.get("from") or "").lower().strip()
                t_n = (r.get("to") or "").lower().strip()
                if f_n and t_n and (f_n in ans_beam_low) and (t_n in ans_beam_low):
                    u_q.add(f"{r['from']}->{r['to']}#{r['relation']}")

        query_metrics: Dict[str, Any] = {}
        c0_keys = per_config_output["C0_Full_BeamRAG"]["triple_keys"]
        c0_set = set(c0_keys)
        c0_top10_set = set(c0_keys[:10])

        all_eval_configs = list(CONFIGS.keys()) + ["Baseline_BFS"]

        for cfg_name in all_eval_configs:
            cfg_keys = per_config_output[cfg_name]["triple_keys"]
            cfg_set = set(cfg_keys)
            cfg_rels = per_config_output[cfg_name]["relations"]

            # 1. Top-25 Jaccard Overlap (%)
            union_len = len(c0_set | cfg_set)
            jaccard = (len(c0_set & cfg_set) / union_len * 100.0) if union_len > 0 else 100.0

            # 2. Top-10 Exact Retention in Top-25 (%) & RBO
            retention_top10 = (len(c0_top10_set & cfg_set) / len(c0_top10_set) * 100.0) if c0_top10_set else 100.0
            rbo = rbo_score(c0_keys, cfg_keys, p=0.90, k=25) * 100.0

            # 3. Cross-Document Drift Contamination Rate (%)
            # For each triple, find associated docs of endpoints and edge
            drift_count = 0
            if cfg_rels:
                for r in cfg_rels:
                    u_node = r.get("from", "")
                    v_node = r.get("to", "")
                    edge_d = graph.get_edge_data(u_node, v_node) or {}
                    edge_doc = edge_d.get("document_id", "")
                    all_docs = get_node_documents(graph, u_node) | get_node_documents(graph, v_node)
                    if edge_doc:
                        all_docs.add(str(edge_doc))

                    max_coh = max((doc_coh.get(d, 0.0) for d in all_docs), default=0.0) if all_docs else 0.0
                    if max_coh < 0.65:
                        drift_count += 1
                drift_rate = (drift_count / len(cfg_rels)) * 100.0
            else:
                drift_rate = 0.0

            # 4. Super-Hub Intrusion Rate (%)
            hub_count = 0
            if cfg_rels:
                for r in cfg_rels:
                    deg_u = degrees.get(r.get("from", ""), 0)
                    deg_v = degrees.get(r.get("to", ""), 0)
                    if deg_u > 30 or deg_v > 30:
                        hub_count += 1
                hub_rate = (hub_count / len(cfg_rels)) * 100.0
            else:
                hub_rate = 0.0

            # 5. Grounding Recall (%) on winning queries with |U_q| > 0
            grounding_recall = None
            if q["is_winning_beam"] and len(u_q) > 0:
                grounding_recall = (len(cfg_set & u_q) / len(u_q)) * 100.0

            query_metrics[cfg_name] = {
                "jaccard_top25": round(jaccard, 2),
                "retention_top10": round(retention_top10, 2),
                "rbo_top25": round(rbo, 2),
                "drift_rate": round(drift_rate, 2),
                "hub_rate": round(hub_rate, 2),
                "grounding_recall": round(grounding_recall, 2) if grounding_recall is not None else None,
                "n_relations": len(cfg_rels)
            }

        query_results.append({
            "id": q["id"],
            "domain": q["domain"],
            "is_holdout": q["is_holdout"],
            "is_winning_beam": q["is_winning_beam"],
            "u_q_size": len(u_q),
            "metrics": query_metrics
        })

        if (q_idx + 1) % 100 == 0:
            print(f"Processed {q_idx + 1} / 500 queries...")

    t_end_traversal = time.time()
    print(f"All 500 queries processed across 8 configurations in {t_end_traversal - t_start_traversal:.2f} seconds.")
    return {"query_results": query_results}


def compute_bootstrap_ci(data: np.ndarray, b: int = 10000, seed: int = 42) -> Tuple[float, float, float]:
    """Computes mean and 95% Bootstrap Confidence Interval with B iterations."""
    if len(data) == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    mean_val = float(np.mean(data))
    if len(data) == 1:
        return mean_val, mean_val, mean_val

    # Bootstrap resamples of the mean
    boot_indices = rng.integers(0, len(data), size=(b, len(data)))
    boot_means = np.mean(data[boot_indices], axis=1)
    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))
    return mean_val, ci_lower, ci_upper


def aggregate_metrics_for_subset(subset_results: List[Dict[str, Any]], subset_name: str) -> Dict[str, Any]:
    """Aggregates metrics for a subset of queries (Global, Holdout, or specific Domain)."""
    n_queries = len(subset_results)
    cfgs = [
        "C0_Full_BeamRAG",
        "C1_No_MacroGating",
        "C2_No_HubPenalty",
        "C3_No_RelationWeights",
        "C4_No_Reranker",
        "C5_No_HopDecay",
        "C6_No_LeafDampening",
        "Baseline_BFS"
    ]

    agg: Dict[str, Any] = {"subset": subset_name, "n_queries": n_queries, "configurations": {}}

    # Winning queries with u_q > 0
    winning_with_uq = [r for r in subset_results if r["is_winning_beam"] and r["u_q_size"] > 0]
    n_win = len(winning_with_uq)
    agg["n_winning_with_uq"] = n_win

    for cfg in cfgs:
        jaccards = [r["metrics"][cfg]["jaccard_top25"] for r in subset_results]
        retentions = [r["metrics"][cfg]["retention_top10"] for r in subset_results]
        rbos = [r["metrics"][cfg]["rbo_top25"] for r in subset_results]
        drifts = [r["metrics"][cfg]["drift_rate"] for r in subset_results]
        hubs = [r["metrics"][cfg]["hub_rate"] for r in subset_results]

        # Grounding recall array
        recalls = [r["metrics"][cfg]["grounding_recall"] for r in winning_with_uq if r["metrics"][cfg]["grounding_recall"] is not None]
        recalls_arr = np.array(recalls, dtype=np.float64)

        # Delta Recall (Loss vs C0 = 100%): delta = 100 - recall
        deltas_arr = 100.0 - recalls_arr if len(recalls_arr) > 0 else np.array([])
        mean_delta, ci_low, ci_high = compute_bootstrap_ci(deltas_arr, b=10000, seed=42)

        # Wilcoxon signed-rank test against C0
        # C0 has 100% on every winning query with U_q
        c0_recalls = np.array([r["metrics"]["C0_Full_BeamRAG"]["grounding_recall"] for r in winning_with_uq], dtype=np.float64)
        if cfg == "C0_Full_BeamRAG":
            wilcoxon_p = 1.0
        else:
            diffs = c0_recalls - recalls_arr
            if np.all(diffs == 0):
                wilcoxon_p = 1.0
            else:
                try:
                    res_w = stats.wilcoxon(diffs, alternative="greater", zero_method="wilcox")
                    wilcoxon_p = float(res_w.pvalue)
                except Exception:
                    wilcoxon_p = 1.0

        agg["configurations"][cfg] = {
            "jaccard_mean": round(float(np.mean(jaccards)), 2),
            "retention_top10_mean": round(float(np.mean(retentions)), 2),
            "rbo_mean": round(float(np.mean(rbos)), 2),
            "drift_rate_mean": round(float(np.mean(drifts)), 2),
            "hub_rate_mean": round(float(np.mean(hubs)), 2),
            "grounding_recall_mean": round(float(np.mean(recalls_arr)), 2) if len(recalls_arr) > 0 else None,
            "delta_recall_loss_mean": round(mean_delta, 2),
            "delta_recall_ci_95": [round(ci_low, 2), round(ci_high, 2)],
            "wilcoxon_p": wilcoxon_p
        }

    return agg


def format_markdown_table(agg_data: Dict[str, Any], title: str) -> str:
    """Formats aggregated metrics into an academic publication-grade Markdown table."""
    lines = []
    lines.append(f"### {title}")
    lines.append(f"**Sample**: {agg_data['subset']} ($N={agg_data['n_queries']}$, Winning Answers with $U_q$: $N={agg_data['n_winning_with_uq']}$)")
    lines.append("")
    lines.append("| Configuration | Jaccard Top-25 (%) | Top-10 Retention (%) | RBO (p=0.90) (%) | Drift Contamination (%) | Hub Intrusion (%) | Grounding Recall (%) | $\\Delta$ Recall Loss (%) [95% CI] | Wilcoxon $p$-val |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    cfg_labels = {
        "C0_Full_BeamRAG": "**C0: Full BeamRAG (Prod)**",
        "C1_No_MacroGating": "C1: No Macro-Gating ($\\tau_{\\text{coh}}=0$)",
        "C2_No_HubPenalty": "C2: No Hub Penalty ($\\deg>30$)",
        "C3_No_RelationWeights": "C3: No Relation Weights ($\\alpha=1.0$)",
        "C4_No_Reranker": "C4: No Cross-Reranker ($\\beta=1.0$)",
        "C5_No_HopDecay": "C5: No Hop Decay ($\\gamma=1.0$)",
        "C6_No_LeafDampening": "C6: No Leaf Dampening ($\\deg\\le1$)",
        "Baseline_BFS": "*Baseline BFS (Reference Lower)*"
    }

    for cfg_key, label in cfg_labels.items():
        c_data = agg_data["configurations"][cfg_key]
        jacc = f"{c_data['jaccard_mean']:.1f}%"
        ret = f"{c_data['retention_top10_mean']:.1f}%"
        rbo = f"{c_data['rbo_mean']:.1f}%"
        drift = f"{c_data['drift_rate_mean']:.1f}%"
        hub = f"{c_data['hub_rate_mean']:.1f}%"
        rec = f"{c_data['grounding_recall_mean']:.1f}%" if c_data["grounding_recall_mean"] is not None else "N/A"

        if cfg_key == "C0_Full_BeamRAG":
            delta_str = "0.0% (Ref)"
            p_str = "---"
        else:
            delta_val = c_data["delta_recall_loss_mean"]
            ci = c_data["delta_recall_ci_95"]
            delta_str = f"+{delta_val:.1f}% [{ci[0]:.1f}, {ci[1]:.1f}]"
            p_val = c_data["wilcoxon_p"]
            if p_val < 1e-4:
                p_str = f"${p_val:.2e}^*$"
            elif p_val < 0.05:
                p_str = f"{p_val:.4f}*"
            else:
                p_str = f"{p_val:.4f}"

        lines.append(f"| {label} | {jacc} | {ret} | {rbo} | {drift} | {hub} | {rec} | {delta_str} | {p_str} |")

    lines.append("")
    return "\n".join(lines)


def main():
    print("=" * 80)
    print("STARTING DETERMINISTIC BEAMRAG RETRIEVAL ABLATION STUDY (ZERO LLM CALLS)")
    print("=" * 80)

    # 1. Load Knowledge Graph
    graph, node_embeddings = load_knowledge_graph()

    # 2. Load benchmark queries
    queries = load_benchmark_queries()

    # 3. Get / cache query embeddings
    query_embeddings = get_query_embeddings(queries)

    # 4. Run pipeline
    results_dict = execute_ablation_pipeline(graph, node_embeddings, queries, query_embeddings)
    query_results = results_dict["query_results"]

    # 5. Compute aggregations for:
    #    - Global (N=500)
    #    - Holdout (N=375: CS, Agri, Legal)
    #    - Computer Science (N=125)
    #    - Agriculture (N=125)
    #    - Legal (N=125)
    #    - MIX (N=125)
    print("\nComputing statistical aggregations, 10,000 Bootstrap CIs, and Wilcoxon signed-rank tests...")

    global_agg = aggregate_metrics_for_subset(query_results, "Global (All Domains)")
    holdout_agg = aggregate_metrics_for_subset([r for r in query_results if r["is_holdout"]], "Holdout Suite (CS, Agri, Legal)")
    cs_agg = aggregate_metrics_for_subset([r for r in query_results if r["domain"] == "computer_science"], "Computer Science")
    agri_agg = aggregate_metrics_for_subset([r for r in query_results if r["domain"] == "agriculture"], "Agricultural Science")
    legal_agg = aggregate_metrics_for_subset([r for r in query_results if r["domain"] == "legal"], "Contractual Legal")
    mix_agg = aggregate_metrics_for_subset([r for r in query_results if r["domain"] == "mix"], "Cross-Domain MIX")

    # 6. Save results to JSON
    output_data = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_queries": len(query_results),
            "configs": [
                "C0_Full_BeamRAG",
                "C1_No_MacroGating",
                "C2_No_HubPenalty",
                "C3_No_RelationWeights",
                "C4_No_Reranker",
                "C5_No_HopDecay",
                "C6_No_LeafDampening",
                "Baseline_BFS"
            ]
        },
        "aggregates": {
            "global": global_agg,
            "holdout": holdout_agg,
            "domains": {
                "computer_science": cs_agg,
                "agriculture": agri_agg,
                "legal": legal_agg,
                "mix": mix_agg
            }
        },
        "queries": query_results
    }

    with open(OUTPUT_JSON_PATH, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Full ablation results saved to: {OUTPUT_JSON_PATH}")

    # 7. Print Markdown Tables
    print("\n" + "=" * 80)
    print("STATISTICAL SUMMARY TABLES FOR PAPER / REBUTTAL")
    print("=" * 80 + "\n")

    md_global = format_markdown_table(global_agg, "Table A.1: Global Retrieval Ablation Summary (Pooled N=500)")
    md_holdout = format_markdown_table(holdout_agg, "Table A.2: Calibration-Holdout Suite Ablation Summary (N=375)")
    md_cs = format_markdown_table(cs_agg, "Table A.3: Domain Ablation — Computer Science (N=125)")
    md_agri = format_markdown_table(agri_agg, "Table A.4: Domain Ablation — Agricultural Science (N=125)")
    md_legal = format_markdown_table(legal_agg, "Table A.5: Domain Ablation — Contractual Legal (N=125)")
    md_mix = format_markdown_table(mix_agg, "Table A.6: Domain Ablation — Cross-Domain MIX (N=125)")

    full_report = "\n\n".join([md_global, md_holdout, md_cs, md_agri, md_legal, md_mix])
    print(full_report)

    # Also save the markdown report to file
    md_report_path = BASE_DIR / "ablation" / "ablation_summary_report.md"
    with open(md_report_path, "w") as f:
        f.write(full_report)
    print(f"\nMarkdown summary report saved to: {md_report_path}")


if __name__ == "__main__":
    main()
