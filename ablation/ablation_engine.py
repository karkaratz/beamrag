"""
BeamRAG Retrieval Ablation Engine (Zero LLM Calls)
Self-contained module implementing deterministic graph traversal and relation reranking
with parameterized switches for all ablation configurations.
"""

import math
import re
from typing import List, Dict, Any, Set, Tuple, Optional

# Standard weights matching production graph_db.py
RELATION_WEIGHTS: Dict[str, float] = {
    "SAME_AS": 1.00,
    "SYNONYM_OF": 1.00,
    "SUPERSEDES": 0.90,
    "DEFINES": 0.80,
    "INFLUENCES": 0.75,
    "PROPOSES": 0.75,
    "IMPLEMENTS": 0.75,
    "USES": 0.75,
    "DEPENDS_ON": 0.75,
    "CONTAINS": 0.70,
    "PART_OF": 0.70,
    "AUTHORS": 0.70,
    "IS_A": 0.70,
    "HAS_TYPE": 0.70,
    "LEADS_TO": 0.70,
    "RELATED_CONCEPT": 0.65,
    "CONNECTS_TO": 0.60,
    "RELATES_TO": 0.50,
    "CONTAINS_TOPIC": 0.40,
}

_FORBIDDEN_CHARS = {'{', '}', '\n', '\t', '<', '>'}
_FORBIDDEN_KEYS = ('"name":', '"summary":', '"description":', '"entity":', 'summary":', 'name":')
_RE_DOC = re.compile(r'\bdocument\s+\d+\b')
_RE_DOC_START = re.compile(r'^(?:doc|document)\s*[:#\d]')
_RE_BIB = re.compile(r'^(?:bibref|ref|table|fig|figure|section|sec|chapter|chap|part|page|module)\b')


def fast_cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Fast in-memory cosine similarity calculation."""
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = 0.0
    n1 = 0.0
    n2 = 0.0
    for a, b in zip(v1, v2):
        dot += a * b
        n1 += a * a
        n2 += b * b
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    sim = dot / ((n1 ** 0.5) * (n2 ** 0.5))
    return max(0.0, min(1.0, sim))


def hub_degree_penalty(degree: int) -> float:
    """Logarithmic penalty for high-degree hub nodes."""
    return min(1.0, 1.0 / math.log(math.e + max(0, degree)))


def is_structural_artifact(name: str) -> bool:
    """Excludes structural placeholders, bib tags, and parser residues."""
    if not name or not isinstance(name, str):
        return True
    raw = name.strip()
    if len(raw) <= 2 or raw.isdigit():
        return True
    if any(c in _FORBIDDEN_CHARS for c in raw):
        return True
    low = raw.lower()
    if "ultradomain" in low or _RE_DOC.search(low) or _RE_DOC_START.match(low):
        return True
    if _RE_BIB.match(low):
        return True
    if any(k in low for k in _FORBIDDEN_KEYS):
        return True
    return False


def get_node_documents(graph, node_id: str) -> Set[str]:
    """Returns set of document IDs associated with a node in the NetworkX graph."""
    attrs = graph.nodes.get(node_id, {})
    doc_ids: Set[str] = set()
    raw_ids = attrs.get("document_ids")
    if raw_ids:
        doc_ids.update([str(x) for x in raw_ids if x])
    main_id = attrs.get("document_id")
    if main_id:
        doc_ids.add(str(main_id))
    return doc_ids


def adaptive_beam_search_ablation(
    graph,
    node_embeddings: Dict[str, List[float]],
    query: str,
    query_embedding: Optional[List[float]],
    candidate_seeds_pgvector: List[Tuple[str, float]],
    matched_entities: List[str],
    doc_coherence: Optional[Dict[str, float]] = None,
    max_hops: int = 4,
    beam_width: int = 6,
    gamma: float = 0.85,
    alpha: float = 0.60,
    min_similarity: float = 0.55,
    tau_node: float = 0.40,
    tau_coherence: float = 0.65,
    use_hub_penalty: bool = True,
    use_leaf_dampening: bool = True,
    use_hop_decay: bool = True
) -> Dict[str, Any]:
    """
    Adaptive Beam Search Traversal with parameterized ablation switches:
    - use_hub_penalty: if False, disables hub penalty (pen=1.0) on seeds (deg>30) and step_gain
    - use_leaf_dampening: if False, leaf dampening (0.1 for deg<=1) is set to 1.0
    - use_hop_decay: if False, step_decay is set to 1.0
    - alpha: semantic vs structural weight (alpha=1.0 ignores relation weights)
    - tau_coherence: macro-gating threshold (0.0 disables gating)
    """
    seeds: List[Tuple[str, float]] = []
    seed_set = set()

    # 1. Seeds from lexical entity match in question & candidate chunk titles
    if matched_entities:
        for ent in matched_entities:
            if ent in graph and ent not in seed_set and not is_structural_artifact(ent):
                sim = fast_cosine_similarity(query_embedding, node_embeddings.get(ent, [])) if query_embedding else 0.8
                seeds.append((ent, sim))
                seed_set.add(ent)

    # 2. Seeds from pgvector candidate matching
    if candidate_seeds_pgvector:
        for node_name, sim in candidate_seeds_pgvector:
            if sim >= min_similarity and node_name not in seed_set and graph.has_node(node_name) and not is_structural_artifact(node_name):
                seeds.append((node_name, sim))
                seed_set.add(node_name)

    # 3. Lexical regex fallback if empty
    if not seeds:
        q_low = query.lower()
        for node_name in graph.nodes:
            if not is_structural_artifact(node_name) and len(node_name) >= 3:
                pattern = r'\b' + re.escape(node_name.strip().lower()) + r'\b'
                if re.search(pattern, q_low):
                    seeds.append((node_name, 0.5))
                    seed_set.add(node_name)

    # 4. Highest-degree fallback if still empty
    if not seeds and graph.number_of_nodes() > 0:
        for n in list(graph.nodes):
            if not is_structural_artifact(n):
                seeds.append((n, 0.3))
            if len(seeds) >= 5:
                break

    # Macro-Document Coherence Filter on candidate seeds (DHT)
    # Active if doc_coherence is given and tau_coherence > 0.0
    if doc_coherence and tau_coherence > 0.0 and seeds:
        coherent_seeds = []
        for s_name, s_sim in seeds:
            s_docs = get_node_documents(graph, s_name)
            if s_docs:
                max_s_coh = max((doc_coherence.get(d, 0.0) for d in s_docs), default=0.0)
                if max_s_coh < tau_coherence:
                    continue
            coherent_seeds.append((s_name, s_sim))
        if coherent_seeds:
            seeds = coherent_seeds

    # Seed Ranking & Truncation
    def get_seed_rank(item):
        s_name, s_sim = item
        deg = graph.degree(s_name)
        if deg <= 1:
            leaf_mult = 0.1 if use_leaf_dampening else 1.0
            return s_sim * leaf_mult
        pen = hub_degree_penalty(deg) if (use_hub_penalty and deg > 30) else 1.0
        return s_sim * pen

    seeds.sort(key=get_seed_rank, reverse=True)
    top_seeds = seeds[:max(beam_width, 4)]

    # Initial active beams
    active_beams: List[Dict[str, Any]] = []
    for s_name, s_sim in top_seeds:
        deg = graph.degree(s_name)
        pen = hub_degree_penalty(deg) if (use_hub_penalty and deg > 30) else 1.0
        init_score = s_sim * pen
        active_beams.append({
            "nodes": [s_name],
            "edges": [],
            "score": init_score,
            "bridges_count": 0
        })

    all_collected_edges: Dict[str, Dict[str, Any]] = {}
    all_visited_nodes = set([s[0] for s in top_seeds])

    # Multi-hop Beam Search
    for hop in range(1, max_hops + 1):
        candidate_beams: List[Dict[str, Any]] = []

        for beam in active_beams:
            curr_node = beam["nodes"][-1]
            path_nodes_set = set(beam["nodes"])

            # Outgoing expansion
            for nxt in graph.neighbors(curr_node):
                if nxt in path_nodes_set:
                    continue

                edge_data = graph.get_edge_data(curr_node, nxt) or {}
                rel_type = edge_data.get("relation_type", "RELATES_TO")
                is_bridge = (rel_type == "SAME_AS")
                rel_weight = RELATION_WEIGHTS.get(rel_type, 0.60)
                nxt_sim = fast_cosine_similarity(query_embedding, node_embeddings.get(nxt, [])) if query_embedding else 0.5
                nxt_deg = graph.degree(nxt)
                pen = hub_degree_penalty(nxt_deg) if use_hub_penalty else 1.0

                if is_structural_artifact(nxt):
                    continue

                # Micro-Level Node Admissibility (tau_node)
                if nxt_sim < tau_node:
                    continue

                # Macro-Level Document Coherence Gating (tau_coherence)
                if doc_coherence and tau_coherence > 0.0:
                    nxt_docs = get_node_documents(graph, nxt)
                    if nxt_docs:
                        max_doc_coh = max((doc_coherence.get(d, 0.0) for d in nxt_docs), default=0.0)
                        if max_doc_coh < tau_coherence:
                            continue

                step_decay = (gamma ** (hop - 1)) if use_hop_decay else 1.0
                effective_sim_struct = (alpha * nxt_sim + (1.0 - alpha) * rel_weight)
                step_gain = step_decay * effective_sim_struct * pen
                new_score = beam["score"] + step_gain

                edge_info = {
                    "from": curr_node,
                    "relation": rel_type,
                    "to": nxt,
                    "target_summary": graph.nodes.get(nxt, {}).get("summary", ""),
                    "context": edge_data.get("context", ""),
                    "hop": hop,
                    "step_score": round(step_gain, 4),
                    "is_bridge": is_bridge
                }

                edge_key = f"{curr_node}->{nxt}#{rel_type}"
                all_collected_edges[edge_key] = edge_info
                all_visited_nodes.add(nxt)

                candidate_beams.append({
                    "nodes": beam["nodes"] + [nxt],
                    "edges": beam["edges"] + [edge_info],
                    "score": new_score,
                    "bridges_count": beam["bridges_count"] + (1 if is_bridge else 0)
                })

            # Incoming expansion (bidirectional)
            for prv in graph.predecessors(curr_node):
                if prv in path_nodes_set:
                    continue

                edge_data = graph.get_edge_data(prv, curr_node) or {}
                rel_type = edge_data.get("relation_type", "RELATES_TO")
                is_bridge = (rel_type == "SAME_AS")
                rel_weight = RELATION_WEIGHTS.get(rel_type, 0.60)
                prv_sim = fast_cosine_similarity(query_embedding, node_embeddings.get(prv, [])) if query_embedding else 0.5
                prv_deg = graph.degree(prv)
                pen = hub_degree_penalty(prv_deg) if use_hub_penalty else 1.0

                if is_structural_artifact(prv):
                    continue

                if prv_sim < tau_node:
                    continue

                if doc_coherence and tau_coherence > 0.0:
                    prv_docs = get_node_documents(graph, prv)
                    if prv_docs:
                        max_doc_coh = max((doc_coherence.get(d, 0.0) for d in prv_docs), default=0.0)
                        if max_doc_coh < tau_coherence:
                            continue

                step_decay = (gamma ** (hop - 1)) if use_hop_decay else 1.0
                effective_sim_struct = (alpha * prv_sim + (1.0 - alpha) * rel_weight)
                step_gain = step_decay * effective_sim_struct * pen
                new_score = beam["score"] + step_gain

                edge_info = {
                    "from": prv,
                    "relation": rel_type,
                    "to": curr_node,
                    "source_summary": graph.nodes.get(prv, {}).get("summary", ""),
                    "context": edge_data.get("context", ""),
                    "hop": hop,
                    "step_score": round(step_gain, 4),
                    "is_bridge": is_bridge
                }

                edge_key = f"{prv}->{curr_node}#{rel_type}"
                all_collected_edges[edge_key] = edge_info
                all_visited_nodes.add(prv)

                candidate_beams.append({
                    "nodes": beam["nodes"] + [prv],
                    "edges": beam["edges"] + [edge_info],
                    "score": new_score,
                    "bridges_count": beam["bridges_count"] + (1 if is_bridge else 0)
                })

        if not candidate_beams:
            break

        candidate_beams.sort(key=lambda b: b["score"], reverse=True)
        active_beams = candidate_beams[:beam_width]

    # Conservative 1-hop fallback if no multi-hop edge validated
    if not all_collected_edges and top_seeds:
        for s_name, s_sim in top_seeds:
            for nxt in graph.neighbors(s_name):
                if not is_structural_artifact(nxt):
                    edge_data = graph.get_edge_data(s_name, nxt) or {}
                    rel_type = edge_data.get("relation_type", "RELATES_TO")
                    all_collected_edges[f"{s_name}->{nxt}#{rel_type}"] = {
                        "from": s_name,
                        "relation": rel_type,
                        "to": nxt,
                        "target_summary": graph.nodes.get(nxt, {}).get("summary", ""),
                        "context": edge_data.get("context", ""),
                        "hop": 1,
                        "step_score": round(s_sim * 0.5, 4),
                        "is_bridge": (rel_type == "SAME_AS")
                    }
                    all_visited_nodes.add(nxt)
                    if len(all_collected_edges) >= beam_width * 2:
                        break

    final_relations = list(all_collected_edges.values())
    final_relations.sort(key=lambda r: r.get("step_score", 0.0), reverse=True)
    return {
        "relations": final_relations,
        "visited_nodes": list(all_visited_nodes)
    }


def traverse_subgraph_bfs(
    graph,
    matched_entities: List[str],
    max_hops: int = 2
) -> Dict[str, Any]:
    """Baseline BFS Graph traversal (2 hops, unweighted, unpruned)."""
    results = []
    visited_edges = set()
    visited_nodes = set()
    queue = [(name, 0) for name in matched_entities if name in graph]

    while queue:
        current, current_hop = queue.pop(0)
        visited_nodes.add(current)
        if current_hop >= max_hops:
            continue

        for n in graph.neighbors(current):
            edge_data = graph.get_edge_data(current, n) or {}
            rel_key = f"{current}->{n}"
            rel_type = edge_data.get("relation_type", "RELATES_TO")
            if rel_key not in visited_edges:
                visited_edges.add(rel_key)
                target_data = graph.nodes.get(n) or {}
                results.append({
                    "from": current,
                    "relation": rel_type,
                    "to": n,
                    "target_summary": target_data.get("summary", ""),
                    "context": edge_data.get("context", ""),
                    "hop": current_hop + 1,
                    "step_score": 0.50
                })
                if n not in visited_nodes and current_hop + 1 < max_hops:
                    queue.append((n, current_hop + 1))

        for p in graph.predecessors(current):
            edge_data = graph.get_edge_data(p, current) or {}
            rel_key = f"{p}->{current}"
            rel_type = edge_data.get("relation_type", "RELATES_TO")
            if rel_key not in visited_edges:
                visited_edges.add(rel_key)
                source_data = graph.nodes.get(p) or {}
                results.append({
                    "from": p,
                    "relation": rel_type,
                    "to": current,
                    "source_summary": source_data.get("summary", ""),
                    "context": edge_data.get("context", ""),
                    "hop": current_hop + 1,
                    "step_score": 0.50
                })
                if p not in visited_nodes and current_hop + 1 < max_hops:
                    queue.append((p, current_hop + 1))

    return {
        "relations": results,
        "visited_nodes": list(visited_nodes)
    }


def lexical_score(query: str, text: str) -> float:
    """Exact lexical match score from production graphrag.py."""
    if not query or not text:
        return 0.0
    words = [w.lower() for w in re.findall(r'\b\w{3,}\b', query)]
    if not words:
        return 0.0
    text_lower = text.lower()
    score = 0.0
    for w in set(words):
        count = text_lower.count(w)
        if count > 0:
            score += (1.0 + 0.2 * count)
    return score


def rerank_chunks(
    query: str,
    chunks: List[Dict[str, Any]],
    graph_entities: List[str],
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """RRF chunk fusion: Vector + Lexical + Graph overlap (k=60)."""
    if not chunks:
        return []

    sorted_by_vector = sorted(chunks, key=lambda c: float(c.get("similarity", 0.0)), reverse=True)
    vector_ranks = {id(c): rank for rank, c in enumerate(sorted_by_vector, 1)}

    sorted_by_lexical = sorted(chunks, key=lambda c: lexical_score(query, c.get("content", "")), reverse=True)
    lexical_ranks = {id(c): rank for rank, c in enumerate(sorted_by_lexical, 1)}

    def count_graph_entities(content: str) -> int:
        cnt = 0
        c_low = content.lower()
        for ent in graph_entities:
            if len(ent) >= 3 and ent.lower() in c_low:
                cnt += 1
        return cnt

    sorted_by_graph = sorted(chunks, key=lambda c: count_graph_entities(c.get("content", "")), reverse=True)
    graph_ranks = {id(c): rank for rank, c in enumerate(sorted_by_graph, 1)}

    k = 60
    scored_chunks = []
    for c in chunks:
        c_id = id(c)
        r_v = vector_ranks.get(c_id, len(chunks))
        r_l = lexical_ranks.get(c_id, len(chunks))
        r_g = graph_ranks.get(c_id, len(chunks))

        rrf_score = (1.0 / (k + r_v)) + (1.0 / (k + r_l)) + (1.0 / (k + r_g))
        c_copy = dict(c)
        c_copy["rrf_score"] = round(rrf_score, 4)
        scored_chunks.append(c_copy)

    scored_chunks.sort(key=lambda x: x["rrf_score"], reverse=True)
    return scored_chunks[:top_k]


def rerank_relations(
    relations: List[Dict[str, Any]],
    top_chunks: List[Dict[str, Any]],
    beta: float = 0.60,
    max_relations: int = 25
) -> List[Dict[str, Any]]:
    """Joint Graph-Text Cross-Reranker with parameter beta."""
    if not relations:
        return []
    if not top_chunks:
        return relations[:max_relations]

    max_rrf = max((float(c.get("rrf_score", 0.0)) for c in top_chunks), default=1.0)
    if max_rrf <= 0.0:
        max_rrf = 1.0

    scored_relations = []
    for r in relations:
        from_n = (r.get("from") or r.get("source") or "").lower()
        to_n = (r.get("to") or r.get("target") or "").lower()
        rel_ctx = (r.get("context") or "").lower()

        s_graph = float(r.get("step_score", 0.50))
        s_graph = max(0.0, min(1.0, s_graph))

        s_text = 0.0
        for c in top_chunks:
            content_low = c.get("content", "").lower()
            chunk_norm_rrf = float(c.get("rrf_score", 0.0)) / max_rrf

            has_from = bool(from_n and len(from_n) >= 3 and from_n in content_low)
            has_to = bool(to_n and len(to_n) >= 3 and to_n in content_low)

            if has_from and has_to:
                match_val = 1.0
            elif has_from or has_to:
                match_val = 0.5
            elif rel_ctx and len(rel_ctx) >= 10 and rel_ctx[:30] in content_low:
                match_val = 0.3
            else:
                match_val = 0.0

            score_c = chunk_norm_rrf * match_val
            if score_c > s_text:
                s_text = score_c

        s_joint = beta * s_graph + (1.0 - beta) * s_text
        r_copy = dict(r)
        r_copy["joint_score"] = round(s_joint, 4)
        scored_relations.append(r_copy)

    scored_relations.sort(key=lambda x: x["joint_score"], reverse=True)
    return scored_relations[:max_relations]
