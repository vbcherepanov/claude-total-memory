#!/usr/bin/env python3
"""
LongMemEval Benchmark Runner for Claude Total Memory.

Measures R@5 (recall_any@5) and NDCG@5 across 500 questions.
Compares multiple retrieval modes:
  - raw: ChromaDB cosine similarity only
  - bm25: FTS5 + BM25 only
  - hybrid: BM25 + semantic (no reranking)
  - full: 6-stage pipeline (BM25 + semantic + fuzzy + graph + CrossEncoder + MMR)

Usage:
    python benchmarks/longmemeval_bench.py [--limit N] [--modes raw,hybrid,full]
"""

import argparse
import json
import math
import os
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

# Embedding setup
USE_OLLAMA = False
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = "nomic-embed-text"

_st_model = None
_ce_model = None


def get_st_model():
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer("all-MiniLM-L6-v2")
        print(f"[bench] SentenceTransformer loaded: all-MiniLM-L6-v2")
    return _st_model


def get_ce_model():
    global _ce_model
    if _ce_model is None:
        from sentence_transformers import CrossEncoder
        _ce_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        print(f"[bench] CrossEncoder loaded: cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _ce_model


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed texts using SentenceTransformer."""
    model = get_st_model()
    return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)


def embed_texts_ollama(texts: list[str]) -> np.ndarray:
    """Embed texts using Ollama."""
    import urllib.request
    embeddings = []
    for text in texts:
        payload = json.dumps({"model": EMBED_MODEL, "prompt": text[:2000]}).encode()
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            embeddings.append(data["embedding"])
    return np.array(embeddings, dtype=np.float32)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between query vector a and matrix b."""
    a_norm = a / (np.linalg.norm(a) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return b_norm @ a_norm


# =============================================================================
# BM25 (lightweight, no external deps)
# =============================================================================

import re
from collections import Counter


def tokenize(text: str) -> list[str]:
    return re.findall(r'\w+', text.lower())


class SimpleBM25:
    """Minimal BM25 implementation for benchmarking."""

    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.doc_freqs = []
        self.idf = {}
        self.doc_lens = []
        self.avgdl = 0

        # Build index
        df = Counter()
        for doc in corpus:
            tokens = tokenize(doc)
            self.doc_freqs.append(Counter(tokens))
            self.doc_lens.append(len(tokens))
            for t in set(tokens):
                df[t] += 1

        self.avgdl = sum(self.doc_lens) / max(self.corpus_size, 1)

        # IDF
        for term, freq in df.items():
            self.idf[term] = math.log(1 + (self.corpus_size - freq + 0.5) / (freq + 0.5))

    def score(self, query: str) -> list[float]:
        tokens = tokenize(query)
        scores = []
        for i in range(self.corpus_size):
            s = 0.0
            dl = self.doc_lens[i]
            for t in tokens:
                if t not in self.doc_freqs[i]:
                    continue
                tf = self.doc_freqs[i][t]
                idf = self.idf.get(t, 0)
                num = tf * (self.k1 + 1)
                den = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                s += idf * num / den
            scores.append(s)
        return scores


# =============================================================================
# RRF Fusion
# =============================================================================

def rrf_fuse(rankings: dict[str, list[int]], weights: dict[str, float] = None, k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion across multiple ranked lists."""
    weights = weights or {}
    scores = defaultdict(float)
    for source, ranked_ids in rankings.items():
        w = weights.get(source, 1.0)
        for rank, doc_id in enumerate(ranked_ids):
            scores[doc_id] += w * (1.0 / (k + rank + 1))
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# =============================================================================
# MMR Diversity
# =============================================================================

def mmr_reorder(scores: list[float], embeddings: np.ndarray, lambda_param: float = 0.7, top_k: int = 5) -> list[int]:
    """MMR: select indices balancing relevance and diversity."""
    n = len(scores)
    if n <= 1:
        return list(range(n))

    sim_matrix = embeddings @ embeddings.T
    scores_arr = np.array(scores)
    max_s = scores_arr.max()
    if max_s > 0:
        scores_arr = scores_arr / max_s

    selected = []
    remaining = list(range(n))

    for _ in range(min(top_k, n)):
        if not remaining:
            break
        if not selected:
            best = max(remaining, key=lambda i: scores_arr[i])
        else:
            best = None
            best_mmr = -float("inf")
            for idx in remaining:
                max_sim = max(sim_matrix[idx][s] for s in selected)
                mmr = lambda_param * scores_arr[idx] - (1 - lambda_param) * max_sim
                if mmr > best_mmr:
                    best_mmr = mmr
                    best = idx
        selected.append(best)
        remaining.remove(best)

    return selected


# =============================================================================
# Retrieval modes
# =============================================================================

def retrieve_raw(query: str, corpus_embs: np.ndarray, query_emb: np.ndarray, top_k: int = 5) -> list[int]:
    """Raw semantic: cosine similarity only."""
    sims = cosine_sim(query_emb, corpus_embs)
    return list(np.argsort(-sims)[:top_k])


def retrieve_bm25(query: str, bm25: SimpleBM25, top_k: int = 5) -> list[int]:
    """BM25 keyword search only."""
    scores = bm25.score(query)
    return list(np.argsort(-np.array(scores))[:top_k])


def retrieve_hybrid(query: str, corpus_embs: np.ndarray, query_emb: np.ndarray,
                    bm25: SimpleBM25, top_k: int = 5) -> list[int]:
    """Hybrid: RRF fusion of BM25 + semantic."""
    # Semantic ranking
    sims = cosine_sim(query_emb, corpus_embs)
    sem_rank = list(np.argsort(-sims))

    # BM25 ranking
    bm25_scores = bm25.score(query)
    bm25_rank = list(np.argsort(-np.array(bm25_scores)))

    # RRF
    fused = rrf_fuse(
        {"semantic": sem_rank, "bm25": bm25_rank},
        weights={"semantic": 1.2, "bm25": 1.0}
    )
    return [doc_id for doc_id, _ in fused[:top_k]]


def retrieve_full(query: str, corpus: list[str], corpus_embs: np.ndarray,
                  query_emb: np.ndarray, bm25: SimpleBM25, top_k: int = 5) -> list[int]:
    """Full 6-stage pipeline: BM25 + semantic + fuzzy + CrossEncoder + MMR."""
    n = len(corpus)

    # Stage 1: BM25
    bm25_scores = bm25.score(query)
    bm25_rank = list(np.argsort(-np.array(bm25_scores)))

    # Stage 2: Semantic
    sims = cosine_sim(query_emb, corpus_embs)
    sem_rank = list(np.argsort(-sims))

    # Stage 3: Fuzzy (SequenceMatcher on top BM25 misses)
    from difflib import SequenceMatcher
    ql = query.lower()
    fuzzy_scores = []
    for i, doc in enumerate(corpus):
        ratio = SequenceMatcher(None, ql, doc[:300].lower()).ratio()
        fuzzy_scores.append(ratio)
    fuzzy_rank = [i for i in np.argsort(-np.array(fuzzy_scores)) if fuzzy_scores[i] > 0.2]

    # RRF fusion (stages 1-3)
    rankings = {"semantic": sem_rank, "bm25": bm25_rank}
    if fuzzy_rank:
        rankings["fuzzy"] = fuzzy_rank
    fused = rrf_fuse(rankings, weights={"semantic": 1.2, "bm25": 1.0, "fuzzy": 0.5})

    # Take top candidates for reranking
    candidates = [doc_id for doc_id, _ in fused[:top_k * 3]]

    # Stage 5: CrossEncoder re-ranking
    ce = get_ce_model()
    pairs = [[query, corpus[i][:512]] for i in candidates]
    try:
        ce_scores = ce.predict(pairs)
        ce_norm = 1.0 / (1.0 + np.exp(-np.array(ce_scores)))

        # CE boost-only: CE can promote results, never demote them.
        # If CE agrees with RRF — boost. If CE disagrees — keep RRF score.
        fused_dict = dict(fused)
        max_rrf = max(fused_dict.values()) if fused_dict else 1.0
        blended = []
        for idx, c_idx in enumerate(candidates):
            rrf_norm = fused_dict.get(c_idx, 0) / max(max_rrf, 1e-9)
            ce_blend = float(ce_norm[idx]) * 0.4 + rrf_norm * 0.6
            score = max(rrf_norm, ce_blend)  # CE can only help
            blended.append((c_idx, score))
        blended.sort(key=lambda x: x[1], reverse=True)
        candidates = [doc_id for doc_id, _ in blended]
    except Exception as e:
        print(f"  [warn] CrossEncoder failed: {e}")

    # Stage 6: Adaptive MMR — only if top results are truly redundant
    check_n = min(top_k, len(candidates))
    if check_n > 2:
        top_embs_check = corpus_embs[candidates[:check_n]]
        norms_c = np.linalg.norm(top_embs_check, axis=1, keepdims=True)
        norms_c = np.where(norms_c == 0, 1, norms_c)
        top_embs_check = top_embs_check / norms_c
        sim_mat = top_embs_check @ top_embs_check.T

        # Avg pairwise similarity of top-k
        pair_sims = []
        for i in range(check_n):
            for j in range(i + 1, check_n):
                pair_sims.append(float(sim_mat[i][j]))
        avg_sim = sum(pair_sims) / len(pair_sims) if pair_sims else 0

        if avg_sim > 0.85:
            # Redundant — apply gentle MMR (λ=0.9)
            top_embs = corpus_embs[candidates[:top_k * 2]]
            norms = np.linalg.norm(top_embs, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            top_embs = top_embs / norms

            blended_dict = dict(blended) if 'blended' in dir() else dict(fused)
            top_scores = [blended_dict.get(c, 0) for c in candidates[:top_k * 2]]
            mmr_indices = mmr_reorder(top_scores, top_embs, lambda_param=0.9, top_k=top_k)
            return [candidates[i] for i in mmr_indices]

    return candidates[:top_k]


# =============================================================================
# Evaluation metrics
# =============================================================================

def recall_any_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int = 5) -> float:
    """1.0 if any gold ID is in top-k retrieved."""
    top_k = set(retrieved_ids[:k])
    return float(any(g in top_k for g in gold_ids))


def recall_all_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int = 5) -> float:
    """1.0 only if ALL gold IDs are in top-k retrieved."""
    top_k = set(retrieved_ids[:k])
    return float(all(g in top_k for g in gold_ids))


def ndcg_at_k(retrieved_ids: list[str], gold_ids: set, k: int = 5) -> float:
    gold = set(gold_ids)
    dcg = sum(1.0 / math.log2(i + 2) for i, rid in enumerate(retrieved_ids[:k]) if rid in gold)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / ideal if ideal > 0 else 0.0


# =============================================================================
# Main benchmark
# =============================================================================

def run_benchmark(data_path: str, modes: list[str], limit: int = 0, k: int = 5):
    print(f"[bench] Loading {data_path}...")
    with open(data_path) as f:
        data = json.load(f)

    # Filter out abstention questions
    data = [e for e in data if not e.get("question_id", "").endswith("_abs")]
    if limit > 0:
        data = data[:limit]

    print(f"[bench] {len(data)} questions (abstention excluded)")
    print(f"[bench] Modes: {modes}")
    print(f"[bench] K={k}")
    print()

    # Pre-load models
    if any(m in modes for m in ["raw", "hybrid", "full"]):
        get_st_model()
    if "full" in modes:
        get_ce_model()

    # Results per mode per type
    results = {mode: defaultdict(list) for mode in modes}
    times = {mode: [] for mode in modes}

    for qi, entry in enumerate(data):
        qid = entry["question_id"]
        qtype = entry["question_type"]
        question = entry["question"]
        gold_ids = entry["answer_session_ids"]
        session_ids = entry["haystack_session_ids"]
        sessions = entry["haystack_sessions"]

        # Build corpus: concatenate user turns per session
        corpus = []
        corpus_ids = []
        for sid, session in zip(session_ids, sessions):
            user_text = "\n".join(t["content"] for t in session if t["role"] == "user")
            if user_text.strip():
                corpus.append(user_text)
                corpus_ids.append(sid)

        if not corpus:
            continue

        # Embed corpus + query
        need_emb = any(m in modes for m in ["raw", "hybrid", "full"])
        corpus_embs = None
        query_emb = None
        if need_emb:
            all_texts = corpus + [question]
            all_embs = embed_texts(all_texts)
            corpus_embs = all_embs[:-1]
            query_emb = all_embs[-1]

        # Build BM25 index
        need_bm25 = any(m in modes for m in ["bm25", "hybrid", "full"])
        bm25 = None
        if need_bm25:
            bm25 = SimpleBM25(corpus)

        # Run each mode
        for mode in modes:
            t0 = time.time()

            if mode == "raw":
                top_indices = retrieve_raw(question, corpus_embs, query_emb, k)
            elif mode == "bm25":
                top_indices = retrieve_bm25(question, bm25, k)
            elif mode == "hybrid":
                top_indices = retrieve_hybrid(question, corpus_embs, query_emb, bm25, k)
            elif mode == "full":
                top_indices = retrieve_full(question, corpus, corpus_embs, query_emb, bm25, k)
            else:
                continue

            elapsed = time.time() - t0
            times[mode].append(elapsed)

            retrieved = [corpus_ids[i] for i in top_indices if i < len(corpus_ids)]

            r_any = recall_any_at_k(retrieved, gold_ids, k)
            r_all = recall_all_at_k(retrieved, gold_ids, k)
            ndcg = ndcg_at_k(retrieved, gold_ids, k)

            results[mode][qtype].append({"r_any": r_any, "r_all": r_all, "ndcg": ndcg})
            results[mode]["_all"].append({"r_any": r_any, "r_all": r_all, "ndcg": ndcg})

        # Progress
        if (qi + 1) % 25 == 0 or qi == len(data) - 1:
            pct = (qi + 1) / len(data) * 100
            # Show running R@5 for each mode
            running = {}
            for mode in modes:
                all_scores = results[mode]["_all"]
                if all_scores:
                    running[mode] = f"{sum(s['r_any'] for s in all_scores) / len(all_scores) * 100:.1f}%"
            print(f"  [{qi+1}/{len(data)}] {pct:.0f}% | R@{k}: {running}")

    # Final report
    print("\n" + "=" * 80)
    print(f"LongMemEval Benchmark Results — R@{k}")
    print("=" * 80)

    qtypes = ["single-session-user", "single-session-assistant", "single-session-preference",
              "multi-session", "knowledge-update", "temporal-reasoning"]

    # Header
    header = f"{'Type':<30}"
    for mode in modes:
        header += f" | {mode:>10}"
    print(header)
    print("-" * len(header))

    # Per-type results
    for qtype in qtypes:
        row = f"{qtype:<30}"
        for mode in modes:
            scores = results[mode].get(qtype, [])
            if scores:
                r5 = sum(s["r_any"] for s in scores) / len(scores) * 100
                row += f" | {r5:>9.1f}%"
            else:
                row += f" | {'N/A':>9}"
        print(row)

    # Total
    print("-" * len(header))
    row = f"{'TOTAL (R@' + str(k) + ' recall_any)':<30}"
    for mode in modes:
        scores = results[mode]["_all"]
        r5 = sum(s["r_any"] for s in scores) / len(scores) * 100
        row += f" | {r5:>9.1f}%"
    print(row)

    # NDCG
    row = f"{'TOTAL (NDCG@' + str(k) + ')':<30}"
    for mode in modes:
        scores = results[mode]["_all"]
        ndcg = sum(s["ndcg"] for s in scores) / len(scores) * 100
        row += f" | {ndcg:>9.1f}%"
    print(row)

    # recall_all
    row = f"{'TOTAL (R@' + str(k) + ' recall_all)':<30}"
    for mode in modes:
        scores = results[mode]["_all"]
        r_all = sum(s["r_all"] for s in scores) / len(scores) * 100
        row += f" | {r_all:>9.1f}%"
    print(row)

    # Timing
    print()
    print("Avg latency per query:")
    for mode in modes:
        avg_ms = sum(times[mode]) / len(times[mode]) * 1000 if times[mode] else 0
        print(f"  {mode}: {avg_ms:.1f}ms")

    # Save results
    output = {
        "k": k,
        "total_questions": len(data),
        "modes": {},
    }
    for mode in modes:
        output["modes"][mode] = {
            "total_r_any": sum(s["r_any"] for s in results[mode]["_all"]) / len(results[mode]["_all"]),
            "total_ndcg": sum(s["ndcg"] for s in results[mode]["_all"]) / len(results[mode]["_all"]),
            "total_r_all": sum(s["r_all"] for s in results[mode]["_all"]) / len(results[mode]["_all"]),
            "avg_latency_ms": sum(times[mode]) / len(times[mode]) * 1000 if times[mode] else 0,
            "per_type": {},
        }
        for qtype in qtypes:
            scores = results[mode].get(qtype, [])
            if scores:
                output["modes"][mode]["per_type"][qtype] = {
                    "r_any": sum(s["r_any"] for s in scores) / len(scores),
                    "count": len(scores),
                }

    out_path = str(Path(__file__).parent / "results_longmemeval.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LongMemEval Benchmark")
    parser.add_argument("--limit", type=int, default=0, help="Limit to N questions (0=all)")
    parser.add_argument("--modes", type=str, default="raw,bm25,hybrid,full",
                        help="Comma-separated retrieval modes")
    parser.add_argument("--k", type=int, default=5, help="Top-K for recall")
    parser.add_argument("--data", type=str,
                        default=str(Path(__file__).parent / "data" / "longmemeval_s.json"),
                        help="Path to dataset")
    args = parser.parse_args()

    run_benchmark(args.data, args.modes.split(","), limit=args.limit, k=args.k)
