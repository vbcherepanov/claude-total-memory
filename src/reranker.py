"""
Advanced RAG: HyDE + CrossEncoder Reranker + MMR Diversity.

- HyDE: generates hypothetical answer, embeds it for semantic search.
- CrossEncoder: true cross-encoder re-ranking (ms-marco-MiniLM-L-6-v2).
  Falls back to LLM-based reranking via Ollama if CrossEncoder unavailable.
- MMR: Maximal Marginal Relevance for result diversity (λ=0.7).
"""

import json
import os
import sys
import urllib.request
from typing import Optional

import numpy as np

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
# Use smaller model for speed — 7b is ~3x faster than 32b for reranking
RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "qwen2.5-coder:7b")
HYDE_MODEL = os.environ.get("HYDE_MODEL", "qwen2.5-coder:7b")
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")

LOG = lambda msg: sys.stderr.write(f"[reranker] {msg}\n")


def _ollama_generate(prompt: str, model: str, max_tokens: int = 200, temperature: float = 0.3) -> Optional[str]:
    """Call Ollama generate API."""
    try:
        payload = json.dumps({
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            }
        }).encode()
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            return data.get("response", "").strip()
    except Exception as e:
        LOG(f"generate error ({model}): {e}")
        return None


def _ollama_embed(text: str, model: str = None) -> Optional[list]:
    """Get embedding via Ollama."""
    model = model or OLLAMA_EMBED_MODEL
    try:
        payload = json.dumps({"model": model, "prompt": text}).encode()
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            return data.get("embedding")
    except Exception as e:
        LOG(f"embed error: {e}")
        return None


# =============================================================================
# HyDE — Hypothetical Document Embeddings
# =============================================================================

def hyde_expand(query: str, project: str = None) -> Optional[list]:
    """
    Generate a hypothetical answer to the query, then embed it.
    Returns the embedding of the hypothetical answer (not the answer itself).

    Why: "auth JWT" as query might not match "Используем Bearer токен с RS256 подписью".
         But a hypothetical answer about auth WILL be semantically close to that record.

    Returns None if no LLM is configured — caller falls back to plain semantic search.
    """
    try:
        from config import has_llm
        if not has_llm():
            return None
    except Exception:
        pass

    context = f" in project {project}" if project else ""
    prompt = f"""You are a developer knowledge base. Write a short factual answer (2-3 sentences) to this query{context}.
If you don't know, write a plausible answer based on common patterns.

Query: {query}

Answer:"""

    hypothetical = _ollama_generate(prompt, HYDE_MODEL, max_tokens=150, temperature=0.5)
    if not hypothetical:
        return None

    # Embed the hypothetical answer (not the original query)
    embedding = _ollama_embed(hypothetical)
    return embedding


# =============================================================================
# CrossEncoder Reranker — true cross-encoder with LLM fallback
# =============================================================================

_cross_encoder = None
_cross_encoder_failed = False
CROSS_ENCODER_MODEL = os.environ.get("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")


def _get_cross_encoder():
    """Lazy-load CrossEncoder model (singleton)."""
    global _cross_encoder, _cross_encoder_failed
    if _cross_encoder_failed:
        return None
    if _cross_encoder is not None:
        return _cross_encoder
    try:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
        LOG(f"CrossEncoder loaded: {CROSS_ENCODER_MODEL}")
        return _cross_encoder
    except Exception as e:
        LOG(f"CrossEncoder unavailable ({e}), will use LLM fallback")
        _cross_encoder_failed = True
        return None


def rerank_results(query: str, results: list, top_k: int = 10) -> list:
    """
    Rerank search results using CrossEncoder (preferred) or LLM fallback.

    CrossEncoder (ms-marco-MiniLM-L-6-v2): ~5ms for 20 pairs, accurate relevance.
    LLM fallback: ~2-5s via Ollama, less accurate but always available.
    """
    if not results:
        return results

    candidates = results[:min(len(results), top_k * 2)]

    ce = _get_cross_encoder()
    if ce is not None:
        ranked = _rerank_cross_encoder(ce, query, candidates, top_k)
    else:
        ranked = _rerank_llm(query, candidates, top_k)

    # Combine reranked + remaining
    reranked_ids = {id(item) for item in ranked}
    remaining = [item for item in results if id(item) not in reranked_ids]
    return ranked[:top_k] + remaining[:max(0, top_k - len(ranked))]


def _rerank_cross_encoder(ce, query: str, candidates: list, top_k: int) -> list:
    """Rerank using sentence-transformers CrossEncoder."""
    pairs = []
    for item in candidates:
        content = item["r"].get("content", "")[:300]
        project = item["r"].get("project", "")
        tags = item["r"].get("tags", "")
        doc = f"[{project}] {content}"
        if tags:
            doc += f" tags:{tags}"
        pairs.append([query, doc])

    try:
        scores = ce.predict(pairs)
        # Normalize: sigmoid to [0, 1]
        scores_norm = 1.0 / (1.0 + np.exp(-np.array(scores)))
    except Exception as e:
        LOG(f"CrossEncoder scoring failed: {e}")
        return candidates[:top_k]

    # CE boost-only: CE can promote results, never demote them.
    # CE is trained on MS-MARCO (web search) — may misjudge conversational data.
    # If CE agrees with original ranking — boost. If disagrees — keep original.
    max_orig = max(item["score"] for item in candidates) or 1.0
    for i, item in enumerate(candidates):
        if i < len(scores_norm):
            ce_score = float(scores_norm[i])
            orig_norm = item["score"] / max_orig
            ce_blend = ce_score * 0.4 + orig_norm * 0.6
            item["score"] = max(orig_norm, ce_blend) * max_orig  # restore scale
            item["reranked"] = True
            item["ce_score"] = round(ce_score, 4)

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates


def _rerank_llm(query: str, candidates: list, top_k: int) -> list:
    """Fallback: rerank using Ollama LLM as cross-encoder."""
    entries = []
    for i, item in enumerate(candidates):
        content = item["r"].get("content", "")[:200]
        project = item["r"].get("project", "")
        rtype = item["r"].get("type", "")
        entries.append(f"{i}. [{rtype}|{project}] {content}")

    prompt = (
        "Rate how relevant each knowledge entry is to the query. Score 0-10.\n"
        "0 = completely irrelevant, 10 = perfect match.\n"
        "Return ONLY a JSON array of scores, e.g. [8, 3, 7, 1, 9]\n\n"
        f"Query: {query}\n\n"
        f"Entries:\n{chr(10).join(entries)}\n\n"
        "Scores (JSON array):"
    )

    response = _ollama_generate(prompt, RERANKER_MODEL, max_tokens=100, temperature=0.1)
    if not response:
        return candidates[:top_k]

    scores = _parse_scores(response, len(candidates))
    if not scores:
        return candidates[:top_k]

    max_orig = max(item["score"] for item in candidates) or 1.0
    for i, item in enumerate(candidates):
        if i < len(scores):
            rerank_score = scores[i] / 10.0
            orig_norm = item["score"] / max_orig
            item["score"] = rerank_score * 0.6 + orig_norm * 0.4
            item["reranked"] = True

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates


def _parse_scores(response: str, expected_count: int) -> Optional[list]:
    """Parse JSON array of scores from LLM response."""
    response = response.strip()

    for attempt in [response, f"[{response}]"]:
        try:
            scores = json.loads(attempt)
            if isinstance(scores, list):
                result = []
                for s in scores[:expected_count]:
                    try:
                        val = float(s)
                        result.append(max(0, min(10, val)))
                    except (TypeError, ValueError):
                        result.append(5.0)
                return result
        except json.JSONDecodeError:
            continue

    import re
    numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', response)
    if numbers:
        return [max(0, min(10, float(n))) for n in numbers[:expected_count]]

    LOG(f"Failed to parse reranker scores: {response[:100]}")
    return None


# =============================================================================
# MMR — Maximal Marginal Relevance (diversity in search results)
# =============================================================================

def mmr_diversify(results: list, embeddings: list, lambda_param: float = 0.7,
                  top_k: int = 10, redundancy_threshold: float = 0.85) -> list:
    """
    Adaptive MMR: only activates when top results are truly redundant.

    Checks avg pairwise similarity of top-k results first.
    If below threshold — returns as-is (no diversity penalty).
    If above — applies MMR with high λ (0.9) to gently remove near-duplicates.

    Args:
        results: list of dicts with "r" (record) and "score" (relevance score)
        embeddings: list of numpy arrays, same length as results
        lambda_param: base λ (overridden to 0.9 when redundancy detected)
        top_k: number of results to return
        redundancy_threshold: avg pairwise sim above which MMR activates

    Returns:
        reordered list (only reordered if redundancy detected)
    """
    if len(results) <= 2 or not embeddings:
        return results[:top_k]

    n = len(results)
    if len(embeddings) != n:
        LOG(f"MMR: embeddings count ({len(embeddings)}) != results count ({n}), skipping")
        return results[:top_k]

    # Normalize embeddings
    emb_matrix = np.array(embeddings, dtype=np.float32)
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    emb_matrix = emb_matrix / norms

    # Pairwise similarity
    sim_matrix = emb_matrix @ emb_matrix.T

    # Check if top results are actually redundant
    check_n = min(top_k, n)
    if check_n > 1:
        top_sims = []
        for i in range(check_n):
            for j in range(i + 1, check_n):
                top_sims.append(float(sim_matrix[i][j]))
        avg_sim = sum(top_sims) / len(top_sims) if top_sims else 0

        if avg_sim < redundancy_threshold:
            # Results are diverse enough — don't apply MMR
            return results[:top_k]

        LOG(f"MMR activated: avg_sim={avg_sim:.3f} > {redundancy_threshold} (top-{check_n} redundant)")

    # Apply MMR with high λ (gentle: mostly relevance, slight diversity push)
    effective_lambda = 0.9

    scores = np.array([item["score"] for item in results], dtype=np.float32)
    max_score = scores.max()
    if max_score > 0:
        scores = scores / max_score

    selected = []
    remaining = list(range(n))

    for _ in range(min(top_k, n)):
        if not remaining:
            break

        if not selected:
            best_idx = remaining[int(np.argmax(scores[remaining]))]
        else:
            best_score = -float("inf")
            best_idx = remaining[0]
            for idx in remaining:
                relevance = scores[idx]
                max_sim = max(sim_matrix[idx][s] for s in selected)
                mmr_score = effective_lambda * relevance - (1 - effective_lambda) * max_sim
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

        selected.append(best_idx)
        remaining.remove(best_idx)

    return [results[i] for i in selected]


# =============================================================================
# Query Analyzer — classify query to choose strategy
# =============================================================================

def analyze_query(query: str) -> dict:
    """
    Classify query type to optimize search strategy.
    Fast heuristic-based (no LLM call).

    Returns:
        {
            "type": "factual"|"solution"|"debug"|"architecture"|"search",
            "expand": True/False (should use HyDE),
            "deep_graph": True/False (should do multi-hop)
        }
    """
    ql = query.lower()

    # Debug/error patterns
    if any(w in ql for w in ["error", "ошибка", "bug", "fix", "broken", "fail", "не работает", "crash"]):
        return {"type": "debug", "expand": True, "deep_graph": False}

    # Factual lookup (check BEFORE solution — "какой стек" is factual, not "как")
    if any(w in ql for w in ["что", "what", "where", "где", "какой", "какая", "какое", "стек",
                              "порт", "url", "пароль", "домен", "версия", "кто", "сколько"]):
        return {"type": "factual", "expand": False, "deep_graph": False}

    # Architecture patterns
    if any(w in ql for w in ["architect", "архитектур", "структур", "pattern", "design", "подход", "выбор", "сравн"]):
        return {"type": "architecture", "expand": True, "deep_graph": True}

    # Solution/how-to patterns
    if any(w in ql for w in ["как", "how", "implement", "реализ", "сделать", "настро", "создать"]):
        return {"type": "solution", "expand": True, "deep_graph": False}

    # Default: try HyDE for better recall
    return {"type": "search", "expand": len(query.split()) >= 3, "deep_graph": False}


# =============================================================================
# Multi-hop graph traversal
# =============================================================================

def multi_hop_expand(store, seed_ids: list, results: dict, depth: int = 2) -> dict:
    """
    Expand graph relations to depth N (default 2 hops).
    Current system does 1 hop — this does 2+ for architecture queries.

    A → B → C: if A is relevant, B is likely relevant, C might be.
    Score decays: hop1 = parent * 0.4, hop2 = parent * 0.2
    """
    visited = set(results.keys())
    current_layer = seed_ids

    for hop in range(depth):
        next_layer = []
        decay = 0.4 / (hop + 1)  # 0.4, 0.2, 0.13...

        for kid in current_layer:
            if kid not in results:
                continue
            parent_score = results[kid]["score"]

            for r in store.q("""
                SELECT k.* FROM relations rel
                JOIN knowledge k ON k.id = CASE WHEN rel.from_id=? THEN rel.to_id ELSE rel.from_id END
                WHERE (rel.from_id=? OR rel.to_id=?) AND k.status='active'
            """, (kid, kid, kid)):
                rid = r["id"]
                if rid not in visited:
                    visited.add(rid)
                    results[rid] = {"r": r, "score": parent_score * decay, "via": ["graph"]}
                    next_layer.append(rid)

        current_layer = next_layer
        if not current_layer:
            break

    return results
