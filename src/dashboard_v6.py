"""Dashboard v6 extensions — new API endpoints + HTML panels.

Lives alongside dashboard.py. Exports:
  - api_v6_savings(db)        — aggregate filter_savings stats + per-filter breakdown
  - api_v6_queues(db)         — counts per status for all 3 v6 queues
  - api_v6_coverage(db)       — % of knowledge records with representations/enrichment
  - api_graph_delta(db, since) — graph nodes/edges created after ISO timestamp
  - V6_PANELS_HTML            — HTML snippet to inject into main dashboard page
  - GRAPH_LIVE_HTML           — standalone /graph/live page
"""

from __future__ import annotations

import sqlite3


# ──────────────────────────────────────────────
# APIs
# ──────────────────────────────────────────────


def api_v6_savings(db: sqlite3.Connection) -> dict:
    """Aggregate filter_savings metrics."""
    try:
        row = db.execute(
            "SELECT COUNT(*) AS n, "
            "       COALESCE(SUM(input_chars), 0) AS inp, "
            "       COALESCE(SUM(output_chars), 0) AS outp "
            "FROM filter_savings"
        ).fetchone()
        n = row["n"] if row else 0
        inp = int(row["inp"]) if row else 0
        outp = int(row["outp"]) if row else 0
        by_filter = [
            dict(r) for r in db.execute(
                "SELECT filter_name AS name, "
                "       COUNT(*) AS uses, "
                "       SUM(input_chars) AS inp, "
                "       SUM(output_chars) AS outp, "
                "       ROUND(AVG(reduction_pct), 1) AS avg_pct "
                "FROM filter_savings "
                "GROUP BY filter_name "
                "ORDER BY uses DESC"
            ).fetchall()
        ]
        return {
            "applied_count": n,
            "chars_saved": inp - outp,
            "tokens_saved_estimate": (inp - outp) // 4,
            "total_reduction_pct": (
                round((1 - outp / inp) * 100, 1) if inp else 0.0
            ),
            "by_filter": by_filter,
        }
    except sqlite3.Error:
        return {
            "applied_count": 0, "chars_saved": 0, "tokens_saved_estimate": 0,
            "total_reduction_pct": 0.0, "by_filter": [],
        }


def api_v6_queues(db: sqlite3.Connection) -> dict:
    """Status counts per v6 async queue."""
    out: dict = {}
    for table in ("triple_extraction_queue", "deep_enrichment_queue", "representations_queue"):
        counts = {"pending": 0, "processing": 0, "done": 0, "failed": 0}
        try:
            for r in db.execute(
                f"SELECT status, COUNT(*) AS c FROM {table} GROUP BY status"
            ).fetchall():
                counts[r["status"]] = r["c"]
        except sqlite3.Error:
            counts["error"] = "table missing"  # type: ignore[assignment]
        out[table] = counts
    return out


def api_v6_coverage(db: sqlite3.Connection) -> dict:
    """Percent of active knowledge records that have v6 data attached."""
    try:
        total = db.execute(
            "SELECT COUNT(*) FROM knowledge WHERE status='active'"
        ).fetchone()[0]
    except sqlite3.Error:
        total = 0
    if not total:
        return {"active_knowledge": 0, "representations_pct": 0, "enrichment_pct": 0}

    try:
        reps = db.execute(
            "SELECT COUNT(DISTINCT knowledge_id) FROM knowledge_representations"
        ).fetchone()[0]
    except sqlite3.Error:
        reps = 0
    try:
        enr = db.execute("SELECT COUNT(*) FROM knowledge_enrichment").fetchone()[0]
    except sqlite3.Error:
        enr = 0

    return {
        "active_knowledge": total,
        "representations_records": reps,
        "enrichment_records": enr,
        "representations_pct": round((reps / total) * 100, 1),
        "enrichment_pct": round((enr / total) * 100, 1),
    }


def api_graph_by_type(
    db: sqlite3.Connection,
    min_mentions: int = 3,
    limit_per_type: int = 80,
) -> dict:
    """Return nodes grouped by type + edges between them — for hive plot."""
    try:
        type_rows = db.execute(
            "SELECT DISTINCT type FROM graph_nodes WHERE mention_count >= ? ORDER BY type",
            (min_mentions,),
        ).fetchall()
        types = [r[0] for r in type_rows if r[0]]
        groups: dict[str, list[dict]] = {}
        all_node_ids: list[str] = []
        for t in types:
            rows = db.execute(
                "SELECT id, name, mention_count FROM graph_nodes "
                "WHERE type = ? AND mention_count >= ? "
                "ORDER BY mention_count DESC LIMIT ?",
                (t, min_mentions, limit_per_type),
            ).fetchall()
            group = [dict(r) | {"type": t} for r in rows]
            groups[t] = group
            all_node_ids.extend(r["id"] for r in rows)
        if not all_node_ids:
            return {"types": [], "groups": {}, "edges": []}
        placeholders = ",".join("?" * len(all_node_ids))
        edge_rows = db.execute(
            f"SELECT source_id, target_id, relation_type, weight FROM graph_edges "
            f"WHERE source_id IN ({placeholders}) AND target_id IN ({placeholders}) "
            f"AND weight >= 2 LIMIT 5000",
            all_node_ids + all_node_ids,
        ).fetchall()
        return {
            "types": types,
            "groups": groups,
            "edges": [dict(r) for r in edge_rows],
        }
    except Exception as e:
        return {"types": [], "groups": {}, "edges": [], "error": str(e)}


def api_graph_matrix(
    db: sqlite3.Connection,
    min_mentions: int = 5,
    limit: int = 200,
) -> dict:
    """Return ordered nodes + adjacency cells for matrix view."""
    try:
        nodes = db.execute(
            "SELECT id, name, type, mention_count FROM graph_nodes "
            "WHERE mention_count >= ? "
            "ORDER BY type, mention_count DESC LIMIT ?",
            (min_mentions, limit),
        ).fetchall()
        node_list = [dict(r) for r in nodes]
        ids = [n["id"] for n in node_list]
        if not ids:
            return {"nodes": [], "cells": []}
        placeholders = ",".join("?" * len(ids))
        edges = db.execute(
            f"SELECT source_id, target_id, weight FROM graph_edges "
            f"WHERE source_id IN ({placeholders}) AND target_id IN ({placeholders})",
            ids + ids,
        ).fetchall()
        idx = {nid: i for i, nid in enumerate(ids)}
        cells: list[list[int | float]] = []
        for e in edges:
            si = idx.get(e["source_id"])
            ti = idx.get(e["target_id"])
            if si is not None and ti is not None:
                cells.append([si, ti, float(e["weight"] or 1)])
        return {"nodes": node_list, "cells": cells}
    except Exception as e:
        return {"nodes": [], "cells": [], "error": str(e)}


def api_graph_delta(
    db: sqlite3.Connection,
    since: str | None = None,
    limit: int = 200,
    offset: int = 0,
    min_mentions: int = 0,
    min_edge_weight: float = 0.0,
) -> dict:
    """Graph nodes + edges created/last_seen after `since` (ISO-8601, optional).

    Guarantees that every returned edge has BOTH its source and target present
    in the returned `nodes` array (fetches the missing endpoints in a second
    query). Without this, vis-network silently drops edges pointing to nodes
    the client has not loaded yet.

    Two modes:
      - delta:  pass `since=<ISO>` — returns items after that timestamp (used
                by the live page for incremental updates).
      - paged:  no `since`, with `limit` + `offset` — used by initial bulk
                load to walk the whole graph in chunks.

    Returns {nodes, edges, max_ts, stats, total_edges, total_nodes} so the
    client knows when it has loaded everything.
    """
    # min_mentions filters NODES by importance. Edges are then restricted to
    # those whose BOTH endpoints survive the filter (server-side, so client
    # never sees edges to nodes it doesn't have).
    node_conds: list[str] = []
    node_params: list = []
    edge_conds: list[str] = []
    edge_params: list = []

    if since:
        node_conds.append("(last_seen_at > ? OR first_seen_at > ?)")
        node_params.extend([since, since])
        edge_conds.append("(last_reinforced_at > ? OR created_at > ?)")
        edge_params.extend([since, since])

    if min_mentions > 0:
        node_conds.append("mention_count >= ?")
        node_params.append(min_mentions)
        # Edge endpoints must also satisfy min_mentions
        edge_conds.append(
            "source_id IN (SELECT id FROM graph_nodes WHERE mention_count >= ?) "
            "AND target_id IN (SELECT id FROM graph_nodes WHERE mention_count >= ?)"
        )
        edge_params.extend([min_mentions, min_mentions])

    if min_edge_weight > 0:
        edge_conds.append("weight >= ?")
        edge_params.append(min_edge_weight)

    node_where = (" WHERE " + " AND ".join(node_conds)) if node_conds else ""
    edge_where = (" WHERE " + " AND ".join(edge_conds)) if edge_conds else ""

    try:
        node_rows = db.execute(
            "SELECT id, type, name, mention_count, first_seen_at, last_seen_at "
            f"FROM graph_nodes{node_where} "
            "ORDER BY mention_count DESC, COALESCE(last_seen_at, first_seen_at) DESC "
            "LIMIT ? OFFSET ?",
            (*node_params, limit, offset),
        ).fetchall()
    except sqlite3.Error:
        node_rows = []
    try:
        edge_rows = db.execute(
            "SELECT id, source_id, target_id, relation_type, weight, created_at, last_reinforced_at "
            f"FROM graph_edges{edge_where} "
            "ORDER BY weight DESC, COALESCE(last_reinforced_at, created_at) DESC "
            "LIMIT ? OFFSET ?",
            (*edge_params, limit, offset),
        ).fetchall()
    except sqlite3.Error:
        edge_rows = []

    # Total counts respecting the filter — so client knows when it's done
    try:
        if min_mentions > 0:
            total_nodes = db.execute(
                "SELECT COUNT(*) FROM graph_nodes WHERE mention_count >= ?",
                (min_mentions,),
            ).fetchone()[0]
            sql = ("SELECT COUNT(*) FROM graph_edges "
                   "WHERE source_id IN (SELECT id FROM graph_nodes WHERE mention_count >= ?) "
                   "  AND target_id IN (SELECT id FROM graph_nodes WHERE mention_count >= ?)")
            params: list = [min_mentions, min_mentions]
            if min_edge_weight > 0:
                sql += " AND weight >= ?"
                params.append(min_edge_weight)
            total_edges = db.execute(sql, params).fetchone()[0]
        else:
            total_nodes = db.execute("SELECT COUNT(*) FROM graph_nodes").fetchone()[0]
            if min_edge_weight > 0:
                total_edges = db.execute(
                    "SELECT COUNT(*) FROM graph_edges WHERE weight >= ?",
                    (min_edge_weight,),
                ).fetchone()[0]
            else:
                total_edges = db.execute("SELECT COUNT(*) FROM graph_edges").fetchone()[0]
    except sqlite3.Error:
        total_nodes = 0
        total_edges = 0

    # Ensure every edge endpoint is in the node set: fetch missing nodes.
    seen_node_ids = {r["id"] for r in node_rows}
    needed_ids: set[str] = set()
    for e in edge_rows:
        for side in ("source_id", "target_id"):
            nid = e[side]
            if nid and nid not in seen_node_ids:
                needed_ids.add(nid)

    extra_nodes: list = []
    if needed_ids:
        placeholders = ",".join("?" * len(needed_ids))
        try:
            extra_nodes = db.execute(
                "SELECT id, type, name, mention_count, first_seen_at, last_seen_at "
                f"FROM graph_nodes WHERE id IN ({placeholders})",
                list(needed_ids),
            ).fetchall()
        except sqlite3.Error:
            extra_nodes = []

    max_ts = ""
    for r in list(node_rows) + list(edge_rows):
        for field in ("last_seen_at", "last_reinforced_at", "first_seen_at", "created_at"):
            v = r[field] if field in r.keys() else None
            if v and v > max_ts:
                max_ts = v

    all_nodes = [dict(r) for r in node_rows] + [dict(r) for r in extra_nodes]
    return {
        "nodes": all_nodes,
        "edges": [dict(r) for r in edge_rows],
        "max_ts": max_ts,
        "since": since or "",
        "offset": offset,
        "limit": limit,
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "stats": {
            "new_nodes": len(node_rows),
            "endpoint_nodes_filled": len(extra_nodes),
            "total_nodes_in_response": len(all_nodes),
            "edges": len(edge_rows),
        },
    }


# ──────────────────────────────────────────────
# HTML fragments
# ──────────────────────────────────────────────


V6_PANELS_HTML = r"""
<section id="v6-panels" style="padding:16px; display:grid; grid-template-columns:repeat(3,1fr); gap:12px;">
  <div class="v6-card" id="v6-savings-card" style="background:#1a1a1a;border:1px solid #2a2a2a;border-radius:8px;padding:12px;">
    <h3 style="margin:0 0 8px 0;font-size:14px;color:#8ad;">💾 Token savings</h3>
    <div id="v6-savings-body" style="font-size:12px;color:#ccc;">loading...</div>
  </div>
  <div class="v6-card" style="background:#1a1a1a;border:1px solid #2a2a2a;border-radius:8px;padding:12px;">
    <h3 style="margin:0 0 8px 0;font-size:14px;color:#8ad;">🔄 v6 queues</h3>
    <div id="v6-queues-body" style="font-size:12px;color:#ccc;">loading...</div>
  </div>
  <div class="v6-card" style="background:#1a1a1a;border:1px solid #2a2a2a;border-radius:8px;padding:12px;">
    <h3 style="margin:0 0 8px 0;font-size:14px;color:#8ad;">📊 v6 coverage</h3>
    <div id="v6-coverage-body" style="font-size:12px;color:#ccc;">loading...</div>
  </div>
</section>
<script>
(function() {
  async function refreshV6() {
    try {
      const [savings, queues, coverage] = await Promise.all([
        fetch('/api/v6/savings').then(r => r.json()),
        fetch('/api/v6/queues').then(r => r.json()),
        fetch('/api/v6/coverage').then(r => r.json()),
      ]);

      // Savings panel
      const sBody = document.getElementById('v6-savings-body');
      if (sBody) {
        const fmt = n => (n||0).toLocaleString();
        let html = `<div style="font-size:20px;color:#4a9;">${fmt(savings.tokens_saved_estimate)}</div>`;
        html += `<div>≈ tokens saved</div>`;
        html += `<div>${fmt(savings.chars_saved)} chars (-${savings.total_reduction_pct||0}%)</div>`;
        html += `<div style="margin-top:6px;font-size:11px;">${savings.applied_count||0} filters applied</div>`;
        if (savings.by_filter && savings.by_filter.length) {
          html += '<div style="margin-top:6px;padding-top:6px;border-top:1px solid #333;font-size:11px;">';
          savings.by_filter.forEach(f => {
            html += `<div>${f.name}: ${f.uses}× avg -${f.avg_pct||0}%</div>`;
          });
          html += '</div>';
        }
        sBody.innerHTML = html;
      }

      // Queues panel
      const qBody = document.getElementById('v6-queues-body');
      if (qBody) {
        let html = '';
        for (const [name, counts] of Object.entries(queues)) {
          const pend = counts.pending || 0;
          const done = counts.done || 0;
          const failed = counts.failed || 0;
          const shortName = name.replace('_queue','').replace('_',' ');
          const color = pend > 10 ? '#fa0' : (pend > 0 ? '#ea5' : '#4a9');
          html += `<div style="margin:4px 0;">
            <span style="color:${color};">●</span> <b>${shortName}</b>:
            ${pend} pending, ${done} done${failed ? `, <span style="color:#f55;">${failed} failed</span>` : ''}
          </div>`;
        }
        qBody.innerHTML = html;
      }

      // Coverage panel
      const cBody = document.getElementById('v6-coverage-body');
      if (cBody) {
        const total = coverage.active_knowledge || 0;
        const rpct = coverage.representations_pct || 0;
        const epct = coverage.enrichment_pct || 0;
        const bar = (pct, color) =>
          `<div style="background:#222;height:6px;border-radius:3px;margin:4px 0;">
             <div style="background:${color};width:${pct}%;height:100%;border-radius:3px;"></div>
           </div>`;
        cBody.innerHTML =
          `<div>${total} active knowledge</div>` +
          `<div style="margin-top:6px;font-size:11px;">representations: ${rpct}%</div>` +
          bar(rpct, '#4a9') +
          `<div style="font-size:11px;">enrichment: ${epct}%</div>` +
          bar(epct, '#8ad');
      }
    } catch (e) {
      console.error('v6 refresh failed', e);
    }
  }
  refreshV6();
  setInterval(refreshV6, 5000);
})();
</script>
"""




GRAPH_LIVE_HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Graph Live 3D — Claude Total Memory</title>
<style>
  html,body { margin:0; padding:0; background:#0e0e0e; color:#ddd;
              font-family:-apple-system,Segoe UI,sans-serif; height:100%; overflow:hidden; }
  #vtabs { position:fixed; top:0; left:0; right:0; height:36px; z-index:20;
           background:rgba(14,14,20,0.92); border-bottom:1px solid #222;
           display:flex; gap:2px; align-items:center; padding:0 12px; backdrop-filter:blur(8px); }
  #vtabs a { color:#888; text-decoration:none; padding:9px 14px; font-size:12px;
             border-bottom:2px solid transparent; }
  #vtabs a:hover { color:#ccc; }
  #vtabs a.active { color:#8ad; border-bottom-color:#4a9; }
  #vtabs .home { color:#666; padding-right:14px; border-right:1px solid #222; margin-right:4px; }
  #top { position:fixed; top:36px; left:0; right:0; z-index:10;
         padding:8px 14px; background:rgba(14,14,14,.85);
         border-bottom:1px solid #222; display:flex; gap:10px;
         align-items:center; flex-wrap:wrap; font-size:12px; backdrop-filter:blur(8px); }
  #top h1 { margin:0; font-size:14px; font-weight:500; color:#8ad; }
  #top label, #top button, #top select, #top input {
    font-size:12px; color:#ccc; background:#1a1a1a; border:1px solid #333;
    padding:4px 8px; border-radius:4px; }
  #top button { cursor:pointer; }
  #top button:hover { background:#2a2a2a; }
  #top button.active { background:#1d3a4a; border-color:#4a9; color:#8ad; }
  #top input[type=text] { width:140px; }
  #top input[type=checkbox] { margin-right:4px; vertical-align:middle; }
  #breadcrumb { display:none; align-items:center; gap:6px; color:#8ad; }
  #breadcrumb.active { display:inline-flex; }
  #btnBack { background:#1d3a4a; border-color:#4a9; color:#8ad; }
  .stats { color:#888; margin-left:auto; }
  .stats b { color:#4a9; }
  #network { position:fixed; top:78px; left:0; right:0; bottom:28px;
             background:#0e0e0e; }
  #legend { position:fixed; bottom:28px; left:14px; z-index:5;
            background:rgba(14,14,14,.85); padding:6px 10px; border-radius:6px;
            border:1px solid #222; font-size:11px; }
  #legend .row { display:inline-block; margin-right:10px; }
  #legend .dot { display:inline-block; width:10px; height:10px;
                 border-radius:50%; vertical-align:middle; margin-right:4px; }
  #footer { position:fixed; bottom:0; left:0; right:0; height:28px;
            padding:6px 14px; font-size:11px; color:#666;
            background:rgba(14,14,14,.85); border-top:1px solid #222;
            display:flex; gap:14px; align-items:center; }
  kbd { font-family:Menlo,monospace; background:#1a1a1a; border:1px solid #333;
        padding:1px 4px; border-radius:3px; font-size:10px; }
</style>
<script src="https://cdn.jsdelivr.net/npm/three@0.155.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/3d-force-graph@1.73.0/dist/3d-force-graph.min.js"></script>
</head>
<body>

<div id="vtabs">
  <a href="/" class="home">← Dashboard</a>
  <a href="/graph/live"   id="t-live" class="active">3D Live</a>
  <a href="/graph/hive"   id="t-hive">Hive Plot</a>
  <a href="/graph/matrix" id="t-matrix">Matrix</a>
</div>

<div id="top">
  <h1>🧠 Graph Live <span style="font-size:10px;color:#666;">3D · WebGL</span></h1>

  <span id="breadcrumb">
    <button id="btnBack" title="Back to full graph (Esc)">← back</button>
    <span>focused: <b id="focusLabel">—</b></span>
  </span>

  <label title="Hide nodes that have no edges">
    <input type="checkbox" id="hideOrphans" checked>hide orphans
  </label>
  <label title="Server-side filter: only nodes with N+ mentions">
    importance ≥
    <input type="range" id="minMentions" min="0" max="20" value="3" step="1" style="width:80px;vertical-align:middle;">
    <span id="minMentionsLbl" style="display:inline-block;width:18px;text-align:right;">3</span>
  </label>
  <label title="Server-side filter: only edges with weight ≥ N (1 = co-occurred once, 5 = strong)">
    edge weight ≥
    <input type="range" id="minEdgeW" min="0" max="10" value="2" step="1" style="width:80px;vertical-align:middle;">
    <span id="minEdgeWLbl" style="display:inline-block;width:18px;text-align:right;">2</span>
  </label>
  <label>type:
    <select id="typeFilter"><option value="">all</option></select>
  </label>
  <input type="text" id="search" placeholder="search name…">
  <button id="btnFit" title="Fit all in view">⛶ fit</button>
  <button id="btnRotate" title="Auto-rotate camera">🔄 rotate</button>
  <button id="btnPause" title="Pause polling">⏸ pause</button>
  <button id="btnClear" title="Reload">↺ reload</button>

  <span class="stats" id="stats">connecting…</span>
</div>

<div id="network"></div>

<div id="legend">
  <span class="row"><span class="dot" style="background:#00ff99"></span>concept</span>
  <span class="row"><span class="dot" style="background:#33aaff"></span>technology</span>
  <span class="row"><span class="dot" style="background:#ffaa00"></span>person</span>
  <span class="row"><span class="dot" style="background:#ff5577"></span>project</span>
  <span class="row"><span class="dot" style="background:#ffdd00"></span>company</span>
  <span class="row"><span class="dot" style="background:#bb55ff"></span>product</span>
  <span class="row"><span class="dot" style="background:#00ddcc"></span>pattern</span>
  <span class="row"><span class="dot" style="background:#ff8800"></span>domain</span>
</div>

<div id="footer">
  <span><kbd>L-drag</kbd> orbit</span>
  <span><kbd>R-drag</kbd> pan</span>
  <span><kbd>scroll</kbd> zoom</span>
  <span><kbd>click</kbd> focus on node</span>
  <span><kbd>esc</kbd> back to full view</span>
</div>

<script>
// Vivid colors that pop on dark background
const NODE_COLORS = {
  concept:    '#00ff99',  // bright green
  technology: '#33aaff',  // electric blue
  person:     '#ffaa00',  // orange
  project:    '#ff5577',  // hot pink
  company:    '#ffdd00',  // yellow
  product:    '#bb55ff',  // purple
  pattern:    '#00ddcc',  // teal
  domain:     '#ff8800',  // tangerine
  default:    '#cccccc',  // light grey
};

// ─── DOM ───
const container    = document.getElementById('network');
const statsEl      = document.getElementById('stats');
const hideOrphans  = document.getElementById('hideOrphans');
const typeFilter   = document.getElementById('typeFilter');
const searchInput  = document.getElementById('search');
const btnFit       = document.getElementById('btnFit');
const btnRotate    = document.getElementById('btnRotate');
const btnPause     = document.getElementById('btnPause');
const btnClear     = document.getElementById('btnClear');
const btnBack      = document.getElementById('btnBack');
const breadcrumb   = document.getElementById('breadcrumb');
const focusLabel   = document.getElementById('focusLabel');
const minMentionsInput = document.getElementById('minMentions');
const minMentionsLbl   = document.getElementById('minMentionsLbl');
const minEdgeWInput    = document.getElementById('minEdgeW');
const minEdgeWLbl      = document.getElementById('minEdgeWLbl');

// ─── State ───
const nodesById = new Map();      // id → node object
const linksById = new Map();      // id → link object
const edgesByNode = new Map();    // id → degree
const typeSet = new Set();
let lastTs = '';
let paused = false;
let focusNodeId = null;
let focusNeighbors = new Set();
let searchTerm = '';
let currentMinMentions = parseInt(minMentionsInput.value, 10);
let currentMinEdgeW = parseInt(minEdgeWInput.value, 10);
let Graph = null;
let autoRotate = false, rotateAngle = 0, rotateTimer = null;
let bulkDone = false, totalEdges = 0, totalNodes = 0;
let nextOffset = 0, activeWorkers = 0;
let bulkStartedAt = performance.now();

// ─── Helpers ───
function isInFocus(id) {
  if (!focusNodeId) return true;
  return id === focusNodeId || focusNeighbors.has(id);
}
function nodeVisible(n) {
  if (typeFilter.value && n.nodeType !== typeFilter.value) return false;
  if (searchTerm && !String(n.label || '').toLowerCase().includes(searchTerm)) return false;
  if (hideOrphans.checked && (edgesByNode.get(n.id) || 0) === 0) return false;
  return true;
}
function linkVisible(l) {
  const sId = typeof l.source === 'object' ? l.source.id : l.source;
  const tId = typeof l.target === 'object' ? l.target.id : l.target;
  const sn = nodesById.get(sId), tn = nodesById.get(tId);
  return sn && tn && nodeVisible(sn) && nodeVisible(tn);
}
function linkInFocus(l) {
  if (!focusNodeId) return false;
  const sId = typeof l.source === 'object' ? l.source.id : l.source;
  const tId = typeof l.target === 'object' ? l.target.id : l.target;
  return sId === focusNodeId || tId === focusNodeId;
}

// ─── 3D Graph init ───
function initGraph() {
  if (Graph) return;
  Graph = ForceGraph3D()(container)
    .backgroundColor('#0a0a14')   // very dark blue (better contrast than pure black)
    .nodeId('id')
    .nodeColor(n => {
      if (focusNodeId === n.id) return '#ffffff';
      if (focusNodeId && !isInFocus(n.id)) return '#444';
      return NODE_COLORS[n.nodeType] || NODE_COLORS.default;
    })
    // nodeVal = sphere VOLUME in 3d-force-graph; rendered radius ∝ cbrt(val).
    // Big values → big visible spheres at any zoom level.
    .nodeVal(n => {
      if (!nodeVisible(n)) return 0.0001;
      const base = 30;
      const boost = Math.log2(1 + (n.mention_count || 0)) * 25;
      return focusNodeId === n.id ? (base + boost) * 3 : base + boost;
    })
    .nodeOpacity(0.95)
    .nodeLabel(n => `<div style="background:rgba(20,20,30,.95);padding:6px 10px;
                         border:1px solid #444;border-radius:4px;color:#fff;font-size:12px;">
                         <b style="color:${NODE_COLORS[n.nodeType] || '#fff'}">${n.label}</b><br>
                         type: ${n.nodeType} · mentions: ${n.mention_count || 0}<br>
                         degree: ${edgesByNode.get(n.id) || 0}
                       </div>`)
    .nodeResolution(12)             // smoother spheres
    // Edges visible against background — was #3a3a3a (dark grey on near-black).
    .linkColor(l => linkInFocus(l) ? '#00ffff' : (focusNodeId ? '#1a1a1a' : '#8aa0c0'))
    .linkOpacity(0.85)
    .linkWidth(l => linkInFocus(l) ? 4 : 1.5)
    .linkDirectionalParticles(l => linkInFocus(l) ? 4 : 0)  // animated dots on focused edges
    .linkDirectionalParticleSpeed(0.008)
    .linkDirectionalParticleColor(() => '#00ffff')
    .onNodeClick(n => focusOn(n.id))
    .onBackgroundClick(() => clearFocus())
    .cooldownTicks(80)               // settle faster, less lifetime jitter
    .warmupTicks(30)
    .d3AlphaDecay(0.04)              // physics cools quickly
    .d3VelocityDecay(0.6)            // strong damping → no twitching
    .width(container.clientWidth)
    .height(container.clientHeight);

  // Stronger repulsion + longer edges → more breathing room
  Graph.d3Force('charge').strength(-300);
  Graph.d3Force('link').distance(80);

  // Three.js default lighting is dim — nodes look black on dark bg.
  // Boost ambient + add a strong front-light so colors actually shine.
  const scene = Graph.scene();
  // Remove dim defaults
  scene.children
    .filter(o => o.isLight)
    .forEach(o => scene.remove(o));
  scene.add(new THREE.AmbientLight(0xffffff, 1.2));
  const dir = new THREE.DirectionalLight(0xffffff, 0.8);
  dir.position.set(1, 1, 1);
  scene.add(dir);

  // Use built-in spheres (no custom nodeThreeObject — caused render bugs).
  // Just brighten ambient and set big nodeVal so default sphere is visible.

  window.addEventListener('resize', () => {
    Graph.width(container.clientWidth);
    Graph.height(container.clientHeight);
  });
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') clearFocus();
  });
}

function refreshView() {
  if (!Graph) return;
  // Re-trigger accessor evaluation
  Graph.nodeColor(Graph.nodeColor());
  Graph.nodeVal(Graph.nodeVal());
  Graph.linkColor(Graph.linkColor());
  Graph.linkWidth(Graph.linkWidth());
}

// ─── Focus / back ───
function focusOn(id) {
  focusNodeId = id;
  focusNeighbors = new Set();
  for (const l of linksById.values()) {
    const sId = typeof l.source === 'object' ? l.source.id : l.source;
    const tId = typeof l.target === 'object' ? l.target.id : l.target;
    if (sId === id) focusNeighbors.add(tId);
    if (tId === id) focusNeighbors.add(sId);
  }
  const node = nodesById.get(id);
  focusLabel.textContent = node ? node.label : id;
  breadcrumb.classList.add('active');
  refreshView();
  if (Graph && node) {
    const nx = Number.isFinite(node.x) ? node.x : 0;
    const ny = Number.isFinite(node.y) ? node.y : 0;
    const nz = Number.isFinite(node.z) ? node.z : 0;
    // Avoid divide-by-zero — fall back to fixed distance from origin
    const mag = Math.hypot(nx, ny, nz) || 1;
    const dist = 100;
    const r = 1 + dist / mag;
    try {
      Graph.cameraPosition(
        {x: nx * r, y: ny * r, z: nz * r},
        {x: nx, y: ny, z: nz},
        600
      );
    } catch (e) { console.warn('cameraPosition failed:', e); }
  }
}

function clearFocus() {
  if (!focusNodeId) return;
  focusNodeId = null;
  focusNeighbors.clear();
  breadcrumb.classList.remove('active');
  refreshView();
  if (Graph) Graph.zoomToFit(800, 80);
}

btnBack.addEventListener('click', clearFocus);

// ─── UI events ───
hideOrphans.addEventListener('change', refreshView);
typeFilter.addEventListener('change', refreshView);
searchInput.addEventListener('input', () => {
  searchTerm = searchInput.value.trim().toLowerCase();
  refreshView();
});
btnPause.addEventListener('click', () => {
  paused = !paused;
  btnPause.textContent = paused ? '▶ resume' : '⏸ pause';
  btnPause.classList.toggle('active', paused);
});
btnFit.addEventListener('click', () => Graph && Graph.zoomToFit(800, 60));
btnRotate.addEventListener('click', () => {
  autoRotate = !autoRotate;
  btnRotate.classList.toggle('active', autoRotate);
  btnRotate.textContent = autoRotate ? '🔄 rotating' : '🔄 rotate';
  if (autoRotate) {
    rotateTimer = setInterval(() => {
      if (!Graph) return;
      rotateAngle += 0.005;
      const dist = 500;
      Graph.cameraPosition({
        x: dist * Math.sin(rotateAngle),
        z: dist * Math.cos(rotateAngle),
        y: 100,
      });
    }, 30);
  } else if (rotateTimer) { clearInterval(rotateTimer); rotateTimer = null; }
});
btnClear.addEventListener('click', () => fullReload());

let mmTimer = null;
minMentionsInput.addEventListener('input', () => {
  minMentionsLbl.textContent = minMentionsInput.value;
  if (mmTimer) clearTimeout(mmTimer);
  mmTimer = setTimeout(() => {
    currentMinMentions = parseInt(minMentionsInput.value, 10);
    fullReload();
  }, 300);
});
let ewTimer = null;
minEdgeWInput.addEventListener('input', () => {
  minEdgeWLbl.textContent = minEdgeWInput.value;
  if (ewTimer) clearTimeout(ewTimer);
  ewTimer = setTimeout(() => {
    currentMinEdgeW = parseInt(minEdgeWInput.value, 10);
    fullReload();
  }, 300);
});

function rebuildTypeFilter() {
  const cur = typeFilter.value;
  typeFilter.innerHTML = '<option value="">all</option>';
  [...typeSet].sort().forEach(t => {
    const o = document.createElement('option');
    o.value = t; o.textContent = t;
    typeFilter.appendChild(o);
  });
  typeFilter.value = cur;
}

// ─── Data flow ───
let pushTimer = null;
function schedulePush() {
  if (pushTimer) return;
  pushTimer = setTimeout(() => {
    pushTimer = null;
    if (!Graph) return;
    Graph.graphData({
      nodes: Array.from(nodesById.values()),
      links: Array.from(linksById.values()),
    });
  }, 250);
}

function ingest(data) {
  let newN = 0, newE = 0;
  for (const n of data.nodes || []) {
    if (!nodesById.has(n.id)) {
      const size = Math.min(20, 4 + Math.log2(1 + (n.mention_count || 0)) * 2);
      nodesById.set(n.id, {
        id: n.id, label: n.name, nodeType: n.type,
        mention_count: n.mention_count, size,
      });
      edgesByNode.set(n.id, 0);
      newN++;
    }
    if (n.type && !typeSet.has(n.type)) typeSet.add(n.type);
  }
  for (const e of data.edges || []) {
    if (linksById.has(e.id)) continue;
    if (!nodesById.has(e.source_id) || !nodesById.has(e.target_id)) continue;
    linksById.set(e.id, {
      id: e.id, source: e.source_id, target: e.target_id,
      label: e.relation_type, weight: e.weight || 1,
    });
    edgesByNode.set(e.source_id, (edgesByNode.get(e.source_id) || 0) + 1);
    edgesByNode.set(e.target_id, (edgesByNode.get(e.target_id) || 0) + 1);
    newE++;
  }
  if (data.max_ts) lastTs = data.max_ts;
  if (typeof data.total_nodes === 'number') totalNodes = data.total_nodes;
  if (typeof data.total_edges === 'number') totalEdges = data.total_edges;
  if (newN || newE) schedulePush();
  if (newN > 5) rebuildTypeFilter();
  return {newN, newE};
}

function fmtSec(ms) { return (ms/1000).toFixed(1) + 's'; }
function updateStats(extra) {
  const total = nodesById.size, eTot = linksById.size;
  const phase = bulkDone
    ? `🟢 live (loaded ${fmtSec(performance.now() - bulkStartedAt)})`
    : `⏳ ${eTot}/${totalEdges}  workers:${activeWorkers}`;
  statsEl.innerHTML = `<b>${total}</b> nodes · <b>${eTot}</b> edges  <span style="color:#888">${phase}</span>` +
    (extra ? ` · ${extra}` : '');
}

// ─── Loader ───
const CHUNK_SIZE = 1000;
const PARALLEL = 4;

async function fetchChunk(offset) {
  const url = `/api/graph/delta?limit=${CHUNK_SIZE}&offset=${offset}&min_mentions=${currentMinMentions}&min_edge_weight=${currentMinEdgeW}`;
  const r = await fetch(url);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

async function bulkWorker() {
  while (!bulkDone && !paused) {
    const myOffset = nextOffset;
    nextOffset += CHUNK_SIZE;
    if (totalEdges > 0 && myOffset >= totalEdges) break;
    activeWorkers++;
    updateStats();
    try {
      const data = await fetchChunk(myOffset);
      ingest(data);
      if (totalEdges > 0 && linksById.size >= totalEdges &&
          nodesById.size >= totalNodes - 100) bulkDone = true;
    } catch (e) {
      console.warn('chunk fail', myOffset, e);
    } finally {
      activeWorkers--;
    }
    updateStats();
  }
}

async function deltaPoll() {
  if (paused || !bulkDone) return;
  try {
    const url = `/api/graph/delta?limit=500&min_mentions=${currentMinMentions}&min_edge_weight=${currentMinEdgeW}` +
                (lastTs ? `&since=${encodeURIComponent(lastTs)}` : '');
    const data = await fetch(url).then(r => r.json());
    const {newN, newE} = ingest(data);
    updateStats(newN || newE ? `+${newN}n/+${newE}e` : '');
  } catch (e) {
    statsEl.textContent = '⚠ delta: ' + e.message;
  }
}

function fullReload() {
  nodesById.clear(); linksById.clear(); edgesByNode.clear();
  typeSet.clear(); rebuildTypeFilter();
  lastTs = ''; nextOffset = 0; bulkDone = false;
  totalEdges = 0; totalNodes = 0;
  if (Graph) Graph.graphData({nodes: [], links: []});
  startBulk();
}

async function startBulk() {
  bulkStartedAt = performance.now();
  initGraph();
  try {
    const initial = await fetchChunk(0);
    nextOffset = CHUNK_SIZE;
    ingest(initial);
    if (totalEdges <= CHUNK_SIZE && nodesById.size >= totalNodes - 100) bulkDone = true;
  } catch (e) {
    statsEl.textContent = '⚠ initial: ' + e.message;
    return;
  }
  const workers = [];
  for (let i = 0; i < PARALLEL; i++) workers.push(bulkWorker());
  await Promise.all(workers);
  bulkDone = true;
  updateStats();
  // Final flush
  if (Graph) {
    Graph.graphData({
      nodes: Array.from(nodesById.values()),
      links: Array.from(linksById.values()),
    });
    // Multiple zoomToFit passes — gives layout time to settle, then frames everything.
    setTimeout(() => Graph.zoomToFit(800, 60), 1500);
    setTimeout(() => Graph.zoomToFit(800, 60), 4000);
    setTimeout(() => Graph.zoomToFit(800, 60), 8000);
    // NOTE: pauseAnimation() also stops render → camera can't animate clicks.
    // Instead, we let physics decay naturally via d3VelocityDecay=0.6.
    // User can still freeze manually via ❄ button if they want zero motion.
  }
  setInterval(deltaPoll, 2500);
}

startBulk();
</script>
</body>
</html>"""


# ──────────────────────────────────────────────
# Shared CSS + view-tabs header
# ──────────────────────────────────────────────


_VIEW_TABS = """<style>
  body { margin:0; background:#0a0a14; color:#ddd; font-family:-apple-system,Segoe UI,sans-serif; }
  #vtabs { position:fixed; top:0; left:0; right:0; height:36px; z-index:20;
           background:rgba(14,14,20,0.92); border-bottom:1px solid #222;
           display:flex; gap:2px; align-items:center; padding:0 12px; backdrop-filter:blur(8px); }
  #vtabs a { color:#888; text-decoration:none; padding:9px 14px; font-size:12px;
             border-bottom:2px solid transparent; }
  #vtabs a:hover { color:#ccc; }
  #vtabs a.active { color:#8ad; border-bottom-color:#4a9; }
  #vtabs .home { color:#666; padding-right:14px; border-right:1px solid #222; margin-right:4px; }
  #vmain { position:fixed; top:36px; left:0; right:0; bottom:0; }
</style>
<div id="vtabs">
  <a href="/" class="home">← Dashboard</a>
  <a href="/graph/live"   id="t-live">3D Live</a>
  <a href="/graph/hive"   id="t-hive">Hive Plot</a>
  <a href="/graph/matrix" id="t-matrix">Matrix</a>
</div>
<script>
(function(){const p=location.pathname;
const map={'/graph/live':'t-live','/graph/hive':'t-hive','/graph/matrix':'t-matrix'};
const id=map[p]; if(id) document.getElementById(id).classList.add('active');})();
</script>"""


# ──────────────────────────────────────────────
# Hive Plot
# ──────────────────────────────────────────────


GRAPH_HIVE_HTML = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Hive Plot — Claude Memory</title>
""" + _VIEW_TABS + r"""
<style>
  #vmain { display:flex; flex-direction:column; }
  #toolbar { padding:10px 14px; background:rgba(14,14,20,0.85); border-bottom:1px solid #222;
             font-size:12px; display:flex; gap:14px; align-items:center; }
  #toolbar input[type=range] { width:100px; vertical-align:middle; }
  #status { color:#888; margin-left:auto; font-size:11px; }
  svg { background:#0a0a14; flex:1; }
  .axis { stroke:#3a3a4a; stroke-width:1; }
  .axis-label { fill:#888; font-size:11px; }
  .node { cursor:pointer; }
  .node:hover { stroke:#fff; stroke-width:2; }
  .edge { fill:none; stroke-opacity:.6; }
</style>
<script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
</head><body>
<div id="vmain">
  <div id="toolbar">
    <label>min mentions:
      <input type="range" id="minM" min="1" max="20" value="3">
      <span id="minMV">3</span>
    </label>
    <label>per axis:
      <input type="range" id="perA" min="20" max="200" value="80" step="10">
      <span id="perAV">80</span>
    </label>
    <button id="reload" style="background:#1a1a2a;color:#ccc;border:1px solid #333;padding:5px 10px;border-radius:4px;cursor:pointer;">↺ reload</button>
    <span id="status">loading…</span>
  </div>
  <svg id="hive"></svg>
</div>
<script>
const TYPE_COLORS = {
  concept:'#00ff99', technology:'#33aaff', project:'#ff5577',
  person:'#ffaa00', company:'#ffdd00', product:'#bb55ff',
  pattern:'#00ddcc', domain:'#ff8800',
};

const svg = d3.select('#hive');
const status = d3.select('#status');
const minM = document.getElementById('minM');
const perA = document.getElementById('perA');
const minMV = document.getElementById('minMV');
const perAV = document.getElementById('perAV');

async function load() {
  status.text('loading…');
  const data = await fetch(`/api/graph/by_type?min_mentions=${minM.value}&limit_per_type=${perA.value}`).then(r=>r.json());
  render(data);
}

function render(data) {
  svg.selectAll('*').remove();
  const W = svg.node().clientWidth, H = svg.node().clientHeight;
  const cx = W / 2, cy = H / 2;
  const R = Math.min(W, H) * 0.42;
  const innerR = 30;

  const types = (data.types || []).filter(t => (data.groups[t] || []).length > 0);
  if (!types.length) { status.text('no data'); return; }

  // Axis layout: spread N types evenly around 360°
  const axes = types.map((t, i) => ({
    type: t,
    angle: (i * 2 * Math.PI / types.length) - Math.PI / 2,
  }));

  // Position each node along its axis (further = more important)
  const positions = new Map();
  axes.forEach(({type, angle}) => {
    const list = data.groups[type] || [];
    const maxM = Math.max(...list.map(n => n.mention_count || 1)) || 1;
    list.forEach((n, idx) => {
      // Importance maps to radial distance (ord by descending in API)
      const r = innerR + ((idx + 1) / list.length) * (R - innerR);
      const x = cx + r * Math.cos(angle);
      const y = cy + r * Math.sin(angle);
      positions.set(n.id, {x, y, r, type, ...n});
    });
  });

  // Draw axes (lines from center)
  svg.selectAll('.axis').data(axes).join('line')
    .attr('class', 'axis')
    .attr('x1', d => cx + innerR * Math.cos(d.angle))
    .attr('y1', d => cy + innerR * Math.sin(d.angle))
    .attr('x2', d => cx + R * Math.cos(d.angle))
    .attr('y2', d => cy + R * Math.sin(d.angle));

  // Axis labels
  svg.selectAll('.axis-label').data(axes).join('text')
    .attr('class', 'axis-label')
    .attr('x', d => cx + (R + 14) * Math.cos(d.angle))
    .attr('y', d => cy + (R + 14) * Math.sin(d.angle))
    .attr('text-anchor', 'middle')
    .attr('dominant-baseline', 'middle')
    .text(d => `${d.type} (${(data.groups[d.type] || []).length})`);

  // Edges as quadratic curves bowing toward center
  svg.selectAll('.edge').data(data.edges || []).join('path')
    .attr('class', 'edge')
    .attr('d', e => {
      const s = positions.get(e.source_id);
      const t = positions.get(e.target_id);
      if (!s || !t) return null;
      // Bow toward center for visual clarity
      const mx = (s.x + t.x) / 2;
      const my = (s.y + t.y) / 2;
      const bx = cx + (mx - cx) * 0.4;
      const by = cy + (my - cy) * 0.4;
      return `M${s.x},${s.y} Q${bx},${by} ${t.x},${t.y}`;
    })
    .attr('stroke', e => {
      const s = positions.get(e.source_id);
      return s ? TYPE_COLORS[s.type] || '#666' : '#666';
    })
    .attr('stroke-width', e => Math.min(3, Math.sqrt(e.weight || 1)));

  // Nodes
  svg.selectAll('.node').data([...positions.values()]).join('circle')
    .attr('class', 'node')
    .attr('cx', d => d.x)
    .attr('cy', d => d.y)
    .attr('r', d => 3 + Math.min(8, Math.log2(1 + (d.mention_count || 0)) * 1.5))
    .attr('fill', d => TYPE_COLORS[d.type] || '#888')
    .append('title')
    .text(d => `${d.name} (${d.type}, mentions: ${d.mention_count})`);

  status.text(`${positions.size} nodes · ${(data.edges||[]).length} edges across ${types.length} axes`);
}

minM.addEventListener('input', () => { minMV.textContent = minM.value; load(); });
perA.addEventListener('input', () => { perAV.textContent = perA.value; load(); });
document.getElementById('reload').onclick = load;
window.addEventListener('resize', load);
load();
</script>
</body></html>"""


# ──────────────────────────────────────────────
# Adjacency Matrix
# ──────────────────────────────────────────────


GRAPH_MATRIX_HTML = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Matrix — Claude Memory</title>
""" + _VIEW_TABS + r"""
<style>
  #vmain { display:flex; flex-direction:column; }
  #toolbar { padding:10px 14px; background:rgba(14,14,20,0.85); border-bottom:1px solid #222;
             font-size:12px; display:flex; gap:14px; align-items:center; }
  #toolbar input[type=range] { width:100px; vertical-align:middle; }
  #status { color:#888; margin-left:auto; font-size:11px; }
  #wrap { flex:1; overflow:auto; background:#0a0a14; padding:20px; }
  canvas { display:block; margin:0 auto; background:#0a0a14; border:1px solid #222; }
  #tooltip { position:fixed; background:rgba(20,20,30,.96); border:1px solid #444;
             color:#ddd; padding:6px 10px; border-radius:4px; font-size:11px;
             pointer-events:none; display:none; z-index:30; }
</style>
</head><body>
<div id="vmain">
  <div id="toolbar">
    <label>min mentions:
      <input type="range" id="minM" min="1" max="30" value="5">
      <span id="minMV">5</span>
    </label>
    <label>limit nodes:
      <input type="range" id="lim" min="50" max="500" value="200" step="50">
      <span id="limV">200</span>
    </label>
    <button id="reload" style="background:#1a1a2a;color:#ccc;border:1px solid #333;padding:5px 10px;border-radius:4px;cursor:pointer;">↺ reload</button>
    <span id="status">loading…</span>
  </div>
  <div id="wrap"><canvas id="mx"></canvas></div>
  <div id="tooltip"></div>
</div>
<script>
const TYPE_COLORS = {
  concept:'#00ff99', technology:'#33aaff', project:'#ff5577',
  person:'#ffaa00', company:'#ffdd00', product:'#bb55ff',
  pattern:'#00ddcc', domain:'#ff8800',
};
const cv = document.getElementById('mx');
const ctx = cv.getContext('2d');
const tooltip = document.getElementById('tooltip');
const status = document.getElementById('status');
const minM = document.getElementById('minM');
const minMV = document.getElementById('minMV');
const lim = document.getElementById('lim');
const limV = document.getElementById('limV');
let nodes = [], cells = [], cellAt = new Map(), CELL = 6, MARGIN = 200;

async function load() {
  status.textContent = 'loading…';
  const d = await fetch(`/api/graph/matrix?min_mentions=${minM.value}&limit=${lim.value}`).then(r=>r.json());
  nodes = d.nodes || [];
  cells = d.cells || [];
  render();
}

function render() {
  if (!nodes.length) { status.textContent = 'no data'; return; }
  CELL = Math.max(3, Math.min(12, Math.floor(900 / nodes.length)));
  const N = nodes.length;
  const W = MARGIN + N * CELL + 20;
  const H = MARGIN + N * CELL + 20;
  cv.width = W; cv.height = H;

  ctx.fillStyle = '#0a0a14';
  ctx.fillRect(0, 0, W, H);

  // Type-color stripes on rows / cols
  for (let i = 0; i < N; i++) {
    const c = TYPE_COLORS[nodes[i].type] || '#888';
    ctx.fillStyle = c + '33';                 // semi-transparent
    ctx.fillRect(MARGIN - 6, MARGIN + i * CELL, 6, CELL);
    ctx.fillRect(MARGIN + i * CELL, MARGIN - 6, CELL, 6);
  }

  // Cells
  cellAt.clear();
  const maxW = Math.max(...cells.map(c => c[2])) || 1;
  for (const [si, ti, w] of cells) {
    const intensity = Math.min(1, w / maxW);
    const c = TYPE_COLORS[nodes[si].type] || '#888';
    ctx.fillStyle = c;
    ctx.globalAlpha = 0.3 + intensity * 0.7;
    ctx.fillRect(MARGIN + ti * CELL, MARGIN + si * CELL, CELL - 0.5, CELL - 0.5);
    cellAt.set(`${si},${ti}`, w);
  }
  ctx.globalAlpha = 1;

  // Row labels (left)
  ctx.fillStyle = '#aaa';
  ctx.font = `${Math.max(8, CELL - 1)}px Menlo,monospace`;
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (let i = 0; i < N; i++) {
    ctx.fillStyle = TYPE_COLORS[nodes[i].type] || '#888';
    const lbl = nodes[i].name.length > 26 ? nodes[i].name.slice(0, 24) + '…' : nodes[i].name;
    ctx.fillText(lbl, MARGIN - 12, MARGIN + i * CELL + CELL / 2);
  }
  // Column labels (top, rotated 90°)
  ctx.textAlign = 'left';
  for (let i = 0; i < N; i++) {
    ctx.fillStyle = TYPE_COLORS[nodes[i].type] || '#888';
    const lbl = nodes[i].name.length > 26 ? nodes[i].name.slice(0, 24) + '…' : nodes[i].name;
    ctx.save();
    ctx.translate(MARGIN + i * CELL + CELL / 2, MARGIN - 12);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(lbl, 0, 4);
    ctx.restore();
  }

  status.textContent = `${N} nodes · ${cells.length} cells (cell=${CELL}px)`;
}

cv.addEventListener('mousemove', e => {
  const rect = cv.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;
  const ti = Math.floor((x - MARGIN) / CELL);
  const si = Math.floor((y - MARGIN) / CELL);
  if (si < 0 || ti < 0 || si >= nodes.length || ti >= nodes.length) {
    tooltip.style.display = 'none'; return;
  }
  const w = cellAt.get(`${si},${ti}`);
  if (!w) { tooltip.style.display = 'none'; return; }
  tooltip.innerHTML = `<b>${nodes[si].name}</b> → <b>${nodes[ti].name}</b><br>weight: ${w}<br>${nodes[si].type} → ${nodes[ti].type}`;
  tooltip.style.display = 'block';
  tooltip.style.left = (e.clientX + 14) + 'px';
  tooltip.style.top = (e.clientY + 14) + 'px';
});
cv.addEventListener('mouseleave', () => { tooltip.style.display = 'none'; });

minM.addEventListener('input', () => { minMV.textContent = minM.value; load(); });
lim.addEventListener('input', () => { limV.textContent = lim.value; load(); });
document.getElementById('reload').onclick = load;
load();
</script>
</body></html>"""
