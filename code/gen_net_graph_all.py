#!/usr/bin/env python3
"""
gen_net_graph_all.py

Build an interactive bipartite network graph from:
    /data/Deep_Angiography/OVC-Analysis/code/data/results/actionable_findings_extracted.csv
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


DEFAULT_INPUT_CSV = Path(
    "/data/Deep_Angiography/OVC-Analysis/code/data/results/actionable_findings_extracted.csv"
)
DEFAULT_OUTPUT_HTML = Path(
    "/data/Deep_Angiography/OVC-Analysis/code/data/results/actionable_dictionary_group_network.html"
)

REQUIRED_COLUMNS = [
    "Actionable",
    "Evidence",
    "Article Title",
    "Article Year",
    "Dictionary Group",
    "Phrases from the dictionary",
]


def normalize_text(value: str) -> str:
    return " ".join((value or "").strip().split())


def split_semicolon_pipe(value: str) -> List[str]:
    if not value:
        return []

    raw_parts = []
    current = []
    for ch in value:
        if ch in [";", "|"]:
            raw_parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    raw_parts.append("".join(current))

    out = []
    for part in raw_parts:
        cleaned = normalize_text(part)
        if cleaned:
            out.append(cleaned)

    deduped = []
    seen = set()
    for item in out:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def build_plain_tooltip_text(text: str) -> str:
    return normalize_text(text or "N/A")


def read_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    rows: List[Dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV appears empty or has no header row.")

        missing = [col for col in REQUIRED_COLUMNS if col not in reader.fieldnames]
        if missing:
            raise ValueError("Missing required column(s): " + ", ".join(missing))

        for row in reader:
            cleaned = {k: normalize_text(v or "") for k, v in row.items()}
            if not cleaned["Actionable"] or not cleaned["Dictionary Group"]:
                continue
            rows.append(cleaned)

    if not rows:
        raise ValueError("No usable rows found in CSV.")
    return rows


def build_graph_payload(rows: List[Dict[str, str]]) -> Tuple[List[dict], List[dict], dict]:
    actionable_key_to_visid: OrderedDict[Tuple[str, str, str, str], str] = OrderedDict()
    actionable_key_to_meta: Dict[Tuple[str, str, str, str], dict] = {}

    dict_group_to_visid: OrderedDict[str, str] = OrderedDict()
    dict_group_to_phrases: Dict[str, List[str]] = defaultdict(list)

    edges_seen = set()
    edges: List[dict] = []

    for row in rows:
        actionable_key = (
            row["Actionable"],
            row["Evidence"],
            row["Article Title"],
            row["Article Year"],
        )
        dict_group = row["Dictionary Group"]

        if actionable_key not in actionable_key_to_visid:
            visid = f"A_{len(actionable_key_to_visid) + 1}"
            actionable_key_to_visid[actionable_key] = visid
            actionable_key_to_meta[actionable_key] = {
                "id": visid,
                "short_label": f"a{len(actionable_key_to_visid)}",
                "node_type": "actionable",
                "actionable": row["Actionable"],
                "evidence": row["Evidence"],
                "article_title": row["Article Title"],
                "article_year": row["Article Year"],
            }

        if dict_group not in dict_group_to_visid:
            visid = f"D_{len(dict_group_to_visid) + 1}"
            dict_group_to_visid[dict_group] = visid

        phrases_val = row["Phrases from the dictionary"]
        if phrases_val:
            dict_group_to_phrases[dict_group].append(phrases_val)

    dict_group_meta = {}
    for dict_group, visid in dict_group_to_visid.items():
        collected = []
        seen = set()

        for phrase_blob in dict_group_to_phrases.get(dict_group, []):
            items = split_semicolon_pipe(phrase_blob)
            if not items:
                fallback = normalize_text(phrase_blob)
                items = [fallback] if fallback else []

            for item in items:
                if item not in seen:
                    collected.append(item)
                    seen.add(item)

        dict_group_meta[dict_group] = {
            "id": visid,
            "short_label": f"d{len(dict_group_meta) + 1}",
            "node_type": "dictionary_group",
            "dictionary_group": dict_group,
            "phrases_from_dictionary": collected,
        }

    edge_counter = 1
    for row in rows:
        actionable_key = (
            row["Actionable"],
            row["Evidence"],
            row["Article Title"],
            row["Article Year"],
        )
        dict_group = row["Dictionary Group"]

        from_id = dict_group_to_visid[dict_group]
        to_id = actionable_key_to_visid[actionable_key]
        edge_key = (from_id, to_id)

        if edge_key in edges_seen:
            continue
        edges_seen.add(edge_key)

        edges.append(
            {
                "id": f"E_{edge_counter}",
                "from": from_id,
                "to": to_id,
                "width": 1.6,
            }
        )
        edge_counter += 1

    nodes = []

    for actionable_key, visid in actionable_key_to_visid.items():
        meta = actionable_key_to_meta[actionable_key]

        tooltip_text = build_plain_tooltip_text(meta["actionable"])

        search_blob = " ".join(
            [
                meta["short_label"],
                meta["actionable"],
                meta["evidence"],
                meta["article_title"],
                meta["article_year"],
            ]
        ).lower()

        nodes.append(
            {
                "id": meta["id"],
                "label": meta["short_label"],
                "shape": "dot",
                "size": 18,
                "color": {
                    "background": "#fb6a4a",
                    "border": "#c2410c",
                    "highlight": {"background": "#fb6a4a", "border": "#9a3412"},
                    "hover": {"background": "#fb6a4a", "border": "#9a3412"},
                },
                "borderWidth": 1.5,
                "font": {"color": "black"},
                "title": tooltip_text,
                "node_type": meta["node_type"],
                "actionable": meta["actionable"],
                "evidence": meta["evidence"],
                "article_title": meta["article_title"],
                "article_year": meta["article_year"],
                "search_blob": search_blob,
            }
        )

    for dict_group, visid in dict_group_to_visid.items():
        meta = dict_group_meta[dict_group]
        phrases_preview = "; ".join(meta["phrases_from_dictionary"]) if meta["phrases_from_dictionary"] else "N/A"

        tooltip_text = build_plain_tooltip_text(meta["dictionary_group"])

        search_blob = " ".join(
            [meta["short_label"], meta["dictionary_group"], phrases_preview]
        ).lower()

        nodes.append(
            {
                "id": meta["id"],
                "label": meta["short_label"],
                "shape": "dot",
                "size": 22,
                "color": {
                    "background": "#6baed6",
                    "border": "#2563eb",
                    "highlight": {"background": "#6baed6", "border": "#1d4ed8"},
                    "hover": {"background": "#6baed6", "border": "#1d4ed8"},
                },
                "borderWidth": 1.5,
                "font": {"color": "black"},
                "title": tooltip_text,
                "node_type": meta["node_type"],
                "dictionary_group": meta["dictionary_group"],
                "phrases_from_dictionary": meta["phrases_from_dictionary"],
                "search_blob": search_blob,
            }
        )

    stats = {
        "row_count": len(rows),
        "actionable_count": len(actionable_key_to_visid),
        "dictionary_group_count": len(dict_group_to_visid),
        "edge_count": len(edges),
    }

    return nodes, edges, stats


def build_html(nodes: List[dict], edges: List[dict], stats: dict, source_csv: Path) -> str:
    nodes_json = json.dumps(nodes, ensure_ascii=False)
    edges_json = json.dumps(edges, ensure_ascii=False)
    stats_json = json.dumps(stats, ensure_ascii=False)
    source_csv_str = html.escape(str(source_csv))

    html_text = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Actionable–Dictionary Group Network Explorer</title>

  <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>

  <style>
    :root {
      --bg: #f6f8fb;
      --card: #ffffff;
      --text: #0f172a;
      --muted: #64748b;
      --border: #dbe4f0;
      --shadow: 0 10px 25px rgba(15, 23, 42, 0.08);
      --dict: #6baed6;
      --action: #fb6a4a;
      --accent: #2563eb;
      --selected: #e0f2fe;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--text);
      background: linear-gradient(180deg, #f8fbff 0%, #f4f7fb 100%);
    }

    div.vis-tooltip {
      position: absolute;
      visibility: hidden;
      padding: 8px 10px;
      max-width: 420px;
      white-space: normal;
      word-break: break-word;
      overflow-wrap: anywhere;
      line-height: 1.45;
      font-size: 14px;
      color: #0f172a;
      background: #ffffff;
      border: 1px solid #dbe4f0;
      border-radius: 10px;
      box-shadow: 0 10px 25px rgba(15, 23, 42, 0.12);
      z-index: 5;
      pointer-events: none;
    }

    .app-shell {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
      gap: 20px;
      min-height: 100vh;
      padding: 24px;
    }

    .main-stage { min-width: 0; }

    .hero-card,
    .network-card,
    .side-panel {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 20px;
      box-shadow: var(--shadow);
    }

    .hero-card {
      padding: 24px 26px;
      margin-bottom: 18px;
    }

    .hero-eyebrow,
    .panel-eyebrow {
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--accent);
      margin-bottom: 8px;
    }

    h1, h2, p { margin: 0; }

    h1 {
      font-size: 32px;
      line-height: 1.1;
      margin-bottom: 10px;
    }

    .hero-card p,
    .panel-copy {
      color: var(--muted);
      line-height: 1.55;
    }

    .network-card { overflow: hidden; }

    .network-toolbar {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 16px;
      padding: 18px 20px;
      border-bottom: 1px solid var(--border);
      background: #fbfdff;
    }

    .toolbar-title {
      font-size: 18px;
      font-weight: 700;
      margin-bottom: 4px;
    }

    .toolbar-subtitle {
      color: var(--muted);
      font-size: 14px;
      line-height: 1.4;
    }

    .legend {
      display: flex;
      gap: 14px;
      flex-wrap: wrap;
      color: var(--muted);
      font-size: 13px;
    }

    .legend-item {
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }

    .legend-dot {
      width: 12px;
      height: 12px;
      border-radius: 999px;
      display: inline-block;
    }

    .dict-dot { background: var(--dict); }
    .action-dot { background: var(--action); }
    .match-ring {
      width: 12px;
      height: 12px;
      border-radius: 999px;
      display: inline-block;
      border: 3px solid #111827;
      background: white;
    }

    #mynetwork {
      width: 100%;
      height: calc(100vh - 210px);
      min-height: 620px;
      background: linear-gradient(180deg, #ffffff 0%, #f9fbfd 100%);
    }

    .side-panel {
      padding: 20px;
      display: flex;
      flex-direction: column;
      gap: 18px;
      position: sticky;
      top: 24px;
      max-height: calc(100vh - 48px);
      overflow: auto;
    }

    .panel-section {
      display: flex;
      flex-direction: column;
      gap: 10px;
    }

    .panel-section h2 {
      font-size: 24px;
      line-height: 1.15;
    }

    .panel-search {
      width: 100%;
      padding: 12px 14px;
      border: 1px solid var(--border);
      border-radius: 12px;
      font-size: 14px;
      outline: none;
    }

    .panel-search:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 4px rgba(37, 99, 235, 0.08);
    }

    .panel-actions {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }

    .primary-btn,
    .secondary-btn {
      border: 0;
      border-radius: 12px;
      padding: 10px 14px;
      font-size: 14px;
      font-weight: 600;
      cursor: pointer;
      transition: 0.15s ease;
    }

    .primary-btn {
      background: var(--accent);
      color: white;
    }

    .secondary-btn {
      background: #eef4fb;
      color: #1e293b;
    }

    .section-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      font-weight: 700;
    }

    .count-badge,
    .muted-mini {
      font-size: 12px;
      color: var(--muted);
    }

    .count-badge {
      background: #eff6ff;
      color: #1d4ed8;
      border-radius: 999px;
      padding: 4px 8px;
      font-weight: 700;
    }

    .stats-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }

    .stat-card {
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 12px;
      background: #fcfdff;
    }

    .stat-label {
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 6px;
    }

    .stat-value {
      font-size: 22px;
      font-weight: 800;
    }

    .chips-wrap,
    .mini-chip-wrap {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }

    .empty-state {
      border: 1px dashed var(--border);
      border-radius: 12px;
      padding: 12px;
      color: var(--muted);
      background: #fafcff;
      font-size: 14px;
    }

    .mini-chip,
    .selected-chip {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 7px 10px;
      border-radius: 999px;
      background: #eff6ff;
      color: #1e3a8a;
      font-size: 12px;
      font-weight: 700;
    }

    .group-list-section {
      min-height: 220px;
    }

    .group-list {
      display: flex;
      flex-direction: column;
      gap: 8px;
      max-height: 320px;
      overflow: auto;
      padding-right: 2px;
    }

    .group-item {
      width: 100%;
      text-align: left;
      border: 1px solid var(--border);
      background: white;
      color: var(--text);
      padding: 11px 12px;
      border-radius: 12px;
      font-size: 14px;
      font-weight: 600;
      cursor: pointer;
    }

    .group-item.selected {
      background: var(--selected);
      border-color: #93c5fd;
      color: #0f3d91;
    }

    .group-item.matched {
      border-color: #1d4ed8;
      box-shadow: inset 0 0 0 1px #1d4ed8;
      background: #eff6ff;
    }

    .info-card {
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 14px;
      background: #fcfdff;
      color: var(--muted);
      line-height: 1.55;
      font-size: 14px;
    }

    .info-title {
      color: var(--text);
      font-size: 16px;
      font-weight: 800;
      margin-bottom: 8px;
    }

    .info-text {
      color: #334155;
      white-space: pre-wrap;
      margin-bottom: 12px;
    }

    .info-subtitle {
      color: var(--muted);
      font-weight: 700;
      font-size: 12px;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      margin-bottom: 8px;
    }

    @media (max-width: 1100px) {
      .app-shell {
        grid-template-columns: 1fr;
      }

      .side-panel {
        position: static;
        max-height: none;
      }

      #mynetwork {
        height: 680px;
      }
    }
  </style>
</head>
<body>
  <div class="app-shell">
    <main class="main-stage">
      <div class="hero-card">
        <div class="hero-eyebrow">Interactive actionable relationship map</div>
        <h1>Actionable–Dictionary Group Network Explorer</h1>
        <p>
          This network was generated from
          <strong>__SOURCE_CSV__</strong>.
          Search works across both sides of the graph, and directly matched nodes are highlighted.
        </p>
        <div id="statusBox" class="loading">Network loaded successfully.</div>
      </div>

      <div class="network-card">
        <div class="network-toolbar">
          <div>
            <div class="toolbar-title">Network view</div>
            <div class="toolbar-subtitle" id="networkSummary">Preparing network…</div>
          </div>
          <div class="legend">
            <span class="legend-item"><span class="legend-dot dict-dot"></span> Dictionary groups</span>
            <span class="legend-item"><span class="legend-dot action-dot"></span> Actionables</span>
          </div>
        </div>
        <div id="mynetwork"></div>
      </div>
    </main>

    <aside class="side-panel">
      <div class="panel-section">
        <div class="panel-eyebrow">Interactive dictionary-group explorer</div>
        <h2>Dictionary Group Panel</h2>
        <p class="panel-copy">
          Search checks both dictionary-group fields and actionable-side fields. Direct matches are highlighted,
          and connected neighbors are included automatically.
        </p>
      </div>

      <div class="panel-section">
        <input id="groupSearch" class="panel-search" type="text" placeholder="Search groups, phrases, actionables, evidence, article..." />
        <div class="panel-actions">
          <button id="selectVisibleBtn" class="secondary-btn">Select visible groups</button>
          <button id="clearSelectionBtn" class="secondary-btn">Clear selection</button>
          <button id="showAllBtn" class="primary-btn">Show full network</button>
        </div>
      </div>

      <div class="panel-section">
        <div class="section-header">
          <span>Selected dictionary groups</span>
          <span id="selectedCountBadge" class="count-badge">0</span>
        </div>
        <div class="stats-grid">
          <div class="stat-card">
            <div class="stat-label">Total groups</div>
            <div class="stat-value" id="totalGroupsStat">0</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Selected</div>
            <div class="stat-value" id="selectedGroupsStat">0</div>
          </div>
        </div>
        <div id="selectedGroupsChips" class="chips-wrap empty-state">No dictionary groups selected</div>
      </div>

      <div class="panel-section group-list-section">
        <div class="section-header">
          <span>Available dictionary groups</span>
          <span id="shownGroupsLabel" class="muted-mini">0 shown</span>
        </div>
        <div id="groupList" class="group-list"></div>
      </div>

      <div class="panel-section">
        <div class="section-header">
          <span>Details</span>
        </div>
        <div id="detailsInfo" class="info-card">
          Click an actionable node to inspect the actionable details. Click a dictionary-group node to inspect the group and phrases.
        </div>
      </div>
    </aside>
  </div>

  <script>
    const ALL_NODES = __NODES_JSON__;
    const ALL_EDGES = __EDGES_JSON__;
    const GRAPH_STATS = __STATS_JSON__;

    let network = null;
    let nodesDS = null;
    let edgesDS = null;

    const nodeMap = new Map(ALL_NODES.map(n => [String(n.id), n]));
    const dictGroupIds = ALL_NODES
      .filter(n => n.node_type === "dictionary_group")
      .map(n => String(n.id))
      .sort((a, b) => {
        const na = nodeMap.get(a)?.dictionary_group || "";
        const nb = nodeMap.get(b)?.dictionary_group || "";
        return na.localeCompare(nb);
      });

    const actionableIds = ALL_NODES
      .filter(n => n.node_type === "actionable")
      .map(n => String(n.id));

    const groupToActionables = new Map();
    const actionableToGroups = new Map();

    for (const gid of dictGroupIds) groupToActionables.set(gid, new Set());
    for (const aid of actionableIds) actionableToGroups.set(aid, new Set());

    for (const edge of ALL_EDGES) {
      const fromId = String(edge.from);
      const toId = String(edge.to);

      if (!groupToActionables.has(fromId)) groupToActionables.set(fromId, new Set());
      if (!actionableToGroups.has(toId)) actionableToGroups.set(toId, new Set());

      groupToActionables.get(fromId).add(toId);
      actionableToGroups.get(toId).add(fromId);
    }

    let selectedGroups = new Set();
    let searchText = "";

    function escapeHtml(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
    }

    function setDefaultInfo() {
      document.getElementById("detailsInfo").innerHTML =
        "Click an actionable node to inspect the actionable details. Click a dictionary-group node to inspect the group and phrases.";
    }

    function nodeMatchesSearch(node, term) {
      if (!term) return false;
      return String(node.search_blob || "").includes(term);
    }

    function getSearchState(term) {
      if (!term) {
        return null;
      }

      const directMatchedGroups = new Set();
      const directMatchedActionables = new Set();

      for (const node of ALL_NODES) {
        if (!nodeMatchesSearch(node, term)) continue;

        if (node.node_type === "dictionary_group") {
          directMatchedGroups.add(String(node.id));
        } else if (node.node_type === "actionable") {
          directMatchedActionables.add(String(node.id));
        }
      }

      const visibleGroups = new Set(directMatchedGroups);
      const visibleActionables = new Set(directMatchedActionables);

      for (const aid of directMatchedActionables) {
        const neighbors = actionableToGroups.get(aid) || new Set();
        for (const gid of neighbors) visibleGroups.add(gid);
      }

      for (const gid of directMatchedGroups) {
        const neighbors = groupToActionables.get(gid) || new Set();
        for (const aid of neighbors) visibleActionables.add(aid);
      }

      return {
        directMatchedGroups,
        directMatchedActionables,
        visibleGroups,
        visibleActionables
      };
    }

    function getSelectionState() {
      if (!selectedGroups.size) {
        return null;
      }

      const visibleGroups = new Set(selectedGroups);
      const visibleActionables = new Set();

      for (const gid of selectedGroups) {
        const neighbors = groupToActionables.get(gid) || new Set();
        for (const aid of neighbors) visibleActionables.add(aid);
      }

      return {
        visibleGroups,
        visibleActionables
      };
    }

    function intersectSets(a, b) {
      return new Set([...a].filter(x => b.has(x)));
    }

    function cloneNode(node) {
      return JSON.parse(JSON.stringify(node));
    }

    function getVisibleState() {
      const term = searchText.trim().toLowerCase();
      const searchState = getSearchState(term);
      const selectionState = getSelectionState();

      let finalGroupIds = null;
      let finalActionableIds = null;

      if (!searchState && !selectionState) {
        finalGroupIds = new Set(dictGroupIds);
        finalActionableIds = new Set(actionableIds);
      } else if (searchState && !selectionState) {
        finalGroupIds = new Set(searchState.visibleGroups);
        finalActionableIds = new Set(searchState.visibleActionables);
      } else if (!searchState && selectionState) {
        finalGroupIds = new Set(selectionState.visibleGroups);
        finalActionableIds = new Set(selectionState.visibleActionables);
      } else {
        finalGroupIds = intersectSets(selectionState.visibleGroups, searchState.visibleGroups);
        finalActionableIds = intersectSets(selectionState.visibleActionables, searchState.visibleActionables);
      }

      const visibleEdges = ALL_EDGES.filter(edge => {
        const gid = String(edge.from);
        const aid = String(edge.to);
        return finalGroupIds.has(gid) && finalActionableIds.has(aid);
      });

      const nodeIdsFromEdges = new Set();
      for (const edge of visibleEdges) {
        nodeIdsFromEdges.add(String(edge.from));
        nodeIdsFromEdges.add(String(edge.to));
      }

      const visibleNodeIds = new Set(nodeIdsFromEdges);

      if (visibleEdges.length === 0) {
        for (const gid of finalGroupIds) visibleNodeIds.add(gid);
        for (const aid of finalActionableIds) visibleNodeIds.add(aid);
      }

      const visibleNodes = ALL_NODES
        .filter(node => visibleNodeIds.has(String(node.id)))
        .map(node => {
          const cloned = cloneNode(node);
          const id = String(cloned.id);

          const directlyMatched =
            searchState &&
            (
              searchState.directMatchedGroups.has(id) ||
              searchState.directMatchedActionables.has(id)
            );

          if (searchState) {
            if (directlyMatched) {
              if (cloned.node_type === "dictionary_group") {
                cloned.size = 28;
                cloned.borderWidth = 5;
                cloned.color = {
                  background: "#6baed6",
                  border: "#1d4ed8",
                  highlight: { background: "#6baed6", border: "#1d4ed8" },
                  hover: { background: "#6baed6", border: "#1d4ed8" }
                };
              } else {
                cloned.size = 24;
                cloned.borderWidth = 5;
                cloned.color = {
                  background: "#fb6a4a",
                  border: "#b91c1c",
                  highlight: { background: "#fb6a4a", border: "#b91c1c" },
                  hover: { background: "#fb6a4a", border: "#b91c1c" }
                };
              }
            } else {
              cloned.opacity = 0.55;
            }
          }

          return cloned;
        });

      const styledEdges = visibleEdges.map(edge => {
        const cloned = JSON.parse(JSON.stringify(edge));
        const gid = String(cloned.from);
        const aid = String(cloned.to);

        const edgeTouchesDirectMatch =
          searchState &&
          (
            searchState.directMatchedGroups.has(gid) ||
            searchState.directMatchedActionables.has(aid)
          );

        if (searchState) {
          if (edgeTouchesDirectMatch) {
            cloned.width = 2.6;
            cloned.color = {
              color: "#334155",
              highlight: "#111827",
              hover: "#111827",
              opacity: 0.95
            };
          } else {
            cloned.width = 1.2;
            cloned.color = {
              color: "#94a3b8",
              highlight: "#64748b",
              hover: "#64748b",
              opacity: 0.5
            };
          }
        }

        return cloned;
      });

      return {
        visibleNodes,
        visibleEdges: styledEdges,
        searchState
      };
    }

    function updateSummary(visibleNodes, visibleEdges, searchState) {
      const visibleGroups = visibleNodes.filter(n => n.node_type === "dictionary_group").length;
      const visibleActionables = visibleNodes.filter(n => n.node_type === "actionable").length;

      let summary = `Showing ${visibleGroups} dictionary groups, ${visibleActionables} actionable items, and ${visibleEdges.length} relationships`;

      if (selectedGroups.size && searchText.trim()) {
        const directGroupMatches = searchState ? searchState.directMatchedGroups.size : 0;
        const directActionableMatches = searchState ? searchState.directMatchedActionables.size : 0;
        summary =
          `Showing selection + search filter: ${visibleGroups} groups, ${visibleActionables} actionables, ${visibleEdges.length} relationships ` +
          `(direct matches: ${directGroupMatches} groups, ${directActionableMatches} actionables)`;
      } else if (selectedGroups.size) {
        summary =
          `Showing ${selectedGroups.size} selected group${selectedGroups.size > 1 ? "s" : ""}, ` +
          `${visibleActionables} connected actionable item${visibleActionables !== 1 ? "s" : ""}, ` +
          `and ${visibleEdges.length} relationship${visibleEdges.length !== 1 ? "s" : ""}`;
      } else if (searchText.trim()) {
        const directGroupMatches = searchState ? searchState.directMatchedGroups.size : 0;
        const directActionableMatches = searchState ? searchState.directMatchedActionables.size : 0;
        summary =
          `Showing search-expanded subgraph with ${visibleGroups} groups, ${visibleActionables} actionables, and ${visibleEdges.length} relationships ` +
          `(direct matches: ${directGroupMatches} groups, ${directActionableMatches} actionables)`;
      }

      document.getElementById("networkSummary").textContent = summary;
    }

    function showActionableInfo(nodeId) {
      const node = nodeMap.get(String(nodeId));
      if (!node) return;

      document.getElementById("detailsInfo").innerHTML = `
        <div class="info-title">${escapeHtml(node.label || node.id)}</div>
        <div class="info-subtitle">Actionable</div>
        <div class="info-text">${escapeHtml(node.actionable || "N/A")}</div>
        <div class="info-subtitle">Evidence</div>
        <div class="info-text">${escapeHtml(node.evidence || "N/A")}</div>
        <div class="info-subtitle">Article title</div>
        <div class="info-text">${escapeHtml(node.article_title || "N/A")}</div>
        <div class="info-subtitle">Article year</div>
        <div class="info-text">${escapeHtml(node.article_year || "N/A")}</div>
      `;
    }

    function showDictionaryGroupInfo(nodeId) {
      const node = nodeMap.get(String(nodeId));
      if (!node) return;

      const phrases = Array.isArray(node.phrases_from_dictionary) ? node.phrases_from_dictionary : [];
      const phrasesHtml = phrases.length
        ? phrases.map(p => `<span class="mini-chip">${escapeHtml(p)}</span>`).join("")
        : '<span class="muted">No phrases available.</span>';

      document.getElementById("detailsInfo").innerHTML = `
        <div class="info-title">${escapeHtml(node.label || node.id)}</div>
        <div class="info-subtitle">Dictionary Group</div>
        <div class="info-text">${escapeHtml(node.dictionary_group || "N/A")}</div>
        <div class="info-subtitle">Phrases from the dictionary</div>
        <div class="mini-chip-wrap">${phrasesHtml}</div>
      `;
    }

    function renderNetwork() {
      const state = getVisibleState();
      const visibleNodes = state.visibleNodes;
      const visibleEdges = state.visibleEdges;
      const searchState = state.searchState;

      nodesDS = new vis.DataSet(visibleNodes);
      edgesDS = new vis.DataSet(visibleEdges);

      const container = document.getElementById("mynetwork");
      const data = { nodes: nodesDS, edges: edgesDS };
      const options = {
        autoResize: true,
        interaction: {
          hover: true,
          tooltipDelay: 120,
          multiselect: false,
          navigationButtons: true
        },
        physics: {
          enabled: true,
          stabilization: { iterations: 220, fit: true },
          barnesHut: {
            gravitationalConstant: -7000,
            centralGravity: 0.18,
            springLength: 160,
            springConstant: 0.035,
            damping: 0.18,
            avoidOverlap: 0.25
          }
        },
        edges: {
          selectionWidth: 3,
          hoverWidth: 2.2,
          smooth: { enabled: true, type: "dynamic" },
          font: {
            size: 12,
            align: "middle"
          }
        },
        nodes: {
          font: {
            face: "Inter, system-ui, sans-serif",
            size: 15
          }
        }
      };

      if (network) {
        network.destroy();
      }

      network = new vis.Network(container, data, options);

      network.once("stabilizationIterationsDone", function() {
        network.fit({ animation: { duration: 450, easingFunction: "easeInOutQuad" } });
      });

      network.on("click", function(params) {
        if (params.nodes && params.nodes.length) {
          const nodeId = String(params.nodes[0]);
          const node = nodeMap.get(nodeId);

          if (!node) {
            setDefaultInfo();
            return;
          }

          if (node.node_type === "actionable") {
            showActionableInfo(nodeId);
          } else if (node.node_type === "dictionary_group") {
            showDictionaryGroupInfo(nodeId);
          } else {
            setDefaultInfo();
          }
          return;
        }

        setDefaultInfo();
      });

      updateSummary(visibleNodes, visibleEdges, searchState);
    }

    function updateSelectedGroupsUI() {
      const chipsEl = document.getElementById("selectedGroupsChips");
      const selected = Array.from(selectedGroups).sort((a, b) => {
        const na = nodeMap.get(a)?.dictionary_group || "";
        const nb = nodeMap.get(b)?.dictionary_group || "";
        return na.localeCompare(nb);
      });

      document.getElementById("selectedGroupsStat").textContent = selected.length;
      document.getElementById("selectedCountBadge").textContent = selected.length;

      if (!selected.length) {
        chipsEl.className = "chips-wrap empty-state";
        chipsEl.textContent = "No dictionary groups selected";
        return;
      }

      chipsEl.className = "chips-wrap";
      chipsEl.innerHTML = selected
        .map(id => `<span class="selected-chip">${escapeHtml(nodeMap.get(id)?.dictionary_group || id)}</span>`)
        .join("");
    }

    function renderGroupPanel() {
      const listEl = document.getElementById("groupList");
      const term = searchText.trim().toLowerCase();
      const searchState = getSearchState(term);

      const visibleGroups = dictGroupIds.filter(id => {
        if (!searchState) return true;
        return searchState.visibleGroups.has(id);
      });

      listEl.innerHTML = "";

      if (!visibleGroups.length) {
        listEl.innerHTML = `<div class="empty-state">No dictionary groups match the current filter.</div>`;
      } else {
        visibleGroups.forEach(id => {
          const node = nodeMap.get(id);
          const isMatched = searchState ? searchState.directMatchedGroups.has(id) : false;

          const item = document.createElement("button");
          item.className =
            "group-item" +
            (selectedGroups.has(id) ? " selected" : "") +
            (isMatched ? " matched" : "");
          item.type = "button";
          item.textContent = node?.dictionary_group || id;
          item.dataset.groupId = id;

          item.addEventListener("click", () => {
            if (selectedGroups.has(id)) {
              selectedGroups.delete(id);
            } else {
              selectedGroups.add(id);
            }
            renderGroupPanel();
            updateSelectedGroupsUI();
            renderNetwork();
          });

          listEl.appendChild(item);
        });
      }

      document.getElementById("shownGroupsLabel").textContent = `${visibleGroups.length} shown`;
      document.getElementById("totalGroupsStat").textContent = dictGroupIds.length;
    }

    function attachUIHandlers() {
      document.getElementById("groupSearch").addEventListener("input", (e) => {
        searchText = e.target.value || "";
        renderGroupPanel();
        renderNetwork();
      });

      document.getElementById("selectVisibleBtn").addEventListener("click", () => {
        const term = searchText.trim().toLowerCase();
        const searchState = getSearchState(term);

        const visibleGroups = !searchState
          ? dictGroupIds
          : [...searchState.visibleGroups];

        visibleGroups.forEach(id => selectedGroups.add(id));
        renderGroupPanel();
        updateSelectedGroupsUI();
        renderNetwork();
      });

      document.getElementById("clearSelectionBtn").addEventListener("click", () => {
        selectedGroups.clear();
        renderGroupPanel();
        updateSelectedGroupsUI();
        renderNetwork();
        setDefaultInfo();
      });

      document.getElementById("showAllBtn").addEventListener("click", () => {
        selectedGroups.clear();
        searchText = "";
        document.getElementById("groupSearch").value = "";
        renderGroupPanel();
        updateSelectedGroupsUI();
        renderNetwork();
        setDefaultInfo();
      });
    }

    function initialize() {
      attachUIHandlers();
      renderGroupPanel();
      updateSelectedGroupsUI();
      renderNetwork();
      setDefaultInfo();

      document.getElementById("statusBox").textContent =
        `Loaded ${GRAPH_STATS.row_count} row(s), ${GRAPH_STATS.dictionary_group_count} dictionary group(s), ${GRAPH_STATS.actionable_count} actionable item(s), and ${GRAPH_STATS.edge_count} relationship(s).`;
    }

    initialize();
  </script>
</body>
</html>
"""

    html_text = html_text.replace("__SOURCE_CSV__", source_csv_str)
    html_text = html_text.replace("__NODES_JSON__", nodes_json)
    html_text = html_text.replace("__EDGES_JSON__", edges_json)
    html_text = html_text.replace("__STATS_JSON__", stats_json)

    return html_text


def write_html(output_path: Path, html_text: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an interactive bipartite network HTML between actionables and dictionary groups."
    )
    parser.add_argument(
        "--input_csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help=f"Path to the actionable findings CSV (default: {DEFAULT_INPUT_CSV})",
    )
    parser.add_argument(
        "--output_html",
        type=Path,
        default=DEFAULT_OUTPUT_HTML,
        help=f"Path to output HTML (default: {DEFAULT_OUTPUT_HTML})",
    )
    args = parser.parse_args()

    rows = read_csv_rows(args.input_csv)
    nodes, edges, stats = build_graph_payload(rows)
    html_text = build_html(nodes, edges, stats, args.input_csv)
    write_html(args.output_html, html_text)

    print("[DONE] Interactive HTML generated successfully.")
    print(f"[INPUT]  CSV:  {args.input_csv}")
    print(f"[OUTPUT] HTML: {args.output_html}")
    print(
        "[STATS] "
        f"rows={stats['row_count']}, "
        f"dictionary_groups={stats['dictionary_group_count']}, "
        f"actionables={stats['actionable_count']}, "
        f"edges={stats['edge_count']}"
    )
    print()
    print("Open the HTML in a browser.")
    print("If needed, serve locally with:")
    print(f"  cd {args.output_html.parent}")
    print("  python3 -m http.server 8000")


if __name__ == "__main__":
    main()