#!/usr/bin/env python3

"""
Generate a bipartite network graph:
    Actionable Recommendation node  --->  Impacted Gene node

Updates:
- Actionable nodes use short labels (AR_1, AR_2, ...)
- Tooltip for actionable nodes contains ONLY the actionable recommendation text
- Tooltip uses plain text (no HTML), so no raw <div style=...> appears
- HTML is written with write_html() instead of show() to avoid pyvis template error
- Static plot hides recommendation text labels to avoid clutter
"""

import os
import textwrap
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network


# ======================
# CONFIG
# ======================

INPUT_CSV = "data/actionable-results/actionable_recommendations_by_article.csv"

OUTPUT_HTML = "data/actionable-results/actionable_gene_network.html"
OUTPUT_PNG = "data/actionable-results/actionable_gene_network.png"

FIGSIZE = (18, 14)
GENE_NODE_SIZE = 900
ACTIONABLE_NODE_SIZE = 700
FONT_SIZE_GENE = 9


# ======================
# HELPERS
# ======================

def safe_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def wrap_text_plain(s: str, width: int = 70) -> str:
    if not s:
        return ""
    return "\n".join(
        textwrap.wrap(
            s,
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        )
    )


# ======================
# LOAD CSV
# ======================

if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

df = pd.read_csv(INPUT_CSV)

required_cols = [
    "Article Title",
    "Publication Year",
    "Actionable Recommendation",
    "Impacted Genes",
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns in CSV: {missing}")

if df.empty:
    raise ValueError("CSV file is empty.")


# ======================
# BUILD GRAPH
# ======================

G = nx.Graph()
actionable_counter = 0

for _, row in df.iterrows():
    article_title = safe_str(row["Article Title"])
    publication_year = safe_str(row["Publication Year"])
    recommendation_text = safe_str(row["Actionable Recommendation"])
    impacted_genes_raw = safe_str(row["Impacted Genes"])

    if not recommendation_text or not impacted_genes_raw:
        continue

    genes = [g.strip() for g in impacted_genes_raw.split(",") if g.strip()]
    if not genes:
        continue

    actionable_counter += 1
    actionable_node_id = f"AR_{actionable_counter}"

    # Plain text tooltip ONLY
    actionable_tooltip = wrap_text_plain(recommendation_text, 70)

    G.add_node(
        actionable_node_id,
        node_type="actionable",
        label=actionable_node_id,
        full_text=recommendation_text,
        article_title=article_title,
        publication_year=publication_year,
        title_text=actionable_tooltip,
    )

    for gene in genes:
        if not G.has_node(gene):
            G.add_node(
                gene,
                node_type="gene",
                label=gene,
                title_text=gene,
            )
        G.add_edge(actionable_node_id, gene)

print(f"Nodes: {len(G.nodes)}")
print(f"Edges: {len(G.edges)}")


# ======================
# STATIC PNG
# ======================

plt.figure(figsize=FIGSIZE)

pos = nx.spring_layout(G, k=0.8, seed=42)

actionable_nodes = [n for n, d in G.nodes(data=True) if d["node_type"] == "actionable"]
gene_nodes = [n for n, d in G.nodes(data=True) if d["node_type"] == "gene"]

nx.draw_networkx_nodes(
    G,
    pos,
    nodelist=actionable_nodes,
    node_color="salmon",
    node_size=ACTIONABLE_NODE_SIZE,
    alpha=0.9,
)

nx.draw_networkx_nodes(
    G,
    pos,
    nodelist=gene_nodes,
    node_color="lightblue",
    node_size=GENE_NODE_SIZE,
    alpha=0.9,
)

nx.draw_networkx_edges(
    G,
    pos,
    alpha=0.35,
    width=0.8,
)

# Only label genes in the static figure
gene_labels = {n: n for n in gene_nodes}
nx.draw_networkx_labels(
    G,
    pos,
    labels=gene_labels,
    font_size=FONT_SIZE_GENE,
)

plt.title("Actionable Recommendation → Gene Network")
plt.axis("off")
plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved static graph: {OUTPUT_PNG}")


# ======================
# INTERACTIVE HTML
# ======================

net = Network(
    height="900px",
    width="100%",
    bgcolor="white",
    font_color="black",
    notebook=False,
    cdn_resources="in_line",
)

net.barnes_hut(
    gravity=-4000,
    central_gravity=0.2,
    spring_length=180,
    spring_strength=0.03,
    damping=0.95,
    overlap=0.1,
)

for node, data in G.nodes(data=True):
    node_type = data.get("node_type", "")
    label = data.get("label", node)
    title_text = data.get("title_text", label)

    if node_type == "actionable":
        net.add_node(
            node,
            label=label,
            title=title_text,
            color="#fb6a4a",
            size=18,
            shape="dot",
        )
    else:
        net.add_node(
            node,
            label=label,
            title=title_text,
            color="#6baed6",
            size=22,
            shape="dot",
        )

for source, target in G.edges():
    net.add_edge(source, target)

net.show_buttons(filter_=["physics"])
net.write_html(OUTPUT_HTML, open_browser=False)

print(f"Saved interactive graph: {OUTPUT_HTML}")