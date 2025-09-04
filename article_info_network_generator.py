import pandas as pd
from pyvis.network import Network
import re
from pathlib import Path

# ----------------------------
# Configuration
# ----------------------------
CSV_PATH = "ovc_data_biomarkers.csv"  # Update if needed
OUTPUT_DIR = "nets"             # Directory where HTMLs will be written

# Association: column name -> output filename
ASSOCIATIONS = {
    "Proteins":  "articles_to_proteins.html",
    "Genes":     "articles_to_genes.html",
    "DNA":       "articles_to_dna.html",
    "RNA":       "articles_to_rna.html",
    "Meth-RNA":  "articles_to_meth_rna.html",
}

# Expected columns
REQUIRED_COLUMNS = [
    "File Name", "Article Name", "Proteins", "Genes", "DNA", "RNA", "Meth-RNA"
]

# Split pattern: comma, semicolon, pipe, newline, slash
SPLIT_PATTERN = r"[,\;\|\n/]+"


# ----------------------------
# Helpers
# ----------------------------
def parse_items_to_upper(value: object) -> list[str]:
    """
    Convert a cell value to a list of separate UPPERCASE items.
    - Handles NaN/empty gracefully.
    - Splits on comma, semicolon, pipe, newline, or slash.
    - Trims whitespace, normalizes internal spaces, deduplicates (order-preserving).
    """
    if value is None:
        return []
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return []

    raw_parts = re.split(SPLIT_PATTERN, s)
    seen = set()
    items = []
    for p in raw_parts:
        t = " ".join(p.strip().split())  # normalize internal whitespace
        if not t:
            continue
        u = t.upper()
        if u not in seen:
            seen.add(u)
            items.append(u)
    return items


def validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in CSV: {missing}. "
            f"Found columns: {list(df.columns)}"
        )


def build_graph(df: pd.DataFrame, entity_col: str, output_file: str) -> None:
    """
    Build one network: Article Name -> entity_col.
    Only adds an Article node if it connects to at least one entity.
    Writes HTML to output_file.
    """
    net = Network(height="750px", width="100%", bgcolor="#ffffff",
                  font_color="black", notebook=False)

    # Physics / layout
    net.barnes_hut()

    # Disable hover tooltips completely (we never set node 'title')
    net.set_options("""
    {
      "interaction": {
        "hover": false,
        "tooltipDelay": 999999
      }
    }
    """)

    # Track created nodes/edges to avoid duplicates
    seen_nodes = set()
    seen_edges = set()

    for _, row in df.iterrows():
        article = str(row["Article Name"]).strip()
        if not article:
            continue

        # Parse multi-valued cell into separate items (uppercased)
        items = parse_items_to_upper(row.get(entity_col))

        # If no entity items, skip creating the article node (prevents isolated nodes)
        if not items:
            continue

        # Add the article node now that we know it will connect
        article_id = f"ARTICLE::{article}"
        if article_id not in seen_nodes:
            net.add_node(article_id, label=article, shape="box", color="lightblue")
            seen_nodes.add(article_id)

        # Add entity nodes + edges
        for item in items:
            entity_id = f"{entity_col.upper()}::{item}"
            if entity_id not in seen_nodes:
                net.add_node(entity_id, label=item, shape="ellipse", color="lightgreen")
                seen_nodes.add(entity_id)

            edge_key = (article_id, entity_id)
            if edge_key not in seen_edges:
                net.add_edge(article_id, entity_id)
                seen_edges.add(edge_key)

    net.save_graph(output_file)


# ----------------------------
# Main
# ----------------------------
def main():
    # Load CSV as strings
    df = pd.read_csv(CSV_PATH, dtype=str, keep_default_na=True)
    validate_columns(df)

    # Ensure output directory exists
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Build each association network
    for col, filename in ASSOCIATIONS.items():
        out_path = str(Path(OUTPUT_DIR) / filename)
        build_graph(df, col, out_path)
        print(f"[OK] Wrote: {out_path}")


if __name__ == "__main__":
    main()
