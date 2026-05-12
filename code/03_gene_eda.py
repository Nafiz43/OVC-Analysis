"""
gene_eda.py

Basic EDA on gene occurrence CSVs produced by count_gene_occurrences.py.

For each list (prognostic + top-5000):
  - Summary stats
  - Top 5 / Least 5 genes (non-zero)
  - Distribution histogram
  - Top-10 bar chart

Saves all figures to the same HGSOCDATA directory as the CSVs.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR  = Path("/Users/nafiz43/Documents/GitHub/OVC-Analysis/code/data/HGSOCDATA")
FILES = {
    "Prognostic Biomarkers": DATA_DIR / "gene_occurrences_prognostic.csv",
    "Top-5000 mRNA":         DATA_DIR / "gene_occurrences_top5000.csv",
}
FIG_DIR = DATA_DIR / "eda_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#0f1117",
    "axes.edgecolor":   "#2e2e3a",
    "axes.labelcolor":  "#e0e0f0",
    "xtick.color":      "#a0a0c0",
    "ytick.color":      "#a0a0c0",
    "text.color":       "#e0e0f0",
    "grid.color":       "#1e1e2e",
    "grid.linestyle":   "--",
    "grid.linewidth":   0.6,
    "font.family":      "monospace",
    "axes.titlesize":   13,
    "axes.labelsize":   11,
})

ACCENT   = "#7b61ff"
ACCENT2  = "#ff6b6b"
BAR_CMAP = "plasma"

# ── Helper ───────────────────────────────────────────────────────────────────

def slug(label: str) -> str:
    return label.lower().replace(" ", "_").replace("-", "")


def print_section(title: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def eda(label: str, csv_path: Path) -> None:
    print_section(label)

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    total_genes   = len(df)
    found         = df[df["occurrence"] > 0]
    not_found     = df[df["occurrence"] == 0]

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n  Total genes         : {total_genes:,}")
    print(f"  Genes with hits     : {len(found):,}  ({100*len(found)/total_genes:.1f}%)")
    print(f"  Genes with 0 hits   : {len(not_found):,}  ({100*len(not_found)/total_genes:.1f}%)")
    print(f"\n  Occurrence stats (all genes):")
    stats = df["occurrence"].describe().rename({
        "count": "count", "mean": "mean", "std": "std",
        "min": "min", "25%": "25th pct", "50%": "median",
        "75%": "75th pct", "max": "max"
    })
    for k, v in stats.items():
        print(f"    {k:<12}: {v:.2f}")

    # ── Top 5 ────────────────────────────────────────────────────────────────
    top5 = df.nlargest(5, "occurrence")[["gene", "occurrence"]]
    print(f"\n  TOP 5 genes:")
    print(top5.to_string(index=False))

    # ── Least 5 (non-zero) ───────────────────────────────────────────────────
    least5 = df[df["occurrence"] > 0].nsmallest(5, "occurrence")[["gene", "occurrence"]]
    print(f"\n  LEAST 5 genes (non-zero occurrences):")
    print(least5.to_string(index=False))

    # ── Figure 1: Distribution histogram ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4.5))
    nonzero = found["occurrence"]
    ax.hist(nonzero, bins=40, color=ACCENT, edgecolor="#0f1117", linewidth=0.4, alpha=0.88)
    ax.set_xlabel("Occurrence count")
    ax.set_ylabel("Number of genes")
    ax.set_title(f"{label} — Occurrence Distribution (genes with ≥1 hit)")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(axis="y")
    med = nonzero.median()
    ax.axvline(med, color=ACCENT2, linewidth=1.4, linestyle="--", label=f"Median = {med:.0f}")
    ax.legend(framealpha=0.2)
    plt.tight_layout()
    hist_path = FIG_DIR / f"{slug(label)}_distribution.png"
    fig.savefig(hist_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n  → Histogram saved : {hist_path}")

    # ── Figure 2: Top-10 bar chart ────────────────────────────────────────────
    top10 = df.nlargest(10, "occurrence")
    colors = plt.colormaps[BAR_CMAP](
        [i / max(len(top10) - 1, 1) for i in range(len(top10))]
    )
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.barh(top10["gene"][::-1], top10["occurrence"][::-1],
                   color=colors[::-1], edgecolor="#0f1117", linewidth=0.4)
    for bar, val in zip(bars, top10["occurrence"][::-1]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                str(int(val)), va="center", ha="left", fontsize=9, color="#e0e0f0")
    ax.set_xlabel("Occurrence count")
    ax.set_title(f"{label} — Top 10 Genes by Occurrence")
    ax.grid(axis="x")
    ax.set_xlim(0, top10["occurrence"].max() * 1.15)
    plt.tight_layout()
    bar_path = FIG_DIR / f"{slug(label)}_top10.png"
    fig.savefig(bar_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  → Bar chart saved : {bar_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("\n╔══════════════════════════════════════════════╗")
    print("║        Gene Occurrence EDA                  ║")
    print("╚══════════════════════════════════════════════╝")

    for label, path in FILES.items():
        if not path.exists():
            print(f"\n  [SKIP] File not found: {path}")
            continue
        eda(label, path)

    print(f"\n\nAll figures saved to: {FIG_DIR}\n")


if __name__ == "__main__":
    main()