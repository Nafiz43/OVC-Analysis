#!/usr/bin/env python3
"""Aggregates biomarker mentions from extracted data into an Excel summary file."""

import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

# ---------------- Configuration ----------------
INPUT_NAME = "extracted_data.csv"
CASE_INSENSITIVE = False  # set to True to merge capitalization variants

CATEGORIES = ["Proteins", "Genes", "DNA", "RNA", "Meth-RNA"]
EMPTY_LIKE = {"", "nan", "none", "null", "[]", "{}", "na", "n/a", "-", "--"}
_SPLIT_RE = re.compile(r"[;,]")


def find_input_file() -> Path:
    candidates = [Path.cwd() / INPUT_NAME, Path("/mnt/data") / INPUT_NAME]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find '{INPUT_NAME}' in cwd or /mnt/data.")


def clean_token(tok: str) -> str:
    t = tok.strip()
    if (t.startswith(("'", '"')) and t.endswith(("'", '"'))) or \
       (t.startswith("[") and t.endswith("]")) or \
       (t.startswith("{") and t.endswith("}")) or \
       (t.startswith("(") and t.endswith(")")):
        t = t[1:-1].strip()
    t = re.sub(r"\s+", " ", t).strip()
    return t


def key_for(tok: str) -> str:
    return tok.lower().strip() if CASE_INSENSITIVE else tok.strip()


def extract_items(series: pd.Series) -> Tuple[Counter, Dict[str, str]]:
    counts: Counter = Counter()
    casing_tracker: Dict[str, Counter] = defaultdict(Counter)

    for val in series.dropna().astype(str):
        parts = _SPLIT_RE.split(val)
        for p in parts:
            name = clean_token(p)
            if not name or name.lower() in EMPTY_LIKE:
                continue
            k = key_for(name)
            counts[k] += 1
            casing_tracker[k][name] += 1

    rep_map: Dict[str, str] = {}
    for k, cts in casing_tracker.items():
        rep_map[k] = cts.most_common(1)[0][0]
    return counts, rep_map


def main() -> None:
    inpath = find_input_file()
    outpath = inpath.with_name("entity_counts_all.xlsx")

    df = pd.read_csv(inpath)

    missing = [c for c in CATEGORIES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected column(s) in CSV: {missing}")

    unique_counts = {}

    print("\n=== Entity Statistics (Excel with multiple sheets) ===")
    print(f"Input:  {inpath}")
    print(f"Output: {outpath}")

    with pd.ExcelWriter(outpath, engine="openpyxl") as writer:
        for cat in CATEGORIES:
            counts, rep_map = extract_items(df[cat])
            unique_count = len(counts)
            total_mentions = sum(counts.values())
            unique_counts[cat] = unique_count

            print(f"\n[{cat}]")
            print(f"  Unique {cat}: {unique_count}")
            print(f"  Total mentions: {total_mentions}")

            rows = [(rep_map.get(k, k), c) for k, c in counts.items()]
            rows.sort(key=lambda x: (-x[1], x[0].lower()))
            cat_df = pd.DataFrame(rows, columns=["Name", "Occurrence"])

            cat_df.to_excel(writer, sheet_name=cat, index=False)

    print("\n=== Final Unique Counts ===")
    print(
        "Number of Unique Proteins: {p}, Genes: {g}, DNA: {d}, RNA: {r}, Meth-RNA: {m}".format(
            p=unique_counts.get("Proteins", 0),
            g=unique_counts.get("Genes", 0),
            d=unique_counts.get("DNA", 0),
            r=unique_counts.get("RNA", 0),
            m=unique_counts.get("Meth-RNA", 0),
        )
    )

    print("\nDone. Wrote Excel file with 5 sheets.")


if __name__ == "__main__":
    main()