#!/usr/bin/env python3
"""
extract_actionables_from_markdowns_ollama_cli.py

Process markdown (.md) files, extract structured findings/actionables using
Ollama CLI (NOT the Ollama server API), and append results row-by-row into
a single CSV.

LLM engine:
    ollama run qwen2.5:72b

Output CSV columns:
- Actionable/Findings
- Evidence
- Article Title
- Article Year
- Dictionary Group
- Phrases from the dictionary
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
import time
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple


# ============================================================
# CONFIG
# ============================================================

DEFAULT_MD_DIR = Path("/Users/nafiz43/Documents/GitHub/OVC-Analysis/code/data/actionable-articles-preprocessed")
DEFAULT_OUTPUT_DIR = Path("/Users/nafiz43/Documents/GitHub/OVC-Analysis/code/data/results")
DEFAULT_OUTPUT_CSV = DEFAULT_OUTPUT_DIR / "actionable_findings_extracted.csv"
DEFAULT_PROGRESS_LOG = DEFAULT_OUTPUT_DIR / "actionable_findings_progress.log"

DEFAULT_MODEL = "qwen3:8b"
DEFAULT_MAX_RETRIES = 3
DEFAULT_SLEEP_BETWEEN_FILES = 0.5
DEFAULT_MAX_CHARS_PER_ARTICLE = 120000
DEFAULT_OLLAMA_TIMEOUT_SEC = 1800  # 30 min per article if needed

CSV_COLUMNS = [
    "Actionable/Findings",
    "Evidence",
    "Article Title",
    "Article Year",
    "Dictionary Group",
    "Phrases from the dictionary",
]

DICTIONARY_GROUPS = {
    "Dictionary A: Chemotherapy & Resistance Findings": [
        "platinum-resistant",
        "platinum-refractory",
        "platinum-sensitive",
        "relapse",
        "early recurrence",
        "disease progression",
        "neoadjuvant chemotherapy",
        "NACT response",
        "macroscopic residual disease",
        "carboplatin",
        "paclitaxel",
        "cisplatin",
    ],
    "Dictionary B: Survival & Clinical Outcome Findings": [
        "Overall Survival",
        "OS",
        "Progression-Free Survival",
        "PFS",
        "Kaplan-Meier",
        "hazard ratio",
        "multivariate analysis",
        "prognostic biomarker",
        "independent prognostic factor",
        "poor prognosis",
        "favorable prognosis",
    ],
    "Dictionary C: Molecular Subtypes & Therapy Findings": [
        "PARP inhibitor",
        "olaparib",
        "niraparib",
        "Homologous Recombination Deficiency",
        "HRD",
        "BRCAness",
        "synthetic lethality",
        "tumor microenvironment",
        "immune infiltration",
        "angiogenesis inhibitor",
        "ADC",
        "antibody-drug conjugate",
    ],
}


# ============================================================
# PROMPT
# ============================================================

def build_extraction_prompt(article_text: str, article_title_guess: str, article_year_guess: str) -> str:
    dictionary_text = "\n".join(
        f"{group}:\n  - " + "\n  - ".join(terms)
        for group, terms in DICTIONARY_GROUPS.items()
    )

    return f"""
You are an expert biomedical literature extraction assistant.

Your task is to read a markdown-formatted scientific article and extract clinically or biologically meaningful findings in a structured way.

The domain is ovarian cancer biomarker and therapy literature. Focus especially on:
1. chemotherapy response and platinum resistance
2. survival and prognostic outcome
3. molecular subtype, targetability, HRD/PARP-related or tumor microenvironment-related findings

Return ONLY a valid JSON array.
Do NOT return markdown fences.
Do NOT return any explanatory text before or after the JSON.

Required JSON schema:
[
  {{
    "Actionable/Findings": "...",
    "Evidence": "...",
    "Article Title": "...",
    "Article Year": "...",
    "Dictionary Group": "...",
    "Phrases from the dictionary": ["...", "..."]
  }}
]

FIELD DEFINITIONS

1. Actionable/Findings
- A concise, meaningful finding or actionable conclusion grounded in the article.
- Prefer findings involving genes, biomarkers, pathways, prognosis, resistance, survival, or therapy response.
- Keep it specific and useful.

2. Evidence
- This field is mandatory and must be detailed.
- Include:
  a) the exact statement(s) from the article that support the finding, quoted as faithfully as possible from the markdown
  b) a concise explanation of how those exact statement(s) support the actionable/finding
- Format exactly like this:
  "Exact statement(s): ... | Inference: ..."

3. Article Title
- Use the actual article title if present in the markdown.
- Otherwise use: "{article_title_guess}"

4. Article Year
- Use the actual year if present in the markdown.
- Otherwise use: "{article_year_guess}"
- If unknown, use an empty string.

5. Dictionary Group
- Must be exactly one of:
  - "Dictionary A: Chemotherapy & Resistance Findings"
  - "Dictionary B: Survival & Clinical Outcome Findings"
  - "Dictionary C: Molecular Subtypes & Therapy Findings"

6. Phrases from the dictionary
- Must be a JSON array.
- Include only exact phrases from the dictionaries below.
- Include only phrases actually relevant to the finding/evidence.

STRICT RULES

- Only extract findings that are directly supported by the article text.
- No hallucinations.
- No unsupported gene claims.
- No duplicate rows that say the same thing in different wording.
- Prefer results over background/introduction material.
- Ignore generic methodological descriptions unless they directly support a clinically meaningful result.
- If there is not enough evidence for a finding, do not include it.
- If the article contains no relevant findings, return [].
- Choose the single best Dictionary Group per finding.

DICTIONARY GROUPS

{dictionary_text}

ARTICLE MARKDOWN

{article_text}
""".strip()


# ============================================================
# FILE / CSV HELPERS
# ============================================================

def safe_read_text(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except Exception:
            continue
    raise ValueError(f"Could not read file: {path}")


def ensure_output_csv_exists(output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if not output_csv.exists():
        with output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()


def normalize_phrase_list(value: Any) -> str:
    if isinstance(value, list):
        return "; ".join(str(x).strip() for x in value if str(x).strip())
    if isinstance(value, str):
        return value.strip()
    return ""


def append_rows_to_csv(output_csv: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    with output_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        for row in rows:
            writer.writerow({
                "Actionable/Findings": str(row.get("Actionable/Findings", "")).strip(),
                "Evidence": str(row.get("Evidence", "")).strip(),
                "Article Title": str(row.get("Article Title", "")).strip(),
                "Article Year": str(row.get("Article Year", "")).strip(),
                "Dictionary Group": str(row.get("Dictionary Group", "")).strip(),
                "Phrases from the dictionary": normalize_phrase_list(row.get("Phrases from the dictionary", [])),
            })


def load_completed_files(progress_log: Path) -> set[str]:
    if not progress_log.exists():
        return set()
    return {line.strip() for line in progress_log.read_text(encoding="utf-8").splitlines() if line.strip()}


def mark_file_completed(progress_log: Path, filename: str) -> None:
    progress_log.parent.mkdir(parents=True, exist_ok=True)
    with progress_log.open("a", encoding="utf-8") as f:
        f.write(filename + "\n")


def guess_title_and_year_from_markdown(md_text: str, fallback_stem: str) -> Tuple[str, str]:
    lines = [x.strip() for x in md_text.splitlines() if x.strip()]
    title_guess = fallback_stem
    year_guess = ""

    for line in lines[:25]:
        if line.startswith("#"):
            candidate = re.sub(r"^#+\s*", "", line).strip()
            if len(candidate) > 8:
                title_guess = candidate
                break

    year_matches = list(re.finditer(r"\b((?:19|20)\d{2})\b", md_text[:5000]))
    if year_matches:
        year_guess = year_matches[0].group(1)

    return title_guess, year_guess


# ============================================================
# JSON / VALIDATION
# ============================================================

def extract_json_array(text: str) -> List[Dict[str, Any]]:
    text = text.strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass

    match = re.search(r"(\[\s*.*\s*\])", text, flags=re.DOTALL)
    if match:
        candidate = match.group(1)
        parsed = json.loads(candidate)
        if isinstance(parsed, list):
            return parsed

    raise ValueError("Could not parse JSON array from model output.")


def validate_dictionary_group(group: str) -> bool:
    return group in DICTIONARY_GROUPS


def filter_and_validate_rows(rows: List[Dict[str, Any]], fallback_title: str, fallback_year: str) -> List[Dict[str, Any]]:
    valid_rows: List[Dict[str, Any]] = []
    seen = set()

    for row in rows:
        if not isinstance(row, dict):
            continue

        finding = str(row.get("Actionable/Findings", "")).strip()
        evidence = str(row.get("Evidence", "")).strip()
        title = str(row.get("Article Title", "")).strip() or fallback_title
        year = str(row.get("Article Year", "")).strip() or fallback_year
        group = str(row.get("Dictionary Group", "")).strip()
        phrases = row.get("Phrases from the dictionary", [])

        if not finding or not evidence or not group:
            continue
        if not validate_dictionary_group(group):
            continue

        if isinstance(phrases, str):
            phrases = [x.strip() for x in re.split(r"[;,]", phrases) if x.strip()]
        elif not isinstance(phrases, list):
            phrases = []

        allowed = set(DICTIONARY_GROUPS[group])
        phrases = [p for p in phrases if p in allowed]

        if not phrases:
            continue

        dedupe_key = (finding.lower(), title.lower(), year, group)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        valid_rows.append({
            "Actionable/Findings": finding,
            "Evidence": evidence,
            "Article Title": title,
            "Article Year": year,
            "Dictionary Group": group,
            "Phrases from the dictionary": phrases,
        })

    return valid_rows


# ============================================================
# OLLAMA CLI CALL
# ============================================================

def check_ollama_available() -> None:
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        version = (result.stdout or result.stderr).strip()
        print(f"[INFO] Ollama detected: {version}")
    except Exception as e:
        raise RuntimeError(
            "Could not run `ollama --version`. Make sure Ollama is installed and available in PATH."
        ) from e


def call_ollama_cli(
    prompt: str,
    model: str,
    timeout_sec: int,
    max_retries: int,
) -> List[Dict[str, Any]]:
    """
    Uses Ollama CLI directly:
        ollama run qwen2.5:72b

    Prompt is passed via stdin.
    """
    last_err = None

    for attempt in range(1, max_retries + 1):
        try:
            proc = subprocess.run(
                ["ollama", "run", model],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                check=False,
            )

            stdout = (proc.stdout or "").strip()
            stderr = (proc.stderr or "").strip()

            if proc.returncode != 0:
                raise RuntimeError(
                    f"Ollama exited with code {proc.returncode}. stderr: {stderr[:1000]}"
                )

            if not stdout:
                raise RuntimeError(f"Ollama returned empty stdout. stderr: {stderr[:1000]}")

            return extract_json_array(stdout)

        except Exception as e:
            last_err = e
            print(f"[WARN] Ollama attempt {attempt}/{max_retries} failed: {e}")
            time.sleep(2 * attempt)

    raise RuntimeError(f"Ollama extraction failed after {max_retries} attempts: {last_err}")


# ============================================================
# ARTICLE PROCESSING
# ============================================================

def process_single_markdown(
    md_path: Path,
    output_csv: Path,
    model: str,
    max_chars_per_article: int,
    timeout_sec: int,
    max_retries: int,
) -> int:
    text = safe_read_text(md_path).strip()
    if not text:
        print(f"[INFO] Empty markdown, skipping: {md_path.name}")
        return 0

    if len(text) > max_chars_per_article:
        print(f"[WARN] {md_path.name} is long ({len(text)} chars). Truncating to {max_chars_per_article} chars.")
        text = text[:max_chars_per_article]

    title_guess, year_guess = guess_title_and_year_from_markdown(text, md_path.stem)
    prompt = build_extraction_prompt(
        article_text=text,
        article_title_guess=title_guess,
        article_year_guess=year_guess,
    )

    raw_rows = call_ollama_cli(
        prompt=prompt,
        model=model,
        timeout_sec=timeout_sec,
        max_retries=max_retries,
    )

    valid_rows = filter_and_validate_rows(
        raw_rows,
        fallback_title=title_guess,
        fallback_year=year_guess,
    )

    append_rows_to_csv(output_csv, valid_rows)
    return len(valid_rows)


# ============================================================
# MAIN
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract structured findings from markdown files using Ollama CLI and append to a single CSV."
    )
    parser.add_argument("--md_dir", type=Path, default=DEFAULT_MD_DIR, help="Directory containing markdown files.")
    parser.add_argument("--output_csv", type=Path, default=DEFAULT_OUTPUT_CSV, help="Output CSV path.")
    parser.add_argument("--progress_log", type=Path, default=DEFAULT_PROGRESS_LOG, help="Progress log path.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Ollama model name.")
    parser.add_argument("--max_retries", type=int, default=DEFAULT_MAX_RETRIES, help="Max retries per file.")
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP_BETWEEN_FILES, help="Sleep between files.")
    parser.add_argument("--max_chars_per_article", type=int, default=DEFAULT_MAX_CHARS_PER_ARTICLE, help="Max characters per markdown file sent to the model.")
    parser.add_argument("--timeout_sec", type=int, default=DEFAULT_OLLAMA_TIMEOUT_SEC, help="Timeout for each ollama invocation.")
    parser.add_argument("--resume", action="store_true", help="Skip files listed in the progress log.")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.md_dir.exists():
        raise FileNotFoundError(f"Markdown directory not found: {args.md_dir}")

    md_files = sorted(args.md_dir.glob("*.md"))
    if not md_files:
        print(f"[INFO] No .md files found in: {args.md_dir}")
        return

    check_ollama_available()
    ensure_output_csv_exists(args.output_csv)

    completed = load_completed_files(args.progress_log) if args.resume else set()

    total_rows = 0
    processed_count = 0
    skipped_count = 0
    error_count = 0

    print(f"[INFO] Found {len(md_files)} markdown files")
    print(f"[INFO] Model: {args.model}")
    print(f"[INFO] Output CSV: {args.output_csv}")

    for idx, md_path in enumerate(md_files, start=1):
        if args.resume and md_path.name in completed:
            skipped_count += 1
            print(f"[SKIP] [{idx}/{len(md_files)}] {md_path.name}")
            continue

        print(f"[INFO] [{idx}/{len(md_files)}] Processing {md_path.name}")

        try:
            n_rows = process_single_markdown(
                md_path=md_path,
                output_csv=args.output_csv,
                model=args.model,
                max_chars_per_article=args.max_chars_per_article,
                timeout_sec=args.timeout_sec,
                max_retries=args.max_retries,
            )
            mark_file_completed(args.progress_log, md_path.name)
            total_rows += n_rows
            processed_count += 1
            print(f"[DONE] {md_path.name} -> {n_rows} row(s)")
        except Exception as e:
            error_count += 1
            print(f"[ERROR] {md_path.name}: {e}")

        time.sleep(args.sleep)

    print("\n===== SUMMARY =====")
    print(f"Processed files : {processed_count}")
    print(f"Skipped files   : {skipped_count}")
    print(f"Errored files   : {error_count}")
    print(f"Rows written    : {total_rows}")
    print(f"CSV             : {args.output_csv}")


if __name__ == "__main__":
    main()