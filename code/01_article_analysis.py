#!/usr/bin/env python3
"""
extract_actionables_from_markdowns_ollama_cli.py

Process markdown (.md) files, extract structured findings/actionables using
Ollama CLI (NOT the Ollama server API), and append results row-by-row into
a single CSV.

LLM engine:
    ollama run qwen3:32b

Outputs:
1. Main CSV with extracted rows
2. Log file for articles with NO findings
3. Run summary stats file

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
from collections import Counter, defaultdict

from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

DEFAULT_MD_DIR = Path("/data/Deep_Angiography/OVC-Analysis/code/data/actionable-articles-preprocessed")
DEFAULT_OUTPUT_DIR = Path("/data/Deep_Angiography/OVC-Analysis/code/data/results")
DEFAULT_OUTPUT_CSV = DEFAULT_OUTPUT_DIR / "actionable_findings_extracted.csv"
DEFAULT_PROGRESS_LOG = DEFAULT_OUTPUT_DIR / "actionable_findings_progress.log"
DEFAULT_NO_FINDINGS_LOG = DEFAULT_OUTPUT_DIR / "articles_with_no_actionables.csv"
DEFAULT_STATS_FILE = DEFAULT_OUTPUT_DIR / "actionable_extraction_stats.txt"

DEFAULT_MODEL = "qwen3:32b"
DEFAULT_MAX_RETRIES = 3
DEFAULT_SLEEP_BETWEEN_FILES = 0.5
DEFAULT_MAX_CHARS_PER_ARTICLE = 120000
DEFAULT_OLLAMA_TIMEOUT_SEC = 1800  # 30 minutes

CSV_COLUMNS = [
    "Actionable/Findings",
    "Evidence",
    "Article Title",
    "Article Year",
    "Dictionary Group",
    "Phrases from the dictionary",
]

NO_FINDINGS_COLUMNS = [
    "Filename",
    "Article Title Guess",
    "Article Year Guess",
    "Reason",
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
# PROMPTS
# ============================================================

def build_extraction_prompt(article_text: str, article_title_guess: str, article_year_guess: str) -> str:
    dictionary_text = "\n".join(
        f"{group}:\n  - " + "\n  - ".join(terms)
        for group, terms in DICTIONARY_GROUPS.items()
    )

    return f"""
You are an expert biomedical literature extraction assistant specializing in ovarian cancer clinical and translational research.

Your task: read a markdown-formatted scientific article and extract every clinically or biologically meaningful finding as structured JSON rows.

═══════════════════════════════════════════════
OUTPUT FORMAT — MANDATORY
═══════════════════════════════════════════════

Return ONLY a valid JSON array. No markdown fences. No preamble. No postamble.

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

If the article contains no relevant findings, return exactly: []

═══════════════════════════════════════════════
EXTRACTION SCOPE — WHERE TO LOOK
═══════════════════════════════════════════════

Priority order for finding evidence:
1. Results section — quantitative outcomes, statistical comparisons, survival curves
2. Conclusions / Discussion conclusions — interpretive findings backed by data
3. Abstract results — only if the full Results section is absent or truncated
4. Methods — ONLY if a novel methodological design directly yields a finding (rare)

IGNORE:
- Introduction / Background statements that describe prior literature, not this study's data
- Generic methodological descriptions with no tied result
- Limitations sections unless a limitation directly invalidates a finding
- Boilerplate clinical trial registration or ethics statements

═══════════════════════════════════════════════
FIELD DEFINITIONS
═══════════════════════════════════════════════

1. Actionable/Findings
   - ONE specific, self-contained, clinically or biologically meaningful finding per row.
   - Must include: the entity (gene, biomarker, drug, pathway, subgroup), the direction of effect,
     and the clinical/biological context.
   - Good: "High BRCA2 mutation rate was associated with improved PFS in platinum-sensitive patients (HR 0.54, p=0.003)"
   - Bad: "BRCA mutations were discussed in relation to treatment"
   - If a finding is relevant to more than one Dictionary Group, emit one row per group
     (same finding text, different Dictionary Group and Phrases from the dictionary).

2. Evidence  <- THIS FIELD IS THE MOST IMPORTANT
   Format exactly:
     "Exact statement(s): <verbatim quote(s) from the article, minimum 20 words each> | Inference: <1-2 sentences explaining how the quote supports the finding>"

   Rules:
   - The verbatim quote MUST be copied character-for-character from the article text.
   - Include numbers, p-values, confidence intervals, and effect sizes whenever present.
   - If multiple non-contiguous sentences jointly support the finding, join them with " [...] ".
   - The Inference must explain causality or clinical significance, not just paraphrase.

3. Article Title
   - Use the actual article title if present in the markdown.
   - Otherwise use: "{article_title_guess}"

4. Article Year
   - Use the actual year if present in the markdown.
   - Otherwise use: "{article_year_guess}"
   - If unknown, use an empty string.

5. Dictionary Group
   - Must be EXACTLY one of:
     * "Dictionary A: Chemotherapy & Resistance Findings"
     * "Dictionary B: Survival & Clinical Outcome Findings"
     * "Dictionary C: Molecular Subtypes & Therapy Findings"
   - Choose the group whose dictionary terms appear most directly in the evidence.

6. Phrases from the dictionary
   - JSON array of exact phrases copied verbatim from the dictionary lists below.
   - Include ONLY phrases that appear in or are directly implied by the Evidence quote.
   - At least one phrase is required; omit the row entirely if none apply.

═══════════════════════════════════════════════
QUALITY RULES
═══════════════════════════════════════════════

DO
  - Emit separate rows for statistically distinct subgroup findings (e.g., platinum-sensitive vs. platinum-resistant)
  - Emit separate rows for each distinct biomarker even if reported in the same table
  - Include hazard ratios, ORs, p-values, confidence intervals verbatim in Evidence
  - Prefer multivariate/independent findings over univariate ones where both are reported

DO NOT
  - Hallucinate gene names, drug names, or statistics not present in the article
  - Emit duplicate rows (same finding + same article + same group)
  - Emit near-duplicate rows that only differ in minor wording -- pick the most precise one
  - Emit rows where Phrases from the dictionary is an empty array
  - Attribute findings from cited prior studies to this article

═══════════════════════════════════════════════
DICTIONARY GROUPS
═══════════════════════════════════════════════

{dictionary_text}

═══════════════════════════════════════════════
ARTICLE MARKDOWN
═══════════════════════════════════════════════

{article_text}
""".strip()


def build_json_repair_prompt(raw_response: str) -> str:
    return f"""
You are a strict JSON formatter.

Your task is to convert the following model output into a valid JSON array.

Rules:
- Return ONLY valid JSON.
- Do NOT include markdown fences.
- Do NOT include explanation.
- Do NOT add new findings that are not already present in the content.
- Preserve the original meaning as much as possible.
- If the content clearly contains no findings, return [].
- The output must be a JSON array of objects with exactly these keys:

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

Here is the raw model output to repair:

{raw_response}
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


def ensure_no_findings_csv_exists(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=NO_FINDINGS_COLUMNS)
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


def append_no_findings_row(path: Path, filename: str, title_guess: str, year_guess: str, reason: str) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=NO_FINDINGS_COLUMNS)
        writer.writerow({
            "Filename": filename,
            "Article Title Guess": title_guess,
            "Article Year Guess": year_guess,
            "Reason": reason,
        })


def load_completed_files(progress_log: Path) -> set[str]:
    """
    Read the progress log and return a SET of already-processed filenames.
    Duplicates in the log are collapsed automatically by the set.
    Lines that are empty or whitespace-only are ignored.
    """
    if not progress_log.exists():
        return set()
    completed: set[str] = set()
    for line in progress_log.read_text(encoding="utf-8").splitlines():
        name = line.strip()
        if name:
            completed.add(name)
    return completed


def compute_pending_files(
    md_files: List[Path],
    progress_log: Path,
) -> Tuple[List[Path], set[str], int, int]:
    """
    SetA = filenames already recorded in the progress log (de-duplicated).
    SetB = filenames that actually exist on disk right now.

    pending  = SetB - SetA   (files on disk NOT yet processed)
    done     = SetA & SetB   (files on disk that ARE already done)
    log_only = SetA - SetB   (entries in log whose files no longer exist on disk)

    Returns:
        pending_paths  : sorted list of Path objects still to process
        set_a          : the de-duplicated completed set (used for live tracking)
        n_done         : len(done)
        n_log_only     : len(log_only) — informational
    """
    set_a: set[str] = load_completed_files(progress_log)
    set_b: set[str] = {p.name for p in md_files}

    done     = set_a & set_b
    pending  = set_b - set_a
    log_only = set_a - set_b

    pending_paths = sorted(
        [p for p in md_files if p.name in pending],
        key=lambda p: p.name,
    )

    return pending_paths, set_a, len(done), len(log_only)


def mark_file_completed(progress_log: Path, filename: str, completed_set: set[str]) -> None:
    """Append filename to the log on disk AND update the in-memory set."""
    progress_log.parent.mkdir(parents=True, exist_ok=True)
    with progress_log.open("a", encoding="utf-8") as f:
        f.write(filename + "\n")
    completed_set.add(filename)


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


def filter_and_validate_rows(
    rows: List[Dict[str, Any]],
    fallback_title: str,
    fallback_year: str,
) -> Tuple[List[Dict[str, Any]], str]:
    valid_rows: List[Dict[str, Any]] = []
    seen = set()

    if not rows:
        return [], "returned_empty_array"

    for row in rows:
        if not isinstance(row, dict):
            continue

        finding = str(row.get("Actionable/Findings", "")).strip()
        evidence = str(row.get("Evidence", "")).strip()
        title    = str(row.get("Article Title", "")).strip() or fallback_title
        year     = str(row.get("Article Year", "")).strip() or fallback_year
        group    = str(row.get("Dictionary Group", "")).strip()
        phrases  = row.get("Phrases from the dictionary", [])

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

    if not valid_rows:
        return [], "all_rows_failed_validation"

    return valid_rows, "valid_rows_found"


# ============================================================
# OLLAMA CLI CALLS
# ============================================================

def check_ollama_available() -> None:
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True, text=True, check=True, timeout=30,
        )
        version = (result.stdout or result.stderr).strip()
        print(f"[INFO] Ollama detected: {version}")
    except Exception as e:
        raise RuntimeError(
            "Could not run `ollama --version`. Make sure Ollama is installed and in PATH."
        ) from e


def run_ollama_raw(prompt: str, model: str, timeout_sec: int) -> str:
    proc = subprocess.run(
        ["ollama", "run", model],
        input=prompt,
        capture_output=True, text=True,
        timeout=timeout_sec, check=False,
    )
    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()

    if proc.returncode != 0:
        raise RuntimeError(f"Ollama exited with code {proc.returncode}. stderr: {stderr[:1000]}")
    if not stdout:
        raise RuntimeError(f"Ollama returned empty stdout. stderr: {stderr[:1000]}")
    return stdout


def parse_with_json_repair(
    raw_output: str,
    model: str,
    timeout_sec: int,
) -> Tuple[List[Dict[str, Any]], str]:
    try:
        return extract_json_array(raw_output), "direct"
    except Exception:
        repair_prompt = build_json_repair_prompt(raw_output)
        repaired = run_ollama_raw(repair_prompt, model, timeout_sec)
        return extract_json_array(repaired), "repaired"


def call_ollama_cli(
    prompt: str,
    model: str,
    timeout_sec: int,
    max_retries: int,
) -> Tuple[List[Dict[str, Any]], str]:
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            raw_output = run_ollama_raw(prompt, model, timeout_sec)
            return parse_with_json_repair(raw_output, model, timeout_sec)
        except Exception as e:
            last_err = e
            tqdm.write(f"[WARN] Ollama attempt {attempt}/{max_retries} failed: {e}")
            time.sleep(2 * attempt)
    raise RuntimeError(f"Ollama extraction failed after {max_retries} attempts: {last_err}")


# ============================================================
# ARTICLE PROCESSING
# ============================================================

def process_single_markdown(
    md_path: Path,
    output_csv: Path,
    no_findings_log: Path,
    model: str,
    max_chars_per_article: int,
    timeout_sec: int,
    max_retries: int,
) -> Dict[str, Any]:
    text = safe_read_text(md_path).strip()
    if not text:
        title_guess, year_guess = md_path.stem, ""
        append_no_findings_row(no_findings_log, md_path.name, title_guess, year_guess, "Empty markdown file")
        return {
            "rows_written": 0, "title_guess": title_guess, "year_guess": year_guess,
            "status": "no_findings", "reason": "Empty markdown file",
            "rows": [], "parse_mode": "not_applicable", "validation_status": "empty_markdown",
        }

    if len(text) > max_chars_per_article:
        tqdm.write(f"[WARN] {md_path.name} is long ({len(text)} chars). Truncating to {max_chars_per_article}.")
        text = text[:max_chars_per_article]

    title_guess, year_guess = guess_title_and_year_from_markdown(text, md_path.stem)
    prompt = build_extraction_prompt(text, title_guess, year_guess)
    raw_rows, parse_mode = call_ollama_cli(prompt, model, timeout_sec, max_retries)
    valid_rows, validation_status = filter_and_validate_rows(raw_rows, title_guess, year_guess)

    if not valid_rows:
        reason_map = {
            "returned_empty_array":      "Model returned empty JSON array",
            "all_rows_failed_validation":"Model returned rows, but all failed validation",
        }
        reason = reason_map.get(validation_status, "Model returned no valid actionable findings")
        append_no_findings_row(no_findings_log, md_path.name, title_guess, year_guess, reason)
        return {
            "rows_written": 0, "title_guess": title_guess, "year_guess": year_guess,
            "status": "no_findings", "reason": reason,
            "rows": [], "parse_mode": parse_mode, "validation_status": validation_status,
        }

    append_rows_to_csv(output_csv, valid_rows)
    return {
        "rows_written": len(valid_rows), "title_guess": title_guess, "year_guess": year_guess,
        "status": "with_findings", "reason": "",
        "rows": valid_rows, "parse_mode": parse_mode, "validation_status": validation_status,
    }


# ============================================================
# STATS
# ============================================================

def compute_stats(all_results: List[Dict[str, Any]]) -> str:
    processed_articles = [r for r in all_results if r["status"] != "error"]
    errored_articles   = [r for r in all_results if r["status"] == "error"]
    with_findings      = [r for r in all_results if r["status"] == "with_findings"]
    without_findings   = [r for r in all_results if r["status"] == "no_findings"]

    total_actionables = sum(r.get("rows_written", 0) for r in with_findings)

    group_counter            = Counter()
    article_year_counter     = Counter()
    phrases_counter          = Counter()
    actionables_per_article  = []
    articles_per_group       = defaultdict(set)
    parse_mode_counter       = Counter()
    no_findings_reason_counter = Counter()
    validation_status_counter  = Counter()

    for result in all_results:
        pm = result.get("parse_mode", "")
        vs = result.get("validation_status", "")
        rn = result.get("reason", "")
        if pm: parse_mode_counter[pm] += 1
        if vs: validation_status_counter[vs] += 1
        if result.get("status") == "no_findings" and rn:
            no_findings_reason_counter[rn] += 1

    for result in with_findings:
        rows = result.get("rows", [])
        actionables_per_article.append(len(rows))
        for row in rows:
            group  = row.get("Dictionary Group", "").strip()
            year   = row.get("Article Year", "").strip()
            title  = row.get("Article Title", "").strip()
            phrases = row.get("Phrases from the dictionary", [])
            if group:
                group_counter[group] += 1
                if title: articles_per_group[group].add(title)
            if year: article_year_counter[year] += 1
            if isinstance(phrases, list):
                for p in phrases:
                    if str(p).strip(): phrases_counter[str(p).strip()] += 1

    avg_act = (sum(actionables_per_article) / len(actionables_per_article)) if actionables_per_article else 0.0
    max_act = max(actionables_per_article) if actionables_per_article else 0
    min_act = min(actionables_per_article) if actionables_per_article else 0

    lines = ["===== ACTIONABLE EXTRACTION SUMMARY =====", ""]
    lines += [
        f"Total articles processed successfully : {len(processed_articles)}",
        f"Total articles with actionables        : {len(with_findings)}",
        f"Total articles without actionables     : {len(without_findings)}",
        f"Total errored articles                 : {len(errored_articles)}",
        f"Total actionables extracted            : {total_actionables}",
        "",
    ]
    if processed_articles:
        lines += [
            f"Percent with actionables    : {100.0*len(with_findings)/len(processed_articles):.2f}%",
            f"Percent without actionables : {100.0*len(without_findings)/len(processed_articles):.2f}%",
            "",
        ]

    lines.append("----- JSON parsing mode counts -----")
    for mode in ["direct", "repaired", "not_applicable"]:
        if mode in parse_mode_counter:
            lines.append(f"{mode}: {parse_mode_counter[mode]}")
    lines.append("")

    lines.append("----- Validation status counts -----")
    for k, v in validation_status_counter.items(): lines.append(f"{k}: {v}")
    lines.append("")

    lines.append("----- No-findings reason counts -----")
    for k, v in no_findings_reason_counter.items(): lines.append(f"{k}: {v}")
    lines.append("")

    lines.append("----- Actionable count per dictionary group -----")
    for g in DICTIONARY_GROUPS: lines.append(f"{g}: {group_counter.get(g, 0)}")
    lines.append("")

    lines.append("----- Unique articles per dictionary group -----")
    for g in DICTIONARY_GROUPS: lines.append(f"{g}: {len(articles_per_group.get(g, set()))}")
    lines.append("")

    lines += [
        "----- Actionables per article -----",
        f"Average : {avg_act:.2f}",
        f"Min     : {min_act}",
        f"Max     : {max_act}",
        "",
    ]

    if article_year_counter:
        lines.append("----- Actionables by article year -----")
        for yr in sorted(article_year_counter): lines.append(f"{yr}: {article_year_counter[yr]}")
        lines.append("")

    if phrases_counter:
        lines.append("----- Most frequently matched dictionary phrases -----")
        for phrase, count in phrases_counter.most_common(20):
            lines.append(f"{phrase}: {count}")
        lines.append("")

    if with_findings:
        lines.append("----- Top articles by extracted actionables -----")
        for item in sorted(with_findings, key=lambda x: x.get("rows_written", 0), reverse=True)[:15]:
            lines.append(f"{item.get('title_guess','')} ({item.get('year_guess','')}) -> {item.get('rows_written',0)}")
        lines.append("")

    if without_findings:
        lines.append("----- Articles without actionables -----")
        for item in without_findings[:50]:
            lines.append(f"{item.get('title_guess','')} ({item.get('year_guess','')}) | {item.get('reason','')}")
        if len(without_findings) > 50:
            lines.append(f"... and {len(without_findings) - 50} more")
        lines.append("")

    if errored_articles:
        lines.append("----- Errored articles -----")
        for item in errored_articles:
            lines.append(f"{item.get('filename','')} | {item.get('reason','')}")
        lines.append("")

    return "\n".join(lines)


def write_stats_file(stats_file: Path, text: str) -> None:
    stats_file.parent.mkdir(parents=True, exist_ok=True)
    stats_file.write_text(text, encoding="utf-8")


# ============================================================
# MAIN
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract structured findings from markdown files using Ollama CLI."
    )
    parser.add_argument("--md_dir",               type=Path,  default=DEFAULT_MD_DIR)
    parser.add_argument("--output_csv",            type=Path,  default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--progress_log",          type=Path,  default=DEFAULT_PROGRESS_LOG)
    parser.add_argument("--no_findings_log",       type=Path,  default=DEFAULT_NO_FINDINGS_LOG)
    parser.add_argument("--stats_file",            type=Path,  default=DEFAULT_STATS_FILE)
    parser.add_argument("--model",                 type=str,   default=DEFAULT_MODEL)
    parser.add_argument("--max_retries",           type=int,   default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--sleep",                 type=float, default=DEFAULT_SLEEP_BETWEEN_FILES)
    parser.add_argument("--max_chars_per_article", type=int,   default=DEFAULT_MAX_CHARS_PER_ARTICLE)
    parser.add_argument("--timeout_sec",           type=int,   default=DEFAULT_OLLAMA_TIMEOUT_SEC)
    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="Ignore the progress log and reprocess ALL files from scratch.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.md_dir.exists():
        raise FileNotFoundError(f"Markdown directory not found: {args.md_dir}")

    all_md_files: List[Path] = sorted(args.md_dir.glob("*.md"))
    if not all_md_files:
        print(f"[INFO] No .md files found in: {args.md_dir}")
        return

    check_ollama_available()
    ensure_output_csv_exists(args.output_csv)
    ensure_no_findings_csv_exists(args.no_findings_log)

    # ── Set-intersection resume logic ─────────────────────────────────────────
    set_b: set[str] = {p.name for p in all_md_files}

    if args.reprocess:
        pending_files:  List[Path] = list(all_md_files)
        completed_set:  set[str]   = set()
        print(f"[INFO] --reprocess: ignoring progress log, queuing all {len(pending_files)} files.")
    else:
        pending_files, completed_set, n_done, n_log_only = compute_pending_files(
            all_md_files, args.progress_log
        )
        print(f"[INFO] SetA (progress log, de-duplicated) : {len(completed_set)} unique entries")
        print(f"[INFO] SetB (markdown files on disk)      : {len(set_b)}")
        print(f"[INFO] Already done  (SetA ∩ SetB)        : {n_done}")
        print(f"[INFO] Pending       (SetB − SetA)        : {len(pending_files)}")
        if n_log_only:
            print(f"[INFO] Log-only ghosts (SetA − SetB)      : {n_log_only}  (in log but no longer on disk — skipped)")
        if not pending_files:
            print("[INFO] SetA already covers all of SetB — nothing left to process. Exiting.")
            return
        print(f"[INFO] (Pass --reprocess to ignore the log and redo everything.)")
    # ──────────────────────────────────────────────────────────────────────────

    print(f"[INFO] Model      : {args.model}")
    print(f"[INFO] Output CSV : {args.output_csv}")
    print(f"[INFO] No-findings: {args.no_findings_log}")
    print(f"[INFO] Stats file : {args.stats_file}")

    total_rows      = 0
    processed_count = 0
    error_count     = 0
    all_results: List[Dict[str, Any]] = []

    pbar = tqdm(
        pending_files,
        total=len(pending_files),
        desc="Extracting",
        unit="article",
        dynamic_ncols=True,
        colour="cyan",
    )

    for md_path in pbar:
        pbar.set_description(f"Extracting · {md_path.stem[:40]}")

        try:
            result = process_single_markdown(
                md_path=md_path,
                output_csv=args.output_csv,
                no_findings_log=args.no_findings_log,
                model=args.model,
                max_chars_per_article=args.max_chars_per_article,
                timeout_sec=args.timeout_sec,
                max_retries=args.max_retries,
            )
            result["filename"] = md_path.name
            all_results.append(result)

            # Update both disk log and in-memory set atomically
            mark_file_completed(args.progress_log, md_path.name, completed_set)
            total_rows      += result.get("rows_written", 0)
            processed_count += 1

            if result["status"] == "with_findings":
                tqdm.write(f"[DONE]        {md_path.name} -> {result['rows_written']} row(s) | parse_mode={result.get('parse_mode','?')}")
            else:
                tqdm.write(f"[NO_FINDINGS] {md_path.name} -> {result.get('reason','')} | parse_mode={result.get('parse_mode','?')}")

        except Exception as e:
            error_count += 1
            all_results.append({
                "filename": md_path.name, "rows_written": 0,
                "title_guess": md_path.stem, "year_guess": "",
                "status": "error", "reason": str(e),
                "rows": [], "parse_mode": "error", "validation_status": "error",
            })
            tqdm.write(f"[ERROR] {md_path.name}: {e}")

        pbar.set_postfix(rows=total_rows, errors=error_count, refresh=True)

        # Early-exit: stop as soon as SetA fully covers SetB
        if completed_set >= set_b:
            tqdm.write("[INFO] SetA now equals SetB — all files on disk are processed. Stopping.")
            break

        time.sleep(args.sleep)

    pbar.close()

    stats_text = compute_stats(all_results)
    write_stats_file(args.stats_file, stats_text)

    print("\n===== SUMMARY =====")
    print(f"Processed this run : {processed_count}")
    print(f"Errored this run   : {error_count}")
    print(f"Rows written       : {total_rows}")
    print(f"CSV                : {args.output_csv}")
    print(f"No-findings log    : {args.no_findings_log}")
    print(f"Stats file         : {args.stats_file}")
    print("")
    print(stats_text)


if __name__ == "__main__":
    main()