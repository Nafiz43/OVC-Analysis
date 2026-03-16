#!/usr/bin/env python3
"""
Extract actionable recommendations tied to a fixed list of genes from PDF articles.

For each PDF:
- extract text
- remove references
- cap text to WORD_CAP
- send to LLM
- extract actionable recommendations tied to target genes

Outputs:
1) JSON per article
2) Markdown preprocessed article
3) Recommendation-level CSV (one row per recommendation)
4) Article-level summary CSV (one row per article)

Behavior:
- writes incrementally to avoid data loss if the script stops midway
- skips already processed PDFs if a valid JSON output already exists
"""

import csv
import json
import os
import re
from typing import Any, Dict, List

import pdfplumber
from dotenv import load_dotenv
load_dotenv()

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# =====================
# Configuration
# =====================

API_KEY = os.getenv("OPENAI_API_KEY", "")

MODEL_NAME = "gpt-4.1-mini"
TEMPERATURE = 0

INPUT_FOLDER = "/Users/nafiz43/Documents/GitHub/OVC-Analysis/code/papers/biomarkers"

OUTPUT_FOLDER = "actionable-results"
PREPROCESSED_FOLDER = "actionable-articles-preprocessed"

RECOMMENDATION_CSV = os.path.join(
    OUTPUT_FOLDER,
    "actionable_recommendations_by_article.csv"
)

ARTICLE_SUMMARY_CSV = os.path.join(
    OUTPUT_FOLDER,
    "article_actionable_recommendations_summary.csv"
)

WORD_CAP = 10000


GOOD_PROGNOSIS_GENES = [
    "FXR2", "EIF4A1", "HSP90B3P", "TUBBP6", "NACA2", "EIF5AL1", "NPIP", "RACGAP1P",
    "DYNC1I2", "KDM3B", "EEA1", "POTEE", "KIAA1751", "TTF1", "GBAP1", "MTX1", "ACPP",
    "RPL19P12", "PSMD14", "PPIG", "C17orf74", "CCDC112", "BBS5", "LEMD2", "SMURF1",
    "SPDYE3", "PCDP1", "CAPN5", "HIST2H2AC", "NUP214", "GPN3", "SS18", "WDR60",
    "C18orf21", "DVL2", "RNASEK", "TEX11", "LRRC37A3", "CDKN2AIPNL", "LSMD1", "WBP11P1",
    "ICT1", "TOR1B", "EPHB2", "PHKA2", "CLK2P", "CLPP", "CYB5D1", "MED31", "ITPR2",
    "ATP5L2", "CRYBG3", "MPDU1", "UBR3", "ORC2L", "LARS2", "GTF2F1", "ZNF511", "SSB",
    "RBM41", "GPR123", "CHCHD3", "GLRX3", "PELP1", "C17orf81"
]

TARGET_GENE_SET = set(GOOD_PROGNOSIS_GENES)

_REF_HEADING_RE = re.compile(r"^\s*(references?|reference)\s*:?\s*$", flags=re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")


# =====================
# Disk-safe write helpers
# =====================

def append_csv_row(csv_path: str, row: List[Any]) -> None:
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def write_text_file(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())


# =====================
# Directory setup
# =====================

def ensure_dirs() -> None:
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(PREPROCESSED_FOLDER, exist_ok=True)


def write_csv_headers_if_needed() -> None:
    if not os.path.exists(RECOMMENDATION_CSV):
        append_csv_row(
            RECOMMENDATION_CSV,
            ["Article Title", "Publication Year", "Actionable Recommendation", "Impacted Genes"]
        )

    if not os.path.exists(ARTICLE_SUMMARY_CSV):
        append_csv_row(
            ARTICLE_SUMMARY_CSV,
            ["Article Title", "Publication Year", "Recommendations Count", "All Impacted Genes", "JSON Path"]
        )


# =====================
# Checkpoint / skip logic
# =====================

def get_json_output_path(pdf_filename: str) -> str:
    base = os.path.splitext(pdf_filename)[0]
    return os.path.join(OUTPUT_FOLDER, base + ".json")


def get_md_output_path(pdf_filename: str) -> str:
    base = os.path.splitext(pdf_filename)[0]
    return os.path.join(PREPROCESSED_FOLDER, base + ".md")


def is_valid_existing_json(json_path: str) -> bool:
    """
    Treat an article as already processed only if:
    - JSON file exists
    - file is non-empty
    - file parses as JSON dict
    """
    if not os.path.exists(json_path):
        return False

    try:
        if os.path.getsize(json_path) == 0:
            return False

        with open(json_path, "r", encoding="utf-8") as f:
            parsed = json.load(f)

        return isinstance(parsed, dict)
    except Exception:
        return False


def already_processed(pdf_filename: str) -> bool:
    json_path = get_json_output_path(pdf_filename)
    return is_valid_existing_json(json_path)


# =====================
# PDF processing
# =====================

def pdf_to_text(pdf_path: str) -> str:
    chunks: List[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text() or ""
            if txt.strip():
                chunks.append(txt)
    return "\n\n".join(chunks).strip()


def truncate_at_references(text: str) -> str:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if _REF_HEADING_RE.match(line):
            return "\n".join(lines[:i])
    return text


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def cap_words(text: str, cap: int) -> str:
    tokens = re.findall(r"\S+", text)
    if len(tokens) <= cap:
        return text
    return " ".join(tokens[:cap])


# =====================
# Metadata extraction
# =====================

def get_pdf_metadata(pdf_path: str) -> Dict[str, Any]:
    try:
        with pdfplumber.open(pdf_path) as pdf:
            return pdf.metadata or {}
    except Exception:
        return {}


def get_article_title(pdf_path: str, fallback_text: str = "") -> str:
    meta = get_pdf_metadata(pdf_path)
    title = str(meta.get("Title") or "").strip()
    if title:
        return title

    base = os.path.splitext(os.path.basename(pdf_path))[0]
    if base:
        return base

    for line in fallback_text.splitlines():
        if line.strip():
            return line[:200]

    return "Untitled Article"


def get_publication_year(pdf_path: str, fallback_text: str = "") -> str:
    meta = get_pdf_metadata(pdf_path)

    for value in meta.values():
        if value:
            m = _YEAR_RE.search(str(value))
            if m:
                return m.group(0)

    head = "\n".join(fallback_text.splitlines()[:80])
    m = _YEAR_RE.search(head)
    if m:
        return m.group(0)

    return ""


# =====================
# JSON parsing helpers
# =====================

def clean_and_parse_json(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw

    if not isinstance(raw, str):
        return {}

    text = raw.strip()

    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]
            if text.startswith("json"):
                text = text[4:].strip()

    try:
        return json.loads(text)
    except Exception:
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(text[start:end + 1])
        except Exception:
            pass
    return {}


def stringify(obj: Any) -> str:
    try:
        return json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception:
        return str(obj)


# =====================
# Prompt
# =====================

def make_prompt(article_text: str) -> str:
    genes = ", ".join(GOOD_PROGNOSIS_GENES)

    return f"""
You are an expert biomedical literature analyst.

Extract actionable recommendations tied to the following gene list:

{genes}

An actionable recommendation means a concrete suggested action
(e.g., therapeutic targeting, biomarker use, monitoring strategy,
experimental manipulation, or clinical decision) involving one or
more genes from the list.

Return ONLY JSON:

{{
 "article_title":"",
 "publication_year":"",
 "recommendations":[
   {{
     "actionable_recommendation":"",
     "impacted_genes":["GENE1","GENE2"],
     "evidence_excerpt":""
   }}
 ],
 "notes":""
}}

Rules:
- impacted_genes must come from the provided gene list
- ignore descriptive findings
- do not hallucinate
- JSON only

Article:
{article_text}
"""


# =====================
# LLM call
# =====================

def call_model(article_text: str) -> str:
    if OpenAI is None:
        raise RuntimeError("Install openai package")
    if not API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    client = OpenAI(api_key=API_KEY)

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You extract biomedical recommendations."},
            {"role": "user", "content": make_prompt(article_text)}
        ]
    )

    return resp.choices[0].message.content


# =====================
# Main pipeline
# =====================

def main() -> None:
    ensure_dirs()
    write_csv_headers_if_needed()

    if not os.path.isdir(INPUT_FOLDER):
        raise FileNotFoundError(f"Input folder does not exist: {INPUT_FOLDER}")

    pdf_files = sorted(
        f for f in os.listdir(INPUT_FOLDER)
        if f.lower().endswith(".pdf")
    )

    if not pdf_files:
        print("No PDF files found.")
        return

    for fname in pdf_files:
        try:
            if already_processed(fname):
                print(f"\nSkipping already processed file: {fname}")
                continue

            print(f"\nProcessing: {fname}")

            path = os.path.join(INPUT_FOLDER, fname)

            text = pdf_to_text(path)
            text = truncate_at_references(text)
            capped = cap_words(text, WORD_CAP)

            title = get_article_title(path, text)
            year = get_publication_year(path, text)

            md_path = get_md_output_path(fname)
            write_text_file(md_path, f"# {title}\n\n{capped}")

            raw = call_model(capped)
            parsed = clean_and_parse_json(raw)

            json_path = get_json_output_path(fname)
            write_text_file(json_path, stringify(parsed))

            recs = parsed.get("recommendations", [])
            if not isinstance(recs, list):
                recs = []

            article_genes: List[str] = []
            valid_recommendation_count = 0

            for rec in recs:
                if not isinstance(rec, dict):
                    continue

                rec_text = str(rec.get("actionable_recommendation", "")).strip()

                impacted = rec.get("impacted_genes", [])
                if not isinstance(impacted, list):
                    impacted = []

                genes = [str(g).strip() for g in impacted if str(g).strip() in TARGET_GENE_SET]

                if not rec_text or not genes:
                    continue

                valid_recommendation_count += 1
                article_genes.extend(genes)

                append_csv_row(
                    RECOMMENDATION_CSV,
                    [title, year, rec_text, ", ".join(genes)]
                )

            article_genes = sorted(set(article_genes))

            append_csv_row(
                ARTICLE_SUMMARY_CSV,
                [title, year, valid_recommendation_count, ", ".join(article_genes), json_path]
            )

            print("Saved results")

        except Exception as e:
            print(f"Error processing {fname}: {e}")


if __name__ == "__main__":
    main()