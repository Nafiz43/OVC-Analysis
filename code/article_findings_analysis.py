"""Extract article findings related to a fixed list of good-prognosis genes.

For each PDF article in INPUT_FOLDER, this script asks an LLM whether the article
contains findings related to any target genes. It writes:
- one JSON response file per article
- one aggregate CSV summary row per article
- one preprocessed markdown dump per article
"""

import csv
import json
import os
import re
from typing import Any, Dict, List, Union

import pdfplumber

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME = "gpt-5-nano"
TEMPERATURE = 0

INPUT_FOLDER = "biomarkers"
OUTPUT_FOLDER = "findings-results"
PREPROCESSED_FOLDER = "findings-articles-preprocessed"
CSV_FILE = os.path.join(OUTPUT_FOLDER, "article_findings_summary.csv")

WORD_CAP = 10_000

GOOD_PROGNOSIS_GENES = [
    "FXR2", "EIF4A1", "HSP90B3P", "TUBBP6", "NACA2", "EIF5AL1", "NPIP", "RACGAP1P",
    "DYNC1I2", "KDM3B", "EEA1", "POTEE", "KIAA1751", "TTF1", "GBAP1", "MTX1", "ACPP",
    "RPL19P12", "PSMD14", "PPIG", "C17orf74", "CCDC112", "BBS5", "LEMD2", "SMURF1",
    "SPDYE3", "PCDP1", "CAPN5", "HIST2H2AC", "NUP214", "GPN3", "SS18", "WDR60",
    "C18orf21", "DVL2", "RNASEK", "TEX11", "LRRC37A3", "CDKN2AIPNL", "LSMD1", "WBP11P1",
    "ICT1", "TOR1B", "EPHB2", "PHKA2", "CLK2P", "CLPP", "CYB5D1", "MED31", "ITPR2",
    "ATP5L2", "CRYBG3", "MPDU1", "UBR3", "ORC2L", "LARS2", "GTF2F1", "ZNF511", "SSB",
    "RBM41", "GPR123", "CHCHD3", "GLRX3", "PELP1", "C17orf81",
]

_REF_HEADING_RE = re.compile(r"^\s*(references?|reference)\s*:?\s*$", flags=re.IGNORECASE)


def ensure_dirs() -> None:
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(PREPROCESSED_FOLDER, exist_ok=True)


def pdf_to_text(pdf_path: str) -> str:
    chunks: List[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text() or ""
            if txt.strip():
                chunks.append(txt)
    return "\n\n".join(chunks).strip()


def get_article_title(pdf_path: str, fallback_text: str = "") -> str:
    try:
        with pdfplumber.open(pdf_path) as pdf:
            meta = pdf.metadata or {}
            title = (meta.get("Title") or meta.get("title") or "").strip()
            if title:
                return title
    except Exception:
        pass

    base = os.path.splitext(os.path.basename(pdf_path))[0]
    if base:
        return base

    for line in (fallback_text or "").splitlines():
        line = line.strip()
        if line:
            return line[:120]
    return "Untitled Article"


def truncate_at_references(text: str) -> str:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if _REF_HEADING_RE.match(line):
            return "\n".join(lines[:i]).rstrip()
    return text


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def cap_words(text: str, cap: int) -> str:
    tokens = re.findall(r"\S+", text)
    if len(tokens) <= cap:
        return text
    return " ".join(tokens[:cap])


def stringify(obj: Any) -> str:
    if isinstance(obj, str):
        return obj
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        return str(obj)


def clean_and_parse_json(raw: Union[str, Dict[str, Any], List[Any]]) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        try:
            return json.loads(json.dumps(raw))
        except Exception:
            return {}

    text = raw.strip()
    if text.startswith("```"):
        if "```json" in text:
            text = text.split("```json", 1)[-1]
        else:
            text = text.split("```", 1)[-1]
        text = text.split("```", 1)[0].strip()

    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except Exception:
                return {}
    return {}


def make_prompt(article_text: str) -> str:
    genes_text = ", ".join(GOOD_PROGNOSIS_GENES)
    return f"""You are an expert biomedical literature analyst.

Task:
Given an ovarian cancer article, identify findings related to this exact target gene list:
{genes_text}

Return ONLY valid JSON with this schema:
{{
  "has_any_gene_findings": true/false,
  "genes_with_findings": [
    {{
      "gene": "<one gene from target list>",
      "finding_summary": "short summary of what article reports about this gene",
      "evidence_excerpt": "brief verbatim phrase or sentence from article if available"
    }}
  ],
  "notes": "optional short note; empty string if none"
}}

Rules:
- Only include genes from the target list.
- If a target gene is absent or has no specific finding, do not include it.
- Do not hallucinate; use only information present in article text.
- If no findings for any target gene, return has_any_gene_findings=false and an empty genes_with_findings array.
- No markdown, no prose, JSON only.

### Article Content (verbatim):
{article_text}
"""


def call_model(article_text: str) -> str:
    if OpenAI is None:
        raise RuntimeError("OpenAI SDK not installed. Please `pip install openai`.")
    if not API_KEY:
        raise RuntimeError("API key is not set. Set OPENAI_API_KEY in your environment.")

    client = OpenAI(api_key=API_KEY)
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": "You are a careful biomedical findings extractor."},
            {"role": "user", "content": make_prompt(article_text)},
        ],
    )
    return resp.choices[0].message.content


def main() -> None:
    ensure_dirs()

    write_header = not os.path.exists(CSV_FILE)
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow([
                "File Name",
                "Article Name",
                "Word Count (no-refs)",
                "Word Count (processed_cap)",
                "Has Any Gene Findings",
                "Genes Found",
                "Findings Count",
                "Raw JSON Path",
            ])

    pdf_files = [f for f in sorted(os.listdir(INPUT_FOLDER)) if f.lower().endswith(".pdf")]
    for fname in pdf_files:
        pdf_path = os.path.join(INPUT_FOLDER, fname)
        try:
            print(f"\n=== Processing: {fname} ===")
            text_full = pdf_to_text(pdf_path)
            text_no_refs = truncate_at_references(text_full)
            wc_no_refs = word_count(text_no_refs)

            text_capped = cap_words(text_no_refs, WORD_CAP)
            wc_processed = word_count(text_capped)
            title = get_article_title(pdf_path, fallback_text=text_no_refs)

            md_path = os.path.join(PREPROCESSED_FOLDER, os.path.splitext(fname)[0] + ".md")
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(
                    f"# {title}\n\n"
                    f"> Word count (no refs): {wc_no_refs}\n"
                    f"> Word count used (cap {WORD_CAP}): {wc_processed}\n\n"
                    f"{text_no_refs}"
                )

            raw_response = call_model(text_capped)
            parsed = clean_and_parse_json(raw_response)

            json_path = os.path.join(OUTPUT_FOLDER, os.path.splitext(fname)[0] + ".json")
            with open(json_path, "w", encoding="utf-8") as f:
                f.write(stringify(parsed if parsed else raw_response))

            findings = parsed.get("genes_with_findings") if isinstance(parsed, dict) else []
            findings = findings if isinstance(findings, list) else []

            genes_found = []
            for item in findings:
                if isinstance(item, dict):
                    gene = str(item.get("gene", "")).strip()
                    if gene:
                        genes_found.append(gene)

            has_any = bool(parsed.get("has_any_gene_findings")) if isinstance(parsed, dict) else False
            genes_joined = ", ".join(dict.fromkeys(genes_found))

            with open(CSV_FILE, "a", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    fname,
                    title,
                    wc_no_refs,
                    wc_processed,
                    has_any,
                    genes_joined,
                    len(findings),
                    json_path,
                ])

            print(f"[OK] Saved findings JSON: {json_path}")
            print(f"[OK] Appended summary row to: {CSV_FILE}")

        except Exception as e:
            print(f"Error processing {fname}: {e}")


if __name__ == "__main__":
    main()
