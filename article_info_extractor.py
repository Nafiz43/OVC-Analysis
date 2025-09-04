import os
import re
import csv
import json
import pdfplumber
from typing import Any, Dict, List, Union

try:
    # New-style OpenAI SDK (>=1.0)
    from openai import OpenAI
except Exception:
    OpenAI = None  # Will raise a clear error if used without installation


# =====================
# Configuration
# =====================
# SECURITY NOTE: Prefer env var over hardcoding.
API_KEY = os.getenv("OPENAI_API_KEY","")
INPUT_FOLDER = "biomarkers"                      # Folder containing your PDF files
OUTPUT_FOLDER = "results"                        # Folder where results (csv, txt) will be saved
PREPROCESSED_FOLDER = "articles-preprocessed"    # Folder for Markdown versions of PDFs
CSV_FILE = os.path.join(OUTPUT_FOLDER, "extracted_entities.csv")

MODEL_NAME = "gpt-5-nano"  # LLM to use
TEMPERATURE = 1            # Deterministic extraction

# Word cap per article for model processing
WORD_CAP = 10_000

# =====================
# Helpers
# =====================
def ensure_dirs() -> None:
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(PREPROCESSED_FOLDER, exist_ok=True)


def pdf_to_text(pdf_path: str) -> str:
    """Extracts text from a PDF using pdfplumber. Returns a single string."""
    chunks: List[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text() or ""
            if txt.strip():
                chunks.append(txt)
    return "\n\n".join(chunks).strip()


def get_article_title(pdf_path: str, fallback_text: str = "") -> str:
    """Returns PDF metadata title if available; otherwise filename (sans .pdf);
    as a final fallback, try first non-empty line of text."""
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


def stringify(obj: Any) -> str:
    """Return a UTF-8 safe string for writing to disk, even if obj is dict/list/etc."""
    if isinstance(obj, str):
        return obj
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        return str(obj)


def clean_and_parse_json(raw: Union[str, Dict[str, Any], List[Any]]) -> Dict[str, Any]:
    """Parse JSON from model output. Accepts dict (returns as-is), or string.
    Strips code fences and tries to load JSON; returns {} on failure."""
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        try:
            return json.loads(json.dumps(raw))
        except Exception:
            return {}

    text = raw.strip()

    # Strip common code-fence patterns
    if text.startswith("```"):
        # Try to find a fenced json block
        if "```json" in text:
            text = text.split("```json", 1)[-1]
        else:
            text = text.split("```", 1)[-1]
        text = text.split("```", 1)[0].strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except Exception:
        # Heuristic: find first '{' and last '}' to isolate JSON
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                candidate = text[start : end + 1]
                return json.loads(candidate)
        except Exception:
            pass

    return {}


def ensure_list(x: Any) -> List[str]:
    """Normalize a field into a list of strings for CSV output."""
    if x is None:
        return []
    if isinstance(x, list):
        return [str(i) for i in x if i is not None]
    return [str(x)]


# =====================
# New helpers for your request
# =====================
_REF_HEADING_RE = re.compile(r'^\s*(references?|reference)\s*:?\s*$', flags=re.IGNORECASE)

def truncate_at_references(text: str) -> str:
    """
    Truncate the article at the first References/REFERENCE/... heading (case-insensitive).
    We detect it as a *standalone heading line* to avoid false positives like "reference genome".
    """
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if _REF_HEADING_RE.match(line):
            return "\n".join(lines[:i]).rstrip()
    return text

def word_count(text: str) -> int:
    """
    Count words conservatively. Use non-whitespace tokens as words.
    """
    return len(re.findall(r'\S+', text))

def cap_words(text: str, cap: int) -> str:
    """
    Keep only the first 'cap' words of 'text'.
    """
    tokens = re.findall(r'\S+', text)
    if len(tokens) <= cap:
        return text
    tokens = tokens[:cap]
    # Reconstruct with single spaces; preserve basic readability
    return " ".join(tokens)


# =====================
# Prompt (definitions only; no examples)
# =====================
EXTRACTION_PROMPT = """You are an expert biomedical text-mining system.
Your task is to carefully read a given article and extract the names of biological entities,
grouped into the following five categories.

Category Definitions (for guidance only; do not hallucinate):
1. DNA – The entire genetic blueprint; usually referred to as chromosomes or specific DNA regions.
2. Genes – Specific sequences of DNA that code for proteins or RNAs.
3. RNA – Transcribed copies of genes, usually messenger RNA (mRNA), but also includes other RNA types (lncRNA, tRNA, etc.).
4. Proteins – Functional molecules built from amino acids, encoded by genes.
5. Meth-RNA – RNA molecules or transcripts influenced by DNA methylation (epigenetic regulation) OR RNAs with direct methylation marks (e.g., m6A).

Instructions:
- Read the provided article carefully.
- Identify and extract all explicitly mentioned entities under these five categories.
- If an entity could fit more than one category (e.g., gene vs. protein), decide based on local context.
- Output the results strictly in valid JSON with the keys: DNA, Genes, RNA, Proteins, Meth-RNA.
- Do not include any explanations or text outside the JSON.
"""


# =====================
# LLM call
# =====================
def call_model(article_text: str) -> str:
    """Send the prompt + article text to the model, return raw string content."""
    if OpenAI is None:
        raise RuntimeError("OpenAI SDK not installed. Please `pip install openai`.")

    api_key = API_KEY or ""
    if not api_key:
        raise RuntimeError("API key is not set. Set OPENAI_API_KEY in your environment.")

    client = OpenAI(api_key=api_key)

    # Build the final input
    final_prompt = (
        EXTRACTION_PROMPT
        + "\n\n### Article Content (verbatim):\n"
        + article_text
    )

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": "You are a careful, rigorous biomedical information extractor."},
            {"role": "user", "content": final_prompt},
        ],
    )
    return resp.choices[0].message.content


# =====================
# Main pipeline
# =====================
def main() -> None:
    ensure_dirs()

    # Prepare CSV header (append if exists; write header only when new file)
    write_header = not os.path.exists(CSV_FILE)
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            # Added two columns for visibility; if your CSV already exists, header won't change.
            writer.writerow([
                "File Name", "Article Name",
                "Proteins", "Genes", "DNA", "RNA", "Meth-RNA",
                "Word Count (no-refs)", "Word Count (processed_cap)"
            ])

    # Iterate PDFs
    pdf_files = [f for f in sorted(os.listdir(INPUT_FOLDER)) if f.lower().endswith(".pdf")]
    for fname in pdf_files:
        pdf_path = os.path.join(INPUT_FOLDER, fname)
        try:
            print(f"\n=== Processing: {fname} ===")

            # Extract text
            text_full = pdf_to_text(pdf_path)

            # Remove everything after References heading (case-insensitive)
            text_no_refs = truncate_at_references(text_full)

            # Count words (after ref trimming)
            wc_no_refs = word_count(text_no_refs)
            print(f"Word count (no refs): {wc_no_refs}")

            # Cap to first N words for model processing
            text_capped = cap_words(text_no_refs, WORD_CAP)
            wc_processed = word_count(text_capped)
            print(f"Word count used (cap {WORD_CAP}): {wc_processed}")

            # Determine title
            title = get_article_title(pdf_path, fallback_text=text_no_refs)

            # Save preprocessed MD (keep the *no-refs* full text for human reading)
            md_path = os.path.join(PREPROCESSED_FOLDER, os.path.splitext(fname)[0] + ".md")
            md_body = f"# {title}\n\n" \
                      f"> Word count (no refs): {wc_no_refs}\n\n" \
                      f"{text_no_refs if text_no_refs else ''}"
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md_body)
            print(f"Saved preprocessed MD -> {md_path}")

            # Call model on the capped text only
            raw_response = call_model(text_capped)

            # Save raw response to .txt (stringified for safety)
            txt_path = os.path.join(OUTPUT_FOLDER, os.path.splitext(fname)[0] + ".txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(stringify(raw_response))
            print(f"Saved raw response -> {txt_path}")

            # Print response to console (truncated for readability)
            preview = stringify(raw_response)
            print("Model response preview:\n" + (preview[:1000] + ("..." if len(preview) > 1000 else "")))

            # Parse JSON
            parsed = clean_and_parse_json(raw_response)

            # Normalize fields
            dna = ", ".join(ensure_list(parsed.get("DNA")))
            genes = ", ".join(ensure_list(parsed.get("Genes")))
            rna = ", ".join(ensure_list(parsed.get("RNA")))
            proteins = ", ".join(ensure_list(parsed.get("Proteins")))
            meth_rna = ", ".join(ensure_list(parsed.get("Meth-RNA")))

            # Append to CSV
            with open(CSV_FILE, "a", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                # If your CSV already existed (older header), these last cols will still be written;
                # consider regenerating CSV for strict schema.
                writer.writerow([fname, title, proteins, genes, dna, rna, meth_rna, wc_no_refs, wc_processed])
            print(f"Appended row to CSV -> {CSV_FILE}")

        except Exception as e:
            print(f"Error processing {fname}: {e}")


if __name__ == "__main__":
    main()




