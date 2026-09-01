import re
import argparse
from pathlib import Path

import fitz  # PyMuPDF
import pandas as pd


ROLE_KEYWORDS = [
    "magistrate", "judge", "court", "minister", "child", "parent",
    "guardian", "person", "officer", "commissioner", "police",
    "defendant", "plaintiff", "director", "secretary", "president",
    "government", "public servant", "wife", "husband", "surgeon"
]

CROSS_REF_PATTERNS = [
    r"\bthis section\b",
    r"\bthat section\b",
    r"\blast preceding section\b",
    r"\bpreceding section\b",
    r"\bsubsection\b",
    r"\bchapter\b",
    r"\bthis code\b",
    r"\bhereinafter\b",
]

SECTION_START_RE = re.compile(r"^(?P<section>\d+[A-Z]?)\.(?P<rest>.*)")
SUBSECTION_RE = re.compile(r"^\((\d+)\)\s*")
PARA_RE = re.compile(r"^\(([a-z])\)\s*", re.IGNORECASE)
ROMAN_RE = re.compile(r"^\(([ivxlcdm]+)\)\s*", re.IGNORECASE)
PROVIDED_RE = re.compile(r"^Provided(?:\s+further)?\s+that\b", re.IGNORECASE)
EXPLANATION_RE = re.compile(r"^Explanation(?:\s*\d+)?(?:\s*[-.:])?", re.IGNORECASE)
ILLUS_RE = re.compile(r"^Illustrations?\.?$", re.IGNORECASE)
CHAPTER_RE = re.compile(r"^CHAPTER\b", re.IGNORECASE)

AMENDMENT_LINE_RE = re.compile(r"^\[\s*\d+.*\]$")
PAGE_MARKER_RE = re.compile(r"^--- PAGE \d+ ---$")


def normalize_line_text(text: str) -> str:
    text = text.replace("\uFFFE", "")
    text = text.replace("\u2018", "'").replace("\u201C", '"').replace("\u201D", '"')
    text = re.sub(r"\s+", " ", text).strip()
    return text


def line_is_margin_noise(text: str, page_width: float, x0: float, x1: float) -> bool:
    t = normalize_line_text(text)
    if not t:
        return True
    if AMENDMENT_LINE_RE.match(t):
        return True
    if t == "PENAL CODE":
        return True
    if x1 < page_width * 0.22 and len(t) < 80:
        return True
    if x0 < page_width * 0.20 and len(t) < 60 and not SECTION_START_RE.match(t) and not CHAPTER_RE.match(t):
        return True
    return False


def extract_clean_page_lines(pdf_path: str):
    doc = fitz.open(pdf_path)
    page_rows = []

    for page_index, page in enumerate(doc):
        page_num = page_index + 1
        page_width = float(page.rect.width)

        words = page.get_text("words", sort=True)

        lines = {}
        for x0, y0, x1, y1, word, block_no, line_no, word_no in words:
            key = (block_no, line_no)
            lines.setdefault(key, {
                "x0": x0, "x1": x1, "y0": y0, "y1": y1, "words": []
            })
            lines[key]["x0"] = min(lines[key]["x0"], x0)
            lines[key]["x1"] = max(lines[key]["x1"], x1)
            lines[key]["y0"] = min(lines[key]["y0"], y0)
            lines[key]["y1"] = max(lines[key]["y1"], y1)
            lines[key]["words"].append((x0, word))

        cleaned_lines = []
        for key, line in sorted(lines.items(), key=lambda kv: (kv[1]["y0"], kv[1]["x0"])):
            ordered_words = [w for _, w in sorted(line["words"], key=lambda t: t[0])]
            text = normalize_line_text(" ".join(ordered_words))

            if line_is_margin_noise(text, page_width, line["x0"], line["x1"]):
                continue

            cleaned_lines.append(text)

        cleaned_page_text = "\n".join(cleaned_lines)
        page_rows.append({
            "page": page_num,
            "text": cleaned_page_text
        })

    return page_rows


def save_page_csv(rows, out_csv: str):
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")


def save_page_txt(rows, out_txt: str):
    parts = []
    for row in rows:
        parts.append(f"--- PAGE {row['page']} ---\n")
        parts.append(row["text"])
        parts.append("\n\n")
    Path(out_txt).write_text("".join(parts), encoding="utf-8")


def merge_pages_to_single_text(page_rows):
    chunks = []
    for row in page_rows:
        text = row["text"].strip()
        if not text:
            continue
        chunks.append(f"--- PAGE {row['page']} ---\n{text}")
    return "\n\n".join(chunks)


def extract_page_number_from_marker(text_chunk: str):
    m = re.search(r"--- PAGE (\d+) ---", text_chunk)
    return int(m.group(1)) if m else None


def find_section_spans(merged_text: str):
    # Match NNN. or NNA. followed by space, newline, or end-of-line
    # Filter out years (4 digits) and numbers outside Penal Code range
    raw_matches = list(re.finditer(r"(?m)^(\d+[A-Z]?)\.(?:\s|$)", merged_text))

    valid = []
    for m in raw_matches:
        sec = m.group(1)
        if re.fullmatch(r"\d{4}", sec):        # year like 1887
            continue
        n = int(re.match(r"\d+", sec).group())
        if n < 1 or n > 500:                   # outside Penal Code range
            continue
        valid.append((sec, m.start()))

    spans = []
    for i, (section_id, start) in enumerate(valid):
        end = valid[i + 1][1] if i + 1 < len(valid) else len(merged_text)
        spans.append((section_id, start, end))

    return spans


def split_section_into_units(section_text: str, section_id: str):
    text = section_text.strip()
    text = re.sub(r"^--- PAGE \d+ ---\s*", "", text).strip()
    text = re.sub(rf"^{re.escape(section_id)}\.(?=\S)", "", text).strip()

    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    units = []

    current = []
    current_label = ""
    current_type = "section_body"

    def flush():
        nonlocal current, current_label, current_type
        if current:
            units.append({
                "unit_label": current_label,
                "unit_type": current_type,
                "text": " ".join(current).strip()
            })
        current = []
        current_label = ""
        current_type = "section_body"

    for line in lines:
        if PAGE_MARKER_RE.match(line):
            continue
        if CHAPTER_RE.match(line):
            continue

        if SUBSECTION_RE.match(line):
            flush()
            current = [line]
            current_label = SUBSECTION_RE.match(line).group(1)
            current_type = "subsection"
            continue

        if PROVIDED_RE.match(line):
            flush()
            current = [line]
            current_label = "provided_that"
            current_type = "proviso"
            continue

        if EXPLANATION_RE.match(line):
            flush()
            current = [line]
            current_label = "explanation"
            current_type = "explanation"
            continue

        if ILLUS_RE.match(line):
            flush()
            current = [line]
            current_label = "illustrations"
            current_type = "illustrations_header"
            continue

        if PARA_RE.match(line) and current_type in {"subsection", "explanation", "illustrations_header", "illustration"}:
            current.append(line)
            if current_type == "illustrations_header":
                current_type = "illustration"
            continue

        if ROMAN_RE.match(line) and current_type in {"subsection", "explanation", "illustrations_header", "illustration"}:
            current.append(line)
            if current_type == "illustrations_header":
                current_type = "illustration"
            continue

        if not current:
            current = [line]
            current_label = ""
            current_type = "section_body"
        else:
            current.append(line)

    flush()

    if not units:
        units = [{
            "unit_label": "",
            "unit_type": "section_body",
            "text": text
        }]

    return units


def clean_unit_text(text: str) -> str:
    text = normalize_line_text(text)
    text = re.sub(r"\s+([;,.:\)])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)
    return text.strip()


def has_modal(text: str) -> int:
    return int(bool(re.search(r"\b(shall|shall not|may|must|must not)\b", text, flags=re.IGNORECASE)))

def has_exception(text: str) -> int:
    return int(bool(re.search(r"\b(if|unless|provided that|provided further that|except|notwithstanding)\b", text, flags=re.IGNORECASE)))

def has_numbers(text: str) -> int:
    return int(bool(re.search(r"\b\d+\b", text)))

def has_roles(text: str) -> int:
    lower = text.lower()
    return int(any(role in lower for role in ROLE_KEYWORDS))

def needs_context(text: str) -> int:
    lower = text.lower()
    return int(any(re.search(p, lower) for p in CROSS_REF_PATTERNS))

def likely_bad_fragment(text: str) -> int:
    t = text.strip()
    if len(t) < 35:
        return 1
    if t.endswith((":", ";")) and len(t) < 120:
        return 1
    return 0


def build_candidate_rows(page_rows, act_name: str):
    merged = merge_pages_to_single_text(page_rows)
    section_spans = find_section_spans(merged)

    rows = []
    counter = 1

    for section_id, start, end in section_spans:
        raw_section = merged[start:end]
        page_start = extract_page_number_from_marker(raw_section)

        units = split_section_into_units(raw_section, section_id)

        for unit in units:
            source_text = clean_unit_text(unit["text"])
            if not source_text:
                continue

            rows.append({
                "example_id": f"SL{counter:05d}",
                "act_name": act_name,
                "section_id": f"Section {section_id}",
                "unit_label": unit["unit_label"],
                "unit_type": unit["unit_type"],
                "page_start": page_start,
                "source_text": source_text,
                "has_modal": has_modal(source_text),
                "has_exception": has_exception(source_text),
                "has_numbers": has_numbers(source_text),
                "has_roles": has_roles(source_text),
                "needs_context": needs_context(source_text),
                "bad_fragment": likely_bad_fragment(source_text),
                "simple_text": "",
                "split": "",
                "notes": ""
            })
            counter += 1

    return rows


def save_candidate_csv(rows, out_csv: str):
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--act", required=True)
    parser.add_argument("--pages_csv", required=True)
    parser.add_argument("--pages_txt", required=True)
    parser.add_argument("--candidates_csv", required=True)
    args = parser.parse_args()

    print("1) Extracting cleaned page text...")
    page_rows = extract_clean_page_lines(args.pdf)
    save_page_csv(page_rows, args.pages_csv)
    save_page_txt(page_rows, args.pages_txt)
    print(f"   Pages: {len(page_rows)}")

    print("2) Building candidate dataset rows...")
    rows = build_candidate_rows(page_rows, args.act)
    save_candidate_csv(rows, args.candidates_csv)
    print(f"   Candidate rows: {len(rows)}")
    print("Done.")

if __name__ == "__main__":
    main()