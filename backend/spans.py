import re
from typing import List, Set
from backend.schemas import Span


# ── Priority patterns (legal markers) ─────────────────────────────────────────
# These fire first. Their positions are marked as "priority split points".
# If a sentence-boundary split lands within MERGE_WINDOW chars of one of these,
# the sentence-boundary split is discarded — the legal marker takes over.
_PRIORITY_PATTERNS = re.compile(
    r'(?m)(?:^|(?<=[\n.;:]))\s*(?=\b(?:Provided\s+that|Explanation|Illustration)\b)',
    re.IGNORECASE,
)

# ── Secondary patterns (structural) ───────────────────────────────────────────
# Subsection markers like (1), (2), (a), (b), (aa), (bb), (i), (ii), (iii), (iv).
# This avoids splitting random bracketed words such as (hereinafter).
_SUBSECTION_PATTERN = re.compile(
    r'(?=\(\s*(?:\d+|[a-z]|([a-z])\1|i{1,3}|iv|v|vi{0,3}|ix|x)\s*\))',
    re.IGNORECASE,
)

# Conjunctive legal words that introduce new clauses.
# These are only split when they appear at the start of the text, a new line,
# or after punctuation. This prevents false splitting in phrases like
# "the place where the offence occurred".
_CONJUNCTIVE_PATTERN = re.compile(
    r'(?m)(?:^|(?<=[\n.;:]))\s*(?=\b(?:Where|Unless|Except|Notwithstanding)\b)',
    re.IGNORECASE,
)

# Sentence boundary: full stop followed by whitespace and an uppercase letter.
# No re.IGNORECASE here intentionally. Uppercase is used as a signal that a
# new sentence is starting.
_SENTENCE_PATTERN = re.compile(
    r'(?<=\.)\s+(?=[A-Z"(])',
)

# If a sentence-boundary split is very close to a priority split, keep the
# priority split and discard the sentence split.
MERGE_WINDOW = 8

# Header-only priority markers. These should be merged with the next chunk.
# Full provisos such as "Provided that the person..." should NOT be merged again.
_PRIORITY_HEADER_ONLY = re.compile(
    r'^\s*(?:Provided\s+that|Explanation|Illustration)\s*[:.\-–—]?\s*$',
    re.IGNORECASE,
)


def make_spans(text: str) -> List[Span]:
    """
    Split provision text into evidence spans P1..Pn.

    Strategy:
    1. Find priority split points (Provided that / Explanation / Illustration).
    2. Find secondary split points (subsections, conjunctives, sentence boundaries).
    3. Drop any secondary split that is within MERGE_WINDOW chars of a priority split.
    4. Merge leftover tiny chunks (< 30 chars) into their neighbour.
    5. Return spans with offsets into the ORIGINAL raw input — never stripped.

    re.IGNORECASE is applied to every pattern compile call above except
    _SENTENCE_PATTERN, which deliberately requires an uppercase letter to
    confirm a new sentence is starting.
    """
    if not text:
        return []

    # ── Step 1: collect priority split positions ───────────────────────────────
    priority_positions: Set[int] = set()
    for m in _PRIORITY_PATTERNS.finditer(text):
        priority_positions.add(m.start())

    # ── Step 2: collect all split positions ───────────────────────────────────
    all_splits: Set[int] = {0, len(text)}

    # Priority splits always included
    all_splits.update(priority_positions)

    # Subsection splits always included
    for m in _SUBSECTION_PATTERN.finditer(text):
        all_splits.add(m.start())

    # Conjunctive splits always included
    for m in _CONJUNCTIVE_PATTERN.finditer(text):
        all_splits.add(m.start())

    # Sentence boundary splits included ONLY if not near a priority split
    for m in _SENTENCE_PATTERN.finditer(text):
        pos = m.start()
        too_close = any(
            abs(pos - p) <= MERGE_WINDOW
            for p in priority_positions
        )
        if not too_close:
            all_splits.add(pos)

    # ── Step 3: sort and build raw chunks ─────────────────────────────────────
    split_points = sorted(all_splits)

    raw_chunks: List[tuple] = []
    for i in range(len(split_points) - 1):
        start = split_points[i]
        end = split_points[i + 1]

        chunk_text = text[start:end]
        stripped_chunk = chunk_text.strip()

        if not stripped_chunk:
            continue

        # Compute actual start offset by skipping leading whitespace.
        # This preserves correct offsets into the original raw input text.
        leading_spaces = len(chunk_text) - len(chunk_text.lstrip())
        actual_start = start + leading_spaces
        actual_end = end

        raw_chunks.append((actual_start, actual_end, stripped_chunk))

    # ── Step 4: merge chunks ───────────────────────────────────────────────────
    merged: List[tuple] = []

    i = 0
    while i < len(raw_chunks):
        actual_start, actual_end, visible = raw_chunks[i]

        # Only merge priority markers when they are header-only.
        # Example: "Provided that" should merge with the next chunk.
        # But "Provided that the person has..." is already a full proviso and should not
        # be force-merged with the following sentence.
        is_priority_header_only = bool(_PRIORITY_HEADER_ONLY.match(visible))

        if is_priority_header_only and i + 1 < len(raw_chunks):
            next_start, next_end, next_visible = raw_chunks[i + 1]
            merged.append((actual_start, next_end, f"{visible.rstrip()} {next_visible.lstrip()}"))
            i += 2

        elif merged and len(visible) < 30:
            # Tiny leftover chunk — merge back into the previous span
            prev_start, _, prev_visible = merged[-1]
            merged[-1] = (prev_start, actual_end, prev_visible + ' ' + visible)
            i += 1

        else:
            merged.append((actual_start, actual_end, visible))
            i += 1

    # ── Step 5: build Span objects ────────────────────────────────────────────
    spans = []
    for idx, (start, end, visible) in enumerate(merged):
        spans.append(Span(
            span_id=f"P{idx + 1}",
            text=visible,
            start=start,
            end=end,
        ))

    return spans