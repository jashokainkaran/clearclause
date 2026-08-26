import re
from typing import List, Set, Tuple
from backend.schemas import Span, StageStatus


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
# The enumerator alternation avoids bracketed words such as (hereinafter).
#
# ANCHORING: the marker must also start the text, start a line, or follow
# . ; : — or – (em dash / en dash). This is the same anchoring style used by
# the priority and conjunctive patterns. Without the anchor the bare lookahead
# fired on INLINE cross-references such as "subsection (1)", "paragraph (b)",
# "section 12(2)" and "section 12 (2)", splitting mid-sentence and cutting a
# cross-reference away from the rule it belongs to.
#
# The ordinary hyphen "-" is deliberately NOT a boundary: it appears inside
# hyphenated words and ranges, where a split would be wrong.
_SUBSECTION_PATTERN = re.compile(
    r'(?m)(?:^|(?<=[\n.;:—–]))\s*'
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

# Semicolon-introduced alternative legal branches: "; or, if", "; and if",
# "; or if", "; if". These separate an ordinary punishment branch from a
# special / aggravated branch.
#
# Deliberately narrow: a bare ";" does NOT split, a comma does NOT split, and
# a bare "or" does NOT split. Only a semicolon immediately introducing a
# conditional branch splits, which keeps spans branch-level and coherent
# instead of shattering a provision at every conjunction.
#
# IMPORTANT: a span produced by this split is HEADLESS — it lacks the subject
# and the operative "shall be punished" phrase, which live in the preceding
# span. evidence.link_claims_to_spans therefore pairs such a span with its
# immediately preceding span before the text is sent to the NLI model. This
# splitter must not be used without that continuation handling.
_SEMICOLON_BRANCH_PATTERN = re.compile(
    r'(?<=;)\s*(?=(?:(?:or|and)\s*,?\s*)?if\b)',
    re.IGNORECASE,
)

# A chunk that opens with a structural enumerator, e.g. "(a) murder;".
# Used to protect deliberate structural splits from the tiny-chunk merge.
_STRUCTURAL_START = re.compile(
    r'^\(\s*(?:\d+|[a-z]|([a-z])\1|i{1,3}|iv|v|vi{0,3}|ix|x)\s*\)',
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
    2. Find secondary split points (anchored subsections, conjunctives,
       semicolon-introduced branches, sentence boundaries).
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

    # Semicolon branch splits always included. m.start() is the offset
    # immediately AFTER the ";", so the semicolon stays with the preceding
    # branch and the leading whitespace is stripped in step 3.
    #
    # Both m.start() and m.end() are recorded: step 3 advances a chunk's start
    # past leading whitespace, so the chunk's actual_start equals m.end().
    # The merge guard in step 4 compares against actual_start, and would miss
    # the branch if only m.start() were stored.
    branch_positions: Set[int] = set()
    for m in _SEMICOLON_BRANCH_PATTERN.finditer(text):
        all_splits.add(m.start())
        branch_positions.add(m.start())
        branch_positions.add(m.end())

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

        elif (
            merged
            and len(visible) < 30
            and actual_start not in branch_positions
            and not _STRUCTURAL_START.match(visible)
        ):
            # Tiny leftover chunk — merge back into the previous span.
            #
            # Chunks that open a semicolon branch or a structural enumerator
            # are exempt. Both are short by nature, and merging them would
            # silently undo a split that was made deliberately (e.g. a short
            # second punishment branch, or "(b) second rule.").
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


class SpanValidationError(Exception):
    """
    Raised by validate_spans when a span list is unusable for downstream
    stages. Callers must not proceed into simplification/evidence linking
    if this fails — make_spans_with_fallback is what decides whether that
    means trying a fallback splitter or truly stopping.
    """
    pass


def validate_spans(spans: List[Span], source_text: str) -> None:
    """
    Lightweight, deterministic sanity check on a span list before it is
    used downstream. Guards against the failure mode where an unexpected
    input format or a splitter regression silently produces an empty,
    malformed, or out-of-bounds span list.
    """
    if not spans:
        raise SpanValidationError("Span generation produced no spans.")

    seen_ids: Set[str] = set()
    text_len = len(source_text)

    for span in spans:
        if not span.span_id:
            raise SpanValidationError("A span is missing its span_id.")
        if span.span_id in seen_ids:
            raise SpanValidationError(f"Duplicate span_id: {span.span_id}")
        seen_ids.add(span.span_id)

        if not span.text or not span.text.strip():
            raise SpanValidationError(f"Span {span.span_id} has empty text.")

        if span.start < 0 or span.end < 0 or span.start >= span.end:
            raise SpanValidationError(
                f"Span {span.span_id} has invalid offsets (start={span.start}, end={span.end})."
            )

        if span.end > text_len:
            raise SpanValidationError(
                f"Span {span.span_id} end offset ({span.end}) exceeds source text length ({text_len})."
            )


def make_spans_sentence_fallback(text: str) -> List[Span]:
    """
    Fallback tier 2 of make_spans_with_fallback. Splits purely on sentence
    boundaries — no subsection/priority/branch markers — reusing the same
    _SENTENCE_PATTERN make_spans() already uses, rather than introducing a
    second sentence-detection mechanism. Deliberately simpler than
    make_spans(): fewer split rules means fewer ways for this tier itself
    to produce something invalid.
    """
    if not text:
        return []

    split_points = sorted({0, len(text)} | {m.start() for m in _SENTENCE_PATTERN.finditer(text)})

    spans: List[Span] = []
    idx = 1
    for i in range(len(split_points) - 1):
        start, end = split_points[i], split_points[i + 1]
        chunk = text[start:end]
        stripped = chunk.strip()
        if not stripped:
            continue
        leading_spaces = len(chunk) - len(chunk.lstrip())
        actual_start = start + leading_spaces
        spans.append(Span(span_id=f"P{idx}", text=stripped, start=actual_start, end=end))
        idx += 1

    return spans


def make_spans_with_fallback(text: str) -> Tuple[List[Span], StageStatus]:
    """
    Shared span-generation entry point used by both the /spans endpoint and
    the full pipeline, so the two can never disagree about what counts as
    a valid split.

    Tier 1: the normal structural splitter (make_spans), validated.
    Tier 2 (only if tier 1 is invalid): a plain sentence-boundary fallback,
            validated the same way.
    Tier 3 (only if tier 2 is also invalid): the entire original text as
            one span, P1.

    No tier ever rewrites or invents source text — every fallback is an
    exact substring (or the whole) of the original. Tier 3 can only fail
    to validate if the source text itself is unusable, which Stage 0 input
    validation (backend/schemas.py) already prevents — so in practice this
    resolves to "success" or "fallback", essentially never "failed".
    """
    primary_spans = make_spans(text)
    try:
        validate_spans(primary_spans, text)
        return primary_spans, StageStatus(status="success")
    except SpanValidationError:
        pass

    fallback_spans = make_spans_sentence_fallback(text)
    try:
        validate_spans(fallback_spans, text)
        return fallback_spans, StageStatus(status="fallback", method="sentence_splitter")
    except SpanValidationError:
        pass

    whole_provision = [Span(span_id="P1", text=text, start=0, end=len(text))]
    try:
        validate_spans(whole_provision, text)
        return whole_provision, StageStatus(status="fallback", method="full_provision")
    except SpanValidationError as final_error:
        # Defensive-only: unreachable in practice given Stage 0 guarantees
        # non-empty input, but handled explicitly rather than assumed away.
        return whole_provision, StageStatus(
            status="failed",
            reason=f"Source text could not be preserved as a usable span: {final_error}",
        )