import re
from decimal import Decimal
from typing import List, Optional

# Common words that carry no legal meaning — excluded from matching.
# NOTE: shall / may / must / not / can are deliberately NOT stopwords — they
# carry legal force (obligation / permission / prohibition) and must
# participate in retrieval. ("can" was previously stopped by mistake here;
# removing it lets permission wording like "may"/"can" be matched.)
STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "by",
    "with", "is", "are", "be", "was", "were", "this", "that",
    "any", "such", "as", "at", "he", "she",
    "they", "who", "which", "been", "have", "has", "from", "its", "their"
}

# How many points a matching DISTINGUISHING token — a NUMBER (digit or
# word-number) or the word "not" — is worth in link_claims_to_spans'
# ranking, versus 1 point for an ordinary matching word. Chosen as the
# smallest whole-number weight that flips the real, documented
# sibling-branch failure (P1 "5 shared words" vs P2 "3 shared words + the
# branch-distinguishing number") with a safety margin, while staying a
# plain int so exact-tie detection is never affected by floating-point
# rounding. See link_claims_to_spans for the full rationale.
#
# "not" is included here as a plain word already present in the token set
# (STOPWORDS above deliberately keeps it) — this does NOT touch
# _extract_modality_tags/_MODAL_PROHIBITION_PATTERN or the
# clear_modal_conflict warning that depends on it, which stay exactly as
# narrow as before. This only changes how much a shared "not" counts
# toward picking which span a claim belongs to.
DISTINGUISHING_WORDS = frozenset({"not"})
DISTINGUISHING_MATCH_WEIGHT = 4

# Canonical modality tokens — added alongside the original words so
# semantically-equivalent modal phrasing (e.g. "shall" vs "must") still
# overlaps even when the surface word differs.
LEGAL_MODAL_OBLIGATION = "LEGAL_MODAL_OBLIGATION"
LEGAL_MODAL_PERMISSION = "LEGAL_MODAL_PERMISSION"
LEGAL_MODAL_PROHIBITION = "LEGAL_MODAL_PROHIBITION"

# Negated / multi-word forms are matched first. Any single-word obligation or
# permission match that falls inside an already-matched negated phrase is
# suppressed — but only for that specific occurrence, so a sentence with a
# genuine obligation *and* a separate prohibition still receives both tags.
_MODAL_PROHIBITION_PATTERN = re.compile(r'\b(?:shall|must|may|can)\s+not\b|\bcannot\b')
_MODAL_OBLIGATION_PATTERN = re.compile(r'\b(?:shall|must)\b')
_MODAL_PERMISSION_PATTERN = re.compile(r'\b(?:may|can)\b')

# Standalone legal numeric values: integers, comma-formatted (5,000) and
# decimals (3.5 / 3.50). Bounded on both sides so digits glued to letters
# (subsection labels like "34A", or "covid19") are left untouched.
_NUMBER_PATTERN = re.compile(
    r'(?<![A-Za-z])(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(?![A-Za-z])'
)

# A span that opens with a continuation phrase is headless — the subject and
# the operative verb phrase live in the PREVIOUS span. Matches the branch
# forms produced by spans._SEMICOLON_BRANCH_PATTERN: "or, if", "and if",
# "or if", "if".
_CONTINUATION_START = re.compile(
    r'^(?:(?:or|and)\s*,?\s*)?if\b',
    re.IGNORECASE,
)

# A span that opens with a LIST-ITEM enumerator is also headless: "(a) theft;"
# has no subject or operative verb — those live in the introducing head span.
#
# Only LETTERED and ROMAN markers count as list items here. NUMERIC markers
# like "(1)"/"(2)" are deliberately excluded: a numbered subsection is normally
# a self-contained rule OR the head that introduces lettered paragraphs
# (e.g. "(1) The following acts are offences: (a) ... (b) ..."). Treating a
# numeric marker as an item would both (a) wrongly merge independent
# subsections and (b) make "(1)" an unusable head for its "(a)/(b)" children.
_ITEM_START = re.compile(
    r'^\(\s*(?:[a-z]|([a-z])\1|i{1,3}|iv|v|vi{0,3}|ix|x)\s*\)',
    re.IGNORECASE,
)


def _is_headless(text: str) -> bool:
    """
    True when a span cannot stand alone as evidence: a continuation branch
    ("or, if ...") or a lettered/roman list item ("(a) ...").
    """
    t = text or ""
    return bool(_CONTINUATION_START.match(t) or _ITEM_START.match(t))


def _canonical_number(raw: str) -> str:
    """Canonicalises a numeric string so 5,000 == 5000 and 3.50 == 3.5."""
    cleaned = raw.replace(",", "")
    return format(Decimal(cleaned).normalize(), "f")


# Word-form cardinal and ordinal numbers, recognised in ADDITION to digit
# numbers (_NUMBER_PATTERN above), so a branch distinguished only by a
# spelled-out number ("six months" vs "one year", "first offence" vs
# "second offence") gets the same numeric weighting/tie-break treatment as
# one distinguished by digits. See link_claims_to_spans for why this
# matters: a claim that repeats a shared head's subject/verb can otherwise
# outscore the one branch-specific number that actually distinguishes it
# (see the "rioting_real_llm_wording" fixture in
# tests/pipeline_tests/test_evidence_units.py for a real, documented case).
#
# Deliberately narrow:
#   - "one" is EXCLUDED from standalone recognition. Unlike the other
#     cardinals, "one" is heavily overloaded in ordinary English as an
#     indefinite pronoun ("one of the offences", "each one") rather than a
#     count, and treating every occurrence as the number 1 produces false
#     numeric agreement between unrelated spans that happen to both use
#     "one" this way — caught by hand against
#     tests/pipeline_tests/test_evidence_units.py's nested_enum fixture,
#     where "Theft is one of the offences listed" is not about subsection
#     "(1)". "one" is still recognised inside an unambiguous hyphenated
#     compound ("twenty-one").
#   - No "hundred"/"thousand" multiplier and no multi-word compounding
#     ("one hundred and five"). Amounts that large are written as digits
#     in real statutes far more often than spelled out; the durations and
#     counts that actually distinguish branches ("six months", "first
#     offence") stay in scope.
_TENS_WORD_VALUES = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_ONES_WORD_VALUES = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9,
}
_TEEN_WORD_VALUES = {
    "zero": 0, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19,
}
_ORDINAL_WORD_VALUES = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14,
    "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18,
    "nineteenth": 19, "twentieth": 20,
}
_STANDALONE_CARDINAL_VALUES = {
    **_TENS_WORD_VALUES,
    **_TEEN_WORD_VALUES,
    **{word: value for word, value in _ONES_WORD_VALUES.items() if word != "one"},
}

_TENS_ALTERNATION = '|'.join(_TENS_WORD_VALUES)
_ONES_ALTERNATION = '|'.join(_ONES_WORD_VALUES)
_STANDALONE_CARDINAL_ALTERNATION = '|'.join(_STANDALONE_CARDINAL_VALUES)
_ORDINAL_ALTERNATION = '|'.join(_ORDINAL_WORD_VALUES)

_WORD_NUMBER_PATTERN = re.compile(
    r'\b(?:'
    rf'(?:{_TENS_ALTERNATION})-(?:{_ONES_ALTERNATION})'
    rf'|{_STANDALONE_CARDINAL_ALTERNATION}'
    rf'|{_ORDINAL_ALTERNATION}'
    r')\b',
    re.IGNORECASE,
)


def _word_number_value(word: str) -> Optional[int]:
    """
    Canonical integer value of a word matched by _WORD_NUMBER_PATTERN, or
    None if it isn't one (defensive — the pattern should only ever match a
    recognised word). A hyphenated tens-ones compound ("twenty-one") sums
    its two parts, including "one" as the ones-part — unlike bare "one",
    a compound like this is unambiguously numeric.
    """
    w = word.lower()
    if "-" in w:
        tens_part, _, ones_part = w.partition("-")
        tens_val = _TENS_WORD_VALUES.get(tens_part)
        ones_val = _ONES_WORD_VALUES.get(ones_part)
        if tens_val is not None and ones_val is not None:
            return tens_val + ones_val
        return None
    return _STANDALONE_CARDINAL_VALUES.get(w, _ORDINAL_WORD_VALUES.get(w))


def extract_numbers(text: str) -> set[str]:
    """
    Extracts every standalone legal numeric value in text, canonicalised —
    both digit numbers (_NUMBER_PATTERN) and recognised word-numbers
    (_WORD_NUMBER_PATTERN, e.g. "six" -> "6"). Used for number-aware
    tie-breaking, ranking weight, and diagnostics — kept separate from the
    general lexical tokenizer (see _tokenize).
    """
    digit_numbers = {_canonical_number(m) for m in _NUMBER_PATTERN.findall(text)}
    word_numbers = set()
    for m in _WORD_NUMBER_PATTERN.finditer(text):
        value = _word_number_value(m.group(0))
        if value is not None:
            word_numbers.add(str(value))
    return digit_numbers | word_numbers


def extract_modality_tags(text: str) -> set[str]:
    """
    Public wrapper around _extract_modality_tags for external callers
    (llm_client.py's claim-extraction warnings, pipeline.py's post-NLI
    diagnostics) that need modality tags without going through the full
    tokenizer.
    """
    return _extract_modality_tags(text.lower())


def lexical_overlap(text_a: str, text_b: str) -> set[str]:
    """
    Lexical token intersection between two texts, using the same
    normalisation/stopword rules as evidence retrieval. Used to detect
    "zero meaningful overlap" between an extracted claim and the
    simplified text it was supposedly extracted from — deliberately an
    exact-zero check, not a tunable similarity threshold.
    """
    lex_a, _ = _tokenize_parts(text_a)
    lex_b, _ = _tokenize_parts(text_b)
    return lex_a & lex_b


def detect_modal_conflict(text_a: str, text_b: str) -> bool:
    """
    True only for an unambiguous modality flip between two texts: one
    contains PROHIBITION and not the other, while the other introduces
    PERMISSION (or the mirror image). Deliberately narrow — a provision or
    its simplification can legitimately mix several clauses with different
    modal words (must / may / must not), so a broad difference in the
    overall modality sets is not, by itself, evidence of an error. Only
    this specific prohibition<->permission flip is unambiguous enough to
    treat as an integrity problem; anything broader is left to the caller
    to record as a non-blocking diagnostic instead.
    """
    tags_a = extract_modality_tags(text_a)
    tags_b = extract_modality_tags(text_b)

    flip_a_to_b = (
        LEGAL_MODAL_PROHIBITION in tags_a
        and LEGAL_MODAL_PROHIBITION not in tags_b
        and LEGAL_MODAL_PERMISSION in tags_b
    )
    flip_b_to_a = (
        LEGAL_MODAL_PROHIBITION in tags_b
        and LEGAL_MODAL_PROHIBITION not in tags_a
        and LEGAL_MODAL_PERMISSION in tags_a
    )
    return flip_a_to_b or flip_b_to_a


def detect_verification_conflicts(
    evidence_text: str, claim_text: str, verification_label: str
) -> List[str]:
    """
    Post-NLI deterministic sanity check — a second, independent safety
    signal alongside the NLI verdict, never a replacement for it. Only
    meaningful for a claim the model called "supported": a model that
    already said unsupported/uncertain has already flagged a problem
    itself, so there is nothing extra to surface here.

    Reuses the same number/modality primitives evidence retrieval already
    relies on, so this is not a second scoring algorithm — just a
    diagnostic comparison of what has already been extracted. Callers
    (pipeline.py) must record these as warnings only; the NLI label itself
    is never changed.
    """
    warnings: List[str] = []
    if verification_label != "supported":
        return warnings

    evidence_numbers = extract_numbers(evidence_text)
    claim_numbers = extract_numbers(claim_text)
    if evidence_numbers and claim_numbers and evidence_numbers != claim_numbers:
        warnings.append("numerical_conflict")

    if detect_modal_conflict(evidence_text, claim_text):
        warnings.append("clear_modal_conflict")

    return warnings


def _extract_modality_tags(text_lower: str) -> set[str]:
    """
    Deterministically tags obligation / permission / prohibition modality.
    """
    tags: set[str] = set()
    consumed_spans = []

    for m in _MODAL_PROHIBITION_PATTERN.finditer(text_lower):
        tags.add(LEGAL_MODAL_PROHIBITION)
        consumed_spans.append(m.span())

    def _is_consumed(span) -> bool:
        start, end = span
        return any(cs <= start and end <= ce for cs, ce in consumed_spans)

    for m in _MODAL_OBLIGATION_PATTERN.finditer(text_lower):
        if not _is_consumed(m.span()):
            tags.add(LEGAL_MODAL_OBLIGATION)

    for m in _MODAL_PERMISSION_PATTERN.finditer(text_lower):
        if not _is_consumed(m.span()):
            tags.add(LEGAL_MODAL_PERMISSION)

    return tags


def _tokenize_parts(text: str) -> tuple[set[str], set[str]]:
    """
    Splits tokenization into (lexical_tokens, number_tokens).

    Numbers are extracted and canonicalised first, then masked out of the
    text, so a comma-formatted or decimal value is never fragmented by the
    generic alphanumeric split below (e.g. "5,000" must not become "5" and
    "000").

    Returning the two sets separately is what fixes the double count
    documented in evidence-retrieval-number-audit.md: a matching number used
    to inflate the raw overlap score AND be re-checked in the tie-break.
    Retrieval now scores lexical overlap only, and applies numeric agreement
    exactly once, as tie-break tiers 2 and 3.
    """
    text_lower = text.lower()
    numbers: set[str] = set()

    def _mask_number(m: "re.Match[str]") -> str:
        numbers.add(_canonical_number(m.group(0)))
        return " "

    masked_text = _NUMBER_PATTERN.sub(_mask_number, text_lower)

    def _mask_word_number(m: "re.Match[str]") -> str:
        value = _word_number_value(m.group(0))
        if value is None:
            return m.group(0)
        numbers.add(str(value))
        return " "

    masked_text = _WORD_NUMBER_PATTERN.sub(_mask_word_number, masked_text)

    lexical: set[str] = set()
    lexical |= _extract_modality_tags(text_lower)

    words = re.findall(r"[a-z0-9]+", masked_text)
    lexical |= {w for w in words if w not in STOPWORDS and len(w) > 2}

    return lexical, numbers


def _tokenize(text: str) -> set[str]:
    """
    Public tokenizer — behaviour unchanged. Returns lexical tokens, modality
    tags and canonical numeric tokens as one set.

    Retrieval scoring deliberately does NOT use this function; it uses
    _tokenize_parts so that numbers are not counted twice.
    """
    lexical, numbers = _tokenize_parts(text)
    return lexical | numbers


def resolve_evidence_units(spans: list) -> list[dict]:
    """
    Resolve each span into its SEMANTIC evidence unit — the smallest run of
    text that can stand on its own as evidence for a claim.

    Rules
    -----
    * A normal (non-headless) span is its own unit.
    * A headless span (a continuation branch "or, if ..." or a lettered/roman
      list item "(a) ...") is joined to the nearest preceding NON-headless
      span — its head. Headless SIBLINGS in between are skipped, so a claim
      about the third branch gets head + third branch, never the second.
    * If no non-headless head exists before a headless span (malformed input
      such as a fragment appearing first), the span is left as its own unit
      rather than reaching past the start of the document.

    Each returned unit carries:
      - selected_id : the span this unit represents (retrieval provenance)
      - ids         : the span ids sent to the NLI model ([head, item] or [id])
      - premise     : the exact text sent to the NLI model
      - own_lex     : lexical tokens of the span's OWN text (used for ranking)
      - own_num     : numeric tokens of the span's OWN text (used for ranking)

    Ranking uses each span's OWN tokens so a claim attaches to the span it is
    actually about; the premise uses the JOINED text so the model always
    receives a grammatically complete sentence.
    """
    span_list = list(spans)
    units: list[dict] = []

    for i, span in enumerate(span_list):
        own_lex, own_num = _tokenize_parts(span.text)

        if i > 0 and _is_headless(span.text):
            j = i - 1
            while j > 0 and _is_headless(span_list[j].text):
                j -= 1
            if _is_headless(span_list[j].text):
                # No usable head before this headless span — keep it alone.
                ids = [span.span_id]
                premise = span.text
            else:
                head = span_list[j]
                ids = [head.span_id, span.span_id]
                premise = f"{head.text} {span.text}"
        else:
            ids = [span.span_id]
            premise = span.text

        units.append({
            "selected_id": span.span_id,
            "ids": ids,
            "premise": premise,
            "own_lex": own_lex,
            "own_num": own_num,
        })

    return units


def _unresolved_result(claim: dict, source_text: Optional[str], method_if_fallback: str) -> dict:
    """
    Shared shape for the two "no single confident match" outcomes: a
    genuine no-match, and an exact unresolved tie. Both prefer the whole
    trusted provision over guessing, when it's available; otherwise the
    claim simply has no evidence and will be reported as unverified
    downstream.
    """
    if source_text:
        return {
            "claim_id": claim["claim_id"],
            "claim_text": claim["claim_text"],
            "evidence_span_id": None,
            "evidence_span_ids": [],
            "evidence_text": source_text,
            "evidence_score": 0,
            "evidence_method": method_if_fallback,
        }
    return {
        "claim_id": claim["claim_id"],
        "claim_text": claim["claim_text"],
        "evidence_span_id": None,
        "evidence_span_ids": [],
        "evidence_text": None,
        "evidence_score": 0,
        "evidence_method": None,
    }


def link_claims_to_spans(
    claims: list[dict], spans: list, source_text: Optional[str] = None
) -> list[dict]:
    """
    For each claim, select the best SEMANTIC EVIDENCE UNIT (see
    resolve_evidence_units) and record what the NLI model should verify.

    Selection scores the claim against each unit's OWN-span tokens. A
    matching NUMBER (digit or recognised word-number — see
    _WORD_NUMBER_PATTERN) counts for DISTINGUISHING_MATCH_WEIGHT points instead of
    1, so it can outrank a shared-vocabulary lead from ordinary words.
    This is not a cosmetic tie-break: two sibling punishment branches
    routinely share their subject and verb ("the person ... must be
    punished") because a claim restates the head, and under a flat word
    count that shared vocabulary can outscore the one or two words that
    actually distinguish the branches ("six months" vs "one year",
    "first offence" vs "second offence") — a real, documented failure
    (see the "rioting_real_llm_wording" fixture in
    tests/pipeline_tests/test_evidence_units.py). evidence_score below IS
    this weighted total, not a plain word count, so what's shown to a
    caller is exactly what decided the winner.

    Ranking key, highest wins:
      1. Weighted score (generic lexical matches x1 + number matches x
         DISTINGUISHING_MATCH_WEIGHT) — this is evidence_score
      2. Whether the span contains all of the claim's normalised numbers
      3. Count of matching normalised numbers
      4. Prefer the more specific unit (a joined head+item) over a bare head
         — this attaches a claim that matches a terse list item to that item
         rather than to the general header it also happens to overlap.

    No match (every unit scores 0) and an exact unresolved tie (two or
    more units achieve the identical ranking key) are both treated as "the
    deterministic rules cannot confidently pick a span" — rather than
    quietly keeping an arbitrary first-seen winner, both cases prefer the
    complete original provision as evidence when `source_text` is given
    (whether it actually fits the NLI model is decided later, in
    nli_client.py — this function stays unaware of tokenizer limits).
    Without `source_text`, the claim simply gets no evidence.

    Output fields (API-stable, extended with two additive fields):
      - evidence_span_id  : the single primary selected span (provenance)
      - evidence_span_ids : the full unit — [head, item] for a headless span,
                            otherwise [selected]. This is what the NLI model
                            receives.
      - evidence_text     : the unit premise — exactly the text sent to DeBERTa
      - evidence_score    : integer weighted overlap against the OWN span
                            (generic lexical matches x1 + number matches x
                            DISTINGUISHING_MATCH_WEIGHT) — always a whole number,
                            never a fraction, so the exact-tie check below
                            stays exact
      - evidence_method   : "lexical_overlap" | "full_provision" |
                            "full_provision_ambiguity_fallback" | None
      - evidence_ambiguity: True only when an exact tie was detected
    """
    units = resolve_evidence_units(spans)
    linked = []

    for claim in claims:
        claim_words, claim_numbers = _tokenize_parts(claim["claim_text"])

        best_key = None
        best_unit = None
        best_score = 0
        tie_detected = False

        for unit in units:
            all_word_matches = claim_words & unit["own_lex"]
            distinguishing_word_matches = all_word_matches & DISTINGUISHING_WORDS
            generic_matches = all_word_matches - DISTINGUISHING_WORDS
            number_matches = claim_numbers & unit["own_num"]
            score = (
                len(generic_matches)
                + len(number_matches) * DISTINGUISHING_MATCH_WEIGHT
                + len(distinguishing_word_matches) * DISTINGUISHING_MATCH_WEIGHT
            )
            if score == 0:
                continue

            candidate_key = (
                score,
                claim_numbers.issubset(unit["own_num"]),
                len(number_matches),
                len(unit["ids"]) > 1,
            )

            if best_key is None or candidate_key > best_key:
                best_key = candidate_key
                best_unit = unit
                best_score = score
                tie_detected = False
            elif candidate_key == best_key:
                tie_detected = True

        if best_unit is not None and tie_detected:
            result = _unresolved_result(claim, source_text, "full_provision_ambiguity_fallback")
            result["evidence_ambiguity"] = True
        elif best_unit is None:
            result = _unresolved_result(claim, source_text, "full_provision")
            result["evidence_ambiguity"] = False
        else:
            result = {
                "claim_id": claim["claim_id"],
                "claim_text": claim["claim_text"],
                "evidence_span_id": best_unit["selected_id"],
                "evidence_span_ids": list(best_unit["ids"]),
                "evidence_text": best_unit["premise"],
                "evidence_score": best_score,
                "evidence_method": "lexical_overlap",
                "evidence_ambiguity": False,
            }

        # Carry forward any diagnostic fields the caller already attached
        # to the claim (e.g. claim-extraction's extraction_warnings) —
        # this function only adds evidence-related fields, it must not
        # silently drop what came in.
        if "extraction_warnings" in claim:
            result["extraction_warnings"] = claim["extraction_warnings"]

        linked.append(result)

    return linked
