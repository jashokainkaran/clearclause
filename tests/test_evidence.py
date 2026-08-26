import pytest

from backend import evidence
from backend.schemas import Span


def _span(span_id: str, text: str) -> Span:
    return Span(span_id=span_id, text=text, start=0, end=len(text))


# ---------------------------------------------------------------------------
# Numeric preservation (A.1)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("number", ["5", "18", "21", "30", "50"])
def test_short_numbers_survive_tokenization(number):
    tokens = evidence._tokenize(f"the limit is {number} units")
    assert number in tokens


# ---------------------------------------------------------------------------
# Number extraction and normalisation (B)
# ---------------------------------------------------------------------------

def test_comma_formatted_number_normalises_to_plain_integer():
    assert evidence.extract_numbers("a fine of 5,000 rupees") == {"5000"}
    assert evidence.extract_numbers("a fine of 5000 rupees") == {"5000"}


def test_decimal_trailing_zero_normalises_same_as_short_form():
    assert evidence.extract_numbers("interest of 3.50 percent") == {"3.5"}
    assert evidence.extract_numbers("interest of 3.5 percent") == {"3.5"}


def test_different_values_remain_different():
    assert evidence.extract_numbers("18 years") != evidence.extract_numbers("21 years")


def test_comma_number_is_not_fragmented_in_general_tokenizer():
    tokens = evidence._tokenize("a fine of 5,000 rupees")
    assert "5000" in tokens
    assert "5" not in tokens
    assert "000" not in tokens


def test_claim_with_multiple_numbers_is_handled_correctly():
    assert evidence.extract_numbers(
        "between 18 and 21 years, fined 5,000 rupees"
    ) == {"18", "21", "5000"}


def test_claim_with_no_numbers_has_empty_number_set():
    assert evidence.extract_numbers("the officer shall report the matter") == set()


# ---------------------------------------------------------------------------
# Stopword / legal-token behaviour (A.3)
# ---------------------------------------------------------------------------

def test_shall_may_must_not_are_retained_as_tokens():
    tokens = evidence._tokenize(
        "the officer shall report and may appeal, must comply, not otherwise"
    )
    for word in ("shall", "may", "must", "not"):
        assert word in tokens


# ---------------------------------------------------------------------------
# Modality normalisation (C)
# ---------------------------------------------------------------------------

def test_shall_and_must_share_obligation_tag():
    assert evidence.LEGAL_MODAL_OBLIGATION in evidence._tokenize("the officer shall report")
    assert evidence.LEGAL_MODAL_OBLIGATION in evidence._tokenize("the officer must report")


def test_may_and_can_share_permission_tag():
    assert evidence.LEGAL_MODAL_PERMISSION in evidence._tokenize("the officer may appeal")
    assert evidence.LEGAL_MODAL_PERMISSION in evidence._tokenize("the officer can appeal")


def test_shall_not_and_must_not_share_prohibition_tag_only():
    for phrase in ("the officer shall not disclose", "the officer must not disclose"):
        tokens = evidence._tokenize(phrase)
        assert evidence.LEGAL_MODAL_PROHIBITION in tokens
        assert evidence.LEGAL_MODAL_OBLIGATION not in tokens


def test_may_not_and_cannot_share_prohibition_tag_only():
    for phrase in (
        "the officer may not disclose",
        "the officer cannot disclose",
        "the officer can not disclose",
    ):
        tokens = evidence._tokenize(phrase)
        assert evidence.LEGAL_MODAL_PROHIBITION in tokens
        assert evidence.LEGAL_MODAL_PERMISSION not in tokens


def test_multiword_modality_detected_before_single_word_rule():
    # One genuine obligation and one separate, negated prohibition in the
    # same sentence must legitimately receive both canonical tags.
    tokens = evidence._tokenize(
        "the officer shall report the matter but shall not disclose it"
    )
    assert evidence.LEGAL_MODAL_OBLIGATION in tokens
    assert evidence.LEGAL_MODAL_PROHIBITION in tokens


# ---------------------------------------------------------------------------
# Retrieval / ranking behaviour (D)
# ---------------------------------------------------------------------------

def test_highest_raw_overlap_span_is_selected():
    claims = [{"claim_id": "C1", "claim_text": "the officer shall report the incident within 30 days"}]
    spans = [
        _span("P1", "the officer shall report the incident"),
        _span("P2", "the weather today is sunny"),
    ]
    linked = evidence.link_claims_to_spans(claims, spans)
    assert linked[0]["evidence_span_id"] == "P1"


def test_number_agreement_used_only_when_lexical_overlap_tied():
    # Both spans now share IDENTICAL lexical overlap (penalty, units), so the
    # raw score genuinely ties and the number tie-break is what decides.
    #
    # This fixture was updated when the number double-count was fixed: numbers
    # no longer inflate the raw overlap score, so the previous fixture (which
    # relied on a number to create the tie) no longer tied at all.
    claims = [{"claim_id": "C1", "claim_text": "the penalty is 18 units exactly"}]
    spans = [
        # Identical lexical overlap, wrong number.
        _span("P1", "the penalty is 21 units precisely"),
        # Identical lexical overlap, correct number.
        _span("P2", "the penalty is 18 units broadly"),
    ]
    linked = evidence.link_claims_to_spans(claims, spans)
    assert linked[0]["evidence_span_id"] == "P2"


def test_numbers_are_not_counted_twice_in_the_raw_score():
    # Regression for evidence-retrieval-number-audit.md: a shared number must
    # contribute to evidence_score exactly ONCE, not twice (once as a plain
    # lexical token, once again in the numeric tie-break tiers). Claim and
    # span share one lexical token ("penalty") plus the number 18.
    #
    # Numbers are now deliberately weighted (DISTINGUISHING_MATCH_WEIGHT) rather than
    # excluded from the score entirely -- see link_claims_to_spans and
    # verification_archive/semantic_evidence_resolver_report.md for why a
    # flat, number-blind score let a shared-vocabulary lead beat the one
    # number that actually distinguishes two sibling branches. So the
    # expected value here is 1 generic match + 1 number match x
    # DISTINGUISHING_MATCH_WEIGHT, not the old flat "1" -- what stays true, and what
    # this test still checks, is that "18" is counted exactly once.
    claims = [{"claim_id": "C1", "claim_text": "the penalty is 18"}]
    spans = [_span("P1", "the penalty was 18")]
    linked = evidence.link_claims_to_spans(claims, spans)
    assert linked[0]["evidence_score"] == 1 + evidence.DISTINGUISHING_MATCH_WEIGHT


def test_number_mismatch_does_not_discard_topical_span():
    claims = [{"claim_id": "C1", "claim_text": "the fine is 50,000 rupees"}]
    spans = [_span("P1", "the fine is 5,000 rupees")]
    linked = evidence.link_claims_to_spans(claims, spans)
    assert linked[0]["evidence_span_id"] == "P1"
    assert linked[0]["evidence_score"] > 0


def test_equal_scores_without_source_text_yield_no_evidence_and_ambiguity_flag():
    # An exact, deterministically unresolved tie is no longer silently
    # resolved by "first span wins" — that's an arbitrary choice the NLI
    # model would then be asked to verify against. Without a source_text
    # to fall back on, the claim simply gets no evidence.
    claims = [{"claim_id": "C1", "claim_text": "the officer shall report the matter"}]
    spans = [
        _span("P1", "the officer shall report the matter today"),
        _span("P2", "the officer shall report the matter tomorrow"),
    ]
    linked = evidence.link_claims_to_spans(claims, spans)
    assert linked[0]["evidence_span_id"] is None
    assert linked[0]["evidence_text"] is None
    assert linked[0]["evidence_ambiguity"] is True


def test_equal_scores_with_source_text_use_full_provision_ambiguity_fallback():
    claims = [{"claim_id": "C1", "claim_text": "the officer shall report the matter"}]
    spans = [
        _span("P1", "the officer shall report the matter today"),
        _span("P2", "the officer shall report the matter tomorrow"),
    ]
    source_text = "the officer shall report the matter today. the officer shall report the matter tomorrow."
    linked = evidence.link_claims_to_spans(claims, spans, source_text=source_text)
    assert linked[0]["evidence_span_id"] is None
    assert linked[0]["evidence_text"] == source_text
    assert linked[0]["evidence_method"] == "full_provision_ambiguity_fallback"
    assert linked[0]["evidence_ambiguity"] is True


def test_no_number_claim_gets_no_artificial_number_bonus():
    claims = [{"claim_id": "C1", "claim_text": "the officer shall report the matter"}]
    spans = [
        _span("P1", "the officer shall report the matter clearly today"),
        _span("P2", "the officer shall report the matter clearly on 5 occasions"),
    ]
    linked = evidence.link_claims_to_spans(claims, spans)
    # Both spans must genuinely tie on raw overlap — P2 must not win
    # outright purely because it happens to contain a number the claim
    # never mentioned. A real tie is now reported as unresolved ambiguity,
    # not silently awarded to either span.
    assert linked[0]["evidence_ambiguity"] is True
    assert linked[0]["evidence_span_id"] is None


def test_no_match_with_source_text_uses_full_provision_fallback():
    claims = [{"claim_id": "C1", "claim_text": "the weather today is sunny"}]
    spans = [_span("P1", "the officer shall report the matter")]
    source_text = "the officer shall report the matter"
    linked = evidence.link_claims_to_spans(claims, spans, source_text=source_text)
    assert linked[0]["evidence_span_id"] is None
    assert linked[0]["evidence_text"] == source_text
    assert linked[0]["evidence_method"] == "full_provision"
    assert linked[0]["evidence_ambiguity"] is False


def test_empty_claims_list_is_handled_safely():
    assert evidence.link_claims_to_spans([], [_span("P1", "some text")]) == []


# ---------------------------------------------------------------------------
# Word-number recognition (E) — _word_number_value / _WORD_NUMBER_PATTERN,
# added alongside the DISTINGUISHING_MATCH_WEIGHT fix. See
# verification_archive/semantic_evidence_resolver_report.md and
# HANDOFF.md's "Evidence-linking: distinguishing-token weighting".
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("word,expected", [
    ("six", 6),
    ("twelve", 12),
    ("thirteen", 13),
    ("twenty", 20),
    ("forty", 40),
    ("ninety", 90),
    ("zero", 0),
])
def test_word_number_value_recognises_cardinals(word, expected):
    assert evidence._word_number_value(word) == expected


@pytest.mark.parametrize("word,expected", [
    ("first", 1),
    ("second", 2),
    ("third", 3),
    ("tenth", 10),
    ("twentieth", 20),
])
def test_word_number_value_recognises_ordinals(word, expected):
    assert evidence._word_number_value(word) == expected


@pytest.mark.parametrize("word,expected", [
    ("twenty-one", 21),
    ("twenty-five", 25),
    ("forty-two", 42),
    ("ninety-nine", 99),
])
def test_word_number_value_recognises_hyphenated_compounds(word, expected):
    assert evidence._word_number_value(word) == expected


def test_word_number_value_excludes_bare_one():
    # Deliberate exclusion: "one" is overloaded as an indefinite pronoun
    # ("one of the offences") far more often than a genuine count in this
    # domain. Recognising it as a number caused a false numeric-agreement
    # collision against an unrelated subsection marker "(1)" — see
    # tests/pipeline_tests/test_evidence_units.py's nested_enum fixture.
    assert evidence._word_number_value("one") is None


def test_word_number_value_still_recognises_one_inside_a_compound():
    # "twenty-one" is unambiguously numeric even though bare "one" is not.
    assert evidence._word_number_value("twenty-one") == 21


def test_word_number_value_returns_none_for_unrecognised_words():
    assert evidence._word_number_value("cat") is None
    assert evidence._word_number_value("hundred") is None  # deliberately out of scope


def test_extract_numbers_includes_word_numbers_alongside_digits():
    result = evidence.extract_numbers(
        "six months, one year, twenty-five days, the first offence, or 6 units"
    )
    # "six" -> 6 (same canonical value as digit "6"), "twenty-five" -> 25,
    # "first" -> 1. "one" contributes nothing (excluded).
    assert result == {"1", "6", "25"}


def test_tokenize_parts_masks_word_numbers_out_of_the_lexical_set():
    lexical, numbers = evidence._tokenize_parts("extend to six months")
    assert "six" not in lexical
    assert numbers == {"6"}
    assert lexical == {"extend", "months"}


# ---------------------------------------------------------------------------
# Distinguishing-token weighting ambiguity (F) — confirms the new weighted
# score (numbers/"not" at DISTINGUISHING_MATCH_WEIGHT, everything else at 1)
# still produces an exact, safely-detected tie rather than an arbitrary
# pick, and that the weighting stays in exact integer arithmetic (no
# floating-point near-misses breaking the tie check).
# ---------------------------------------------------------------------------

def test_weighted_tie_on_shared_number_is_still_reported_as_ambiguous():
    # Both spans share the same 2 generic words ("penalty", "months") and
    # the same distinguishing number ("six"/"6") with the claim — an exact
    # tie under the new weighted score (2 + 1*DISTINGUISHING_MATCH_WEIGHT
    # for each), not just under the old flat count.
    claims = [{"claim_id": "C1", "claim_text": "the penalty is six months exactly"}]
    spans = [
        _span("P1", "the penalty was six months precisely"),
        _span("P2", "the sentence penalty is six months broadly"),
    ]
    linked = evidence.link_claims_to_spans(claims, spans)
    assert linked[0]["evidence_ambiguity"] is True
    assert linked[0]["evidence_span_id"] is None


def test_distinguishing_number_breaks_a_tie_that_would_otherwise_be_ambiguous():
    # Same setup as above, but P2's number now disagrees with the claim.
    # The shared distinguishing number must let P1 win outright instead of
    # falling into the ambiguity fallback.
    claims = [{"claim_id": "C1", "claim_text": "the penalty is six months exactly"}]
    spans = [
        _span("P1", "the penalty was six months precisely"),
        _span("P2", "the sentence penalty is twelve months broadly"),
    ]
    linked = evidence.link_claims_to_spans(claims, spans)
    assert linked[0]["evidence_ambiguity"] is False
    assert linked[0]["evidence_span_id"] == "P1"


def test_empty_span_list_is_handled_safely():
    claims = [{"claim_id": "C1", "claim_text": "the officer shall report the matter"}]
    linked = evidence.link_claims_to_spans(claims, [])
    assert linked[0]["evidence_span_id"] is None
    assert linked[0]["evidence_text"] is None
    assert linked[0]["evidence_score"] == 0
