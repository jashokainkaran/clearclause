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
    # not inflate evidence_score. Claim and span share exactly one lexical
    # token ("penalty") plus the number 18 — the score must be 1, not 2.
    claims = [{"claim_id": "C1", "claim_text": "the penalty is 18"}]
    spans = [_span("P1", "the penalty was 18")]
    linked = evidence.link_claims_to_spans(claims, spans)
    assert linked[0]["evidence_score"] == 1


def test_number_mismatch_does_not_discard_topical_span():
    claims = [{"claim_id": "C1", "claim_text": "the fine is 50,000 rupees"}]
    spans = [_span("P1", "the fine is 5,000 rupees")]
    linked = evidence.link_claims_to_spans(claims, spans)
    assert linked[0]["evidence_span_id"] == "P1"
    assert linked[0]["evidence_score"] > 0


def test_equal_scores_retain_first_span():
    claims = [{"claim_id": "C1", "claim_text": "the officer shall report the matter"}]
    spans = [
        _span("P1", "the officer shall report the matter today"),
        _span("P2", "the officer shall report the matter tomorrow"),
    ]
    linked = evidence.link_claims_to_spans(claims, spans)
    assert linked[0]["evidence_span_id"] == "P1"


def test_no_number_claim_gets_no_artificial_number_bonus():
    claims = [{"claim_id": "C1", "claim_text": "the officer shall report the matter"}]
    spans = [
        _span("P1", "the officer shall report the matter clearly today"),
        _span("P2", "the officer shall report the matter clearly on 5 occasions"),
    ]
    linked = evidence.link_claims_to_spans(claims, spans)
    # Both spans tie on raw overlap; P2 must not win purely because it
    # happens to contain a number the claim never mentioned.
    assert linked[0]["evidence_span_id"] == "P1"


def test_empty_claims_list_is_handled_safely():
    assert evidence.link_claims_to_spans([], [_span("P1", "some text")]) == []


def test_empty_span_list_is_handled_safely():
    claims = [{"claim_id": "C1", "claim_text": "the officer shall report the matter"}]
    linked = evidence.link_claims_to_spans(claims, [])
    assert linked[0]["evidence_span_id"] is None
    assert linked[0]["evidence_text"] is None
    assert linked[0]["evidence_score"] == 0
