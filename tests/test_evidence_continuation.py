"""
Continuation-branch (multi-span evidence) tests.

A span produced by the semicolon branch splitter is headless: it lacks the
subject and the operative "shall be punished" phrase, which live in the
preceding span. Retrieval must join the immediately PRECEDING span (by index,
not by score) before the text reaches the NLI model.

Fully deterministic — no LLM, no model download.
"""

import pytest

from backend import evidence
from backend.spans import make_spans


ABSCONDING = (
    "Whoever absconds in order to avoid being served with a summons, notice, or order "
    "proceeding from any public servant legally competent, as such public servant, to "
    "issue such summons, notice, or order, shall be punished with simple imprisonment "
    "for a term which may extend to one month, or with fine which may extend to fifty "
    "rupees, or with both; or, if the summons, notice, or order is to attend in person "
    "or by agent, or to produce a document in a Court of Justice, with simple "
    "imprisonment for a term which may extend to six months, or with fine which may "
    "extend to one hundred rupees, or with both."
)


# ---------------------------------------------------------------------------
# The continuation matcher itself
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    ["or, if the summons is to attend", "and if the offence be not committed",
     "or if the person fails", "if the person fails"],
)
def test_continuation_start_matches_branch_openers(text):
    assert evidence._CONTINUATION_START.match(text)


@pytest.mark.parametrize(
    "text",
    ["Whoever absconds in order to avoid", "The Court may make an order",
     "(b) second rule", "ordinary text that merely contains if inside it"],
)
def test_continuation_start_does_not_match_ordinary_spans(text):
    assert not evidence._CONTINUATION_START.match(text)


# ---------------------------------------------------------------------------
# Multi-span evidence on the real absconding provision
# ---------------------------------------------------------------------------

def test_special_branch_claim_gets_previous_plus_selected_evidence():
    spans = make_spans(ABSCONDING)
    assert len(spans) == 2  # precondition

    claims = [{
        "claim_id": "C1",
        "claim_text": (
            "If the summons is to produce a document in a Court of Justice, the "
            "punishment may extend to six months or one hundred rupees."
        ),
    }]
    linked = evidence.link_claims_to_spans(claims, spans)[0]

    # Retrieval provenance still points at the branch span it selected.
    assert linked["evidence_span_id"] == "P2"
    # But the evidence actually sent to the model is P1 + P2.
    assert linked["evidence_span_ids"] == ["P1", "P2"]
    assert linked["evidence_text"] == f"{spans[0].text} {spans[1].text}"
    # The combined premise now contains the operative phrase the branch lacks.
    assert "shall be punished" in linked["evidence_text"]


def test_ordinary_branch_claim_keeps_single_span_evidence():
    spans = make_spans(ABSCONDING)
    claims = [{
        "claim_id": "C1",
        "claim_text": "A person who absconds to avoid being served with a summons commits this offence.",
    }]
    linked = evidence.link_claims_to_spans(claims, spans)[0]

    assert linked["evidence_span_id"] == "P1"
    assert linked["evidence_span_ids"] == ["P1"]
    assert linked["evidence_text"] == spans[0].text


def test_first_span_is_never_expanded_backwards():
    # A continuation-looking span at index 0 has no predecessor to join.
    from backend.schemas import Span

    spans = [Span(span_id="P1", text="if the person fails to appear, the fine applies",
                  start=0, end=46)]
    claims = [{"claim_id": "C1", "claim_text": "the person fails to appear and the fine applies"}]
    linked = evidence.link_claims_to_spans(claims, spans)[0]

    assert linked["evidence_span_ids"] == ["P1"]
    assert linked["evidence_text"] == spans[0].text


def test_unlinked_claim_has_empty_span_id_list():
    spans = make_spans(ABSCONDING)
    claims = [{"claim_id": "C1", "claim_text": "zzzz qqqq wwww"}]
    linked = evidence.link_claims_to_spans(claims, spans)[0]

    assert linked["evidence_span_id"] is None
    assert linked["evidence_span_ids"] == []
    assert linked["evidence_text"] is None


def test_previous_span_is_chosen_by_index_not_by_score():
    # P2 is lexically closer to the claim than P1, but the continuation span
    # P3 must still be joined to its IMMEDIATE predecessor P2 by index.
    from backend.schemas import Span

    spans = [
        Span(span_id="P1", text="The Registrar shall maintain a register of applications.", start=0, end=10),
        Span(span_id="P2", text="Whoever fails to appear shall be punished with a fine of fifty rupees;", start=10, end=20),
        Span(span_id="P3", text="or, if the failure continues, with imprisonment for six months.", start=20, end=30),
    ]
    claims = [{
        "claim_id": "C1",
        "claim_text": "If the failure continues, imprisonment for six months may be imposed.",
    }]
    linked = evidence.link_claims_to_spans(claims, spans)[0]

    assert linked["evidence_span_id"] == "P3"
    assert linked["evidence_span_ids"] == ["P2", "P3"]
    assert linked["evidence_text"].startswith("Whoever fails to appear")
