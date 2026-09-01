"""
Deterministic pipeline regression tests for the semantic evidence resolver.

These are the Phase-0 regression harness: real provisions, hand-written claims,
and EXACT expected evidence_span_ids. No LLM calls, no model download — every
assertion is computed from make_spans + link_claims_to_spans, so results are
identical on every machine and every run.

The goal is to lock evidence-linking behaviour so future changes can't silently
regress the headless-branch and list-item handling.
"""

import pytest

from backend.spans import make_spans
from backend.evidence import link_claims_to_spans, resolve_evidence_units


# --------------------------------------------------------------------------- #
# Provision texts
# --------------------------------------------------------------------------- #

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

RIOTING = (
    "Whoever maliciously or wantonly, by doing anything which is illegal, gives "
    "provocation to any person intending or knowing it to be likely that such "
    "provocation, will cause the offence of rioting to be committed, shall, if the "
    "offence of rioting be committed in consequence of such provocation, be punished "
    "with imprisonment of either description for a term which may extend to one year, "
    "or with fine, or with both; and if the offence of rioting be not committed, with "
    "imprisonment of either description for a term which may extend to six months, or "
    "with fine, or with both."
)

MAINTENANCE = (
    "However, an application under this Act must not be rejected because of any error, "
    "omission, or irregularity in the application, in the affidavit required under "
    "subsection (1), in the summons issued under it, or in other proceedings before or "
    "during an inquiry under this Act, unless the error, omission, or irregularity has "
    "caused material prejudice to a party."
)

THREE_BRANCH = (
    "Whoever fails to appear shall be punished with a fine which may extend to fifty "
    "rupees; or, if the failure continues for one month, with imprisonment which may "
    "extend to six months; or, if the failure continues beyond one year, with "
    "imprisonment which may extend to two years."
)

ENUM_LIST = (
    "The following are offences under this section— (a) theft of property belonging to "
    "another; (b) dishonest misappropriation of property; (c) criminal breach of trust "
    "by a public servant."
)

NESTED_ENUM = "(1) The following acts are offences: (a) theft; (b) misappropriation."

INLINE_SUBSECTION = (
    "A person referred to in subsection (1) shall report the matter to the Court "
    "without delay."
)

PARAGRAPH_B = "The duty described in paragraph (b) applies to every public servant named above."

SECTION_12_2 = "Nothing in section 12(2) shall affect the powers of the Magistrate under this Act."

STRUCTURAL_MARKERS = "(1) The Court may make an order. (2) The Magistrate shall record reasons."

PROVIDED_THAT = (
    "Every application shall be made within thirty days. Provided that the Court may "
    "extend the period where good cause is shown."
)


# --------------------------------------------------------------------------- #
# (text, expected_span_count, [(claim_id, claim_text, expected_evidence_ids)])
# --------------------------------------------------------------------------- #

FIXTURES = {
    "absconding": (ABSCONDING, 2, [
        ("C1", "A person who absconds to avoid being served with a summons commits this offence.", ["P1"]),
        ("C2", "The ordinary punishment is simple imprisonment which may extend to one month.", ["P1"]),
        ("C3", "In the Court of Justice case, the imprisonment may extend to six months.", ["P1", "P2"]),
        ("C4", "In the Court of Justice case, the fine may extend to one hundred rupees.", ["P1", "P2"]),
    ]),
    "rioting": (RIOTING, 2, [
        ("C1", "If the riot is committed, the person may be punished with imprisonment which may extend to one year.", ["P1"]),
        ("C2", "If the riot is not committed, the person may be punished with imprisonment which may extend to six months.", ["P1", "P2"]),
    ]),
    "maintenance": (MAINTENANCE, 1, [
        ("C1", "An application under this Act must not be rejected for an error, omission, or irregularity.", ["P1"]),
        ("C2", "The rule does not apply if the irregularity caused material prejudice to a party.", ["P1"]),
    ]),
    "three_branch": (THREE_BRANCH, 3, [
        ("C1", "A person who fails to appear may be punished with a fine which may extend to fifty rupees.", ["P1"]),
        ("C2", "If the failure continues for one month, imprisonment may extend to six months.", ["P1", "P2"]),
        ("C3", "If the failure continues beyond one year, imprisonment may extend to two years.", ["P1", "P3"]),
    ]),
    "enum_list": (ENUM_LIST, 4, [
        ("Ca", "Theft of property belonging to another is an offence under this section.", ["P1", "P2"]),
        ("Cb", "Dishonest misappropriation of property is an offence under this section.", ["P1", "P3"]),
        ("Cc", "Criminal breach of trust by a public servant is an offence under this section.", ["P1", "P4"]),
    ]),
    "nested_enum": (NESTED_ENUM, 3, [
        ("Ct", "Theft is one of the offences listed.", ["P1", "P2"]),
        ("Cm", "Misappropriation is one of the offences listed.", ["P1", "P3"]),
    ]),
    "inline_subsection": (INLINE_SUBSECTION, 1, [
        ("C1", "A person referred to in subsection (1) must report the matter to the Court without delay.", ["P1"]),
    ]),
    "paragraph_b": (PARAGRAPH_B, 1, [
        ("C1", "The duty described in paragraph (b) applies to every public servant named above.", ["P1"]),
    ]),
    "section_12_2": (SECTION_12_2, 1, [
        ("C1", "Section 12(2) does not affect the powers of the Magistrate under this Act.", ["P1"]),
    ]),
    "structural_markers": (STRUCTURAL_MARKERS, 2, [
        ("C1", "The Court may make an order.", ["P1"]),
        ("C2", "The Magistrate shall record reasons.", ["P2"]),
    ]),
    "provided_that": (PROVIDED_THAT, 2, [
        ("C1", "Every application must be made within thirty days.", ["P1"]),
        ("C2", "The Court may extend the period where good cause is shown.", ["P2"]),
    ]),
    # Real claim wording from an actual live pipeline run
    # (outputs/runs/run_20260722_091308_55c782.json), not hand-written.
    # This is the exact regression the "rioting" fixture above did NOT
    # catch: the real LLM's claims repeat the head's subject/verb ("the
    # person ... must be punished"), which a flat word count let outweigh
    # the branch-specific "six months"/"not". Before the
    # DISTINGUISHING_MATCH_WEIGHT fix, C5/C6/C7 all resolved to ["P1"]
    # instead of ["P1", "P2"] and showed red/unsupported (see
    # link_claims_to_spans in backend/evidence.py for the full rationale).
    "rioting_real_llm_wording": (RIOTING, 2, [
        ("C1", "If someone acts maliciously or wantonly and does something illegal, intending or knowing that this action might cause a riot, they are responsible if a riot actually happens.", ["P1"]),
        ("C2", "If a riot occurs because of their provocation, they must be punished with imprisonment for up to one year.", ["P1"]),
        ("C3", "If a riot occurs because of their provocation, they must be punished with a fine.", ["P1"]),
        ("C4", "If a riot occurs because of their provocation, they must be punished with both imprisonment for up to one year and a fine.", ["P1"]),
        ("C5", "If the person's provocation does not lead to a riot, they must be punished with imprisonment for up to six months.", ["P1", "P2"]),
        ("C6", "If the person's provocation does not lead to a riot, they must be punished with a fine.", ["P1", "P2"]),
        ("C7", "If the person's provocation does not lead to a riot, they must be punished with both imprisonment for up to six months and a fine.", ["P1", "P2"]),
        ("C9", "The person's actions must be illegal and intended to cause a riot.", ["P1"]),
    ]),
}


def _claims(cases):
    return [{"claim_id": cid, "claim_text": text} for cid, text, _ in cases]


# --------------------------------------------------------------------------- #
# Span counts
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", list(FIXTURES))
def test_span_count(name):
    text, expected_spans, _ = FIXTURES[name]
    assert len(make_spans(text)) == expected_spans


# --------------------------------------------------------------------------- #
# Exact evidence_span_ids per claim
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", list(FIXTURES))
def test_evidence_span_ids(name):
    text, _, cases = FIXTURES[name]
    spans = make_spans(text)
    linked = link_claims_to_spans(_claims(cases), spans)
    got = {c["claim_id"]: c["evidence_span_ids"] for c in linked}
    expected = {cid: ids for cid, _, ids in cases}
    assert got == expected


# --------------------------------------------------------------------------- #
# API-contract invariants that must hold for every claim in every fixture
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", list(FIXTURES))
def test_api_field_contract(name):
    text, _, cases = FIXTURES[name]
    spans = make_spans(text)
    span_ids = {s.span_id for s in spans}
    linked = link_claims_to_spans(_claims(cases), spans)

    for c in linked:
        # evidence_score is always a plain int.
        assert isinstance(c["evidence_score"], int)
        # evidence_span_id is the primary SELECTED span (the item for a headless
        # unit), so it is the LAST id — the head is prepended before it.
        assert c["evidence_span_id"] == c["evidence_span_ids"][-1]
        # for a headless unit the head is the first id.
        assert len(c["evidence_span_ids"]) in (1, 2)
        # every id is a real span id.
        for sid in c["evidence_span_ids"]:
            assert sid in span_ids
        # evidence_text is exactly the joined text of the unit's spans.
        by_id = {s.span_id: s.text for s in spans}
        expected_text = " ".join(by_id[sid] for sid in c["evidence_span_ids"])
        assert c["evidence_text"] == expected_text


# --------------------------------------------------------------------------- #
# Targeted acceptance checks (named, so a failure is self-describing)
# --------------------------------------------------------------------------- #

def _ids_by_claim(name):
    text, _, cases = FIXTURES[name]
    linked = link_claims_to_spans(_claims(cases), make_spans(text))
    return {c["claim_id"]: c["evidence_span_ids"] for c in linked}


def test_rioting_no_riot_claim_uses_p1_p2():
    # The bug this whole change targets: the "riot not committed" claim must
    # be verified against P1 + P2, not P1 alone.
    assert _ids_by_claim("rioting")["C2"] == ["P1", "P2"]


def test_absconding_court_claims_use_p1_p2():
    ids = _ids_by_claim("absconding")
    assert ids["C3"] == ["P1", "P2"]
    assert ids["C4"] == ["P1", "P2"]


def test_absconding_ordinary_claims_stay_single_span():
    ids = _ids_by_claim("absconding")
    assert ids["C1"] == ["P1"]
    assert ids["C2"] == ["P1"]


def test_maintenance_does_not_split_subsection_one():
    assert len(make_spans(MAINTENANCE)) == 1


def test_three_branch_third_branch_skips_sibling():
    # Must be head + third branch, NOT head + second + third.
    assert _ids_by_claim("three_branch")["C3"] == ["P1", "P3"]


def test_enum_list_item_b_skips_sibling_a():
    # Head + (b), never head + (a) + (b).
    assert _ids_by_claim("enum_list")["Cb"] == ["P1", "P3"]


def test_nested_numeric_head_is_usable_for_lettered_items():
    # (1) must act as the head for its (a)/(b) children.
    ids = _ids_by_claim("nested_enum")
    assert ids["Ct"] == ["P1", "P2"]
    assert ids["Cm"] == ["P1", "P3"]


def test_numeric_markers_stay_standalone_when_not_a_list_head():
    # Two independent numbered subsections must NOT merge.
    ids = _ids_by_claim("structural_markers")
    assert ids["C1"] == ["P1"]
    assert ids["C2"] == ["P2"]


def test_real_llm_rioting_wording_no_riot_claims_use_p1_p2():
    # The actual documented failure, locked in with real (not hand-written)
    # claim wording: before the DISTINGUISHING_MATCH_WEIGHT fix, these three
    # resolved to ["P1"] and showed red/unsupported against the wrong branch.
    ids = _ids_by_claim("rioting_real_llm_wording")
    assert ids["C5"] == ["P1", "P2"]
    assert ids["C6"] == ["P1", "P2"]
    assert ids["C7"] == ["P1", "P2"]


def test_real_llm_rioting_wording_committed_claims_stay_on_p1():
    # The fix must not overcorrect: claims genuinely about the "riot
    # committed" branch must stay on P1 alone.
    ids = _ids_by_claim("rioting_real_llm_wording")
    assert ids["C1"] == ["P1"]
    assert ids["C2"] == ["P1"]
    assert ids["C3"] == ["P1"]
    assert ids["C4"] == ["P1"]
    assert ids["C9"] == ["P1"]


# --------------------------------------------------------------------------- #
# Resolver-level edge cases
# --------------------------------------------------------------------------- #

def test_resolver_handles_empty_spans():
    assert resolve_evidence_units([]) == []


def test_headless_first_span_is_left_alone():
    # A continuation fragment with nothing before it must not reach past index 0.
    from backend.schemas import Span
    spans = [Span(span_id="P1", text="or, if the person fails, with a fine.", start=0, end=37)]
    units = resolve_evidence_units(spans)
    assert units[0]["ids"] == ["P1"]
    assert units[0]["premise"] == spans[0].text


def test_unlinked_claim_returns_nulls():
    spans = make_spans(ABSCONDING)
    linked = link_claims_to_spans([{"claim_id": "C1", "claim_text": "zzzz qqqq wwww"}], spans)[0]
    assert linked["evidence_span_id"] is None
    assert linked["evidence_span_ids"] == []
    assert linked["evidence_text"] is None
    assert linked["evidence_score"] == 0
