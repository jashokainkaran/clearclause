"""
Deterministic span-splitting regression tests.

No LLM calls and no model downloads — make_spans is pure regex logic, so
every assertion here is exact and offline.
"""

import pytest

from backend.spans import make_spans


# ---------------------------------------------------------------------------
# Shared fixtures (real statutory wording)
# ---------------------------------------------------------------------------

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

RIOTING_PROVOCATION = (
    "Whoever maliciously or wantonly, by doing anything which is illegal, gives "
    "provocation to any person intending or knowing it to be likely that such "
    "provocation, will cause the offence of rioting to be committed, shall, if the "
    "offence of rioting be committed in consequence of such provocation, be punished "
    "with imprisonment of either description for a term which may extend to one year, "
    "or with fine, or with both; and if the offence of rioting be not committed, with "
    "imprisonment of either description for a term which may extend to six months, or "
    "with fine, or with both."
)


# ---------------------------------------------------------------------------
# 1. Inline legal references must NOT split
#    (regression: the unanchored subsection pattern split mid-sentence)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "A person referred to in subsection (1) shall report the matter to the Court without delay.",
        "The duty described in paragraph (b) applies to every public servant named above.",
        "Nothing in section 12(2) shall affect the powers of the Magistrate under this Act.",
        "Nothing in section 12 (2) shall affect the powers of the Magistrate under this Act.",
    ],
)
def test_inline_reference_does_not_split(text):
    spans = make_spans(text)
    assert len(spans) == 1
    assert spans[0].text == text


def test_fixture_e_nested_inline_references_do_not_split():
    text = "An order made under paragraph (b) of subsection (2) shall apply."
    spans = make_spans(text)
    assert len(spans) == 1
    assert spans[0].text == text


# ---------------------------------------------------------------------------
# 2. Real structural markers MUST still split
# ---------------------------------------------------------------------------

def test_line_start_subsection_markers_still_split():
    text = (
        "(1) Every person who fails to comply shall be liable to a fine.\n"
        "(2) The Court may extend the time allowed for compliance."
    )
    spans = make_spans(text)
    assert len(spans) == 2
    assert spans[0].text.startswith("(1)")
    assert spans[1].text.startswith("(2)")


def test_fixture_d_sentence_separated_subsections_split():
    text = "(1) The Court may make an order. (2) The Magistrate shall record reasons."
    spans = make_spans(text)
    assert len(spans) == 2
    assert spans[0].text.startswith("(1)")
    assert spans[1].text.startswith("(2)")


def test_subsection_marker_after_semicolon_still_splits():
    text = (
        "The following offences are covered: (a) theft of property belonging to another; "
        "(b) dishonest misappropriation of property found by chance."
    )
    spans = make_spans(text)
    assert any(s.text.startswith("(b)") for s in spans)


# ---------------------------------------------------------------------------
# 3. Dash enumerators — em dash and en dash are boundaries, hyphen is not
# ---------------------------------------------------------------------------

def test_em_dash_enumerator_splits():
    text = "as follows— (a) first rule; (b) second rule."
    spans = make_spans(text)
    assert any(s.text.startswith("(a)") for s in spans)
    assert any(s.text.startswith("(b)") for s in spans)


def test_en_dash_enumerator_splits():
    text = "as follows– (a) first rule; (b) second rule."
    spans = make_spans(text)
    assert any(s.text.startswith("(a)") for s in spans)
    assert any(s.text.startswith("(b)") for s in spans)


def test_ordinary_hyphen_is_not_a_boundary():
    # A hyphen must NOT introduce a structural split — it appears inside
    # hyphenated words and ranges.
    text = "the follow-up (a) reference stays inline within this single sentence."
    spans = make_spans(text)
    assert len(spans) == 1


# ---------------------------------------------------------------------------
# 4. Semicolon branch splitting
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("joiner", ["; or, if", "; and if", "; or if", "; if"])
def test_semicolon_branch_variants_split(joiner):
    text = (
        "Whoever fails to appear shall be punished with a fine which may extend to fifty rupees"
        f"{joiner} the failure continues for more than one month, with simple imprisonment "
        "for a term which may extend to six months."
    )
    spans = make_spans(text)
    assert len(spans) == 2


def test_short_branch_is_not_swallowed_by_the_tiny_merge():
    # A branch chunk under 30 chars must survive: the tiny-merge would
    # otherwise silently undo the split.
    text = "Whoever fails to appear shall be punished with a fine; or, if late, with both."
    spans = make_spans(text)
    assert len(spans) == 2
    assert spans[1].text.startswith("or, if")


# ---------------------------------------------------------------------------
# 5 & 6. Things that must NOT split
# ---------------------------------------------------------------------------

def test_bare_semicolon_does_not_split():
    text = (
        "The officer shall record the statement; the statement shall then be read over "
        "to the person who made it."
    )
    spans = make_spans(text)
    assert len(spans) == 1


def test_comma_or_alone_does_not_split():
    text = (
        "The Court may impose a fine, or order imprisonment, or both, depending on the "
        "circumstances of the case."
    )
    spans = make_spans(text)
    assert len(spans) == 1


# ---------------------------------------------------------------------------
# 7. Provided that / unless / except must preserve the exception intact
# ---------------------------------------------------------------------------

def test_provided_that_proviso_is_its_own_span_and_is_complete():
    text = (
        "Every application shall be made within thirty days. Provided that the Court may "
        "extend the period where good cause is shown."
    )
    spans = make_spans(text)
    proviso = [s for s in spans if s.text.lower().startswith("provided that")]
    assert len(proviso) == 1
    assert "extend the period" in proviso[0].text
    assert "good cause is shown" in proviso[0].text


def test_unless_clause_keeps_the_whole_exception_together():
    text = (
        "The application shall not be rejected for an irregularity. Unless the "
        "irregularity has caused material prejudice to a party, the inquiry shall proceed."
    )
    spans = make_spans(text)
    unless = [s for s in spans if s.text.lower().startswith("unless")]
    assert len(unless) == 1
    assert "material prejudice" in unless[0].text


def test_except_clause_is_not_left_as_a_dangling_fragment():
    text = (
        "Every person shall attend the inquiry in person. Except where the Court permits "
        "attendance by an agent, no substitution is allowed."
    )
    spans = make_spans(text)
    for span in spans:
        # No span may be a bare header with no operative content.
        assert len(span.text.split()) > 2


def test_fixture_a_exception_qualifiers_survive_in_one_span():
    # Fixture A: the "before or during" qualifier and the "unless ... material
    # prejudice" exception must both remain present in the spans.
    text = (
        "However, an application under this Act must not be rejected because of any error, "
        "omission, or irregularity in the application, in the affidavit required under "
        "subsection (1), in the summons issued under it, or in other proceedings before or "
        "during an inquiry under this Act, unless the error, omission, or irregularity has "
        "caused material prejudice to a party."
    )
    spans = make_spans(text)
    joined = " ".join(s.text for s in spans)
    assert "before or during" in joined
    assert "material prejudice" in joined
    # The inline "subsection (1)" must not have caused a split.
    assert all("subsection (1)" not in s.text or len(s.text.split()) > 5 for s in spans)


# ---------------------------------------------------------------------------
# 9. Multi-branch punishment provisions stay branch-level
# ---------------------------------------------------------------------------

def test_absconding_provision_yields_exactly_two_branch_spans():
    spans = make_spans(ABSCONDING)
    assert len(spans) == 2


def test_absconding_first_span_is_offence_plus_ordinary_punishment():
    p1 = make_spans(ABSCONDING)[0].text
    assert p1.startswith("Whoever absconds")
    assert "one month" in p1
    assert "fifty rupees" in p1
    assert p1.rstrip().endswith(";")
    assert "six months" not in p1
    assert "one hundred rupees" not in p1


def test_absconding_second_span_is_the_special_branch():
    p2 = make_spans(ABSCONDING)[1].text
    assert p2.startswith("or, if")
    assert "Court of Justice" in p2
    assert "six months" in p2
    assert "one hundred rupees" in p2
    assert "fifty rupees" not in p2


def test_rioting_provocation_branches_are_separated():
    spans = make_spans(RIOTING_PROVOCATION)
    assert len(spans) == 2
    committed, not_committed = spans[0].text, spans[1].text
    assert "one year" in committed
    assert "six months" not in committed
    assert not_committed.startswith("and if")
    assert "be not committed" in not_committed
    assert "six months" in not_committed


# ---------------------------------------------------------------------------
# 10. Role-sensitive text must not be mixed across spans
# ---------------------------------------------------------------------------

def test_distinct_legal_roles_are_not_merged_into_one_span():
    text = (
        "(1) The Court may make an order against a public servant.\n"
        "(2) The Magistrate shall record the reasons given by the police officer.\n"
        "(3) A private citizen may apply for a copy of the order."
    )
    spans = make_spans(text)
    assert len(spans) == 3
    assert "Court" in spans[0].text and "Magistrate" not in spans[0].text
    assert "Magistrate" in spans[1].text and "private citizen" not in spans[1].text
    assert "private citizen" in spans[2].text and "Magistrate" not in spans[2].text


# ---------------------------------------------------------------------------
# Offsets remain within the original text
# ---------------------------------------------------------------------------

def test_span_offsets_stay_within_the_source_text():
    spans = make_spans(ABSCONDING)
    for span in spans:
        assert 0 <= span.start < span.end <= len(ABSCONDING)


def test_empty_text_returns_no_spans():
    assert make_spans("") == []
