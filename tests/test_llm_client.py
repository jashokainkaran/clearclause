"""
Deterministic tests for claim JSON parsing, validation and renumbering.

No live Qwen calls. Importing backend.llm_client constructs an InferenceClient
object, but that performs no network request, so these tests stay offline.
"""

import json

import pytest

from backend import llm_client


# ---------------------------------------------------------------------------
# _parse_claims_json — strict validation
# ---------------------------------------------------------------------------

def test_valid_json_is_parsed_and_renumbered():
    raw = json.dumps([
        {"claim_id": "X9", "claim_text": "A person who absconds commits this offence."},
        {"claim_id": "zzz", "claim_text": "The punishment may extend to one month."},
    ])
    claims = llm_client._parse_claims_json(raw)
    assert [c["claim_id"] for c in claims] == ["C1", "C2"]
    assert claims[0]["claim_text"] == "A person who absconds commits this offence."


def test_markdown_fenced_json_is_still_parsed():
    raw = '```json\n[{"claim_id": "C1", "claim_text": "The Court may impose a fine."}]\n```'
    claims = llm_client._parse_claims_json(raw)
    assert claims == [{"claim_id": "C1", "claim_text": "The Court may impose a fine."}]


def test_claim_text_is_stripped():
    raw = json.dumps([{"claim_id": "C1", "claim_text": "   padded claim text   "}])
    assert llm_client._parse_claims_json(raw)[0]["claim_text"] == "padded claim text"


@pytest.mark.parametrize(
    "bad_claim_text",
    [None, "", "   ", 42, 3.5, ["a"], {"a": 1}],
)
def test_invalid_claim_text_is_rejected(bad_claim_text):
    # Regression: a null / empty / non-string claim_text previously passed
    # validation and then crashed in evidence._tokenize on text.lower().
    raw = json.dumps([{"claim_id": "C1", "claim_text": bad_claim_text}])
    assert llm_client._parse_claims_json(raw) is None


@pytest.mark.parametrize("bad_claim_id", [None, 1, 2.0, ["C1"]])
def test_non_string_claim_id_is_rejected(bad_claim_id):
    raw = json.dumps([{"claim_id": bad_claim_id, "claim_text": "A valid claim."}])
    assert llm_client._parse_claims_json(raw) is None


def test_missing_keys_are_rejected():
    assert llm_client._parse_claims_json(json.dumps([{"claim_text": "no id"}])) is None
    assert llm_client._parse_claims_json(json.dumps([{"claim_id": "C1"}])) is None


def test_non_list_and_empty_list_are_rejected():
    assert llm_client._parse_claims_json(json.dumps({"claim_id": "C1", "claim_text": "x"})) is None
    assert llm_client._parse_claims_json("[]") is None


def test_malformed_json_is_rejected():
    assert llm_client._parse_claims_json("not json at all") is None
    assert llm_client._parse_claims_json('[{"claim_id": "C1", "claim_text": ') is None


def test_one_bad_entry_rejects_the_whole_response():
    raw = json.dumps([
        {"claim_id": "C1", "claim_text": "A valid claim."},
        {"claim_id": "C2", "claim_text": None},
    ])
    assert llm_client._parse_claims_json(raw) is None


def test_parsed_claims_never_crash_the_tokenizer():
    # End-to-end guard for the AttributeError path: anything that survives
    # _parse_claims_json must be safe to tokenize.
    from backend import evidence

    raw = json.dumps([{"claim_id": "C1", "claim_text": "The Court may impose a fine."}])
    for claim in llm_client._parse_claims_json(raw):
        assert evidence._tokenize(claim["claim_text"])


# ---------------------------------------------------------------------------
# Fallback path — must renumber consistently
# ---------------------------------------------------------------------------

def test_fallback_split_renumbers_sequentially():
    claims = llm_client._fallback_claim_split(
        "The first rule applies. The second rule applies. The third rule applies."
    )
    assert [c["claim_id"] for c in claims] == ["C1", "C2", "C3"]
    assert all(isinstance(c["claim_text"], str) and c["claim_text"] for c in claims)


def test_fallback_ignores_empty_fragments():
    claims = llm_client._fallback_claim_split("Only one sentence here.   ")
    assert len(claims) == 1
    assert claims[0] == {"claim_id": "C1", "claim_text": "Only one sentence here."}


# ---------------------------------------------------------------------------
# Prompt content — the rules the pipeline depends on must be present
# ---------------------------------------------------------------------------

def test_simplification_prompt_forbids_lists_and_requires_paragraphs():
    prompt = llm_client.SIMPLIFY_USER
    assert "No bullet points" in prompt
    assert "No numbered lists" in prompt
    assert "plain paragraphs" in prompt
    assert "blank line between paragraphs" in prompt


def test_simplification_prompt_preserves_legal_qualifiers():
    prompt = llm_client.SIMPLIFY_USER
    for word in ("before", "during", "unless", "except", "not", "must", "may", "shall"):
        assert word in prompt
    assert "Do not merge different punishment branches" in prompt


def test_claim_prompt_requires_punishment_splitting():
    prompt = llm_client.CLAIM_USER
    assert "split separate punishments into separate atomic claims" in prompt
    assert 'Split imprisonment, fine, and "both" into separate claims' in prompt
    assert "Keep different punishment branches separate" in prompt
    assert "independently checkable" in prompt


def test_claim_prompt_no_longer_encourages_broad_compound_claims():
    # The removed rule pushed everything into one claim.
    assert (
        "Section numbers, Act names, dates, ages, fines, imprisonment terms and legal "
        "roles must stay inside the relevant claim." not in llm_client.CLAIM_USER
    )


def test_claim_prompt_protects_legal_roles():
    assert "Keep the legal role exact" in llm_client.CLAIM_USER
