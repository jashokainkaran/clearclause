"""
Deterministic tests for claim JSON parsing, validation and renumbering.

No live Qwen calls. Importing backend.llm_client constructs an InferenceClient
object, but that performs no network request, so these tests stay offline.
"""

import json

import pytest

from backend import llm_client


class _FakeResponse:
    def __init__(self, status_code):
        self.status_code = status_code


class _FakeHTTPError(Exception):
    """Stands in for the HfHub HTTP error shape llm_client inspects (`.response.status_code`)."""
    def __init__(self, status_code, message="http error"):
        super().__init__(message)
        self.response = _FakeResponse(status_code)


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


def test_simplify_prompt_delimits_source_text_as_data():
    assert "<<<PROVISION_TEXT_START>>>" in llm_client.SIMPLIFY_USER
    assert "<<<PROVISION_TEXT_END>>>" in llm_client.SIMPLIFY_USER
    assert "never as instructions" in llm_client.SIMPLIFY_USER


def test_claim_prompt_delimits_simplified_text_as_data():
    assert "<<<SIMPLIFIED_TEXT_START>>>" in llm_client.CLAIM_USER
    assert "<<<SIMPLIFIED_TEXT_END>>>" in llm_client.CLAIM_USER
    assert "never as instructions" in llm_client.CLAIM_USER


# ---------------------------------------------------------------------------
# simplify_with_attempts — bounded technical retry only, no content checks
# ---------------------------------------------------------------------------

def test_simplify_retries_once_on_transient_failure_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def _fake_raw_call(messages):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _FakeHTTPError(429)
        return "Simplified output."

    monkeypatch.setattr(llm_client, "_raw_chat_completion", _fake_raw_call)
    monkeypatch.setattr(llm_client, "SIMPLIFY_RETRY_BACKOFF_SECONDS", 0)

    text, attempts = llm_client.simplify_with_attempts("some provision text")

    assert text == "Simplified output."
    assert attempts == 2
    assert calls["n"] == 2


def test_simplify_does_not_retry_permanent_failure(monkeypatch):
    calls = {"n": 0}

    def _fake_raw_call(messages):
        calls["n"] += 1
        raise _FakeHTTPError(401)

    monkeypatch.setattr(llm_client, "_raw_chat_completion", _fake_raw_call)
    monkeypatch.setattr(llm_client, "SIMPLIFY_RETRY_BACKOFF_SECONDS", 0)

    with pytest.raises(llm_client.SimplificationFailedError) as exc_info:
        llm_client.simplify_with_attempts("some provision text")

    assert calls["n"] == 1
    assert exc_info.value.attempts == 1
    assert "token is invalid" in str(exc_info.value)


def test_simplify_fails_after_exhausting_retries(monkeypatch):
    calls = {"n": 0}

    def _fake_raw_call(messages):
        calls["n"] += 1
        raise _FakeHTTPError(503)

    monkeypatch.setattr(llm_client, "_raw_chat_completion", _fake_raw_call)
    monkeypatch.setattr(llm_client, "SIMPLIFY_RETRY_BACKOFF_SECONDS", 0)

    with pytest.raises(llm_client.SimplificationFailedError) as exc_info:
        llm_client.simplify_with_attempts("some provision text")

    assert calls["n"] == llm_client.SIMPLIFY_MAX_ATTEMPTS
    assert exc_info.value.attempts == llm_client.SIMPLIFY_MAX_ATTEMPTS


def test_simplify_does_not_validate_content(monkeypatch):
    # Deliberate: Stage 2 only asks "did the technical call work?", never
    # "is the output faithful?" — that is claim extraction + evidence +
    # NLI's job, downstream. A technically successful call must be
    # accepted even if it obviously dropped or changed a number.
    monkeypatch.setattr(llm_client, "_raw_chat_completion", lambda messages: "999999999")
    text, attempts = llm_client.simplify_with_attempts("The fine is 5,000 rupees.")
    assert text == "999999999"
    assert attempts == 1


# ---------------------------------------------------------------------------
# extract_claims_with_status — total call budget, structural validation,
# fallback hierarchy, semantic warnings
# ---------------------------------------------------------------------------

def test_claim_extraction_succeeds_via_llm_on_first_call(monkeypatch):
    calls = {"n": 0}

    def _fake_raw_call(messages):
        calls["n"] += 1
        return json.dumps([{"claim_id": "C1", "claim_text": "The Court may impose a fine."}])

    monkeypatch.setattr(llm_client, "_raw_chat_completion", _fake_raw_call)

    claims, status = llm_client.extract_claims_with_status("The Court may impose a fine.")

    assert status == {"status": "success", "method": "llm"}
    assert calls["n"] == 1
    assert claims[0]["claim_text"] == "The Court may impose a fine."
    assert claims[0]["extraction_warnings"] == []


def test_claim_extraction_falls_back_after_malformed_json_twice(monkeypatch):
    calls = {"n": 0}

    def _fake_raw_call(messages):
        calls["n"] += 1
        return "not valid json"

    monkeypatch.setattr(llm_client, "_raw_chat_completion", _fake_raw_call)
    monkeypatch.setattr(llm_client, "CLAIM_RETRY_BACKOFF_SECONDS", 0)

    text = "The first rule applies. The second rule applies."
    claims, status = llm_client.extract_claims_with_status(text)

    assert status == {"status": "fallback", "method": "sentence_splitter"}
    assert len(claims) == 2
    # Malformed JSON is not a transient failure, so each prompt variant is
    # tried exactly once: CLAIM_SYSTEM + CLAIM_RETRY_SYSTEM = 2 calls.
    assert calls["n"] == 2


def test_claim_extraction_retries_transient_failure_then_falls_back(monkeypatch):
    calls = {"n": 0}

    def _fake_raw_call(messages):
        calls["n"] += 1
        raise _FakeHTTPError(503)

    monkeypatch.setattr(llm_client, "_raw_chat_completion", _fake_raw_call)
    monkeypatch.setattr(llm_client, "CLAIM_RETRY_BACKOFF_SECONDS", 0)

    text = "The first rule applies. The second rule applies."
    claims, status = llm_client.extract_claims_with_status(text)

    assert status == {"status": "fallback", "method": "sentence_splitter"}
    assert len(claims) == 2
    # Total budget is 3: CLAIM_SYSTEM attempt 1, its transient retry
    # (attempt 2), then CLAIM_RETRY_SYSTEM gets only the remaining budget
    # (attempt 3) before the budget is exhausted.
    assert calls["n"] == llm_client.CLAIM_MAX_TOTAL_CALLS


def test_claim_extraction_permanent_failure_stops_without_trying_second_prompt(monkeypatch):
    calls = {"n": 0}

    def _fake_raw_call(messages):
        calls["n"] += 1
        raise _FakeHTTPError(401)

    monkeypatch.setattr(llm_client, "_raw_chat_completion", _fake_raw_call)
    monkeypatch.setattr(llm_client, "CLAIM_RETRY_BACKOFF_SECONDS", 0)

    text = "The first rule applies. The second rule applies."
    claims, status = llm_client.extract_claims_with_status(text)

    assert status == {"status": "fallback", "method": "sentence_splitter"}
    # A permanent failure will not be fixed by a different prompt, so only
    # one call is made in total, not the full budget.
    assert calls["n"] == 1


def test_claim_extraction_never_exceeds_total_call_budget(monkeypatch):
    calls = {"n": 0}

    def _fake_raw_call(messages):
        calls["n"] += 1
        raise _FakeHTTPError(429)  # always transient

    monkeypatch.setattr(llm_client, "_raw_chat_completion", _fake_raw_call)
    monkeypatch.setattr(llm_client, "CLAIM_RETRY_BACKOFF_SECONDS", 0)

    llm_client.extract_claims_with_status("Some text with no punctuation markers at all")

    assert calls["n"] == llm_client.CLAIM_MAX_TOTAL_CALLS


def test_empty_claims_list_is_structurally_invalid():
    assert llm_client._apply_structural_validation([]) is None
    assert llm_client._apply_structural_validation(None) is None


def test_claim_extraction_dedups_exact_duplicate_claims():
    claims = [
        {"claim_id": "C1", "claim_text": "The fine is 5,000 rupees."},
        {"claim_id": "C2", "claim_text": "The fine is 5,000 rupees."},
        {"claim_id": "C3", "claim_text": "Imprisonment may also be ordered."},
    ]
    result = llm_client._apply_structural_validation(claims)
    assert [c["claim_text"] for c in result] == [
        "The fine is 5,000 rupees.",
        "Imprisonment may also be ordered.",
    ]
    assert [c["claim_id"] for c in result] == ["C1", "C2"]


# ---------------------------------------------------------------------------
# Fallback tier selection — each tier must be individually reachable
# ---------------------------------------------------------------------------

def test_clause_splitter_tier_used_when_it_actually_splits_something(monkeypatch):
    monkeypatch.setattr(llm_client, "_attempt_claim_calls", lambda t: None)
    text = "Fine shall be paid; and imprisonment may be ordered."
    claims, status = llm_client.extract_claims_with_status(text)
    assert status["method"] == "clause_splitter"
    assert len(claims) == 2


def test_sentence_splitter_tier_used_when_no_clause_markers_exist(monkeypatch):
    monkeypatch.setattr(llm_client, "_attempt_claim_calls", lambda t: None)
    text = "The first rule applies. The second rule applies."
    claims, status = llm_client.extract_claims_with_status(text)
    assert status["method"] == "sentence_splitter"
    assert len(claims) == 2


def test_whole_text_claim_tier_used_when_text_has_no_clause_or_sentence_boundaries(monkeypatch):
    monkeypatch.setattr(llm_client, "_attempt_claim_calls", lambda t: None)
    text = "This is one single sentence with no punctuation markers"
    claims, status = llm_client.extract_claims_with_status(text)
    assert status["method"] == "whole_text_claim"
    assert len(claims) == 1
    assert claims[0]["claim_text"] == text


# ---------------------------------------------------------------------------
# Per-claim semantic extraction_warnings — diagnostic only, never block
# ---------------------------------------------------------------------------

def test_extraction_warning_for_unsupported_number(monkeypatch):
    monkeypatch.setattr(
        llm_client, "_attempt_claim_calls",
        lambda t: [{"claim_id": "C1", "claim_text": "The fine is 50,000 rupees."}],
    )
    claims, _status = llm_client.extract_claims_with_status("The fine is 5,000 rupees.")
    assert "unsupported_number" in claims[0]["extraction_warnings"]
    # Diagnostic only — the claim is still returned unchanged.
    assert claims[0]["claim_text"] == "The fine is 50,000 rupees."


def test_extraction_warning_for_zero_lexical_overlap(monkeypatch):
    monkeypatch.setattr(
        llm_client, "_attempt_claim_calls",
        lambda t: [{"claim_id": "C1", "claim_text": "Completely unrelated wording here."}],
    )
    claims, _status = llm_client.extract_claims_with_status("The Court may impose a fine.")
    assert "zero_lexical_overlap" in claims[0]["extraction_warnings"]


def test_extraction_warning_for_clear_modal_conflict(monkeypatch):
    monkeypatch.setattr(
        llm_client, "_attempt_claim_calls",
        lambda t: [{"claim_id": "C1", "claim_text": "The officer may disclose the report."}],
    )
    simplified = "The officer must not disclose the report."
    claims, _status = llm_client.extract_claims_with_status(simplified)
    assert "clear_modal_conflict" in claims[0]["extraction_warnings"]


def test_no_warnings_for_a_faithful_claim(monkeypatch):
    monkeypatch.setattr(
        llm_client, "_attempt_claim_calls",
        lambda t: [{"claim_id": "C1", "claim_text": "The fine is 5,000 rupees."}],
    )
    claims, _status = llm_client.extract_claims_with_status("The fine is 5,000 rupees.")
    assert claims[0]["extraction_warnings"] == []
