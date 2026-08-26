import math
import types

import pytest
import torch

from backend import nli_client


class _FakeConfig:
    def __init__(self, id2label):
        self.id2label = id2label


class _FakeModel:
    def __init__(self, logits, id2label):
        self.config = _FakeConfig(id2label)
        self._logits = logits

    def __call__(self, **kwargs):
        return types.SimpleNamespace(logits=self._logits)


class _FakeTokenizer:
    # **kwargs so the fake keeps working as the real call gains arguments.
    # Returns a small, well-under-MAX_LENGTH input_ids tensor so the
    # pre-inference length gate passes by default; tests targeting the
    # length gate itself construct their own tokenizer with a longer one.
    def __call__(self, evidence_text, claim_text, **kwargs):
        return {"input_ids": torch.tensor([[1] * 10])}


def _patch_model(monkeypatch, logits, id2label):
    fake_model = _FakeModel(logits, id2label)
    fake_tokenizer = _FakeTokenizer()
    monkeypatch.setattr(
        nli_client, "get_model_and_tokenizer", lambda: (fake_tokenizer, fake_model)
    )


# ---------------------------------------------------------------------------
# NLI probability behaviour (E) — all fully mocked, no model download.
# ---------------------------------------------------------------------------

def test_all_three_probabilities_returned(monkeypatch):
    logits = torch.tensor([[1.0, 5.0, 0.5]])
    _patch_model(monkeypatch, logits, {0: "contradiction", 1: "entailment", 2: "neutral"})
    _, _, _, probs = nli_client.verify_pair("evidence", "claim")
    assert set(probs.keys()) == {"contradiction", "entailment", "neutral"}


def test_probability_labels_follow_int_id2label(monkeypatch):
    logits = torch.tensor([[5.0, 1.0, 0.5]])
    _patch_model(monkeypatch, logits, {0: "contradiction", 1: "entailment", 2: "neutral"})
    label, raw_label, confidence, probs = nli_client.verify_pair("evidence", "claim")
    assert raw_label == "contradiction"
    assert label == "unsupported"
    assert confidence == pytest.approx(probs["contradiction"])


def test_string_id2label_keys_are_handled(monkeypatch):
    logits = torch.tensor([[0.5, 5.0, 1.0]])
    _patch_model(monkeypatch, logits, {"0": "contradiction", "1": "entailment", "2": "neutral"})
    label, raw_label, confidence, probs = nli_client.verify_pair("evidence", "claim")
    assert raw_label == "entailment"
    assert label == "supported"
    assert confidence == pytest.approx(probs["entailment"])


def test_probabilities_sum_to_approximately_one(monkeypatch):
    logits = torch.tensor([[2.0, 0.1, 1.0]])
    _patch_model(monkeypatch, logits, {0: "contradiction", 1: "entailment", 2: "neutral"})
    _, _, _, probs = nli_client.verify_pair("evidence", "claim")
    assert math.isclose(sum(probs.values()), 1.0, abs_tol=1e-6)


def test_verification_confidence_equals_winning_class_probability(monkeypatch):
    logits = torch.tensor([[0.2, 0.3, 4.0]])
    _patch_model(monkeypatch, logits, {0: "contradiction", 1: "entailment", 2: "neutral"})
    _, raw_label, confidence, probs = nli_client.verify_pair("evidence", "claim")
    assert raw_label == "neutral"
    assert confidence == probs["neutral"]


def test_raw_and_normalised_labels_are_correct():
    assert nli_client.normalise_label("entailment") == "supported"
    assert nli_client.normalise_label("contradiction") == "unsupported"
    assert nli_client.normalise_label("neutral") == "uncertain"


def test_no_threshold_changes_the_predicted_label(monkeypatch):
    # A near-tie, low-confidence win must still be reported as the winning
    # class as-is — no confidence threshold exists yet to downgrade it.
    logits = torch.tensor([[0.34, 0.33, 0.33]])
    _patch_model(monkeypatch, logits, {0: "contradiction", 1: "entailment", 2: "neutral"})
    label, raw_label, confidence, probs = nli_client.verify_pair("evidence", "claim")
    assert raw_label == "contradiction"
    assert label == "unsupported"
    assert confidence < 0.5


# ---------------------------------------------------------------------------
# Input-length gate — checked BEFORE inference, never silently truncated
# ---------------------------------------------------------------------------

class _LengthTokenizer:
    """Reports a fixed token count for the (evidence, claim) pair, ignoring content."""
    def __init__(self, length):
        self._length = length

    def __call__(self, evidence_text, claim_text, **kwargs):
        return {"input_ids": torch.tensor([[1] * self._length])}


def test_length_at_exact_maximum_is_accepted(monkeypatch):
    logits = torch.tensor([[0.1, 5.0, 0.1]])
    fake_model = _FakeModel(logits, {0: "contradiction", 1: "entailment", 2: "neutral"})
    monkeypatch.setattr(
        nli_client, "get_model_and_tokenizer",
        lambda: (_LengthTokenizer(nli_client.MAX_LENGTH), fake_model),
    )
    label, raw_label, confidence, probs = nli_client.verify_pair("evidence", "claim")
    assert raw_label == "entailment"


def test_length_over_maximum_is_rejected(monkeypatch):
    fake_model = _FakeModel(torch.tensor([[0.1, 5.0, 0.1]]), {0: "contradiction", 1: "entailment", 2: "neutral"})
    monkeypatch.setattr(
        nli_client, "get_model_and_tokenizer",
        lambda: (_LengthTokenizer(nli_client.MAX_LENGTH + 1), fake_model),
    )
    with pytest.raises(nli_client.InputTooLongError):
        nli_client.verify_pair("evidence", "claim")


# ---------------------------------------------------------------------------
# Output-sanity validation — a call can succeed but return garbage
# ---------------------------------------------------------------------------

def test_nan_probabilities_raise_invalid_output_error(monkeypatch):
    logits = torch.tensor([[float("nan"), 1.0, 0.5]])
    _patch_model(monkeypatch, logits, {0: "contradiction", 1: "entailment", 2: "neutral"})
    with pytest.raises(nli_client.InvalidOutputError):
        nli_client.verify_pair("evidence", "claim")


# ---------------------------------------------------------------------------
# verify_pair_safe — the shared safe path used by /verify AND verify_claims
# ---------------------------------------------------------------------------

def test_verify_pair_safe_returns_no_evidence_reason_without_calling_model(monkeypatch):
    called = {"n": 0}
    monkeypatch.setattr(nli_client, "get_model_and_tokenizer", lambda: called.update(n=called["n"] + 1) or (None, None))
    result = nli_client.verify_pair_safe(None, "claim text")
    assert result["verification_label"] == "unverified"
    assert result["verification_reason"] == "no_evidence"
    assert called["n"] == 0


def test_verify_pair_safe_success_has_no_reason(monkeypatch):
    logits = torch.tensor([[0.1, 5.0, 0.1]])
    _patch_model(monkeypatch, logits, {0: "contradiction", 1: "entailment", 2: "neutral"})
    result = nli_client.verify_pair_safe("evidence", "claim")
    assert result["verification_label"] == "supported"
    assert result["verification_reason"] is None


def test_verify_pair_safe_maps_model_unavailable(monkeypatch):
    def _raise():
        raise nli_client.ModelUnavailableError("boom")
    monkeypatch.setattr(nli_client, "get_model_and_tokenizer", _raise)
    result = nli_client.verify_pair_safe("evidence", "claim")
    assert result["verification_label"] == "unverified"
    assert result["verification_reason"] == "nli_model_unavailable"


def test_verify_pair_safe_maps_input_too_long(monkeypatch):
    fake_model = _FakeModel(torch.tensor([[0.1, 5.0, 0.1]]), {0: "contradiction", 1: "entailment", 2: "neutral"})
    monkeypatch.setattr(
        nli_client, "get_model_and_tokenizer",
        lambda: (_LengthTokenizer(nli_client.MAX_LENGTH + 1), fake_model),
    )
    result = nli_client.verify_pair_safe("evidence", "claim")
    assert result["verification_label"] == "unverified"
    assert result["verification_reason"] == "nli_input_too_long"


def test_verify_pair_safe_maps_invalid_output(monkeypatch):
    logits = torch.tensor([[float("nan"), 1.0, 0.5]])
    _patch_model(monkeypatch, logits, {0: "contradiction", 1: "entailment", 2: "neutral"})
    result = nli_client.verify_pair_safe("evidence", "claim")
    assert result["verification_label"] == "unverified"
    assert result["verification_reason"] == "nli_invalid_output"


def test_verify_pair_safe_maps_generic_inference_failure(monkeypatch):
    def _raise(**kwargs):
        raise RuntimeError("tokenizer exploded")
    fake_model = _FakeModel(torch.tensor([[0.1, 5.0, 0.1]]), {0: "contradiction", 1: "entailment", 2: "neutral"})
    monkeypatch.setattr(nli_client, "get_model_and_tokenizer", lambda: (_raise, fake_model))
    result = nli_client.verify_pair_safe("evidence", "claim")
    assert result["verification_label"] == "unverified"
    assert result["verification_reason"] == "nli_inference_failed"


# ---------------------------------------------------------------------------
# verify_claims — per-claim isolation, and per-run (not permanent) model
# load-failure isolation
# ---------------------------------------------------------------------------

def test_no_evidence_sets_no_evidence_reason():
    claims = [{"claim_id": "C1", "claim_text": "x", "evidence_text": None}]
    verified = nli_client.verify_claims(claims)
    assert verified[0]["verification_label"] == "unverified"
    assert verified[0]["verification_reason"] == "no_evidence"


def test_one_claims_inference_failure_does_not_affect_siblings(monkeypatch):
    good_logits = torch.tensor([[0.1, 5.0, 0.1]])
    good_model = _FakeModel(good_logits, {0: "contradiction", 1: "entailment", 2: "neutral"})

    class _RaisingTokenizer:
        def __call__(self, *args, **kwargs):
            raise RuntimeError("tokenizer exploded")

    calls = {"n": 0}

    def _get(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return _RaisingTokenizer(), good_model
        return _FakeTokenizer(), good_model

    monkeypatch.setattr(nli_client, "get_model_and_tokenizer", _get)

    claims = [
        {"claim_id": "C1", "claim_text": "x", "evidence_text": "evidence 1"},
        {"claim_id": "C2", "claim_text": "y", "evidence_text": "evidence 2"},
    ]
    verified = nli_client.verify_claims(claims)

    assert verified[0]["verification_label"] == "unverified"
    assert verified[0]["verification_reason"] == "nli_inference_failed"
    assert verified[1]["verification_label"] == "supported"
    assert verified[1]["verification_reason"] is None


def test_model_load_failure_is_isolated_per_run_not_process(monkeypatch):
    call_count = {"n": 0}

    def _boom():
        call_count["n"] += 1
        raise OSError("model files not found")

    monkeypatch.setattr(nli_client, "_model", None)
    monkeypatch.setattr(nli_client, "_tokenizer", None)
    monkeypatch.setattr(nli_client, "_load_model_and_tokenizer", _boom)

    claims = [
        {"claim_id": "C1", "claim_text": "x", "evidence_text": "evidence 1"},
        {"claim_id": "C2", "claim_text": "y", "evidence_text": "evidence 2"},
    ]

    verified_run1 = nli_client.verify_claims(claims)
    assert all(c["verification_reason"] == "nli_model_unavailable" for c in verified_run1)
    # Only the first claim in this run attempted the doomed load; the
    # second claim skipped straight to the same reason.
    assert call_count["n"] == 1

    verified_run2 = nli_client.verify_claims(claims)
    assert all(c["verification_reason"] == "nli_model_unavailable" for c in verified_run2)
    # A later /pipeline request's verify_claims() call gets its own fresh
    # attempt — failure is never cached across calls, only within one.
    assert call_count["n"] == 2
