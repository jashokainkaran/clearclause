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
    # **kwargs so the fake keeps working as the real call gains arguments
    # (max_length was added when the tokenizer limit was made explicit).
    def __call__(self, evidence_text, claim_text, **kwargs):
        return {}


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
