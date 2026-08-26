import math
import os

import torch
import torch.nn.functional as F
from dotenv import load_dotenv

# Load .env explicitly from the backend directory, matching llm_client.py's
# approach — keeps this module independently configurable even if it's
# imported before llm_client.py (e.g. directly in a test).
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path)

# Model source is configurable so a locally stored fine-tuned checkpoint can
# be swapped in later with no code change — just point this at a local
# directory path instead of a Hugging Face hub id.
NLI_MODEL_PATH = os.getenv("NLI_MODEL_PATH", "cross-encoder/nli-deberta-v3-small")

# Lazy loaded globals
_tokenizer = None
_model = None

# Explicit tokenizer limit. cross-encoder/nli-deberta-v3-small accepts 512
# tokens; stating it here makes the limit visible and keeps it aligned with
# the offline evaluation harness instead of relying on a model default (some
# Hugging Face tokenizers report a huge sentinel value for
# model_max_length that is not the model's real usable limit).
#
# This matters more now that a continuation branch is verified against the
# PREVIOUS span joined to the selected span, and now that a full-provision
# evidence fallback can be selected — both make a longer premise more
# likely.
MAX_LENGTH = 512

# Standard cross-encoder NLI class ordering — used only when the model
# config doesn't provide a usable id2label mapping.
_FALLBACK_ID2LABEL = {0: "contradiction", 1: "entailment", 2: "neutral"}


class ModelUnavailableError(Exception):
    """
    Raised when the NLI model/tokenizer cannot be loaded. Distinguished
    from a normal per-claim inference failure so callers can report
    verification_reason="nli_model_unavailable" instead of
    "nli_inference_failed".
    """
    pass


class InputTooLongError(Exception):
    """
    Raised when an (evidence, claim) pair exceeds MAX_LENGTH tokens.
    Previously this only produced a warning and the pair was still
    silently truncated and scored; that risked dropping a condition,
    exception, or number that changes the legal meaning without anyone
    knowing. Now verification is refused instead.
    """
    pass


class InvalidOutputError(Exception):
    """
    Raised when the model call returns without error but the output is
    numerically invalid (missing class, NaN/inf, out of [0,1], doesn't sum
    to ~1, or confidence doesn't match the winning class). A model call
    can technically succeed while still producing garbage output.
    """
    pass


def _load_model_and_tokenizer():
    """
    The actual model/tokenizer load, isolated into its own function so:
    (a) tests can mock a load failure without a real model download, and
    (b) a load failure can be distinguished from a normal inference
    failure by get_model_and_tokenizer.
    """
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    print(f"Loading NLI model and tokenizer from '{NLI_MODEL_PATH}'...")
    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_PATH)
    print("NLI model and tokenizer loaded successfully.")
    return tokenizer, model


def get_model_and_tokenizer():
    """
    Lazy singleton: loaded once per process on success (unchanged from
    before — not re-downloaded per run). On failure, _model/_tokenizer are
    left as None, so the very next call — in this run or a later one —
    simply retries the load. Failure is never cached at this level; the
    "don't retry the same doomed load for every claim in one run" behaviour
    lives in verify_claims() instead, scoped to that one call only, so a
    transient failure never becomes permanent until a process restart.
    """
    global _tokenizer, _model
    if _model is None or _tokenizer is None:
        try:
            _tokenizer, _model = _load_model_and_tokenizer()
        except Exception as e:
            raise ModelUnavailableError(str(e)) from e
    return _tokenizer, _model


def normalise_label(label: str) -> str:
    """
    Normalises raw NLI labels to ClearClause status labels.
    entailment -> supported
    contradiction -> unsupported
    neutral -> uncertain

    Any future confidence-threshold rule belongs in verify_pair, after the
    class and confidence are known — not here.
    """
    lbl = label.strip().lower()
    if lbl == "entailment":
        return "supported"
    elif lbl == "contradiction":
        return "unsupported"
    elif lbl == "neutral":
        return "uncertain"
    return "uncertain"


def _resolve_id2label(model_config) -> dict[int, str]:
    """
    Reads the model's actual id2label mapping, normalising keys (which may
    be int or str depending on how the config was loaded) and values
    (lowercased label names). Falls back to the standard cross-encoder
    ordering only if the config doesn't provide a usable mapping.
    """
    raw = getattr(model_config, "id2label", None)
    if raw and isinstance(raw, dict):
        return {int(idx): str(label).strip().lower() for idx, label in raw.items()}
    return dict(_FALLBACK_ID2LABEL)


def _tokenize_pair_or_raise(tok, evidence_text: str, claim_text: str):
    """
    Tokenizes the (evidence, claim) pair WITHOUT truncation first, so the
    real length can be checked against MAX_LENGTH before any inference
    happens. Raises InputTooLongError rather than truncating — this
    replaces the old behaviour of warning and still running inference on
    truncated text. If the pair fits, this same tokenize result is reused
    directly as the model input (no second, redundant tokenize call).
    """
    inputs = tok(evidence_text, claim_text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    length = input_ids.shape[-1] if hasattr(input_ids, "shape") else len(input_ids[0])
    if length > MAX_LENGTH:
        raise InputTooLongError(
            f"Evidence+claim pair is {length} tokens, exceeding the model's "
            f"usable input length of {MAX_LENGTH}."
        )
    return inputs


def _validate_output_or_raise(nli_probabilities: dict, confidence: float) -> None:
    """
    Sanity-checks the model output after inference. A model call can
    technically return without raising while still producing numerically
    invalid output (e.g. NaN) — this catches that rather than reporting a
    garbage label as if it were a real verdict.
    """
    if len(nli_probabilities) != 3:
        raise InvalidOutputError(f"Expected 3 class probabilities, got {len(nli_probabilities)}.")

    total = 0.0
    for label, value in nli_probabilities.items():
        if value is None or not math.isfinite(value):
            raise InvalidOutputError(f"Non-finite probability for '{label}': {value}")
        if value < 0.0 or value > 1.0:
            raise InvalidOutputError(f"Probability for '{label}' out of range: {value}")
        total += value

    if not math.isclose(total, 1.0, abs_tol=1e-3):
        raise InvalidOutputError(f"Probabilities sum to {total}, expected ~1.0.")

    if not math.isclose(confidence, max(nli_probabilities.values()), abs_tol=1e-6):
        raise InvalidOutputError("Reported confidence is not the maximum class probability.")


def verify_pair(evidence_text: str, claim_text: str):
    """
    Tokenizes the evidence and claim as a pair, runs the NLI model, and
    returns (verification_label, nli_raw_label, verification_confidence,
    nli_probabilities).

    Raises ModelUnavailableError if the model/tokenizer cannot be loaded,
    InputTooLongError if the pair exceeds MAX_LENGTH (checked BEFORE
    inference — never silently truncated), or InvalidOutputError if the
    model call succeeds but the output is numerically invalid. Any other
    failure propagates as a plain exception. verify_pair_safe is what
    turns all of these into a stable dict — call that instead unless you
    specifically need the exceptions.

    nli_probabilities carries the full 3-class softmax distribution keyed
    by lowercase label name (contradiction / entailment / neutral), with
    values summing to ~1. verification_confidence is the winning class's
    probability from that same distribution. Model, logits, softmax, and
    label-mapping logic are unchanged from before this fault-tolerance
    work.
    """
    tok, mdl = get_model_and_tokenizer()

    inputs = _tokenize_pair_or_raise(tok, evidence_text, claim_text)

    with torch.no_grad():
        outputs = mdl(**inputs)

    logits = outputs.logits
    probs = F.softmax(logits, dim=1).squeeze()

    id2label = _resolve_id2label(mdl.config)
    max_idx = int(torch.argmax(logits, dim=1).item())

    nli_raw_label = id2label[max_idx]
    nli_probabilities = {id2label[i]: float(probs[i].item()) for i in range(len(id2label))}
    confidence = nli_probabilities[nli_raw_label]

    _validate_output_or_raise(nli_probabilities, confidence)

    verification_label = normalise_label(nli_raw_label)

    return verification_label, nli_raw_label, confidence, nli_probabilities


def verify_pair_safe(evidence_text: str, claim_text: str) -> dict:
    """
    Single-claim-safe wrapper around verify_pair. Used by BOTH
    verify_claims() (the full pipeline) and the standalone /verify
    endpoint, so the two interpret every NLI failure mode identically
    instead of main.py duplicating its own exception handling. Never
    raises: every failure becomes verification_label="unverified" with a
    specific verification_reason.
    """
    if not evidence_text:
        return {
            "verification_label": "unverified",
            "nli_raw_label": None,
            "verification_confidence": None,
            "nli_probabilities": None,
            "verification_reason": "no_evidence",
        }

    try:
        label, raw_label, confidence, probabilities = verify_pair(evidence_text, claim_text)
        return {
            "verification_label": label,
            "nli_raw_label": raw_label,
            "verification_confidence": confidence,
            "nli_probabilities": probabilities,
            "verification_reason": None,
        }
    except ModelUnavailableError as e:
        print(f"NLI model unavailable: {e}")
        reason = "nli_model_unavailable"
    except InputTooLongError as e:
        print(f"NLI input too long: {e}")
        reason = "nli_input_too_long"
    except InvalidOutputError as e:
        print(f"NLI produced invalid output: {e}")
        reason = "nli_invalid_output"
    except Exception as e:
        print(f"Error during NLI verification: {e}")
        reason = "nli_inference_failed"

    return {
        "verification_label": "unverified",
        "nli_raw_label": None,
        "verification_confidence": None,
        "nli_probabilities": None,
        "verification_reason": reason,
    }


def verify_claims(claims: list[dict]) -> list[dict]:
    """
    Loops through linked claims and runs NLI verification via
    verify_pair_safe.

    Model-load failure is isolated to this one call only: once a claim
    comes back with verification_reason="nli_model_unavailable", every
    remaining claim in THIS call skips straight to the same reason without
    re-attempting the load. This is a variable local to this function
    call, not module-level state — the next /pipeline request's
    verify_claims() call starts fresh and will retry loading, so a
    transient failure never becomes permanent until a process restart.
    """
    verified = []
    model_unavailable = False

    for claim in claims:
        c = dict(claim)
        evidence_text = c.get("evidence_text")

        if evidence_text and model_unavailable:
            result = {
                "verification_label": "unverified",
                "nli_raw_label": None,
                "verification_confidence": None,
                "nli_probabilities": None,
                "verification_reason": "nli_model_unavailable",
            }
        else:
            result = verify_pair_safe(evidence_text, c["claim_text"])
            if result["verification_reason"] == "nli_model_unavailable":
                model_unavailable = True

        c.update(result)
        verified.append(c)

    return verified
