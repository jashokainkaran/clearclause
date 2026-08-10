import torch
import torch.nn.functional as F

# Lazy loaded globals
_tokenizer = None
_model = None

# Explicit tokenizer limit. cross-encoder/nli-deberta-v3-small accepts 512
# tokens; stating it here makes the limit visible and keeps it aligned with
# the offline evaluation harness instead of relying on a model default.
#
# This matters more now that a continuation branch is verified against the
# PREVIOUS span joined to the selected span — the premise is longer, so
# truncation is more likely and must be visible when it happens.
MAX_LENGTH = 512

# Standard cross-encoder NLI class ordering — used only when the model
# config doesn't provide a usable id2label mapping.
_FALLBACK_ID2LABEL = {0: "contradiction", 1: "entailment", 2: "neutral"}


def get_model_and_tokenizer():
    global _tokenizer, _model
    if _model is None or _tokenizer is None:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        print("Loading cross-encoder/nli-deberta-v3-small model and tokenizer...")
        _tokenizer = AutoTokenizer.from_pretrained("cross-encoder/nli-deberta-v3-small")
        _model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/nli-deberta-v3-small")
        print("Model and tokenizer loaded successfully.")
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


def _warn_if_truncated(tok, evidence_text: str, claim_text: str) -> None:
    """
    Logs a warning when the (evidence, claim) pair exceeds MAX_LENGTH and is
    therefore truncated before the model sees it.

    Truncation is otherwise silent, which means the model can score less text
    than the UI displays for that span — especially for a combined
    previous+continuation premise.

    Best-effort and defensive: any tokenizer that does not support an
    untruncated call (including test doubles) is skipped silently, so this can
    never break verification itself.
    """
    try:
        full = tok(evidence_text, claim_text)
        input_ids = full["input_ids"] if isinstance(full, dict) else full.get("input_ids")
        if input_ids is None:
            return
        if len(input_ids) > 0 and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        full_len = len(input_ids)
        if full_len > MAX_LENGTH:
            print(
                f"WARNING: NLI input truncated — {full_len} tokens reduced to "
                f"{MAX_LENGTH}. The model scored less text than is displayed "
                f"as evidence for this claim."
            )
    except Exception:
        return


def verify_pair(evidence_text: str, claim_text: str):
    """
    Tokenizes the evidence and claim as a pair, runs NLI model, and returns
    (verification_label, nli_raw_label, verification_confidence,
    nli_probabilities).

    nli_probabilities carries the full 3-class softmax distribution keyed by
    lowercase label name (contradiction / entailment / neutral), with values
    summing to ~1. verification_confidence is the winning class's
    probability from that same distribution.
    """
    tok, mdl = get_model_and_tokenizer()

    # Tokenize as pair, with an explicit length cap.
    inputs = tok(
        evidence_text,
        claim_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )

    # Surface truncation — otherwise the model silently scores less text than
    # the UI shows as the evidence for this claim.
    _warn_if_truncated(tok, evidence_text, claim_text)

    with torch.no_grad():
        outputs = mdl(**inputs)

    logits = outputs.logits
    # Apply softmax to get probabilities
    probs = F.softmax(logits, dim=1).squeeze()

    id2label = _resolve_id2label(mdl.config)
    max_idx = int(torch.argmax(logits, dim=1).item())

    nli_raw_label = id2label[max_idx]
    nli_probabilities = {id2label[i]: float(probs[i].item()) for i in range(len(id2label))}
    confidence = nli_probabilities[nli_raw_label]

    verification_label = normalise_label(nli_raw_label)

    return verification_label, nli_raw_label, confidence, nli_probabilities


def verify_claims(claims: list[dict]) -> list[dict]:
    """
    Loops through linked claims and runs NLI verification on claim+evidence pairs.
    Handles missing evidence and errors gracefully.
    """
    verified = []
    for claim in claims:
        # copy to avoid mutating inputs
        c = dict(claim)
        evidence_text = c.get("evidence_text")

        if not evidence_text:
            c["verification_label"] = "unverified"
            c["nli_raw_label"] = None
            c["verification_confidence"] = None
            c["nli_probabilities"] = None
        else:
            try:
                verification_label, nli_raw_label, confidence, nli_probabilities = verify_pair(
                    evidence_text, c["claim_text"]
                )
                c["verification_label"] = verification_label
                c["nli_raw_label"] = nli_raw_label
                c["verification_confidence"] = confidence
                c["nli_probabilities"] = nli_probabilities
            except Exception as e:
                # Log error and mark claim as unverified
                print(f"Error during NLI verification for claim {c.get('claim_id')}: {e}")
                c["verification_label"] = "unverified"
                c["nli_raw_label"] = None
                c["verification_confidence"] = None
                c["nli_probabilities"] = None
        verified.append(c)
    return verified
