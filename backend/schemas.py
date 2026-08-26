import os
import unicodedata

from pydantic import BaseModel, field_validator
from typing import List, Optional, Literal

# Application-level request-size guard. Distinct from and unrelated to the
# NLI model's own tokenizer length limit (checked separately, at inference
# time, in nli_client.py) — this just stops someone pasting an entire Act
# into an endpoint designed for a single provision before it causes a
# confusing failure several stages later.
MAX_INPUT_LENGTH_CHARS = int(os.getenv("MAX_INPUT_LENGTH_CHARS", "20000"))

# Typed status vocabulary for the pipeline fault-tolerance contract. Using
# Literal instead of free strings prevents accidental inconsistent values
# ("fail" / "failure" / "partal") from ever entering the API contract.
OverallStatus = Literal["success", "partial", "failed"]
StageStatusValue = Literal[
    "success", "partial", "fallback", "failed", "skipped", "warning", "ambiguous"
]


class ProvisionRequest(BaseModel):
    text: str
    act_name: Optional[str] = "unknown"
    provision_id: Optional[str] = "p001"

    @field_validator("text")
    @classmethod
    def _validate_and_normalise_text(cls, value: str) -> str:
        """
        Stage 0 input validation, enforced once at the API boundary so
        every route sharing this schema rejects the same requests the same
        way. Normalises to Unicode NFC (not NFKC, which can alter
        characters that carry legal meaning) before length is checked, so
        validation behaviour is consistent regardless of input encoding.
        """
        normalised = unicodedata.normalize("NFC", value)
        if not normalised.strip():
            raise ValueError("text must not be empty or whitespace-only")
        if len(normalised) > MAX_INPUT_LENGTH_CHARS:
            raise ValueError(
                f"text exceeds the maximum allowed length of {MAX_INPUT_LENGTH_CHARS} characters"
            )
        return normalised


class Span(BaseModel):
    span_id: str        # e.g. "P1", "P2"
    text: str
    start: int          # char offset in original text
    end: int


class SpanResponse(BaseModel):
    provision_id: str
    act_name: str
    original_text: str
    spans: List[Span]


class StageStatus(BaseModel):
    status: StageStatusValue
    attempts: Optional[int] = None
    method: Optional[str] = None
    reason: Optional[str] = None


class PipelineStatus(BaseModel):
    overall: OverallStatus
    span_generation: StageStatus
    simplification: StageStatus
    claim_extraction: StageStatus
    evidence_retrieval: StageStatus
    nli_verification: StageStatus
    run_logging: StageStatus


class Claim(BaseModel):
    claim_id: str
    claim_text: str
    evidence_span_id: Optional[str] = None
    evidence_text: Optional[str] = None
    evidence_score: Optional[int] = None
    evidence_method: Optional[str] = None
    verification_label: str
    nli_raw_label: Optional[str] = None
    verification_confidence: Optional[float] = None
    nli_probabilities: Optional[dict[str, float]] = None

    # Why a claim ended up "unverified" — distinguishes no evidence found,
    # NLI inference failing, the NLI model being unavailable, and the
    # evidence/claim pair exceeding the model's usable input length. Never
    # changes the label mapping, purely diagnostic.
    verification_reason: Optional[str] = None

    # Non-destructive post-NLI diagnostics (e.g. "numerical_conflict",
    # "clear_modal_conflict") — a deterministic sanity signal alongside the
    # NLI verdict. Never overrides verification_label.
    verification_warnings: Optional[List[str]] = None

    # Non-destructive claim-extraction diagnostics (e.g. "unsupported_number",
    # "zero_lexical_overlap", "clear_modal_conflict") — flags a claim that
    # may not faithfully represent the simplified text it was extracted
    # from. Does not block or alter the claim; it still flows to evidence
    # linking and NLI verification unchanged.
    extraction_warnings: Optional[List[str]] = None

    # True when evidence selection hit an exact, deterministically
    # unresolved tie between two or more spans (see evidence.py).
    evidence_ambiguity: Optional[bool] = None

    # Keep compatibility with existing/old fields
    evidence_span_ids: Optional[List[str]] = None
    label: Optional[str] = "not_verified"
    label_confidence: Optional[float] = 0.0


class VerifyRequest(BaseModel):
    evidence_text: str
    claim_text: str


class VerifyResponse(BaseModel):
    verification_label: str
    nli_raw_label: Optional[str] = None
    verification_confidence: Optional[float] = None
    nli_probabilities: Optional[dict[str, float]] = None
    verification_reason: Optional[str] = None


class SimplifyResponse(BaseModel):
    provision_id: str
    simplified_text: Optional[str] = None
    claims: List[Claim]
    spans: List[Span]
    run_id: str
    pipeline_status: Optional[PipelineStatus] = None
    provenance: Optional[dict] = None


class ClaimsRequest(BaseModel):
    simplified_text: str


class EvidenceRequest(BaseModel):
    spans: List[Span]
    claims: List[dict]
    # Optional: enables the full-provision fallback/ambiguity behaviour on
    # this standalone debug endpoint too. Without it, behaviour is
    # unchanged from before (no fallback offered).
    source_text: Optional[str] = None
