from pydantic import BaseModel
from typing import List, Optional

class ProvisionRequest(BaseModel):
    text: str
    act_name: Optional[str] = "unknown"
    provision_id: Optional[str] = "p001"

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

class Claim(BaseModel):
    claim_id: str
    claim_text: str
    evidence_span_id: Optional[str] = None
    evidence_text: Optional[str] = None
    evidence_score: Optional[int] = None
    verification_label: str
    nli_raw_label: Optional[str] = None
    verification_confidence: Optional[float] = None

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

class SimplifyResponse(BaseModel):
    provision_id: str
    simplified_text: str
    claims: List[Claim]
    spans: List[Span]
    run_id: str

class ClaimsRequest(BaseModel):
    simplified_text: str

class EvidenceRequest(BaseModel):
    spans: List[Span]
    claims: List[dict]