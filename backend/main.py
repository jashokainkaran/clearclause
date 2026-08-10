from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import (
    ProvisionRequest, SpanResponse, SimplifyResponse,
    ClaimsRequest, EvidenceRequest, Span,
    VerifyRequest, VerifyResponse,
)
from backend.pipeline import run_spans, run_pipeline
from backend.llm_client import simplify, extract_claims
from backend.evidence import link_claims_to_spans

app = FastAPI(
    title="Faithful Legal Simplification API",
    description="Hallucination-controlled simplification of Sri Lankan statutes.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "version": "0.1.0"}


@app.post("/spans", response_model=SpanResponse)
def get_spans(req: ProvisionRequest):
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="text must not be empty")
    return run_spans(req)


@app.post("/pipeline", response_model=SimplifyResponse)
def pipeline(req: ProvisionRequest):
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="text must not be empty")
    return run_pipeline(req)


# ── Standalone testing endpoints ──────────────────────────────────────────────
# These let you test each pipeline step independently (e.g. via Postman or
# the FastAPI docs at /docs). The frontend only uses /pipeline.

@app.post("/simplify")
def simplify_text(req: ProvisionRequest):
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="text must not be empty")
    result = simplify(req.text)
    return {"simplified_text": result}


@app.post("/claims")
def extract_claims_endpoint(req: ClaimsRequest):
    if not req.simplified_text.strip():
        raise HTTPException(status_code=422, detail="simplified_text must not be empty")
    result = extract_claims(req.simplified_text)
    return {"claims": result}


@app.post("/evidence")
def link_evidence(req: EvidenceRequest):
    if not req.claims or not req.spans:
        raise HTTPException(status_code=422, detail="claims and spans must not be empty")
    result = link_claims_to_spans(req.claims, req.spans)
    return {"claims": result}


@app.post("/verify", response_model=VerifyResponse)
def verify_endpoint(req: VerifyRequest):
    from backend.nli_client import verify_pair
    if not req.evidence_text.strip() or not req.claim_text.strip():
        raise HTTPException(status_code=422, detail="evidence_text and claim_text must not be empty")
    try:
        verification_label, nli_raw_label, verification_confidence, nli_probabilities = verify_pair(
            req.evidence_text.strip(), req.claim_text.strip()
        )
        return VerifyResponse(
            verification_label=verification_label,
            nli_raw_label=nli_raw_label,
            verification_confidence=verification_confidence,
            nli_probabilities=nli_probabilities,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


