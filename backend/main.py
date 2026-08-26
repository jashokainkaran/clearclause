from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import (
    ProvisionRequest, SpanResponse, SimplifyResponse,
    ClaimsRequest, EvidenceRequest,
    VerifyRequest, VerifyResponse,
)
from backend.pipeline import run_spans, run_pipeline
from backend.llm_client import simplify, extract_claims_with_status
from backend.evidence import link_claims_to_spans
from backend.nli_client import verify_pair_safe

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


# Text validity (non-empty, within MAX_INPUT_LENGTH_CHARS, Unicode-normalised)
# is enforced by ProvisionRequest's own field validator (backend/schemas.py).
# FastAPI rejects an invalid request with its own 422 response before these
# endpoint bodies ever run, so /spans, /pipeline and /simplify do not
# duplicate that check.

@app.post("/spans", response_model=SpanResponse)
def get_spans(req: ProvisionRequest):
    return run_spans(req)


@app.post("/pipeline", response_model=SimplifyResponse)
def pipeline(req: ProvisionRequest):
    return run_pipeline(req)


# ── Standalone testing endpoints ──────────────────────────────────────────────
# These let you test each pipeline step independently (e.g. via Postman or
# the FastAPI docs at /docs). The frontend only uses /pipeline. Each reuses
# the exact same shared functions the pipeline uses, so a standalone call
# and a full-pipeline call never silently disagree about the same input.

@app.post("/simplify")
def simplify_text(req: ProvisionRequest):
    result = simplify(req.text)
    return {"simplified_text": result}


@app.post("/claims")
def extract_claims_endpoint(req: ClaimsRequest):
    if not req.simplified_text.strip():
        raise HTTPException(status_code=422, detail="simplified_text must not be empty")
    claims, status = extract_claims_with_status(req.simplified_text)
    return {"claims": claims, "claim_extraction_status": status}


@app.post("/evidence")
def link_evidence(req: EvidenceRequest):
    if not req.claims or not req.spans:
        raise HTTPException(status_code=422, detail="claims and spans must not be empty")
    result = link_claims_to_spans(req.claims, req.spans, source_text=req.source_text)
    return {"claims": result}


@app.post("/verify", response_model=VerifyResponse)
def verify_endpoint(req: VerifyRequest):
    if not req.evidence_text.strip() or not req.claim_text.strip():
        raise HTTPException(status_code=422, detail="evidence_text and claim_text must not be empty")
    # verify_pair_safe is the same function verify_claims() uses in the full
    # pipeline, so /verify interprets every NLI failure mode (model
    # unavailable, input too long, invalid output, inference crash)
    # identically — no separate exception handling duplicated here.
    result = verify_pair_safe(req.evidence_text.strip(), req.claim_text.strip())
    return VerifyResponse(**result)
