import json
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import HTTPException
from backend.schemas import (
    ProvisionRequest, SpanResponse, SimplifyResponse, Claim
)
from backend.spans import make_spans
from backend.llm_client import simplify, extract_claims
from backend.evidence import link_claims_to_spans

RUNS_DIR = Path("outputs/runs")
RUNS_DIR.mkdir(parents=True, exist_ok=True)


def run_spans(req: ProvisionRequest) -> SpanResponse:
    spans = make_spans(req.text)
    return SpanResponse(
        provision_id=req.provision_id,
        act_name=req.act_name,
        original_text=req.text,
        spans=spans,
    )


def run_pipeline(req: ProvisionRequest) -> SimplifyResponse:
    """
    Full pipeline:
      1. Split provision into evidence spans
      2. Simplify with Qwen/Qwen3-8B via HuggingFace Inference API
      3. Extract atomic claims from simplified text (Qwen JSON output)
      4. Link each claim to its best matching source span
      5. Log the full run to outputs/runs/
    """
    run_id = f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    # Step 1: spans
    spans = make_spans(req.text)

    # Step 2: simplify
    try:
        simplified_text = simplify(req.text)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    # Step 3: extract atomic claims
    try:
        raw_claims = extract_claims(simplified_text)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    # Step 4: link claims to spans
    linked = link_claims_to_spans(raw_claims, spans)

    # Step 5: verify claims with NLI
    from backend.nli_client import verify_claims
    verified_claims = verify_claims(linked)

    # Step 6: build Claim objects using updated schema
    claims = [
        Claim(
            claim_id=c["claim_id"],
            claim_text=c["claim_text"],
            evidence_span_id=c.get("evidence_span_id"),
            evidence_text=c.get("evidence_text"),
            evidence_score=c.get("evidence_score"),
            verification_label=c["verification_label"],
            nli_raw_label=c.get("nli_raw_label"),
            verification_confidence=c.get("verification_confidence"),
            nli_probabilities=c.get("nli_probabilities"),

            # evidence_span_ids now carries the spans actually sent to the NLI
            # model. For an ordinary claim that is a single id; for a
            # continuation branch it is [previous_id, selected_id], because
            # evidence.link_claims_to_spans joined the preceding span to the
            # headless branch. Falls back to the single selected id so older
            # linked payloads (e.g. from the standalone /evidence endpoint)
            # still populate this field.
            evidence_span_ids=c.get("evidence_span_ids")
            or ([c["evidence_span_id"]] if c.get("evidence_span_id") else []),
            label=c["verification_label"],
            label_confidence=c.get("verification_confidence") or 0.0,
        )
        for c in verified_claims
    ]

    result = SimplifyResponse(
        provision_id=req.provision_id,
        simplified_text=simplified_text,
        claims=claims,
        spans=spans,
        run_id=run_id,
    )

    # Persist run log
    log_path = RUNS_DIR / f"{run_id}.json"
    log_path.write_text(result.model_dump_json(indent=2))

    return result
