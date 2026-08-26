import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import HTTPException
from backend.schemas import (
    ProvisionRequest, SpanResponse, SimplifyResponse, Claim,
    PipelineStatus, StageStatus,
)
from backend.spans import make_spans_with_fallback
from backend.llm_client import (
    simplify_with_attempts, SimplificationFailedError,
    extract_claims_with_status, MODEL as SIMPLIFICATION_MODEL,
)
from backend.evidence import link_claims_to_spans, detect_verification_conflicts
from backend.nli_client import verify_claims, NLI_MODEL_PATH

logger = logging.getLogger(__name__)

RUNS_DIR = Path("outputs/runs")
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# Claim-extraction/verification signals that indicate a real fidelity
# concern and should pull overall status down to "partial". Diagnostic-only
# signals (zero_lexical_overlap, broad modality differences) are still
# recorded on the claim but never affect overall — otherwise a routine
# diagnostic observation would make "partial" fire far more often than it
# should.
_INTEGRITY_WARNINGS = {"unsupported_number", "numerical_conflict", "clear_modal_conflict"}
_DEGRADED_STAGE_STATUSES = {"fallback", "partial", "ambiguous"}


def run_spans(req: ProvisionRequest) -> SpanResponse:
    """
    Standalone /spans endpoint. Uses the exact same fallback hierarchy as
    the full pipeline (make_spans_with_fallback), so the two routes can
    never disagree about what counts as a valid split for the same input.
    """
    spans, _status = make_spans_with_fallback(req.text)
    return SpanResponse(
        provision_id=req.provision_id,
        act_name=req.act_name,
        original_text=req.text,
        spans=spans,
    )


def _skipped() -> StageStatus:
    return StageStatus(status="skipped")


def _write_run_log(run_id: str, json_text: str) -> str:
    """
    Persists a run outcome to outputs/runs/ via a temp-file-then-replace
    sequence, so a crash mid-write can never leave a corrupted/partial run
    file behind. Logging is secondary to the user-facing result: any
    failure here is caught, logged, and reported as "warning" — it never
    affects overall, and the caller's result is returned regardless.
    """
    log_path = RUNS_DIR / f"{run_id}.json"
    tmp_path = RUNS_DIR / f"{run_id}.json.tmp"
    try:
        tmp_path.write_text(json_text)
        os.replace(tmp_path, log_path)
        return "success"
    except OSError as e:
        logger.warning("Failed to write run log %s: %s", run_id, e)
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        return "warning"


def _evidence_stage_status(linked: list[dict]) -> StageStatus:
    """
    Worst-wins priority across all claims in one run:
      partial   — some claim has no evidence at all (fallback unavailable
                  or, on the standalone /evidence endpoint, not requested)
      fallback  — any claim used the full-provision fallback (no match, or
                  an exact unresolved tie)
      ambiguous — a tie occurred but no fallback was available to resolve it
      success   — every claim matched a span directly
    """
    if not linked:
        return _skipped()
    if any(c.get("evidence_text") is None for c in linked):
        return StageStatus(status="partial")
    if any(
        c.get("evidence_method") in ("full_provision", "full_provision_ambiguity_fallback")
        for c in linked
    ):
        return StageStatus(status="fallback", method="full_provision")
    if any(c.get("evidence_ambiguity") for c in linked):
        return StageStatus(status="ambiguous")
    return StageStatus(status="success", method="lexical_overlap")


def _nli_stage_status(verified: list[dict]) -> StageStatus:
    """
    "partial" if any claim couldn't be verified for a reason attributable
    to the NLI stage itself (inference crash, model unavailable, input too
    long, invalid output). "no_evidence" does NOT count here — that's
    Stage 4's concern, not the verifier's fault.
    """
    if not verified:
        return _skipped()
    degraded_reasons = {
        "nli_inference_failed", "nli_model_unavailable",
        "nli_input_too_long", "nli_invalid_output",
    }
    if any(c.get("verification_reason") in degraded_reasons for c in verified):
        return StageStatus(status="partial")
    return StageStatus(status="success")


def _overall_status(
    span_status: StageStatus,
    claim_status: StageStatus,
    evidence_status: StageStatus,
    nli_status: StageStatus,
    verified_claims: list[dict],
) -> str:
    """
    Only reachable once span generation and simplification have both
    already succeeded (their failure short-circuits with a raised
    HTTPException earlier) — so this only ever decides success vs.
    partial, never failed.
    """
    if (
        span_status.status in _DEGRADED_STAGE_STATUSES
        or claim_status.status in _DEGRADED_STAGE_STATUSES
        or evidence_status.status in _DEGRADED_STAGE_STATUSES
        or nli_status.status in _DEGRADED_STAGE_STATUSES
    ):
        return "partial"

    for claim in verified_claims:
        all_warnings = (claim.get("extraction_warnings") or []) + (claim.get("verification_warnings") or [])
        if any(w in _INTEGRITY_WARNINGS for w in all_warnings):
            return "partial"

    return "success"


def run_pipeline(req: ProvisionRequest) -> SimplifyResponse:
    """
    Full pipeline:
      1. Split the provision into evidence spans, with a deterministic
         fallback hierarchy if the primary structural splitter's output
         doesn't validate.
      2. Simplify with a bounded retry on transient provider failures
         only — content faithfulness is judged downstream, by claim
         extraction + evidence retrieval + NLI, not here (see
         llm_client.simplify_with_attempts for why).
      3. Extract atomic claims (LLM, with a total-call-budgeted retry and
         a deterministic fallback hierarchy).
      4. Link each claim to its best matching source span, falling back to
         the whole provision when no span matches or ranking is
         genuinely tied.
      5. Verify claims with NLI (per-run model-load isolation, an
         input-length gate, output-sanity validation).
      6. Compute post-NLI deterministic conflict warnings (diagnostic
         only — never overrides the NLI label).
      7. Persist the full run to outputs/runs/ (non-critical, atomic).

    A failure in span generation (defensive-only — Stage 0 input
    validation already prevents an unusable source) or simplification
    stops the pipeline early via a raised HTTPException carrying whatever
    was safely produced. Nothing downstream is ever fabricated. A degraded
    claim-extraction, evidence-retrieval, or NLI outcome instead returns
    normally with pipeline_status.overall == "partial".
    """
    run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    # Step 1: spans, with fallback hierarchy
    spans, span_status = make_spans_with_fallback(req.text)

    if span_status.status == "failed":
        status = PipelineStatus(
            overall="failed",
            span_generation=span_status,
            simplification=_skipped(),
            claim_extraction=_skipped(),
            evidence_retrieval=_skipped(),
            nli_verification=_skipped(),
            run_logging=StageStatus(status="success"),
        )
        error_payload = {
            "message": f"Span generation could not preserve a usable representation of the source text: {span_status.reason}",
            "provision_id": req.provision_id,
            "run_id": run_id,
            "spans": [s.model_dump() for s in spans],
            "pipeline_status": status.model_dump(),
        }
        status.run_logging = StageStatus(
            status=_write_run_log(run_id, json.dumps(error_payload, indent=2, default=str))
        )
        error_payload["pipeline_status"] = status.model_dump()
        raise HTTPException(status_code=500, detail=error_payload)

    # Step 2: simplify (bounded retry on transient provider failures only)
    try:
        simplified_text, attempts = simplify_with_attempts(req.text)
        simplify_status = StageStatus(status="success", attempts=attempts)
    except SimplificationFailedError as e:
        status = PipelineStatus(
            overall="failed",
            span_generation=span_status,
            simplification=StageStatus(status="failed", attempts=e.attempts, reason=str(e)),
            claim_extraction=_skipped(),
            evidence_retrieval=_skipped(),
            nli_verification=_skipped(),
            run_logging=StageStatus(status="success"),
        )
        error_payload = {
            "message": str(e),
            "provision_id": req.provision_id,
            "run_id": run_id,
            "spans": [s.model_dump() for s in spans],
            "pipeline_status": status.model_dump(),
        }
        status.run_logging = StageStatus(
            status=_write_run_log(run_id, json.dumps(error_payload, indent=2, default=str))
        )
        error_payload["pipeline_status"] = status.model_dump()
        raise HTTPException(status_code=502, detail=error_payload)

    # Step 3: extract atomic claims (LLM, budgeted retry + deterministic fallback)
    raw_claims, extraction_status_info = extract_claims_with_status(simplified_text)
    claim_status = StageStatus(
        status=extraction_status_info["status"], method=extraction_status_info["method"]
    )

    # Step 4: link claims to spans (full-provision fallback available)
    linked = link_claims_to_spans(raw_claims, spans, source_text=req.text)
    evidence_status = _evidence_stage_status(linked)

    # Step 5: verify claims with NLI
    verified_claims = verify_claims(linked)
    nli_status = _nli_stage_status(verified_claims)

    # Step 6: post-NLI deterministic conflict warnings — diagnostic only,
    # never overrides verification_label.
    for claim in verified_claims:
        claim["verification_warnings"] = detect_verification_conflicts(
            claim.get("evidence_text") or "",
            claim["claim_text"],
            claim["verification_label"],
        )

    # Step 7: build Claim objects
    claims = [
        Claim(
            claim_id=c["claim_id"],
            claim_text=c["claim_text"],
            evidence_span_id=c.get("evidence_span_id"),
            evidence_text=c.get("evidence_text"),
            evidence_score=c.get("evidence_score"),
            evidence_method=c.get("evidence_method"),
            verification_label=c["verification_label"],
            nli_raw_label=c.get("nli_raw_label"),
            verification_confidence=c.get("verification_confidence"),
            nli_probabilities=c.get("nli_probabilities"),
            verification_reason=c.get("verification_reason"),
            verification_warnings=c.get("verification_warnings") or None,
            extraction_warnings=c.get("extraction_warnings") or None,
            evidence_ambiguity=c.get("evidence_ambiguity"),

            # evidence_span_ids carries the spans actually sent to the NLI
            # model. Falls back to the single selected id so older linked
            # payloads (e.g. from the standalone /evidence endpoint)
            # still populate this field.
            evidence_span_ids=c.get("evidence_span_ids")
            or ([c["evidence_span_id"]] if c.get("evidence_span_id") else []),
            label=c["verification_label"],
            label_confidence=c.get("verification_confidence") or 0.0,
        )
        for c in verified_claims
    ]

    overall = _overall_status(span_status, claim_status, evidence_status, nli_status, verified_claims)
    status = PipelineStatus(
        overall=overall,
        span_generation=span_status,
        simplification=simplify_status,
        claim_extraction=claim_status,
        evidence_retrieval=evidence_status,
        nli_verification=nli_status,
        run_logging=StageStatus(status="success"),
    )

    provenance = {
        "simplification_model": SIMPLIFICATION_MODEL,
        "claim_extraction_method": claim_status.method,
        "evidence_retrieval_method": evidence_status.method or "lexical_overlap",
        "nli_model": NLI_MODEL_PATH,
    }

    result = SimplifyResponse(
        provision_id=req.provision_id,
        simplified_text=simplified_text,
        claims=claims,
        spans=spans,
        run_id=run_id,
        pipeline_status=status,
        provenance=provenance,
    )

    result.pipeline_status.run_logging = StageStatus(
        status=_write_run_log(run_id, result.model_dump_json(indent=2))
    )

    return result
