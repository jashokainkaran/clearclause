"""
Integration-level fault-tolerance tests for backend/pipeline.py.

All external calls (LLM, NLI, evidence linking) are mocked at the names
pipeline.py imports them under (pipeline.simplify_with_attempts,
pipeline.extract_claims_with_status, pipeline.verify_claims,
pipeline.link_claims_to_spans) — pipeline.py uses `from X import Y`, so the
name bound inside the pipeline module is what must be patched, not the name
inside llm_client/nli_client/evidence themselves. Evidence linking is mocked
in most tests so a stage-specific test isn't accidentally affected by the
real lexical-overlap algorithm picking an unrelated fallback for the
deliberately mismatched sample claim/source text used here.
"""

import pytest
from fastapi import HTTPException

from backend import pipeline
from backend.llm_client import SimplificationFailedError
from backend.schemas import ProvisionRequest, StageStatus


SAMPLE_TEXT = "Whoever commits theft shall be punished with imprisonment."


def _req(text=SAMPLE_TEXT):
    return ProvisionRequest(text=text, act_name="Test Act", provision_id="T1")


def _fake_link_claims_to_spans(claims, spans, source_text=None):
    """A clean, deterministic 'every claim matched span 1' stand-in, that
    still carries extraction_warnings through — exactly what the real
    function is required to do."""
    out = []
    for c in claims:
        out.append({
            "claim_id": c["claim_id"],
            "claim_text": c["claim_text"],
            "evidence_span_id": spans[0].span_id if spans else None,
            "evidence_span_ids": [spans[0].span_id] if spans else [],
            "evidence_text": spans[0].text if spans else None,
            "evidence_score": 1,
            "evidence_method": "lexical_overlap",
            "evidence_ambiguity": False,
            "extraction_warnings": c.get("extraction_warnings"),
        })
    return out


def _fake_verify_claims_supported(linked):
    out = []
    for c in linked:
        c = dict(c)
        c["verification_label"] = "supported"
        c["nli_raw_label"] = "entailment"
        c["verification_confidence"] = 0.9
        c["nli_probabilities"] = {"entailment": 0.9, "neutral": 0.05, "contradiction": 0.05}
        c["verification_reason"] = None
        out.append(c)
    return out


def _stub_success_chain(monkeypatch, simplified_text="Plain text version.", claim_text="Plain text version."):
    """Wires simplify/claims/evidence/nli to a fully normal, successful path."""
    monkeypatch.setattr(pipeline, "simplify_with_attempts", lambda text: (simplified_text, 1))
    monkeypatch.setattr(
        pipeline,
        "extract_claims_with_status",
        lambda text: (
            [{"claim_id": "C1", "claim_text": claim_text, "extraction_warnings": []}],
            {"status": "success", "method": "llm"},
        ),
    )
    monkeypatch.setattr(pipeline, "link_claims_to_spans", _fake_link_claims_to_spans)
    monkeypatch.setattr(pipeline, "verify_claims", _fake_verify_claims_supported)


# ---------------------------------------------------------------------------
# Normal successful run — response shape unchanged
# ---------------------------------------------------------------------------

def test_normal_successful_pipeline(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "RUNS_DIR", tmp_path)
    _stub_success_chain(monkeypatch)

    result = pipeline.run_pipeline(_req())

    assert result.simplified_text == "Plain text version."
    assert len(result.claims) == 1
    assert result.pipeline_status.overall == "success"
    assert result.pipeline_status.span_generation.status == "success"
    assert result.pipeline_status.simplification.status == "success"
    assert result.pipeline_status.claim_extraction.status == "success"
    assert result.pipeline_status.evidence_retrieval.status == "success"
    assert result.pipeline_status.nli_verification.status == "success"
    assert result.pipeline_status.run_logging.status == "success"
    assert result.provenance["nli_model"] == pipeline.NLI_MODEL_PATH
    assert result.provenance["simplification_model"] == pipeline.SIMPLIFICATION_MODEL


# ---------------------------------------------------------------------------
# Span generation — defensive-only failure path
# ---------------------------------------------------------------------------

def test_span_generation_failure_raises_500_and_preserves_spans(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "RUNS_DIR", tmp_path)

    monkeypatch.setattr(
        pipeline, "make_spans_with_fallback",
        lambda text: ([], StageStatus(status="failed", reason="invariant violated")),
    )

    called = {"simplify": False}
    def _should_not_run(*args, **kwargs):
        called["simplify"] = True
        raise AssertionError("simplify must not run after span generation fails")
    monkeypatch.setattr(pipeline, "simplify_with_attempts", _should_not_run)

    with pytest.raises(HTTPException) as exc_info:
        pipeline.run_pipeline(_req())

    assert not called["simplify"]
    assert exc_info.value.status_code == 500
    detail = exc_info.value.detail
    assert detail["pipeline_status"]["overall"] == "failed"
    assert detail["pipeline_status"]["span_generation"]["status"] == "failed"
    assert detail["spans"] == []


# ---------------------------------------------------------------------------
# Simplification failure after the permitted retry
# ---------------------------------------------------------------------------

def test_simplification_failure_raises_502_preserves_spans_skips_downstream(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "RUNS_DIR", tmp_path)

    def _always_fails(text):
        raise SimplificationFailedError("provider down", attempts=2)
    monkeypatch.setattr(pipeline, "simplify_with_attempts", _always_fails)

    called = {"claims": False}
    def _should_not_run(*args, **kwargs):
        called["claims"] = True
        raise AssertionError("claim extraction must not run after simplification fails")
    monkeypatch.setattr(pipeline, "extract_claims_with_status", _should_not_run)

    with pytest.raises(HTTPException) as exc_info:
        pipeline.run_pipeline(_req())

    assert not called["claims"]
    assert exc_info.value.status_code == 502
    detail = exc_info.value.detail
    assert detail["pipeline_status"]["overall"] == "failed"
    assert detail["pipeline_status"]["simplification"]["status"] == "failed"
    assert detail["pipeline_status"]["simplification"]["attempts"] == 2
    assert detail["pipeline_status"]["claim_extraction"]["status"] == "skipped"
    assert detail["pipeline_status"]["evidence_retrieval"]["status"] == "skipped"
    assert detail["pipeline_status"]["nli_verification"]["status"] == "skipped"
    assert len(detail["spans"]) > 0  # spans preserved even though simplify failed
    assert "simplified_text" not in detail  # nothing fabricated in its place


# ---------------------------------------------------------------------------
# Degraded-but-usable outcomes → overall = partial (never failed)
# ---------------------------------------------------------------------------

def test_claim_extraction_fallback_marks_overall_partial(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "simplify_with_attempts", lambda text: ("Plain text.", 1))
    monkeypatch.setattr(
        pipeline,
        "extract_claims_with_status",
        lambda text: (
            [{"claim_id": "C1", "claim_text": "Plain text.", "extraction_warnings": []}],
            {"status": "fallback", "method": "sentence_splitter"},
        ),
    )
    monkeypatch.setattr(pipeline, "link_claims_to_spans", _fake_link_claims_to_spans)
    monkeypatch.setattr(pipeline, "verify_claims", _fake_verify_claims_supported)

    result = pipeline.run_pipeline(_req())

    assert result.pipeline_status.claim_extraction.status == "fallback"
    assert result.pipeline_status.claim_extraction.method == "sentence_splitter"
    assert result.pipeline_status.overall == "partial"


def test_nli_degraded_verification_marks_overall_partial(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "simplify_with_attempts", lambda text: ("Plain text.", 1))
    monkeypatch.setattr(
        pipeline,
        "extract_claims_with_status",
        lambda text: (
            [{"claim_id": "C1", "claim_text": "Plain text.", "extraction_warnings": []}],
            {"status": "success", "method": "llm"},
        ),
    )
    monkeypatch.setattr(pipeline, "link_claims_to_spans", _fake_link_claims_to_spans)

    def _degraded_verify(linked):
        out = []
        for c in linked:
            c = dict(c)
            c["verification_label"] = "unverified"
            c["nli_raw_label"] = None
            c["verification_confidence"] = None
            c["nli_probabilities"] = None
            c["verification_reason"] = "nli_model_unavailable"
            out.append(c)
        return out
    monkeypatch.setattr(pipeline, "verify_claims", _degraded_verify)

    result = pipeline.run_pipeline(_req())

    assert result.pipeline_status.nli_verification.status == "partial"
    assert result.pipeline_status.overall == "partial"
    assert result.claims[0].verification_reason == "nli_model_unavailable"


def test_integrity_warning_marks_overall_partial_even_if_stages_all_succeeded(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "simplify_with_attempts", lambda text: ("Plain text.", 1))
    monkeypatch.setattr(
        pipeline,
        "extract_claims_with_status",
        lambda text: (
            [{"claim_id": "C1", "claim_text": "Plain text.", "extraction_warnings": ["unsupported_number"]}],
            {"status": "success", "method": "llm"},
        ),
    )
    monkeypatch.setattr(pipeline, "link_claims_to_spans", _fake_link_claims_to_spans)
    monkeypatch.setattr(pipeline, "verify_claims", _fake_verify_claims_supported)

    result = pipeline.run_pipeline(_req())

    assert result.pipeline_status.claim_extraction.status == "success"
    assert result.pipeline_status.evidence_retrieval.status == "success"
    assert result.pipeline_status.nli_verification.status == "success"
    assert result.pipeline_status.overall == "partial"
    assert "unsupported_number" in result.claims[0].extraction_warnings


def test_diagnostic_only_warning_does_not_mark_overall_partial(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "simplify_with_attempts", lambda text: ("Plain text.", 1))
    monkeypatch.setattr(
        pipeline,
        "extract_claims_with_status",
        lambda text: (
            [{"claim_id": "C1", "claim_text": "Plain text.", "extraction_warnings": ["zero_lexical_overlap"]}],
            {"status": "success", "method": "llm"},
        ),
    )
    monkeypatch.setattr(pipeline, "link_claims_to_spans", _fake_link_claims_to_spans)
    monkeypatch.setattr(pipeline, "verify_claims", _fake_verify_claims_supported)

    result = pipeline.run_pipeline(_req())

    # A diagnostic-only signal is still recorded on the claim...
    assert "zero_lexical_overlap" in result.claims[0].extraction_warnings
    # ...but must not, by itself, drag the whole run down to partial.
    assert result.pipeline_status.overall == "success"


# ---------------------------------------------------------------------------
# Run logging — non-critical, never affects overall
# ---------------------------------------------------------------------------

def test_run_log_write_failure_does_not_lose_successful_result(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "RUNS_DIR", tmp_path)
    _stub_success_chain(monkeypatch)

    def _boom(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(pipeline.os, "replace", _boom)

    result = pipeline.run_pipeline(_req())

    assert result.simplified_text == "Plain text version."
    assert result.pipeline_status.run_logging.status == "warning"
    assert result.pipeline_status.overall == "success"
