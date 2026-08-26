# HANDOFF — ClearClause current state

Last updated: fault-tolerance implementation pass (backend stages 0–6, debug
telemetry, `start.bat` auto-open) + switching the NLI verifier from the
pretrained baseline to a fine-tuned checkpoint. Backend test suite:
**205 passing** (`venv\Scripts\python.exe -m pytest -q`).

**NLI model in use**: `backend/.env`'s `NLI_MODEL_PATH` now points to
`D:/nli_baseline_test/finetuning/finetuned_deberta` — a fine-tune of
`cross-encoder/nli-deberta-v3-small` (same base architecture, verified
`id2label` mapping). Selected over three later `_v3_base` checkpoints
because it scores highest on every held-out metric checked, notably
contradiction recall (0.86 on the external test set vs 0.58 for the most
recently trained `_boosted` checkpoint, which uses a different, larger base
model and is a regression despite being trained last — see
`THESIS_FAULT_TOLERANCE_MATERIAL.md` §10 for the full comparison). Verified
live: correctly flags a 5,000→50,000 number substitution as a contradiction
with ~100% confidence — the exact failure mode `numerical_conflict` (§4.1)
exists to catch when it *isn't* caught upstream.

This documents what exists right now, not the history of how it got here —
see git log for that.

## What this system does

Generate → decompose → verify. A statute provision is split into evidence
spans, simplified by an LLM, decomposed into atomic claims, each claim is
linked back to a source span, and an NLI model (`cross-encoder/nli-deberta-v3-small`
by default) checks whether the claim is actually entailed by that span. The
point of the whole pipeline is that the NLI stage — not any earlier
deterministic rule — is what's evaluated on catching unfaithful
simplification; earlier stages deliberately do NOT pre-check content
faithfulness (see "Design decisions" below).

## Fault-tolerance model

Every stage answers: did the preferred method run technically? Is the
output structurally valid? If not, is there a safe deterministic
alternative? If every recovery fails, does the pipeline degrade or stop?
"No exception raised" is never treated as proof a stage worked.

### Overall status (`pipeline_status.overall`)

- **success** — every stage completed via its normal method.
- **partial** — a usable result exists but something degraded: a fallback
  ran, an evidence match was ambiguous or missing, NLI couldn't verify some
  claims, or an *integrity* warning fired (see below). Returned as a normal
  `200`.
- **failed** — span generation (defensive-only) or simplification could not
  produce a trustworthy result. Represented as a raised `HTTPException`
  carrying whatever was safely produced (e.g. spans), never a `200`, never
  a fabricated substitute.

### Stage-by-stage

| Stage | Normal method | Fallback hierarchy | On total failure |
|---|---|---|---|
| 0. Input validation | Pydantic `field_validator` on `ProvisionRequest.text` (non-empty, NFC-normalised, ≤`MAX_INPUT_LENGTH_CHARS`) | — | FastAPI's own `422`, before `run_pipeline` ever executes. Not a `pipeline_status` stage. |
| 1. Span generation | `spans.make_spans()` (structural regex) | → sentence-boundary splitter → whole text as one span `P1` | `500` (defensive-only; Stage 0 already guarantees non-empty source, so in practice this never fires) |
| 2. Simplification | Qwen3-8B via `llm_client.simplify_with_attempts` | 1 bounded retry on transient failures only (`SIMPLIFY_MAX_ATTEMPTS=2`) | `502`, spans preserved, downstream skipped |
| 3. Claim extraction | Structured LLM JSON (`CLAIM_MAX_TOTAL_CALLS=3` shared budget across both prompt variants + retries) | → clause splitter → sentence splitter → whole simplified text as one claim | Can't fail — final tier always returns something |
| 4. Evidence retrieval | Deterministic lexical/number/modality overlap (unchanged algorithm) | No match or an exact unresolved tie → whole provision offered as evidence | Claim gets no evidence → `unverified` |
| 5. NLI verification | `cross-encoder/nli-deberta-v3-small` (or `NLI_MODEL_PATH`) | Per-claim isolation; per-run (not permanent) model-load-failure isolation | Claim `unverified` with a specific reason |
| 6. Run logging | Temp-file write + `os.replace` (atomic) | — | `run_logging: "warning"`, result still returned, `overall` unaffected |

### `verification_reason` values

`no_evidence` (Stage 4 never found anything, model never called) ·
`nli_model_unavailable` (load failed; isolated per `verify_claims()` call,
not cached across requests) · `nli_input_too_long` (pair exceeds
`MAX_LENGTH=512` tokens — refused, never silently truncated) ·
`nli_invalid_output` (model returned without error but probabilities were
NaN/out-of-range/didn't sum to ~1) · `nli_inference_failed` (anything else).

### Warnings — integrity vs. diagnostic

Two classes, deliberately not lumped together:

- **Integrity** (`unsupported_number`, `numerical_conflict`,
  `clear_modal_conflict`) — pulls `overall` to `partial`.
- **Diagnostic** (`zero_lexical_overlap`, broad modality differences) —
  recorded on the claim, logged, but never changes `overall`. Otherwise
  routine observations would make "partial" fire far too often.

`clear_modal_conflict` is deliberately narrow: only an unambiguous flip
(prohibition disappears *and* permission appears, or the reverse) counts —
a provision legitimately mixing several modal clauses is not, by itself,
evidence of an error.

Claim-extraction warnings (`extraction_warnings`) check claim-vs-simplified-text
fidelity — a distinct failure mode from simplification-vs-source fidelity,
kept so a claim-extraction fault is attributable to claim extraction, not
silently blamed on the simplifier. Post-NLI warnings (`verification_warnings`)
check evidence-vs-claim and never override the NLI label — they're a
secondary safety net for a documented NLI weak spot (see below), not a
competing classifier.

## Design decisions worth remembering

- **Simplification (Stage 2) does not validate content.** No number check,
  no modality check, no repair loop. This was deliberate, not an oversight:
  pre-checking simplification faithfulness would pre-empt the exact thing
  the NLI verification stage exists to be evaluated on catching. Stage 2
  only asks "did the technical call work?" Semantic checks that *do* exist
  (Stage 3's `extraction_warnings`, Stage 5's `verification_warnings`) are
  either checking a different link in the chain (claim vs. simplified text,
  not source vs. simplified text) or running *after* the NLI verdict as a
  non-overriding diagnostic — neither pre-empts the auditor.
- **Why a deterministic post-NLI number check exists at all**: general NLI
  models lean heavily on structural/lexical similarity. Two near-identical
  sentences differing only in a digit ("5,000 rupees" vs "50,000 rupees")
  often don't move the model's confidence the way you'd want — it hasn't
  strongly learned "different number = contradiction" the way it's learned
  negation-based contradiction. The `numerical_conflict` warning exists to
  catch this specific, documented blind spot, not to duplicate the model.
- **Evidence ties are never resolved arbitrarily.** If two spans score
  identically and no deterministic tie-break can separate them, the pipeline
  does not silently keep "whichever came first" — it prefers the full
  provision (`evidence_method="full_provision_ambiguity_fallback"`) if
  available, or leaves the claim unevidenced (`evidence_ambiguity=true`) if
  not. Asking the NLI auditor to verify against an arbitrary pick would be
  asking it to grade against a coin-flip.
- **NLI model source is configurable** via `NLI_MODEL_PATH` (env var,
  defaults to `cross-encoder/nli-deberta-v3-small`). Accepts a local
  checkpoint directory path exactly the same way it accepts a hub id — no
  code change needed to swap to a fine-tuned checkpoint later.
- **Model-load failure isolation is per-run, not permanent.** A local
  variable inside one `verify_claims()` call remembers a load failure for
  the rest of that call only. The next `/pipeline` request gets a fresh
  attempt. There is deliberately no module-level "sticky" failure cache —
  that would make a transient failure permanent until a process restart.

## HTTP semantics

| Outcome | Status |
|---|---|
| `overall = success` or `partial` | `200` |
| Stage 0 input validation failure | `422` (Pydantic's own shape) |
| Span generation truly unusable (defensive-only) | `500` |
| Simplification failed after retry budget | `502`, `detail={message, spans, pipeline_status}` |
| Unexpected internal bug | `500` (unhandled, not caught anywhere in `run_pipeline`) |

## Provenance

Additive `SimplifyResponse.provenance`: `simplification_model`,
`claim_extraction_method`, `evidence_retrieval_method`, `nli_model`. Exists
so a stored run can later say whether it came from the pretrained or a
fine-tuned NLI model, and whether any fallback was involved.

## Debug telemetry (development only)

Every `/pipeline` call — success, partial, or a structured failure —
publishes to `frontend/js/app.js`:

- **Live**: `BroadcastChannel("clearclause_pipeline_debug")` — full detail
  including claim/evidence text, in-memory only, reaches only a debug tab
  open *right now*.
- **Persisted**: `localStorage["clearclause_pipeline_debug_runs"]` — last 5
  runs, **sanitised** (no raw provision/claim/evidence text — only ids,
  statuses, scores, warnings, probabilities, provenance).

`frontend/debug.html` + `frontend/js/debug.js` render this. Standalone page,
not linked from the normal app, reached only by opening it directly. The
normal UI never shows fallback/status/reason-code detail — a claim that
can't be verified just says "Unable to verify"; a failed simplification
says "The simplification could not be completed. Please try again."
`start.bat` now opens both the main app and the debug page automatically.

## Standalone endpoints

`/spans` uses the exact same fallback hierarchy as `/pipeline`
(`spans.make_spans_with_fallback`) — the two routes can't disagree about
what's valid. `/claims` and `/evidence` reuse the same shared functions and
additively expose method/status info. `/verify` uses `nli_client.verify_pair_safe`
— the same safe wrapper `verify_claims()` uses — so it interprets every NLI
failure mode identically instead of duplicating exception handling in
`main.py`.

## New environment variables (all optional, in `backend/.env.example`)

`MAX_INPUT_LENGTH_CHARS` (default `20000`), `NLI_MODEL_PATH` (default
`cross-encoder/nli-deberta-v3-small`), `SIMPLIFY_RETRY_BACKOFF_SECONDS` /
`CLAIM_RETRY_BACKOFF_SECONDS` (default `1.0` each).

## Verified end-to-end

Real backend + frontend servers started, real `/pipeline` call against a
live provision succeeded (`overall: success`, all six stages `success`,
`provenance` populated correctly, 4 claims extracted). Stage 0 rejection
(`422`) and `/spans` fallback-hierarchy reuse confirmed via real HTTP calls.
Frontend static files (`debug.html`, `js/debug.js`, `js/app.js`) confirmed
served correctly. `node --check` passes on both JS files.

**Not verified**: no browser automation tool was available this session,
so the actual live cross-tab `BroadcastChannel` behavior and the rendered
debug-page UI were not visually confirmed in a real browser — only the
server-side/static-file layer. Worth a manual click-through before relying
on it for a viva demo.

## Remaining single points of failure

- The LLM provider itself (Qwen3-8B / whatever `HF_MODEL`/`LLM_BASE_URL`
  resolves to) has no failover — by design (see CLAUDE.md: changing the
  generation model mid-evaluation would affect reproducibility). Documented
  as future work, not a bug.
- `outputs/runs/` and the debug `localStorage` history have no size cap
  beyond the debug page's "last 5 runs" — the run-log directory itself
  grows unbounded over time (pre-existing behavior, unchanged here).

## Explicitly out of scope / future work (documented, not built)

Provider/model failover, semantic or top-k evidence retrieval, multi-span
NLI, confidence thresholds, automatic NLI label overriding, database
persistence, circuit breakers, checkpoint/resume between pipeline stages, a
JS test framework.
