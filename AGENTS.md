# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## What this is

ClearClause is an academic prototype that simplifies Sri Lankan statute provisions into plain English while keeping every claim traceable back to the source text. The pipeline is designed to control hallucination: it forces every simplified claim to be linked to a source span and verified against it with an NLI model, rather than trusting the LLM's output directly.

## Commands

**Recommended way to run locally:**
Simply run `start.bat` from the root `app/` folder. This automatically opens two separate terminal windows to run the FastAPI backend (port 8000) and the Python static HTTP server for the frontend (port 3000), avoiding CORS and Live Server reloading issues.

**Manual Backend (FastAPI):**
Run from the `app/` root so the `backend` package resolves:

```
uvicorn backend.main:app --reload
```

Runs on `http://localhost:8000` (hardcoded as `API_BASE` in `frontend/js/app.js`). Interactive API docs at `/docs`.

Install backend deps:

```
pip install -r backend/requirements.txt
```

Frontend is static HTML/CSS/vanilla JS (Tailwind via CDN, no build step). **Do not open `frontend/index.html` directly via the file:// protocol**, as local `fetch()` calls for HTML components will fail due to strict browser CORS policies. You must serve the `frontend/` directory with a static file server (e.g., `python -m http.server 3000` from the `frontend` directory, or by using VS Code Live Server). There is no bundler.

**Tests:** there IS a pytest suite now. Run with the venv python:

```
venv\Scripts\python.exe -m pytest -q
```

Current state: 144 passing. Files: `tests/test_evidence.py`, `tests/test_spans.py`,
`tests/test_llm_client.py`, `tests/test_evidence_continuation.py`,
`tests/test_nli_client.py` (NLI mocked, no model download), and
`tests/pipeline_tests/test_evidence_units.py` (deterministic fixtures that lock
evidence-linking behaviour — real provisions, hand-written claims, exact expected
`evidence_span_ids`). No linter or CI. `HANDOFF.md` in the app root has a short
current-state summary.

### Environment

Backend reads `backend/.env` (loaded explicitly relative to `backend/llm_client.py`,
not the process cwd). **Changing `.env` requires a full server restart** — a
uvicorn `--reload` file-save is NOT enough, and if a change appears not to apply
it usually means an old backend is still holding port 8000 (kill python, re-run
`start.bat`).

Model source is selectable:

- `HF_TOKEN` — Hugging Face Inference API token (used only on the HF path).
- `HF_PROVIDER` — HF inference provider (default `"auto"`).
- `HF_MODEL` — model id used in BOTH paths (HF model id, or the model name for a
  custom endpoint). `build_chat_messages` has a special-case for `google/gemma*`.
- `LLM_BASE_URL` — **the switch.** If set, calls go to this OpenAI-compatible
  endpoint (Groq, Ollama, vLLM, …) via `LLM_API_KEY`, spending NO HF credits. If
  empty, the backend uses the Hugging Face Inference API. Current repo default is
  Groq (`llama-3.1-8b-instant`) for free testing; for the eval-validated setup /
  viva, remove the `LLM_*` lines and set `HF_MODEL=Qwen/Qwen3-8B`.
- `LLM_API_KEY` — key for the custom endpoint (dummy value fine for local Ollama).
- `HF_MAX_OUTPUT_TOKENS` — output token cap (default 4096). Qwen3 is a "thinking"
  model and can exhaust the budget on long provisions ("max token limit while
  thinking"); Groq's Llama has no thinking mode and avoids that.
- `HF_CLAIM_NO_THINK` — default `1`. Appends Qwen3's `/no_think` to the CLAIM
  extraction call only (mechanical JSON task, not part of the simplifier eval), to
  save tokens. Inert for non-Qwen3 models. Simplification keeps thinking ON.

## Architecture

### Pipeline (`backend/pipeline.py: run_pipeline`)

The `/pipeline` endpoint is the only one the frontend calls. It runs six steps and persists the full result as JSON to `outputs/runs/run_<timestamp>_<hex>.json` for every request:

1. **Span splitting** (`backend/spans.py: make_spans`) — deterministic, regex-based split of the raw provision text into evidence spans (`P1`, `P2`, ...) with exact char offsets into the *original, unstripped* input. Splitting is layered: priority legal markers (`Provided that`, `Explanation`, `Illustration`) win over structural subsection markers, which win over conjunctive clause starts (`Where`, `Unless`, `Except`, `Notwithstanding`), which win over semicolon-introduced branches (`; or, if` / `; and if` / `; if`), which win over generic sentence-boundary splits. The subsection pattern is ANCHORED (must follow line start or `. ; : — –`, but NOT a plain hyphen) so it does not split inside inline references like `subsection (1)`, `section 12(2)`, `paragraph (b)`. A tiny-chunk merge reabsorbs stray fragments (< 30 chars) EXCEPT semicolon branches and structural enumerators, so deliberate branch splits are not undone. This ordering is load-bearing.
2. **Simplification** (`backend/llm_client.py: simplify`) — one LLM call (HF or the custom endpoint) using a system+user prompt that forbids summarizing/adding info, requires preserving qualifiers (`before`, `during`, `unless`, `except`, `not`, `must`, `may`, `shall`), keeps offence elements separate from punishment, and keeps different punishment branches separate. Thinking stays ON here (matches the simplifier evaluation).
3. **Claim extraction** (`backend/llm_client.py: extract_claims`) — a second LLM call asks for atomic claims as strict JSON, with `/no_think` for Qwen3 (see env). Prompt now instructs splitting imprisonment/fine/both and separate branches into separate claims. `_parse_claims_json` is strict: rejects non-list/empty, non-string `claim_id`, and null/empty/non-string `claim_text` (this last one previously crashed `evidence._tokenize`), and renumbers all claims to `C1..Cn`. Retries once with `CLAIM_RETRY_SYSTEM`, then falls back to rule-based sentence splitting. Never raises on malformed output.
4. **Evidence linking** (`backend/evidence.py: link_claims_to_spans` + `resolve_evidence_units`) — first resolves each span into its SEMANTIC evidence unit: a normal span is itself; a headless span (a continuation branch `or, if …` or a lettered/roman list item `(a) …`) is joined to its nearest non-headless HEAD, siblings skipped. Numeric `(1)`/`(2)` are NOT list items — they can be a standalone rule or the head for lettered children. Each claim is then ranked against each unit's OWN-span tokens (numbers counted once, as tie-break tiers), and the winning unit's JOINED text is what goes to DeBERTa. Score 0 → no link (`None`). Output: `evidence_span_id` = selected span (provenance), `evidence_span_ids` = full unit `[head, item]` or `[id]` (what the model receives), `evidence_text` = the joined premise, `evidence_score` = int overlap.
5. **NLI verification** (`backend/nli_client.py: verify_claims` / `verify_pair`) — runs `cross-encoder/nli-deberta-v3-small` (loaded lazily once per process from the local HF cache; not re-downloaded per run) on each (evidence, claim) pair: `entailment → supported`, `contradiction → unsupported`, `neutral → uncertain`. Tokenizes with explicit `max_length=512` and logs a warning when input is truncated (more likely now that a continuation premise is two spans long). No linked evidence → `"unverified"` without running the model.
6. **Claim assembly** — builds `Claim` objects. `evidence_span_ids` now carries the real evidence unit (may be `[head, item]`), not just a one-element legacy list; `label`/`label_confidence` legacy fields are still populated.

### API surface (`backend/main.py`)

`/pipeline` is the only endpoint the frontend uses. `/spans`, `/simplify`, `/claims`, `/evidence`, `/verify` expose each pipeline step standalone for manual testing (Postman, `/docs`) — they are not wired into any frontend flow, so changes to the combined pipeline logic must be mirrored there if those steps are still expected to match.

### Frontend (`frontend/`)

Static, framework-free HTML/CSS/JS. `frontend/js/app.js` fetches partial HTML components (`components/navbar.html`, `components/disclaimer.html`) at runtime and rewrites their links via `data-nav`/`data-asset` attributes so the same partials work from both `index.html` (base path `.`) and `pages/simplify.html` (base path `..`). The simplify page posts to `/pipeline` once and derives both the "claims" and "spans" UI sections from that single response (see the comments in `runSimplification`/`runClaimExtraction` explaining why there's no second `/claims` call despite the function existing). Source spans render with `white-space: pre-wrap` (preserved whitespace); simplified text is split into paragraphs. When a claim's evidence is a combined unit, the badge shows `P1 + P2` and an expandable "Evidence sent to model" block renders the exact joined premise — so what the user sees is what DeBERTa scored. The per-claim number is labelled "NLI label confidence" (uncalibrated max-softmax, not a correctness probability).

## KNOWN OPEN ISSUE — rioting-style false red (highest priority)

For a two-branch punishment provision (e.g. rioting "committed" vs "not
committed"), a correct claim about the SECOND branch can be linked to the first
span and shown red/unsupported. Root cause: the real LLM writes claims that
repeat the head's subject/verb ("the person … must be punished"), so lexical
overlap favours the head span over the branch. Deterministic fixtures pass but
real wording defeats it; the number tie-break is inert because amounts are words
("six months"), not digits. NOT yet fixed — full diagnosis in
`verification_archive/semantic_evidence_resolver_report.md`. Proposed fix (needs
approval before coding): word-number recognition PLUS number-weighted ranking for
sibling branches, acceptance-tested against the REAL rioting output.

## Out of scope (do NOT do without explicit approval)

No fine-tuning; no dataset CSV changes; do NOT add the original source text to
claim extraction; no reverse completeness checking; no broad refactors. The
sibling NLI baseline / fine-tuning work lives in `D:\nli_baseline_test`.

### Run logs (`outputs/runs/`)

Every `/pipeline` call writes a full `SimplifyResponse` JSON snapshot here, named `run_<UTC timestamp>_<6-hex>.json`. Useful for inspecting what a given request actually produced (spans, claims, evidence links, verification labels) without re-running the LLM.

> [!WARNING]
> **VS Code Live Server Bug:** Because the backend writes a new JSON file to this folder on every request, running VS Code Live Server from the root `app/` directory will detect the file change and force-reload the browser *right as the frontend is trying to render the results*, causing the output to disappear instantly. To avoid this, either use `start.bat`, serve only from the `frontend/` directory, or add `"liveServer.settings.ignoreFiles": ["outputs/**"]` to your VS Code settings.
