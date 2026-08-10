# Semantic Evidence Resolver — Phase 0 + Phase 1 Report

## Summary

Applied the approved Phase 0 (deterministic fixture harness) and Phase 1
(`resolve_evidence_units` + rank semantic units, score-own-text/send-joined-text,
skip siblings, numeric heads usable). All unit tests pass. **But a live run
exposed that Phase 1 does not fully fix the real rioting case**, because the
live LLM writes claims that repeat the head's subject and verb. This report
records exactly what passed, what did not, and why — with real numbers.

Honest headline: **deterministic acceptance = PASS; live rioting acceptance =
FAIL.** No regression versus the previous behaviour. The remaining failure is a
retrieval-ranking limitation, not a span-splitting bug.

---

## What was applied

- `backend/evidence.py`
  - `_ITEM_START` (lettered/roman only; numeric `(1)`/`(2)` excluded so a
    numbered subsection can act as a head), `_is_headless(text)`.
  - `resolve_evidence_units(spans)` — head + this item, siblings skipped,
    walk back to nearest non-headless head, never reach past index 0.
  - `link_claims_to_spans` rewritten to rank units: score against each span's
    OWN tokens, send the unit's JOINED text as the premise. Tie-break prefers
    the more specific (joined) unit; `evidence_score` stays an integer.
- `tests/pipeline_tests/test_evidence_units.py` — 11 fixtures (incl. nested
  enumerator), exact `evidence_span_ids` per claim, API-contract invariants,
  resolver edge cases.
- API fields unchanged (`evidence_span_id`, `evidence_span_ids`,
  `evidence_text`, `evidence_score`). `evidence_span_id` = selected span = last
  id of a `[head, item]` unit.

Not touched: dataset CSVs, fine-tuning, Qwen prompts, reverse completeness,
word-number recognition, claim linter, frontend, `nli_client.py`, `spans.py`.

---

## Test results — venv (`venv\Scripts\python.exe -m pytest -q`)

```
144 passed in 5.97s
```

Includes `tests/test_nli_client.py` (real torch import) and the new
`tests/pipeline_tests/`. Sandbox run (torch file excluded): 137 passed.

### Deterministic acceptance — ALL PASS

| Criterion | Result |
|---|---|
| rioting no-riot claim → P1+P2 (fixture wording) | PASS |
| absconding Court claims → P1+P2 | PASS |
| absconding ordinary claims → P1 | PASS |
| maintenance does not split subsection (1) | PASS |
| three-branch claim about branch 3 → P1+P3 (not P1+P2+P3) | PASS |
| enum list claim about (b) → head+(b) (not head+(a)+(b)) | PASS |
| nested enumerator: (1) usable as head → P1+P2 / P1+P3 | PASS |
| numeric (1)/(2) stay standalone when not a list head | PASS |

---

## Live runs (real Qwen + DeBERTa via /pipeline)

Backend up (`/health` ok). HF credits were partially available.

| Provision | File | Result |
|---|---|---|
| maintenance | `manual_pipeline_maintenance_v3.json` | OK — 1 span, 2 claims, both supported |
| rioting | `manual_pipeline_rioting_v3.json` | Ran — **but C5–C7 wrong (see below)** |
| absconding | — | NOT OBTAINED — HTTP 502 "Model hit the max token limit while thinking" (Qwen thinking-token exhaustion, unrelated to linking) |

### The live rioting failure (acceptance criterion FAILS)

The run produced (`run_20260722_091308_55c782`):

```
C5 ids=['P1'] unsupported   "If the person's provocation does not lead to a riot,
                             they must be punished with imprisonment for up to six months."
C6 ids=['P1'] unsupported   "...they must be punished with a fine."
C7 ids=['P1'] unsupported   "...both imprisonment for up to six months and a fine."
```

These are correct claims about the "riot not committed" branch (P2), but they
were linked to P1 ("...one year") and so show **red / unsupported**. The
acceptance target was `['P1','P2']`.

**Why — measured token scores (own-text):**

```
C5 vs P1(own): 5  overlap = person, provocation, punished, imprisonment, MODAL
C5 vs P2(own): 4  overlap = imprisonment, months, not, six
```

The live claim restates the offence's subject and verb ("the person's
provocation … they must be punished") — those words live in the **head** (P1).
They outweigh the branch's distinctive words ("six", "months") by one point, so
the claim links to P1. My fixture claims were terser and branch-focused, so they
passed; the real LLM wording defeats own-text ranking.

**This is not a regression.** Before this change the same claims also linked to
P1 and showed unsupported. Phase 1 fixes the well-worded cases and does not make
the live case worse.

---

## Why the quick alternatives don't fix it either (tested on the live claims)

- **Joined-text scoring** (score the claim against head+item): pulls **every**
  claim onto `['P1','P2']`, including the P1-only "riot occurs" claims C1–C4.
  A joined unit contains all the head's tokens, so it dominates every claim.
  This is the "superset" failure — worse than own-text.
- **Distinctive-token scoring** (ignore tokens shared by sibling spans): tried
  on the live claims, still returned P1 for C5–C7 in a quick prototype. Even
  the distinctive branch tokens ("six", "months") do not reliably outrank the
  distinctive head tokens the claim also repeats ("provocation", "person").

The root problem is deeper than span structure: **lexical bag-of-words
retrieval cannot reliably tell two punishment branches apart when the simplified
claim repeats the shared offence subject and verb.** The only strong
discriminators are the branch-specific facts ("one year" vs "six months",
"committed" vs "not committed"), and those are 2–3 tokens against a shared
head of 5+ tokens.

---

## Recommendation (needs your decision — beyond approved Phase 1 scope)

Phase 1 is a correct, tested foundation and should stay. To actually fix the
live branch case, one of these is needed, and each is a new mechanism I did NOT
implement without approval:

1. **Weight branch-discriminating signal above shared-head signal** — e.g. when
   two candidate units share a head, decide between them using only the tokens
   that differ, and weight negation/number/time tokens ("not", "six", "months",
   "one year") more heavily. This is where **word-number recognition (Phase 2)**
   would finally pay off, but it must also up-weight numbers above raw lexical
   overlap, not just use them as a tie-break.
2. **Ask the extractor to tag each claim with its branch/condition** — a
   claim-side change (prompt), explicitly out of scope here.
3. **Score with the NLI model itself** for ambiguous multi-branch provisions
   (run the claim against each candidate premise, pick the most entailed) —
   heavier, but semantically correct.

My engineering recommendation: keep Phase 1, then do Phase 2 as
"word-numbers **plus** number-weighted ranking for sibling branches", and
re-test against this exact live rioting output as the acceptance case — not a
hand-written fixture, since hand-written claims hid this failure.

---

## Files

- `manual_pipeline_maintenance_v3.json` — real output, OK
- `manual_pipeline_rioting_v3.json` — real output, C5–C7 mislinked (documented)
- `manual_pipeline_absconding_v3.json` — NOT written (Qwen 502 token limit)
- `tests/pipeline_tests/test_evidence_units.py` — new harness (all pass)
