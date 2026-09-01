# Model Evaluation Log

Working notes on the simplifier-model comparison for ClearClause, covering all four
`notebooks/baseline_eval*` rounds. Built by reading the actual notebook source +
saved cell outputs (not re-run) on 2026-08-05, so it reflects exactly what's committed
in the repo right now, including gaps and stale results.

---

## 1. What's being evaluated

Six candidate models are asked to rewrite a Sri Lankan legal provision (`source_text`)
into plain English, scored against a human gold simplification (`simple_text`) on a
20-row pilot set: `data/pilot/pilot_20_baseline_eval.csv`.

**Metrics**, computed per row and averaged:
- **FKGL** (Flesch-Kincaid Grade Level) and **ARI** (Automated Readability Index) on the model's own output — lower = easier to read.
- **ROUGE-L F1** and **BERTScore F1** against `simple_text` — higher = closer to the gold rewrite.

**Source baseline** (`results/source_reference_metrics_summary.csv`, computed in
`notebooks/source_metrics.ipynb`, 20 rows):

| | Avg FKGL | Avg ARI |
|---|---|---|
| Raw source legal text | 22.389 | 25.892 |
| Human gold simplification | 11.88 | 12.487 |

The source text sits around graduate/professional reading level; the human gold
rewrites land around high-school level. That ~10-11 point FKGL/ARI drop is the
target a good simplifier should approach.

---

## 2. Models across the four rounds

Each round is a folder of 6 notebooks, one per model. `baseline_eval/` is the
original round; `_2`, `_3`, `_4` are later iterations.

| Slot | Round 1 (`baseline_eval`) | Round 2 (`_2`) | Round 3 (`_3`) | Round 4 (`_4`) |
|---|---|---|---|---|
| BART | `facebook/bart-large` (raw) | `facebook/bart-large` (raw) | **`eugenesiow/bart-paraphrase`** | `eugenesiow/bart-paraphrase` |
| FLAN-T5 | `google/flan-t5-large` | `google/flan-t5-large` | `google/flan-t5-large` | `google/flan-t5-large` |
| Gemma | `google/gemma-3-4b-it` | **`google/gemma-3n-E4B-it`** | `google/gemma-3-4b-it` | `google/gemma-3-4b-it` |
| Llama | `meta-llama/Llama-3.2-3B-Instruct` | **`meta-llama/Llama-3.1-8B-Instruct`** | `meta-llama/Llama-3.2-3B-Instruct` | `meta-llama/Llama-3.2-3B-Instruct` |
| Mistral | `mistralai/Mistral-7B-Instruct-v0.3` | **`mistralai/Mistral-7B-Instruct-v0.2`** | `mistralai/Mistral-7B-Instruct-v0.3` | `mistralai/Mistral-7B-Instruct-v0.3` |
| Qwen | `Qwen/Qwen3-8B` | **`Qwen/Qwen3-4B-Instruct-2507`** | `Qwen/Qwen3-8B` | `Qwen/Qwen3-8B` |

**Reading this table:** Round 2 was a one-off detour testing alternate sizes/versions
of Gemma, Llama, Mistral, and Qwen (leaving BART and FLAN-T5 untouched). It was not
carried forward — Round 3 reverts to the Round 1 model lineup. The one change that
*did* stick from Round 3 onward is BART: swapped from the raw, non-fine-tuned
`bart-large` to `eugenesiow/bart-paraphrase` (bart-large fine-tuned on Quora/PAWS/MSR
paraphrase corpora), because raw `bart-large` has no incentive to rewrite clean text
(see `review.md`, and §5 below).

---

## 3. Per-model generation setup (as of Round 3/4, i.e. current)

| Model | Loading | Decoding | Prompt style | Stop-token handling |
|---|---|---|---|---|
| BART (`eugenesiow/bart-paraphrase`) | fp16, no quant | beam search, `num_beams=4`, `max_new_tokens=256` | raw `source_text`, no instruction (not instruction-tuned) | default |
| FLAN-T5-large | fp16, no quant | beam search, `num_beams=4`, `max_new_tokens=256` | full natural-language instruction + rule list (FLAN-T5 *is* instruction-tuned, unlike raw T5/BART) | default |
| Gemma-3-4b-it | 4-bit nf4 (bitsandbytes, double quant) | greedy, `num_beams=1`, `max_new_tokens=192` | chat template, system+user messages | **fixed in Round 3**: `eos_token_id=[eos_token_id, <end_of_turn>]` (was previously just base `eos`, id 1, missing the real turn-end token id 106) |
| Llama-3.2-3B-Instruct | 4-bit nf4 | greedy, `num_beams=1`, `max_new_tokens=192` | chat template, system+user messages, **refined rule list** (see below) | `tokenizer.eos_token_id` — correct as-is (resolves to `<\|eot_id\|>`) |
| Mistral-7B-Instruct-v0.3 | 4-bit nf4 | greedy, `num_beams=1`, `max_new_tokens=192` | chat template, system+user messages | `tokenizer.eos_token_id` — correct (`</s>`, id 2) |
| Qwen3-8B | 4-bit nf4 | greedy, `num_beams=1`, `max_new_tokens=192`, `enable_thinking=False` | chat template, system+user messages | `tokenizer.eos_token_id` — correct (`<\|im_end\|>`) |

Gemma, Mistral, and Qwen share the **exact same** system + user prompt text (rule
list: preserve SHALL/MAY/SHALL NOT, numbers, dates, fines, conditions, exceptions,
legal roles; no examples/commentary) — a controlled-comparison choice, not a
per-model-tuned one; see `review.md`'s "Cross-model prompt note" if the thesis
write-up wants to discuss it as a limitation. **Llama's rule list was refined** to
additionally forbid preamble/label text (e.g. "Here is the simplified version") and
bullet-point formatting, and to require multi-part conditions to be written as
separate short sentences rather than one long one — its outputs were diverging from
the others' plain-prose style in ways that hurt its similarity to the gold
references. See §4 for details on the changes and their measured effect, and §6 for
how it stacks up against the other models.

`MAX_INPUT_TOKENS = 1024`, `MAX_NEW_TOKENS = 192` for all four chat models.

---

## 4. Prompt and code changes applied to Llama (Round 4)

A row-by-row scan of all 20 outputs from all four chat models (checking for
preamble text, prompt/rule leakage, and markdown formatting) found two problems
unique to Llama — Gemma, Mistral, and Qwen3 came back clean (0/20) on the same
checks:

- **Preamble text**: 6/20 outputs (30%) opened with boilerplate like *"Here's a
  simplified version:"* before the actual simplification — never present in the
  gold references, and directly hurting ROUGE-L/BERTScore since the reference text
  never contains that framing.
- **Bullet-point reformatting**: 4/20 outputs restructured multi-part punishments
  as markdown bullet lists instead of the prose the gold references use.

### Changes made

1. **Prompt rule list extended** (`build_messages` in
   `notebooks/baseline_eval_4/baseline_eval-llama3.2-3b-instruct.ipynb`) — two rules
   added to the existing list:
   - *"Do not include any preamble, introduction, or label such as 'Here is the
     simplified version' or 'Plain English:' — output only the simplified provision
     itself."*
   - A prose-formatting rule banning bullet points/lists (revised once — see below).
2. **Post-processing safety net added**: a `strip_preamble()` regex function (new
   cell, plus `import re` in the imports cell) that strips a known preamble pattern
   from the decoded text as a backstop, in case the prompt rule alone doesn't fully
   stick.
3. **Prose rule revised once**: the first version told the model to write "a single
   paragraph." Row-level inspection showed this pushed multi-condition clauses into
   one very long run-on sentence instead of the several short sentences the gold
   references actually use. Revised to explicitly allow multiple sentences: *"Write
   the answer as plain prose using full sentences. Break multi-part conditions or
   punishments into separate short sentences instead of one long sentence."*

### Measured effect

| | Avg FKGL | Avg ARI | Avg ROUGE-L | Avg BERTScore F1 | Preamble | Bullets |
|---|---|---|---|---|---|---|
| Before (original shared prompt) | 17.10 | 19.41 | 0.381 | 0.906 | 6/20 | 4/20 |
| After (current — both fixes applied) | 17.70 | 20.25 | 0.404 | 0.918 | 0/20 | 0/20 |

Preamble and bullet formatting are both fully eliminated, and ROUGE-L (+0.023) and
BERTScore (+0.012) improved — Llama's outputs are closer to the gold wording now
that the boilerplate/formatting divergence is gone. FKGL/ARI moved slightly the
wrong way (+0.6 / +0.8), not closer to the gold target. An intermediate version
(rules 1+2 only, before the prose rule was revised) actually had *better*
ROUGE-L/BERTScore (0.432 / 0.923) but *worse* FKGL/ARI (18.17 / 20.58) — revising
the prose rule traded some of that gold-similarity gain back for a small
readability improvement, without fully solving the underlying issue: one clause
(`SL00184`) still comes out as a single 30+ FKGL run-on sentence despite the rule
explicitly asking for shorter sentences. A 3B model may simply have limited
steerability on this specific instruction; further prompt iteration on this point
wasn't pursued. The numbers used everywhere else in this document (§6) are from
the current, fully-fixed version.

**Other changes made across rounds** (not Llama-specific) — the BART
paraphrase-checkpoint swap and the Gemma stop-token/truncation bug fixes — are
covered in §5 below.

---

## 5. What changed between rounds, and why

**Round 1 → Round 2:** Explored swapping in different-sized/newer variants of four
of the six models (bigger Llama, smaller Qwen and Mistral, a different Gemma
family member). This round appears abandoned — only the untouched FLAN-T5 notebook
has a saved execution result; BART, the three swapped models, and Gemma all have no
saved outputs. Round 3 reverts to the Round 1 model choices, so none of the Round 2
alternates ended up as the "kept" versions.

**Round 1/2 → Round 3:** Two fixes applied, both traceable to `review.md`:

1. **BART swapped to a paraphrase-tuned checkpoint.** Raw `facebook/bart-large` is a
   denoising autoencoder with no instruction-following ability and, per its own model
   card, is "mostly meant to be fine-tuned" — fed clean text, it has little reason to
   change anything. `eugenesiow/bart-paraphrase` is fine-tuned specifically to rewrite
   input sentences, which is a much fairer "best this architecture can do" comparison
   point. Code comment in the notebook explicitly credits this reasoning.
2. **Gemma's stop-token bug fixed.** Gemma's chat turns end on `<end_of_turn>`
   (token id 106), not the base `<eos>` (id 1) that the notebook was previously
   passing alone as `eos_token_id`. That meant generation could run past the real
   stop point and append trailing junk to every output, silently skewing all four
   metrics for Gemma. Round 3's `build_messages`/`generate_output` cells now pass
   `eos_token_id=[eos_token_id, end_of_turn_id]`, and separately fix the truncation
   direction bug (previously truncating the tokenized prompt from the *front*, which
   would strip the system instructions on a long input — now the source text is
   capped *before* the prompt is built, at `MAX_SOURCE_TOKENS = 700`).

**Round 3 → Round 4:** No model/checkpoint changes for any of the six models —
Round 4 started as a clean copy of Round 3, prepared for a fresh full run, then had
the Llama-specific prompt changes from §4 applied on top. It has since been executed
to completion for all six models, with per-row outputs saved to
`notebooks/baseline_eval_4/outputs/<model>/` — see §6.

---

## 6. Actual results — Round 4 (real, post-fix, all 6 models)

Rounds 1–3 only ever produced partial or stale results (see the git history of this
file for that earlier analysis). **Round 4 has since been run to completion**, with
full per-row outputs saved to `notebooks/baseline_eval_4/outputs/<model>/` — this is
the first trustworthy, complete comparison across all six models with the Gemma
stop-token/truncation fix actually in effect. All numbers below are computed fresh
from those CSVs, N = 20.

### Quantitative

| Model | Avg FKGL | Avg ARI | Avg ROUGE-L | Avg BERTScore F1 | Exact copies of source |
|---|---|---|---|---|---|
| *Gold reference (target)* | *11.88* | *12.49* | *—* | *—* | *—* |
| **Qwen3-8B** | 15.73 | 18.36 | 0.425 | 0.923 | 0/20 |
| **Mistral-7B-Instruct-v0.3** | 15.85 | 17.42 | **0.442** | **0.924** | 0/20 |
| Gemma-3-4b-it | 15.86 | 17.73 | 0.415 | 0.921 | 0/20 |
| Llama-3.2-3B-Instruct | 17.70 | 20.25 | 0.404 | 0.918 | 0/20 |
| BART (`eugenesiow/bart-paraphrase`) | 20.14 | 22.94 | 0.317 | 0.898 | 0/20 |
| FLAN-T5-Large | 23.64 | 27.44 | 0.334 | 0.898 | **9/20 (45%)** |

*Llama's numbers reflect its refined prompt (§4), not the original shared one. On
that original prompt it scored 17.10 / 19.41 / 0.381 / 0.906 — see §4 for the full
before/after breakdown.*

**Mistral-7B-v0.3 and Qwen3-8B have the best FKGL/ARI and are essentially tied for
best ROUGE-L/BERTScore too**, closest to the gold target (~12) and closest to the
gold wording. Gemma is a clear third on both fronts. Llama's prompt refinement
closed most of its gold-similarity gap versus its original prompt (§4), but it's
still last of the four instruct models on every one of the four metrics. BART and
FLAN-T5 are far behind everything else.

**But averages hide how often each model actually wins a given row.** Counting the
single best model per row (20 rows) tells a different story:

| | ROUGE-L wins | BERTScore wins |
|---|---|---|
| **Qwen3-8B** | 6 | **9** |
| Gemma-3-4b-it | 6 | 5 |
| Mistral-7B-v0.3 | 5 | 4 |
| FLAN-T5-Large | 2 | 0 |
| Llama-3.2-3B | 1 | 1 |
| BART | 0 | 1 |

Qwen3 wins the most individual rows on both metrics — nearly double Mistral's count
on BERTScore (9 vs. 4) — despite Mistral's average edging it out. Mistral's higher
average comes from being consistently close-behind rather than from winning outright
most often.

**FLAN-T5's core failure mode: it echoes the source verbatim on 9/20 rows (45%).**
Not a borderline case — on nearly half the pilot it returns the input unchanged
instead of simplifying, consistent with `review.md`'s prediction that the
multi-constraint rule-list prompt is too much for a 780M model to reliably follow.

**Every model degrades on clauses with conditional/exception logic**
(`has_exception=1`, 7/20 rows) — the legally highest-stakes cases — but by very
different amounts:

| Model | ROUGE-L, exception rows | ROUGE-L, other rows | Drop |
|---|---|---|---|
| Llama-3.2-3B | 0.400 | 0.450 | -0.050 |
| Qwen3-8B | 0.412 | 0.432 | -0.020 |
| Gemma-3-4b-it | 0.364 | 0.443 | -0.079 |
| Mistral-7B-v0.3 | 0.386 | 0.473 | -0.087 |
| BART | 0.242 | 0.357 | -0.115 |
| FLAN-T5-Large | 0.253 | 0.378 | -0.125 |

### Qualitative

Manually read all 6 models' output on all 7 `has_exception=1` rows — the rows with a
real conditional or carve-out clause, and the legally highest-stakes cases in the
pilot. Scored each as "preserved" (the condition/exception survives, however
reworded) or "dropped/garbled" (the model states the rule as unconditional, or
mistranslates the exception).

| Model | Conditional/exception preserved |
|---|---|
| **Qwen3-8B** | **7/7** — correct on every exception row |
| Mistral-7B-v0.3 | 6/7 clean, 1 garbled-but-present (the property-forfeiture row — keeps the "except for the government's benefit" concept but the phrasing is awkward) |
| Gemma-3-4b-it | 4/7 — drops the conditional entirely on the mutiny row, partially drops the exception on the property-forfeiture row |
| Llama-3.2-3B | 4/7 — drops the exception entirely on the intoxication row (`SL00098`, the one clause built *around* that exception), mistranslates the exception on the property-forfeiture row ("cannot own any property for the benefit of anyone else" inverts the source's meaning), drops the conditional on the mutiny row |
| BART / FLAN-T5 | Trivially "preserve" everything since they mostly copy the source verbatim rather than actually simplifying — not a meaningful comparison here |

The clearest single example is the mutiny/abetment clause (`SL00193`): source reads
"...shall, **if mutiny be committed in consequence of that abetment**, be
punished..." — Mistral and Qwen3 both keep that conditional ("if mutiny occurs due
to that help" / "and mutiny actually happens because of that help"), while Gemma and
Llama state the punishment as unconditional.

**Correction to an earlier read of this data:** I'd previously generalized from that
one row to "Gemma and Llama silently drop conditionals" as if it were a consistent
per-model trait. The full 7-row check shows that's row-specific, not universal —
Gemma and Llama get 4 of the 7 exception rows right, they just each fail on a
different subset. Qwen3 is the only model that got all 7 right.

**One new defect found in this pass**: Gemma leaks a literal prompt-template label
into row 14's output — `**Plain English:**` appears at the start of the generated
text, ahead of the actual simplification. It's isolated (1/20 rows) and the earlier
automated regex scan for prompt-leakage missed it — the check looked for `Plain
English:` with a trailing word boundary, which silently fails to match when the
colon is immediately followed by markdown asterisks (`:**`), since neither character
is a "word" character for `\b` to anchor on. Not worth a code fix given how rare it
is, but worth knowing it's there if quoting Gemma outputs directly.

### Bottom line

**Qwen3-8B and Mistral-7B-Instruct-v0.3 are the two strongest simplifiers in this
pilot, with Qwen3 the stronger of the two.** On raw averages they're close enough to
call a tie (Mistral edges ahead on 3 of 4 metrics, but by small margins). What
separates them: Qwen3 wins the most individual rows outright (9/20 on BERTScore vs.
Mistral's 4/20), and it's the only model that correctly preserved the
conditional/exception logic on **all 7** of the pilot's exception rows — Mistral got
6/7. Both are well ahead of Gemma and Llama on this dimension (4/7 each). If the
thesis needs a single recommended model, that's Qwen3-8B; Mistral is the clear
second choice and a reasonable fallback (e.g. if Qwen3's larger size is a deployment
concern). Gemma is a reasonable middle option. Llama's refined prompt closed most of
its gold-similarity gap but still reads more complex than the other three instruct
models and drops exception clauses more often. BART (even the paraphrase-tuned
checkpoint) and FLAN-T5 are not viable as-is — FLAN-T5 in particular is failing
outright on almost half the pilot by returning the input unchanged.

### Other things worth flagging

- **FLAN-T5's doc-comment inaccuracy from `review.md` was never fixed**: the
  comment still claims a "1024 input / 256 output" training window, while the
  actual `tokenizer_config.json` spec is 512. `max_length=1024` is still used for
  tokenization. Likely harmless given how short the pilot clauses are, but the
  comment misstates the model's documented spec.
- **Decoding isn't held constant across models**: BART/FLAN-T5 use beam search
  (`num_beams=4`), Gemma/Llama/Mistral/Qwen use greedy decoding (`num_beams=1`).
  Worth a line in the thesis methodology section so low BART/FLAN-T5 scores aren't
  misread as purely architecture-driven.
- **`data/pilot/pilot_20_baseline_eval.csv`'s own output columns are still empty**
  — the real per-row outputs live in `notebooks/baseline_eval_4/outputs/<model>/`,
  not in the shared pilot CSV. Use the per-model CSVs there for any per-row quoting.

---

## 7. File map

| Path | Contents |
|---|---|
| `model_notes.md` | One-paragraph rationale for each of the 6 simplifier candidates + the planned separate NLI verifier (`microsoft/deberta-v3-small`, not yet built) |
| `review.md` | Code review of the Round 1 notebooks — source of the Gemma stop-token/truncation bug fixes and the BART-checkpoint-swap reasoning applied in Round 3 |
| `data/pilot/pilot_20_baseline_eval.csv` | The 20-row pilot set (source clause + gold simplification + metadata); output columns present but empty |
| `notebooks/baseline_eval/` | Round 1 — original 6 notebooks, partial results (FLAN-T5, Gemma, Llama, Mistral) |
| `notebooks/baseline_eval_2/` | Round 2 — alternate model-size exploration, effectively abandoned, only FLAN-T5 has a saved result |
| `notebooks/baseline_eval_3/` | Round 3 — Gemma bug fixes + BART checkpoint swap applied in code; saved outputs for Gemma appear stale/pre-fix |
| `notebooks/baseline_eval_4/` | Round 4 — clean copy of Round 3's fixes, **fully executed**; see `outputs/` below for the real per-model results |
| `notebooks/baseline_eval_4/outputs/<model>/` | Per-model results CSV (all 20 rows, source + gold + model output + FKGL/ARI/ROUGE-L/BERTScore per row) and summary-table PNG — the source of truth for §6 |
| `notebooks/source_metrics.ipynb` | Computes source-vs-gold FKGL/ARI baseline (§1 table) |
| `notebooks/output/`, `results/` | Duplicate copies of `source_reference_metrics_{summary,per_row}.csv` and `avg_source_readability.png` — no per-model results are saved here |
| `requirements-colab.txt` | Colab pip pins: `transformers`, `datasets`, `accelerate`, `peft`, `trl`, `evaluate`, `textstat`, `bert-score`, `sacrebleu`, `sentencepiece`, `pandas`, `numpy`, `scikit-learn`, `jupyter` |
| `checkpoints/`, `scripts/` | Currently empty |
| `fyp hugging face token.txt` | **Plaintext HF token in the repo — not covered here further, but should be rotated/gitignored before any push.** |
