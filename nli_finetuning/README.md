# ClearClause NLI — statutory faithfulness verifier

Tools for evaluating and fine-tuning NLI checkpoints as a faithfulness
verifier for **ClearClause**, which simplifies Sri Lankan statutory
provisions into plain English. The verifier checks whether a simplified claim is
faithful to the original law.

`cross-encoder/nli-deberta-v3-small` was the original baseline checkpoint.
Five checkpoints have since been evaluated head-to-head on the same fixed
dataset — see "Multi-model comparison" below for results and methodology.
Fine-tuning (`finetuning/finetune_deberta.py`) still targets
`nli-deberta-v3-small` specifically. All four planned fine-tuning runs have
been completed — see "DeBERTa Fine-Tuning Experiment" and
"DeBERTa-v3-base Fine-Tuning Experiments" below for the full run reports,
and "3. Fine-tuning (scaffold)" for how to reproduce or rerun them.

*(Merged 27 August 2026 from this file plus `README-multi-model-nli.md` and*
*`README-finetuning-experiments.md`, after confirming their content was*
*genuinely unique rather than duplicated — those two files, along with*
*`README-sequential-nli-run.md`, described a launcher script*
*(`run_all_nli_evaluations.bat` / `run_all_nli_sequential.bat`) that no*
*longer exists on disk and a "not yet executed" status that was no longer*
*true; those stale parts were dropped rather than merged.)*

- **premise** = original statutory span
- **hypothesis** = simplified atomic legal claim
- No RAG. The model sees only the (premise, hypothesis) pair.

## Label mapping

```
0 = contradiction
1 = entailment
2 = neutral
```

`gold_label` may be given as an integer id (`0`/`1`/`2`) or a label name
(`contradiction`/`entailment`/`neutral`, case-insensitive).

## Folder contents

```
evaluation_scripts/            Nine standalone evaluation scripts (inference only)
  evaluate_deberta_baseline.py   cross-encoder/nli-deberta-v3-small (zero-shot)
  evaluate_deberta_v3_base.py    MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli (zero-shot)
  evaluate_roberta_large_mnli.py FacebookAI/roberta-large-mnli (zero-shot)
  evaluate_distilroberta_nli.py  cross-encoder/nli-distilroberta-base (zero-shot)
  evaluate_modernbert_nli.py     tasksource/ModernBERT-base-nli (zero-shot)
  evaluate_finetuned_deberta.py  finetuning/finetuned_deberta on frozen internal_test v2 /
                                 external_test v2 / heldout_combined only (leakage-free)
  evaluate_finetuned_deberta_v3_base.py          finetuning/finetuned_deberta_v3_base (1st attempt)
  evaluate_finetuned_deberta_v3_base_retuned.py  finetuning/finetuned_deberta_v3_base_retuned
  evaluate_finetuned_deberta_v3_base_boosted.py  finetuning/finetuned_deberta_v3_base_boosted
  nli_evaluation_utils.py        shared dataset/inference/report/metric logic
make_v2_datasets.py             Remove train->eval leakage, write "v2" data files
finetuning/                    Fine-tuning scaffold + artifacts (starting point, editable)
  finetune_deberta.py            Scaffold targeting cross-encoder/nli-deberta-v3-small
  finetune_deberta_v3_base.py    Same scaffold targeting MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli.
                                 Reused/edited in place across 3 attempts (currently holds the
                                 "boosted" config) -- see "DeBERTa-v3-base Fine-Tuning Experiments"
                                 below for the hyperparameters used for each of the 3 output folders.
  finetuned_deberta/                    Saved fine-tuned nli-deberta-v3-small -- the best model overall
  finetuned_deberta_v3_base/            Saved fine-tuned DeBERTa-v3-base, 1st attempt (best of the 3 attempts)
  finetuned_deberta_v3_base_retuned/    Saved fine-tuned DeBERTa-v3-base, retuned attempt (worst of the 3)
  finetuned_deberta_v3_base_boosted/    Saved fine-tuned DeBERTa-v3-base, boosted attempt (2nd of the 3)
  data/                          Copy of data/ used by all fine-tuning runs
  outputs/finetune_epoch_metrics.csv                    Per-epoch metrics, nli-deberta-v3-small run
  outputs/finetune_v3_base_epoch_metrics.csv            Per-epoch metrics, DeBERTa-v3-base 1st attempt
  outputs/finetune_v3_base_retuned_epoch_metrics.csv    Per-epoch metrics, DeBERTa-v3-base retuned attempt
  outputs/finetune_v3_base_boosted_epoch_metrics.csv    Per-epoch metrics, DeBERTa-v3-base boosted attempt
finetune_deberta_colab.py       A Colab-adapted variant of the fine-tuning scaffold, referenced
                                 in earlier project notes; does not currently exist in this
                                 project (nothing currently depends on it)
cache_one_nli_model.py          Download/cache one HF checkpoint by name (no data read)
run_all_nli_visible_progress.bat / run_and_log_visible_progress.ps1
                                 Double-click launcher: download + evaluate all 5 models in sequence
requirements.txt               Core dependencies
data/                          Datasets (originals + v2)
outputs/                       Generated predictions, reports, metrics — one subfolder per model
```

### Data files

Original splits (built from Sri Lankan statutes; Penal Code in-domain,
Maintenance Act out-of-domain):

```
data/nli final clean train.csv              (2667 rows) — training only
data/nli final clean validation.csv         ( 530 rows)
data/nli final clean internal test.csv      ( 634 rows) — in-domain test
data/nli final clean external test.csv      ( 175 rows) — out-of-domain test
```

Leakage-free "v2" splits produced by `make_v2_datasets.py` (originals are never
modified). Use these for fine-tuning so post-training scores are not inflated by
memorised examples:

```
data/nli final clean train v2.csv           (2667 rows) — copy of train
data/nli final clean validation v2.csv      ( 429 rows) — 101 leaked rows removed
data/nli final clean internal test v2.csv   ( 547 rows) — 87 leaked rows removed
data/nli final clean external test v2.csv   ( 175 rows) — copy (0 overlap)
```

Row columns: `nli_id, source_id, statute, section_number, split, span_ids,
premise, claim_id, hypothesis, gold_label, error_type`. Mandatory:
`premise`, `hypothesis`, `gold_label`; the rest are carried through into the
prediction outputs and error analysis.

## Setup

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` covers evaluation (transformers, torch, pandas,
scikit-learn). **Fine-tuning additionally needs `accelerate`:**

```
pip install accelerate
```

The scripts load the model from the local Hugging Face cache with offline mode
enforced (`HF_HUB_OFFLINE=1`, `local_files_only=True`), so the model must be
cached once beforehand and no data ever leaves the machine. See "Running on
Google Colab" below if the model is not cached locally.

## 1. Baseline evaluation

```
python evaluation_scripts/evaluate_deberta_baseline.py
```

(Run from the project root or from `evaluation_scripts/` — paths resolve
relative to the script's own location either way.)

Evaluates four sets independently — `validation`, `internal_test`,
`external_test`, and `heldout_combined` (internal + external, concatenated in
memory; no combined CSV is written). To evaluate the leakage-free splits or a
fine-tuned model, edit the `DATA_FILES` / `MODEL_NAME` constants near the top.

Per set it writes to `outputs/deberta_v3_small/`:

- `<set>_predictions.csv` — all original columns plus `pred_label`,
  `confidence` (max softmax probability), and `correct`.
- `<set>_report.txt` — model name, row count, label distribution, duplicate
  `nli_id` count, accuracy, per-label precision/recall/F1, macro F1,
  entailment/neutral/contradiction recall, contradiction F1, confusion matrix,
  total incorrect, and a structured incorrect-prediction table
  (`nli_id, source_id, split, gold_label, pred_label, confidence, error_type,
  hypothesis`).
- `<set>_confusion_matrix.csv` — the same confusion matrix as its own file.

Plus one combined `outputs/deberta_v3_small/summary_metrics.csv` (one row per
dataset).

Contradiction detection is the primary case of interest (an unfaithful claim
conflicting with the statute), so contradiction recall and F1 are surfaced
separately. Judge models on those and macro F1 — not overall accuracy.

The other four checkpoints in `evaluation_scripts/` follow the identical
pattern, each writing to its own `outputs/<model>/` folder — see
"Multi-model comparison" below.

## Multi-model comparison

Five checkpoints have been run zero-shot against the same fixed `validation` /
`internal_test` / `external_test` / `heldout_combined` sets (results already
in `outputs/`, last run 2026-07-23). Contradiction detection is the metric
that matters for this verifier, so the table below is sorted by
`heldout_combined` contradiction F1 (from each model's `summary_metrics.csv`):

| Model | Contradiction F1 | Contradiction recall | Macro F1 |
|---|---|---|---|
| **fine-tuned `nli-deberta-v3-small`** (see caveat below) | **0.942** | **0.918** | **0.967** |
| fine-tuned `DeBERTa-v3-base-mnli-fever-anli` (1st attempt, see caveat below) | 0.891 | 0.854 | 0.949 |
| fine-tuned `DeBERTa-v3-base-mnli-fever-anli` (boosted attempt) | 0.880 | 0.835 | 0.944 |
| fine-tuned `DeBERTa-v3-base-mnli-fever-anli` (retuned attempt) | 0.862 | 0.772 | 0.936 |
| `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` (zero-shot) | 0.869 | 0.781 | 0.894 |
| `FacebookAI/roberta-large-mnli` (zero-shot) | 0.717 | 0.562 | 0.821 |
| `tasksource/ModernBERT-base-nli` (zero-shot) | 0.636 | 0.472 | 0.787 |
| `cross-encoder/nli-deberta-v3-small` (zero-shot, original baseline) | 0.634 | 0.478 | 0.757 |
| `cross-encoder/nli-distilroberta-base` (zero-shot) | 0.550 | 0.399 | 0.629 |

Among the **zero-shot** checkpoints, `DeBERTa-v3-base-mnli-fever-anli` wins by
a wide margin on every split, including the out-of-domain `external_test`, and
remains the best zero-shot candidate.

**Caveat on the fine-tuned rows** (numbers from
`evaluation_scripts/evaluate_finetuned_deberta.py` and the three
`evaluate_finetuned_deberta_v3_base*.py` scripts; full run reports and
run-to-run analysis under "DeBERTa Fine-Tuning Experiment" /
"DeBERTa-v3-base Fine-Tuning Experiments" below): these are not an
apples-to-apples comparison with the five zero-shot rows above.

- All fine-tuned models have seen 2,667 in-domain training pairs; none of
  the zero-shot models have seen any ClearClause data. This measures whether
  fine-tuning helps each specific checkpoint, not a pure zero-shot
  architecture comparison.
- Row counts differ: the zero-shot `heldout_combined` above is 809 rows (634
  `internal_test` + 175 `external_test`, original files); every fine-tuned
  model's `heldout_combined` is 722 rows (547 `internal_test v2` + 175
  `external_test v2`) — the 87-row gap is train-overlapping rows removed so
  no fine-tuned model is scored on memorized pairs, not a different test
  set in spirit, but the numbers aren't drawn from identical rows.
- There's a visible in-domain/out-of-domain gap within **every** fine-tuned
  model, but it is far more severe for every `DeBERTa-v3-base` attempt than
  for `nli-deberta-v3-small`: `nli-deberta-v3-small` goes from 0.960
  (`internal_test v2`) to 0.892 (`external_test v2`) contradiction F1, while
  the best `DeBERTa-v3-base` attempt (1st) goes from 0.948 to 0.704 — and its
  out-of-domain contradiction recall (0.581) actually *regressed below its
  own zero-shot score on the same file* (0.628, from the zero-shot row
  above).
- Two follow-up attempts tried to fix that regression — training more
  conservatively (retuned) and up-weighting contradiction further (boosted)
  — and **neither worked**: both score *worse* than the 1st attempt on every
  metric above, not better. See "DeBERTa-v3-base Fine-Tuning Experiments"
  below for the full experiment-by-experiment analysis of why both
  hypotheses (overfitting, then under-boosted contradiction weight) turned
  out to be wrong or ineffective.

**Bottom line, final**: starting from a stronger *zero-shot* checkpoint did
not produce a stronger *fine-tuned* model, across three separate attempts —
fine-tuned `nli-deberta-v3-small` (`finetuning/finetuned_deberta/`) is the
best model produced by this project and the one to actually use.

### Comparison methodology

All five zero-shot scripts share the same evaluation discipline, enforced by
`nli_evaluation_utils.py`:

**Models must already be cached.** Every script forces fully offline
behaviour (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`,
`local_files_only=True`). **No script will download a model.** If a model
isn't cached, the script prints a clear error and exits — it never falls
back to downloading. All five run on CPU only.

**No truncation — overlength rows stop evaluation, not silently drop.**
Before any inference, `check_token_lengths()` tokenizes every premise-
hypothesis pair *without* truncation and measures its real combined token
length. If every row fits within `MAX_LENGTH` (512 for all five), inference
proceeds — still with `truncation=False`, since nothing needs cutting. If
**any** row exceeds `MAX_LENGTH`, that dataset's evaluation stops *before
inference runs*: a clear error is printed, every offending row is written to
`outputs/<model>/<dataset>_overlength_rows.csv`, and the script exits before
computing any metric for that dataset. A dataset's report/predictions/
confusion-matrix files only exist if every row in it was evaluated in full —
metrics are never computed on silently truncated or silently dropped text.

**Per-model label-mapping verification, not assumed.** No script assumes any
checkpoint's output order matches the DeBERTa-small mapping. Each script
reads `model.config.id2label` *at runtime* and calls
`resolve_label_permutation()`, which requires exactly 3 labels that
normalize (case-insensitively) to contradiction/entailment/neutral, builds a
permutation reordering the checkpoint's native probabilities into canonical
order, and **exits immediately with a clear error** if the mapping can't be
safely resolved — it never guesses. Documented expected mappings (confirmed
against each real config when the script runs):

```
cross-encoder/nli-deberta-v3-small              0=contradiction 1=entailment 2=neutral
cross-encoder/nli-distilroberta-base            0=contradiction 1=entailment 2=neutral
FacebookAI/roberta-large-mnli                   0=contradiction 1=neutral    2=entailment
MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli    0=entailment    1=neutral    2=contradiction
```

`tasksource/ModernBERT-base-nli`'s mapping is **not** documented above or
hard-coded anywhere — `tasksource` hosts many multi-task checkpoints with no
guarantee of clean label names, so `evaluate_modernbert_nli.py` runs an extra
`check_modernbert_output_head()` safeguard (requires `num_labels == 3` and a
3-entry `id2label`) before the standard permutation check.

`FacebookAI/roberta-large-mnli` is a "large" checkpoint (~1.4 GB), noticeably
heavier on CPU RAM than the other four — its default `BATCH_SIZE` is
deliberately set to 1 to keep peak memory down on a CPU-only machine.

**Output files.** Every `<dataset>_predictions.csv` keeps all original
columns plus `gold_label_id, gold_label_name, pred_label, confidence,
correct, prob_contradiction, prob_entailment, prob_neutral` (probabilities
always in canonical order, regardless of the checkpoint's native order).
Every `<dataset>_report.txt` includes model/device/batch/sequence-length
info, row count, label distribution, accuracy, macro F1, weighted F1,
per-label precision/recall/F1/support, contradiction recall/F1, the
confusion matrix, and a structured incorrect-prediction table. Every
confusion matrix is *also* written independently as
`<dataset>_confusion_matrix.csv`, always in `gold\pred` order
(contradiction/entailment/neutral, both axes). No timing or latency figures
are collected anywhere — these scripts evaluate correctness only, never
speed.

### Running all five by double-click

Instead of running the five scripts by hand, `run_all_nli_visible_progress.bat`
(project root) runs the whole suite, heaviest model first (RoBERTa-large →
DeBERTa-v3-base → ModernBERT → DeBERTa-v3-small → DistilRoBERTa), via
`run_and_log_visible_progress.ps1`. Logs go under `outputs\run_logs\`: one
timestamped master log (start/finish times, execution order, each model's
exit code and status, any overlength reports, the final summary table) plus
one per-model log with full stdout+stderr. One model failing doesn't stop the
rest — each exit code is captured and recorded, then the batch moves on. No
model is auto-downloaded and no failed evaluation is auto-retried.

## 2. Remove train/eval leakage (make the v2 files)

```
python make_v2_datasets.py
```

The original train file shares identical (premise, hypothesis) pairs with the
validation and internal_test files. That is harmless for a zero-shot baseline
but would inflate scores after fine-tuning. This script removes those exact
train-overlapping rows from validation and internal_test only, writes the `v2`
files listed above (originals untouched, nothing relabelled, no random
re-splitting), and saves `outputs/dataset_v2_dedup_report.txt` detailing exactly
what was removed.

## 3. Fine-tuning (scaffold)

```
pip install accelerate
python finetuning/finetune_deberta.py             # targets cross-encoder/nli-deberta-v3-small
python finetuning/finetune_deberta_v3_base.py      # targets MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli
```

(Run from the project root, or `cd finetuning` first — paths resolve
relative to each script's own location either way.)

Two scripts, identical plumbing — only the base checkpoint, hyperparameters,
and output paths differ. `finetune_deberta.py` was run once
(`finetuning/finetuned_deberta/`, see "DeBERTa Fine-Tuning Experiment"
below). `finetune_deberta_v3_base.py` was run **three times**, edited in
place between runs to change hyperparameters and its output paths (so each
attempt's model/checkpoints/metrics are preserved separately rather than
overwritten) — `finetuning/finetuned_deberta_v3_base/` (1st attempt),
`finetuned_deberta_v3_base_retuned/`, and `finetuned_deberta_v3_base_boosted/`.
The file currently on disk holds the *last* (boosted) attempt's
hyperparameters; see "DeBERTa-v3-base Fine-Tuning Experiments" below for the
exact `HYPERPARAMETERS` values used for each of the three runs and why each
follow-up attempt was tried.

Both scripts train on `train v2` + `validation v2` only, never touch the test
sets, do no random splitting, select the best checkpoint by **macro F1**
(`METRIC_FOR_BEST_MODEL` near the top of each — deliberately not contradiction
F1, since the loss already up-weights contradiction via `CONTRADICTION_BOOST`;
selecting the checkpoint by that same class again would double down on the
bias rather than checking it), and offer optional class weighting to raise
contradiction recall. Tunable knobs (epochs, learning rate, batch size, weight
decay, class weighting, `METRIC_FOR_BEST_MODEL`) are grouped in the
`HYPERPARAMETERS` block near the top of each. After training each prints a
per-epoch metrics table and which epoch was selected and why, and saves that
table to `finetuning/outputs/`. Evaluate any saved model on the frozen test
sets with the matching `evaluation_scripts/evaluate_finetuned_deberta*.py`
script (see "Frozen-test evaluation" below), not `evaluate_deberta_baseline.py`
/ `evaluate_deberta_v3_base.py` (those scripts' `DATA_FILES` point at the
*original*, non-v2 test files, which would include rows these models were
trained on).

Why fine-tune `DeBERTa-v3-base-mnli-fever-anli` at all: it was the strongest
**zero-shot** checkpoint (see "Multi-model comparison" above), and fine-tuning
`nli-deberta-v3-small` (a weaker zero-shot checkpoint) already lifted its
contradiction F1 well past the zero-shot leader. `finetune_deberta_v3_base.py`
exists to test whether fine-tuning the *already-stronger* checkpoint does even
better — **it doesn't**, in any of the three attempts tried; see
"Multi-model comparison" above and "DeBERTa-v3-base Fine-Tuning Experiments"
below for the full story of why.

Note on `finetune_deberta_v3_base.py` specifically: `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`'s
`config.json` declares `torch_dtype: float16` (unlike `nli-deberta-v3-small`,
which is float32), so `from_pretrained` would load it in fp16 by default. fp16
isn't usable for CPU training and crashed the custom weighted-loss trainer
(`RuntimeError: expected scalar type Half but found Float` — fp32 class
weights vs fp16 logits) the first time this was run. The script now passes
`dtype=torch.float32` explicitly to force fp32 regardless of what the
checkpoint declares; `finetune_deberta.py` doesn't need this fix because
`nli-deberta-v3-small`'s config is already float32.

All four fine-tuning runs, and all four frozen-test evaluations, have now
been completed. See "DeBERTa Fine-Tuning Experiment" and "DeBERTa-v3-base
Fine-Tuning Experiments" below for the complete log of all four runs; and
"Multi-model comparison" above for how everything stacks up together.

A GPU is strongly recommended — CPU fine-tuning of DeBERTa on ~2.6k rows at
sequence length 512 is slow (the completed runs took ~1h52m for
nli-deberta-v3-small and roughly 2-3h per DeBERTa-v3-base attempt, all on
CPU).

### Frozen-test evaluation

```
python evaluation_scripts/evaluate_finetuned_deberta.py                   # nli-deberta-v3-small fine-tune
python evaluation_scripts/evaluate_finetuned_deberta_v3_base.py           # DeBERTa-v3-base, 1st attempt
python evaluation_scripts/evaluate_finetuned_deberta_v3_base_retuned.py   # DeBERTa-v3-base, retuned attempt
python evaluation_scripts/evaluate_finetuned_deberta_v3_base_boosted.py   # DeBERTa-v3-base, boosted attempt
```

Each loads its respective checkpoint (path resolved relative to the project
root, never hard-coded) and evaluates it on `internal_test v2`,
`external_test v2`, and an in-memory `heldout_combined` of the two — the only
splits none of these models saw during training or checkpoint selection.
Each writes the same per-set `predictions.csv` / `report.txt` /
`confusion_matrix.csv` plus a `summary_metrics.csv`, to its own
`outputs/run_<timestamp>/<model-folder-name>/` folder — so all four runs'
results coexist and are independently comparable. These are the scripts to
run for a fair generalisation estimate; the validation numbers in the
"Fine-Tuning Experiment" sections below are not a substitute for any of them.
(The copies checked into this repo have had the `run_<timestamp>/` wrapper
flattened away — each `<model-folder-name>/` now sits directly under
`outputs/` — purely to keep path lengths short; a fresh run of these scripts
still produces the nested `run_<timestamp>/` form described above.)

# DeBERTa Fine-Tuning Experiment (nli-deberta-v3-small)

**Script**: `finetuning/finetune_deberta.py` · **Output**: `finetuning/finetuned_deberta/`

### Training setup

- Base checkpoint: `cross-encoder/nli-deberta-v3-small`
- Device: CPU
- Training rows: 2,667
- Validation rows: 429
- Hyperparameters: 4 epochs, LR 2e-5, batch size 8, weight decay 0.01, `CONTRADICTION_BOOST` 1.5
- Total training steps: 1,336
- Final model path: `finetuning/finetuned_deberta`
- Epoch metrics path: `finetuning/outputs/finetune_epoch_metrics.csv`
- Class order: contradiction, entailment, neutral
- Class weights (inverse-frequency x boost, from `train v2`):
  - contradiction: 2.483240223463687
  - entailment: 0.6260563380281691
  - neutral: 1.2521126760563381
- Training runtime: 6,674 seconds, approximately 1 hour 51 minutes 14 seconds
- The complete process, including the final validation check and saving, took approximately 1 hour 52 minutes.
- Training was completed on CPU, so the runtime is an implementation detail and not a model-quality evaluation metric.

### Per-epoch validation results

| Epoch | Step | Validation Loss | Accuracy | Macro F1 | Contradiction Recall | Contradiction F1 | Entailment Recall | Neutral Recall |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 334 | 0.0870 | 0.9907 | 0.9901 | 0.9681 | 0.9838 | 0.9951 | 1.0000 |
| 2 | 668 | 0.1587 | 0.9883 | 0.9869 | 0.9468 | 0.9727 | 1.0000 | 1.0000 |
| 3 | 1002 | 0.1038 | 0.9930 | 0.9922 | 0.9681 | 0.9838 | 1.0000 | 1.0000 |
| 4 | 1336 | 0.1338 | 0.9907 | 0.9895 | 0.9574 | 0.9783 | 1.0000 | 1.0000 |

### Checkpoint selection

- All four epochs were completed normally.
- The model was not reset between epochs. Each epoch continued learning from the weights produced by the previous epoch.
- Validation data was used only for evaluation and checkpoint selection, not for weight updates.
- **Epoch 3, step 1002, was selected.**
- Selection criterion: highest validation macro F1.
- Selected value: 0.9922.
- `metric_for_best_model` was macro F1, and higher values were treated as better.
- Macro F1 was selected because the NLI classes are imbalanced and it gives contradiction, entailment, and neutral equal importance. This is particularly suitable for ClearClause because contradiction detection is important for identifying legal meaning changes.
- Epoch 1 had the lowest validation loss, but Epoch 3 had the highest macro F1 and accuracy — loss and macro F1 do not always agree on the same "best" epoch, which is exactly why loss is not used as the selection criterion.
- The final re-evaluation matched the Epoch 3 results, confirming that the correct checkpoint was restored.
- The `epoch: 4.0000` value printed during the final sanity check refers to the Trainer state after completing training (all 4 epochs ran). It does not mean Epoch 4 was selected — Epoch 3's weights are what was actually restored and saved to `finetuning/finetuned_deberta`.

### Final validation result

- Validation accuracy: 0.9930
- Validation macro F1: 0.9922
- Contradiction recall: 0.9681
- Contradiction F1: 0.9838
- Entailment recall: 1.0000
- Neutral recall: 1.0000
- Validation loss: 0.1038

These are **validation** results and must not be presented as unbiased final test performance, because the validation set was used to select the checkpoint (early-stopping bias). They confirm the checkpoint-selection mechanism worked correctly, not that the model generalises to unseen data. The primary evidence of generalisation must come from the frozen internal test set, the frozen external test set, the held-out combined set, per-class metrics, confusion matrices, and manual review of incorrect predictions. Do not overstate the validation result, and do not claim the system achieves 99% performance on unseen legal data on the strength of validation numbers alone — see "Frozen test result" below for the actual unseen-data numbers.

### Frozen test result

From `evaluation_scripts/evaluate_finetuned_deberta.py`, run against `internal_test v2`, `external_test v2`, and in-memory `heldout_combined` — splits this model never saw during training or checkpoint selection:

| Split | Rows | Accuracy | Macro F1 | Contradiction Recall | Contradiction F1 |
|---|---:|---:|---:|---:|---:|
| internal_test v2 (in-domain) | 547 | 0.9817 | 0.9792 | 0.9391 | 0.9600 |
| external_test v2 (out-of-domain) | 175 | 0.9371 | 0.9321 | 0.8605 | 0.8916 |
| **heldout_combined** | 722 | 0.9709 | 0.9674 | 0.9177 | 0.9416 |

This is the real, unbiased generalisation estimate — and it holds up well: fine-tuning improved contradiction F1 both in-domain (0.960, vs zero-shot `nli-deberta-v3-small`'s 0.650 on the original `internal_test`) and out-of-domain (0.892, vs zero-shot's 0.581 on `external_test`). This is the healthy pattern: fine-tuning helped generalization, not just memorization of the training statute. See "Multi-model comparison" above for the full picture against all other models, including the fine-tuned `DeBERTa-v3-base-mnli-fever-anli` attempts (which, surprisingly, all do *worse* than this model on every frozen-test metric — see below).

### Training warnings

Three warnings were observed during training. None indicate a training failure:

1. **`pin_memory=True` with no accelerator** — harmless on CPU; pinned memory was simply not used. Does not invalidate training.
2. **`warmup_ratio` deprecation** — training completed correctly using `warmup_ratio`; future `transformers` versions should use `warmup_steps` instead.
3. **Tokenizer PAD/BOS/EOS alignment** — Transformers aligned the model configuration with the tokenizer's special-token ids at load time. This is routine configuration alignment, not a training failure.

# DeBERTa-v3-base Fine-Tuning Experiments (3 attempts)

Three attempts at fine-tuning the stronger *zero-shot* checkpoint
(`DeBERTa-v3-base-mnli-fever-anli`), to test whether starting from a
stronger base produces a stronger fine-tuned model than
`nli-deberta-v3-small` did above. **It doesn't, in any of the three.**

## Attempt 1 (1st attempt)

**Script**: `finetuning/finetune_deberta_v3_base.py` (as first configured) · **Output**: `finetuning/finetuned_deberta_v3_base/`

### Why this experiment

`DeBERTa-v3-base-mnli-fever-anli` was the strongest **zero-shot** checkpoint
evaluated (contradiction F1 0.869 vs `nli-deberta-v3-small`'s 0.634 — see
"Multi-model comparison" above). The previous experiment showed fine-tuning
helps; this one tests whether fine-tuning the *already-stronger* checkpoint
helps even more.

### Training setup

- Base checkpoint: `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`
- Device: CPU
- Training rows: 2,667
- Validation rows: 429
- Hyperparameters: 4 epochs, LR 2e-5, batch size 8, weight decay 0.01, `CONTRADICTION_BOOST` 1.5 — identical to the `nli-deberta-v3-small` run except the base checkpoint
- Total training steps: 1,336
- Final model path: `finetuning/finetuned_deberta_v3_base`
- Epoch metrics path: `finetuning/outputs/finetune_v3_base_epoch_metrics.csv`
- Class order: contradiction, entailment, neutral
- Class weights (identical to the `nli-deberta-v3-small` run — computed fresh from the same `train v2` file, so naturally the same): contradiction 2.483240223463687, entailment 0.6260563380281691, neutral 1.2521126760563381
- Training runtime: approximately 2 hours 49 minutes (wall-clock) — longer than the small-model run (~1h52m) because `DeBERTa-v3-base` is a bigger checkpoint.
- **One extra fix needed here that the small-model run didn't need**: this checkpoint's `config.json` declares `torch_dtype: float16` (unlike `nli-deberta-v3-small`, which is float32), so `from_pretrained` loaded it in fp16 by default on the first attempt, which crashed the custom weighted-loss trainer at step 1 (`RuntimeError: expected scalar type Half but found Float` — fp32 class weights vs fp16 logits). Fixed by passing `dtype=torch.float32` explicitly. The results below are from the run after that fix.

### Per-epoch validation results

| Epoch | Step | Validation Loss | Accuracy | Macro F1 | Contradiction Recall | Contradiction F1 | Entailment Recall | Neutral Recall |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 334 | 0.1691 | 0.9744 | 0.9714 | 0.9043 | 0.9444 | 0.9903 | 1.0000 |
| 2 | 668 | 0.1413 | 0.9860 | 0.9842 | 0.9362 | 0.9670 | 1.0000 | 1.0000 |
| 3 | 1002 | 0.0886 | 0.9883 | 0.9871 | 0.9681 | 0.9733 | 0.9903 | 1.0000 |
| 4 | 1336 | 0.1746 | 0.9883 | 0.9869 | 0.9468 | 0.9727 | 1.0000 | 1.0000 |

**Epoch 3 selected** (macro F1 0.9871 — highest). Unlike the small-model run
(where epoch 1 had the lowest loss but epoch 3 had the best macro F1), here
epoch 3 has both the lowest validation loss (0.0886) *and* the highest macro
F1 — they happened to agree this time; `metric_for_best_model` (macro F1) is
still what actually drove selection. Epoch 4 regressed slightly on both
macro F1 (0.9869 vs 0.9871) and loss (0.1746 vs 0.0886) relative to epoch
3 — consistent with the model starting to overfit the training set in the
final epoch, which per-epoch checkpoint selection exists to guard against.

### Final validation result

(epoch 3, the selected checkpoint)

- Validation accuracy: 0.9883
- Validation macro F1: 0.9871
- Contradiction recall: 0.9681
- Contradiction F1: 0.9733
- Entailment recall: 0.9903
- Neutral recall: 1.0000
- Validation loss: 0.0886

These are **validation** results and must not be presented as unbiased final test performance, for the same reason as the small-model run: validation was used to select the checkpoint. See "Frozen test result" below for what actually happened on unseen data — it tells a different story than these validation numbers suggest.

### Frozen test result

From `evaluation_scripts/evaluate_finetuned_deberta_v3_base.py`, run against `internal_test v2`, `external_test v2`, and in-memory `heldout_combined`:

| Split | Rows | Accuracy | Macro F1 | Contradiction Recall | Contradiction F1 |
|---|---:|---:|---:|---:|---:|
| internal_test v2 (in-domain) | 547 | 0.9781 | 0.9756 | 0.9565 | 0.9483 |
| external_test v2 (out-of-domain) | 175 | 0.8800 | 0.8648 | 0.5814 | 0.7042 |
| **heldout_combined** | 722 | 0.9543 | 0.9490 | 0.8544 | 0.8911 |

**This is the key finding of this experiment, and it's a negative result.** Two problems:

1. Every frozen-test metric here is *worse* than the small-model run's, despite starting from a stronger zero-shot checkpoint. Starting stronger didn't translate to ending stronger.
2. On `external_test` specifically, fine-tuning made the model *worse than not fine-tuning it at all*: contradiction recall 0.628 zero-shot → 0.581 fine-tuned; contradiction F1 0.771 zero-shot → 0.704 fine-tuned (compare "Multi-model comparison" above's zero-shot row for the same 175-row file). In-domain, fine-tuning helped (contradiction F1 0.896 zero-shot → 0.948 fine-tuned) — so this looks like the model specialized on Penal-Code-specific patterns at the cost of general contradiction detection elsewhere.

### Training warnings

Same three as the small-model run, plus the fp16/fp32 dtype crash described above (a hard crash on the very first attempt, fixed before this run).

## Attempt 2 (retuned)

**Script**: `finetuning/finetune_deberta_v3_base.py` (as reconfigured) · **Output**: `finetuning/finetuned_deberta_v3_base_retuned/`

### Why this experiment

Attempt 1's out-of-domain regression looked like classic overfitting: the model got better in-domain and worse out-of-domain, and `DeBERTa-v3-base` has more parameters/capacity than `nli-deberta-v3-small` to overfit the same ~2,667-row training set. The standard fix for overfitting is to train less aggressively.

### What changed from Attempt 1

| Hyperparameter | Attempt 1 | Attempt 2 (retuned) | Rationale |
|---|---:|---:|---|
| `EPOCHS` | 4 | 3 | Epoch 3 was already the validation peak in Attempt 1; a 4th pass that regresses even in-domain suggested stopping earlier. |
| `LEARNING_RATE` | 2e-5 | 1e-5 | Smaller steps, less aggressive adaptation away from the checkpoint's pretrained NLI competence. |
| `WEIGHT_DECAY` | 0.01 | 0.05 | Stronger L2 regularisation against overfitting the training set. |
| `CONTRADICTION_BOOST` | 1.5 | 1.5 (unchanged) | Not touched in this experiment. |

### Per-epoch validation results

| Epoch | Step | Validation Loss | Accuracy | Macro F1 | Contradiction Recall | Contradiction F1 | Entailment Recall | Neutral Recall |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 334 | 0.1787 | 0.9720 | 0.9684 | 0.9043 | 0.9392 | 0.9854 | 1.0000 |
| 2 | 668 | 0.1919 | 0.9744 | 0.9708 | 0.8936 | 0.9385 | 0.9951 | 1.0000 |
| 3 | 1002 | 0.1812 | 0.9814 | 0.9790 | 0.9255 | 0.9560 | 0.9951 | 1.0000 |

(Only 3 epochs, per the retuned `EPOCHS=3`.) **Epoch 3 selected** (macro F1 0.9790 — also the last epoch, since macro F1 rose every epoch here).

### Frozen test result

| Split | Rows | Accuracy | Macro F1 | Contradiction Recall | Contradiction F1 |
|---|---:|---:|---:|---:|---:|
| internal_test v2 (in-domain) | 547 | 0.9726 | 0.9672 | 0.8783 | 0.9309 |
| external_test v2 (out-of-domain) | 175 | 0.8629 | 0.8357 | 0.4884 | 0.6364 |
| **heldout_combined** | 722 | 0.9460 | 0.9358 | 0.7722 | 0.8622 |

**This experiment made things worse, not better — the overfitting hypothesis was wrong (or at least incomplete).** Every metric regressed further from Attempt 1, not just held steady:

| Metric | Attempt 1 | Attempt 2 (retuned) | Direction |
|---|---:|---:|---|
| internal_test contradiction recall | 0.957 | 0.878 | worse |
| internal_test contradiction F1 | 0.948 | 0.931 | worse |
| external_test contradiction recall | 0.581 | 0.488 | worse |
| external_test contradiction F1 | 0.704 | 0.636 | worse |
| internal_test entailment recall | 0.990 | 0.996 | *up* |
| external_test entailment recall | 0.955 | 0.977 | *up* |

**Interpretation**: training more conservatively (less LR, fewer epochs, more weight decay) made the model lean *further* toward the safe/majority classes (entailment, neutral) and further *away* from flagging contradiction — the opposite of what "reduce overfitting" was supposed to achieve. This suggests Attempt 1's weakness was never "the model trained too hard and memorized the training set" — it looks more like "the model isn't being pushed hard enough toward the minority contradiction class in the first place," and training more conservatively simply amplified that. `CONTRADICTION_BOOST` — the one dial that specifically counteracts this — was left unchanged in this experiment, which is the gap Attempt 3 addresses.

### Training warnings

Same three as Attempt 1's small-model counterpart. No new crashes.

## Attempt 3 (boosted)

**Script**: `finetuning/finetune_deberta_v3_base.py` (as reconfigured again) · **Output**: `finetuning/finetuned_deberta_v3_base_boosted/`

### Why this experiment

Attempt 2 pointed away from "overfitting" and toward "the model isn't being pushed hard enough toward contradiction specifically." Rather than change training conservatism again, this experiment reverts `EPOCHS` / `LEARNING_RATE` / `WEIGHT_DECAY` back to Attempt 1's values (the best-performing v3-base config so far) and changes exactly one thing: `CONTRADICTION_BOOST` 1.5 → 2.0 — isolating that variable so its effect can be read cleanly against Attempt 1's numbers, rather than changing four things at once as Attempt 2 did.

### Training setup

- `EPOCHS` 4, `LEARNING_RATE` 2e-5, `WEIGHT_DECAY` 0.01 — same as Attempt 1
- `CONTRADICTION_BOOST` 2.0 — the one change from Attempt 1

### Per-epoch validation results

| Epoch | Step | Validation Loss | Accuracy | Macro F1 | Contradiction Recall | Contradiction F1 | Entailment Recall | Neutral Recall |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 334 | 0.1824 | 0.9790 | 0.9761 | 0.9043 | 0.9497 | 1.0000 | 1.0000 |
| 2 | 668 | 0.1740 | 0.9860 | 0.9842 | 0.9362 | 0.9670 | 1.0000 | 1.0000 |
| 3 | 1002 | 0.0793 | 0.9883 | 0.9870 | 0.9574 | 0.9730 | 0.9951 | 1.0000 |
| 4 | 1336 | 0.1589 | 0.9860 | 0.9843 | 0.9468 | 0.9674 | 0.9951 | 1.0000 |

**Epoch 3 selected** (macro F1 0.9870, confirmed in `trainer_state.json`) — the same epoch selected in both prior `DeBERTa-v3-base` attempts.

### Frozen test result

| Split | Rows | Accuracy | Macro F1 | Contradiction Recall | Contradiction F1 |
|---|---:|---:|---:|---:|---:|
| internal_test v2 (in-domain) | 547 | 0.9744 | 0.9712 | 0.9304 | 0.9386 |
| external_test v2 (out-of-domain) | 175 | 0.8743 | 0.8596 | 0.5814 | 0.6944 |
| **heldout_combined** | 722 | 0.9501 | 0.9440 | 0.8354 | 0.8800 |

**Raising `CONTRADICTION_BOOST` did not help — it's a wash at best, and slightly worse overall than Attempt 1.**

| Metric | Attempt 1 (boost 1.5) | Attempt 3 (boost 2.0) | Direction |
|---|---:|---:|---|
| external_test contradiction recall | 0.5814 | 0.5814 | **identical** — no improvement |
| external_test contradiction F1 | 0.7042 | 0.6944 | slightly worse |
| internal_test contradiction recall | 0.9565 | 0.9304 | worse |
| heldout_combined contradiction F1 | 0.8911 | 0.8800 | worse |

The out-of-domain contradiction recall is *exactly* unchanged (0.5814 in both runs) — more boost did not make the model catch more out-of-domain contradictions, it just cost some precision (lower F1) and some in-domain recall. So the revised hypothesis from Attempt 2 (under-boosted contradiction weight) doesn't fully explain the regression either, at least not at this boost level.

**Ranking of all three `DeBERTa-v3-base` attempts** (heldout_combined contradiction F1): Attempt 1 (0.891) > Attempt 3 (0.880) > Attempt 2 (0.862). The simplest, first configuration remains the best of the three — neither follow-up hypothesis (overfitting, then under-boosting) improved on it. If `finetuning/finetuned_deberta_v3_base/` (Attempt 1's output) is ever used for anything, note it still underperforms `finetuning/finetuned_deberta/` (the `nli-deberta-v3-small` run) by a wide margin.

### Training warnings

Same three as the earlier attempts. No new crashes.

## Methodology notes (all four fine-tuning runs)

- **Why fine-tuned models are evaluated on `internal_test v2` / `external_test v2`, not the originals**: every fine-tuning run trains on `train v2`. The original `internal_test.csv` shares 87 exact (premise, hypothesis) rows with `train v2` (see `outputs/dataset_v2_dedup_report.txt`) — rows a fine-tuned model could have memorized. Evaluating on those would silently inflate the score. `internal_test v2` removes exactly those 87 rows; `external_test v2` is an unchanged copy of `external_test` (zero overlap to begin with, nothing to remove). Zero-shot models were never trained on anything, so they're evaluated on the original, larger files — hence the 809-vs-722-row difference in the "Multi-model comparison" table above. `external_test`/`external_test v2` are identical either way (175 rows, no leakage ever existed there), so any external_test-only comparison across zero-shot and fine-tuned models is apples-to-apples regardless.
- **Why checkpoint selection uses macro F1, not contradiction F1**: the loss function already up-weights contradiction via `CONTRADICTION_BOOST`. Also selecting the checkpoint by contradiction F1 would apply that same bias twice, risking a checkpoint that maximizes contradiction at a disproportionate cost to entailment/neutral. Macro F1 keeps the loss-level push toward contradiction recall during training while checkpoint choice reflects balanced performance.
- **Why validation can't catch out-of-domain regression**: `validation v2` is in-domain (Penal Code, same statute family as `train`). Per-epoch checkpoint selection only ever sees in-domain performance, so a checkpoint can look excellent on validation (as every epoch did in Attempts 1 and 2) while quietly getting worse on a genuinely different statute (`external_test`, Maintenance Act). This is exactly why the frozen test evaluation step is not optional — validation metrics alone would have missed Attempts 1 and 2's real weaknesses entirely.
- **`nli-deberta-v3-small`'s config is float32; `DeBERTa-v3-base-mnli-fever-anli`'s is float16.** `from_pretrained` respects each checkpoint's declared dtype unless told otherwise, which is why only the v3-base script needed the explicit `dtype=torch.float32` fix.

## Running on Google Colab

Colab provides a free GPU and already has torch/transformers/accelerate. Two
adjustments are needed because the scripts default to offline/local-cache mode:

1. Set Runtime → Change runtime type → GPU.
2. Allow the base model to download once (a fresh Colab VM has no local cache):
   remove the `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` lines at the top of the
   script and change `local_files_only=True` to `False` in the
   `from_pretrained(...)` calls.

Upload the `data/` folder (or mount Google Drive), run, then download the
resulting `finetuned_deberta/` folder. Downloading model weights is fine; only
the statute CSVs must stay off external services.

## Tests

There is currently no automated test suite. To run tests once they exist:

```
pip install pytest
pytest -v
```
