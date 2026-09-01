"""
Frozen-test evaluation for the fine-tuned ClearClause NLI checkpoint.

Inference only. No training, no fine-tuning, no random splitting, no
modification of the source CSV data or labels.

Model:
  finetuning/finetuned_deberta  (cross-encoder/nli-deberta-v3-small,
  fine-tuned on train v2 + validation v2 -- see finetuning/finetune_deberta.py)
  Loaded from the LOCAL CHECKPOINT FOLDER ONLY (never the Hugging Face hub),
  path resolved relative to this script's own location so it works
  regardless of the project's location on disk or the current working
  directory. Runs on CPU.

Gold-label mapping (fixed for this project; independent of the checkpoint):
  0 = contradiction
  1 = entailment
  2 = neutral

Why the v2 test files, specifically (unlike evaluate_deberta_baseline.py,
which evaluates the zero-shot base checkpoint on the ORIGINAL files):
  This checkpoint was fine-tuned on `train v2`. The original
  `internal_test.csv` shares 87 exact (premise, hypothesis) rows with
  `train v2` (see outputs/dataset_v2_dedup_report.txt) -- rows this model has
  already seen and could have memorized during training. Evaluating on those
  rows would silently inflate the internal_test score and no longer measure
  generalisation to unseen data. `internal_test v2` has those 87 rows removed;
  `external_test v2` is an unchanged copy of `external_test` (zero overlap to
  begin with). This script therefore evaluates ONLY the leakage-free,
  genuinely frozen sets:
    - internal_test v2   (in-domain, Penal Code)
    - external_test v2   (out-of-domain, Maintenance Act)
    - heldout_combined   (internal_test v2 + external_test v2, built in
                          memory; no combined CSV is written)
  `validation v2` is deliberately NOT evaluated here: it was used during
  fine-tuning for per-epoch checkpoint selection (see
  finetuning/finetune_deberta.py), so a score on it reflects early-stopping
  bias, not generalisation -- it is not a fair test-set number.
  `train`/`train v2` are never read here.

No premise or hypothesis is ever truncated. Every pair is tokenized once,
WITHOUT truncation, and its combined token length is checked against
MAX_LENGTH before any inference runs (see
nli_evaluation_utils.check_token_lengths). If any pair in a dataset exceeds
MAX_LENGTH, that dataset's evaluation stops before inference -- metrics are
never computed on a silently truncated or silently reduced dataset -- and
the offending rows are written to <dataset>_overlength_rows.csv instead.

Outputs (written to outputs/run_<timestamp>/finetuned_deberta/):
  <set>_predictions.csv         all original columns + gold_label_id,
                                 gold_label_name, pred_label, confidence,
                                 correct, prob_contradiction, prob_entailment,
                                 prob_neutral
  <set>_report.txt              full metrics + incorrect-prediction table
  <set>_confusion_matrix.csv    the confusion matrix as its own CSV file
  <set>_overlength_rows.csv     only written if that dataset has any pair
                                 exceeding MAX_LENGTH (evaluation then stops
                                 before inference for that dataset)
  summary_metrics.csv           one row per dataset

premise    = original statutory span
hypothesis = simplified atomic legal claim
The verifier checks whether the simplified claim is faithful to the law.

These are the primary, unbiased generalisation numbers for the fine-tuned
model -- unlike the validation metrics reported during training (which
guided checkpoint selection and are therefore optimistic), these frozen sets
were never read by finetuning/finetune_deberta.py.

Legal CSV data is never sent to any external API; the model is loaded only
from the local checkpoint folder on disk.
"""

import os
from datetime import datetime
from pathlib import Path
import sys

# Force fully-offline behaviour BEFORE importing transformers, so no network
# call is ever attempted for the model or tokenizer.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import pandas as pd
import torch

import nli_evaluation_utils as util


# --------------------------------------------------------------------------- #
# Constants (this script's own model / output folder / batch size)
# --------------------------------------------------------------------------- #

# Paths are resolved relative to the project root (one level up from this
# script's own folder, evaluation_scripts/), so the script works regardless
# of the current working directory and never hard-codes an absolute path.
BASE_DIR = Path(__file__).resolve().parent.parent
FINETUNED_MODEL_DIR = BASE_DIR / "finetuning" / "finetuned_deberta"
MODEL_NAME = str(FINETUNED_MODEL_DIR)

# Inference settings -- kept near the top so they are easy to change.
# BATCH_SIZE is only a memory/throughput setting; it is never used to report
# or evaluate speed. MAX_LENGTH is the token-length ceiling checked (without
# truncation) before inference; a pair exceeding it stops evaluation for
# that dataset rather than being truncated. Matches the value used to train
# this checkpoint (finetuning/finetune_deberta.py's MAX_LENGTH).
BATCH_SIZE = 16
MAX_LENGTH = 512

DATA_DIR = BASE_DIR / "data"
_RUN_STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_ROOT = Path(os.environ.get("NLI_OUTPUT_ROOT", str(BASE_DIR / "outputs" / f"run_{_RUN_STAMP}")))
OUTPUT_DIR = OUTPUT_ROOT / "finetuned_deberta"

# Only the leakage-free v2 test files -- see module docstring for why.
DATA_FILES = {
    "internal_test": DATA_DIR / "nli final clean internal test v2.csv",
    "external_test": DATA_DIR / "nli final clean external test v2.csv",
}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    if not FINETUNED_MODEL_DIR.exists():
        print(
            f"ERROR: fine-tuned checkpoint not found at: {FINETUNED_MODEL_DIR}\n"
            "       Run finetuning/finetune_deberta.py first to produce it.",
            file=sys.stderr,
        )
        sys.exit(1)

    device = torch.device("cpu")
    print(f"Using device: {device}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load and validate the two frozen source datasets (read-only).
    datasets = {name: util.load_dataset(name, path) for name, path in DATA_FILES.items()}

    # Build heldout_combined in memory (internal_test v2 + external_test v2). No CSV written.
    heldout_combined = pd.concat(
        [datasets["internal_test"], datasets["external_test"]],
        ignore_index=True,
    )
    print(f"\n--- Built heldout_combined in memory: {len(heldout_combined)} rows "
          "(internal_test v2 + external_test v2) ---")

    tokenizer, model = util.load_model_offline(MODEL_NAME, device)
    torch_version, transformers_version = util.get_environment_versions()

    # Verify this checkpoint's own label order against the canonical
    # ClearClause mapping. The fine-tuning script trains with
    # id2label/label2id already set to the canonical mapping, so this is
    # expected to resolve to the identity permutation -- but it is still
    # verified fresh here rather than assumed.
    permutation = util.resolve_label_permutation(model.config.id2label, MODEL_NAME)

    eval_order = [
        ("internal_test", datasets["internal_test"]),
        ("external_test", datasets["external_test"]),
        ("heldout_combined", heldout_combined),
    ]

    summary_rows = [
        util.evaluate_set(MODEL_NAME, name, df, tokenizer, model, device,
                           permutation, BATCH_SIZE, MAX_LENGTH, OUTPUT_DIR,
                           torch_version, transformers_version)
        for name, df in eval_order
    ]

    summary_df = pd.DataFrame(summary_rows, columns=util.SUMMARY_COLUMNS)
    summary_path = OUTPUT_DIR / "summary_metrics.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nWrote summary metrics to: {summary_path}")

    print("\nDone. Summary (frozen test sets -- unbiased generalisation estimate):")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
