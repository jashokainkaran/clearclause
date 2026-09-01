"""
ModernBERT NLI baseline evaluation for ClearClause.

Inference only. No training, no fine-tuning, no random splitting, no
modification of the source CSV data or labels.

Model:
  tasksource/ModernBERT-base-nli
  Loaded via Hugging Face Transformers + PyTorch, forced to LOCAL CACHE ONLY
  (local_files_only=True). Runs on CPU.

  WARNING -- unlike the other four checkpoints in this project, this one's
  label ordering has NOT been manually confirmed in advance and must not be
  assumed. `tasksource` hosts many multi-task checkpoints, and its NLI heads
  are not guaranteed to expose clean "contradiction"/"entailment"/"neutral"
  names (a generic head can show up as "LABEL_0"/"LABEL_1"/"LABEL_2", or the
  checkpoint may not be a plain 3-way NLI head at all). This script therefore
  runs an extra pre-flight safeguard (see check_modernbert_output_head below)
  in addition to the standard nli_evaluation_utils.resolve_label_permutation()
  check, and refuses to produce any metrics if the output head is ambiguous.
  These checks only run, and can only be verified, when the script is
  actually executed later -- they are not evaluated at authoring time.

Gold-label mapping (fixed for this project; independent of the checkpoint):
  0 = contradiction
  1 = entailment
  2 = neutral

This checkpoint's own output order is NOT hard-coded anywhere in this file.
It is read from model.config at runtime, checked by
check_modernbert_output_head() and then by
nli_evaluation_utils.resolve_label_permutation(), either of which will exit
with a clear error (refusing to guess) rather than silently apply another
model's mapping or produce unreliable metrics from an ambiguous head.

Evaluates three sets, each independently, plus one in-memory combination:
  - validation        (data/nli final clean validation.csv)
  - internal_test      (data/nli final clean internal test.csv)
  - external_test      (data/nli final clean external test.csv)
  - heldout_combined  (internal_test + external_test, concatenated IN MEMORY)

The train file (data/nli final clean train.csv) is intentionally NOT used.
No combined dataset CSV is written to disk.

No premise or hypothesis is ever truncated. Every pair is tokenized once,
WITHOUT truncation, and its combined token length is checked against
MAX_LENGTH before any inference runs (see
nli_evaluation_utils.check_token_lengths). If any pair in a dataset exceeds
MAX_LENGTH, that dataset's evaluation stops before inference -- metrics are
never computed on a silently truncated or silently reduced dataset -- and
the offending rows are written to <dataset>_overlength_rows.csv instead.

Outputs (written to outputs/modernbert_nli/ only):
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

Legal CSV data is never sent to any external API; only the model name is
resolved, and that is forced to the local cache.
"""

import os
import sys
from datetime import datetime
from pathlib import Path

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

MODEL_NAME = "tasksource/ModernBERT-base-nli"

# Inference settings -- kept near the top so they are easy to change.
# BATCH_SIZE is only a memory/throughput setting; it is never used to report
# or evaluate speed. MAX_LENGTH is the token-length ceiling checked (without
# truncation) before inference; a pair exceeding it stops evaluation for
# that dataset rather than being truncated.
BATCH_SIZE = 2
MAX_LENGTH = 512

# Paths are resolved relative to the project root (one level up from this
# script's own folder, evaluation_scripts/), so the script works regardless
# of the current working directory and never reaches outside D:\nli_baseline_test.
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
# Each model's evaluation artifacts live in their own outputs/<model> folder
# so results from different models never overwrite each other. By default
# every run writes into a fresh, timestamped folder (outputs/run_<stamp>/)
# so rerunning never overwrites a prior run's results -- no env var needed.
# NLI_OUTPUT_ROOT can still be set (e.g. by a launcher orchestrating several
# scripts) to make multiple scripts share ONE timestamped folder instead of
# each picking its own.
_RUN_STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_ROOT = Path(os.environ.get("NLI_OUTPUT_ROOT", str(BASE_DIR / "outputs" / f"run_{_RUN_STAMP}")))
OUTPUT_DIR = OUTPUT_ROOT / "modernbert_nli"

# Source CSV file names (exact, with spaces). The train file is deliberately
# absent from this mapping.
DATA_FILES = {
    "validation": DATA_DIR / "nli final clean validation.csv",
    "internal_test": DATA_DIR / "nli final clean internal test.csv",
    "external_test": DATA_DIR / "nli final clean external test.csv",
}


# --------------------------------------------------------------------------- #
# ModernBERT-specific safeguard (runs only when this script is executed)
# --------------------------------------------------------------------------- #

def check_modernbert_output_head(model, model_name: str) -> None:
    """
    Extra pre-flight check for this specific checkpoint, on top of the
    standard resolve_label_permutation() check every script runs.

    tasksource checkpoints are not guaranteed to be clean 3-way NLI heads,
    so this explicitly:
      1. Prints the loaded model configuration (num_labels, id2label).
      2. Checks num_labels == 3.
      3. Checks config.id2label has exactly 3 entries.
      4. Confirms there are exactly three usable NLI outputs (delegated to
         resolve_label_permutation, called right after this by main()).
      5. Refuses to continue (sys.exit(1)) at the first sign of an
         ambiguous or incompatible output head, before any inference or
         metric is computed.
    """
    cfg = model.config
    print(f"\n--- ModernBERT output-head safeguard check for '{model_name}' ---")
    print(f"  num_labels: {getattr(cfg, 'num_labels', None)}")
    print(f"  id2label:   {cfg.id2label}")

    num_labels = getattr(cfg, "num_labels", None)
    if num_labels != 3:
        print(
            f"ERROR: '{model_name}' has num_labels={num_labels}; expected "
            "exactly 3 (contradiction/entailment/neutral). This does not "
            "look like a plain 3-way NLI head. Refusing to evaluate, since "
            "the resulting metrics would be unreliable.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not hasattr(cfg, "id2label") or len(cfg.id2label) != 3:
        print(
            f"ERROR: '{model_name}' config.id2label has "
            f"{len(getattr(cfg, 'id2label', {}))} entries; expected exactly 3. "
            "Refusing to evaluate an ambiguous output head.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("  [ok] exactly 3 output labels present in the model config.")
    print("  -> proceeding to resolve_label_permutation() for the final "
          "contradiction/entailment/neutral mapping check.")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    device = torch.device("cpu")
    print(f"Using device: {device}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load and validate the three source datasets (read-only).
    datasets = {name: util.load_dataset(name, path) for name, path in DATA_FILES.items()}

    # Build heldout_combined in memory (internal_test + external_test). No CSV written.
    heldout_combined = pd.concat(
        [datasets["internal_test"], datasets["external_test"]],
        ignore_index=True,
    )
    print(f"\n--- Built heldout_combined in memory: {len(heldout_combined)} rows "
          "(internal_test + external_test) ---")

    tokenizer, model = util.load_model_offline(MODEL_NAME, device)
    torch_version, transformers_version = util.get_environment_versions()

    # ModernBERT-specific safeguard: confirm num_labels/id2label are usable
    # before attempting to resolve a canonical label mapping or run inference.
    check_modernbert_output_head(model, MODEL_NAME)

    # Verify this checkpoint's own label order against the canonical
    # ClearClause mapping. Exits with a clear error if it cannot be safely
    # resolved -- never silently reuses another model's mapping, and never
    # hard-codes a ModernBERT-specific mapping.
    permutation = util.resolve_label_permutation(model.config.id2label, MODEL_NAME)

    eval_order = [
        ("validation", datasets["validation"]),
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

    print("\nDone. Summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
