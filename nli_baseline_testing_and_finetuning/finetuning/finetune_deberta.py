"""
Fine-tuning scaffold for the ClearClause NLI faithfulness verifier.

WHAT THIS IS
------------
A complete, runnable *starting point* for fine-tuning
`cross-encoder/nli-deberta-v3-small` on the ClearClause statutory-NLI data.
It is deliberately a scaffold: the plumbing is correct and reproducible, and
the knobs you will actually tune (epochs, learning rate, class weighting) are
grouped in one HYPERPARAMETERS block near the top.

DESIGN DECISIONS (aligned with the baseline eval)
-------------------------------------------------
* Trains ONLY on the leakage-free v2 files:
    - train:      data/nli final clean train v2.csv
    - validation: data/nli final clean validation v2.csv   (used for early
                  model selection during training)
* The TEST sets (internal_test, external_test) are NEVER read here. You keep
  them frozen and evaluate the fine-tuned model with evaluate_deberta_baseline.py.
* No random splitting: the given splits are used as-is.
* Label mapping is fixed: 0=contradiction, 1=entailment, 2=neutral.
* Model selection optimises MACRO F1 (not accuracy or loss), so checkpoint
  choice reflects balanced performance across all three classes rather than
  double-counting the contradiction boost already applied in the loss
  (see CONTRADICTION_BOOST below). Contradiction recall/F1 are still
  computed and reported every epoch for inspection.
* Optional class weighting up-weights contradiction to push its recall up.
* Model loads from the local HF cache (local_files_only=True); training writes
  nothing outside this project folder.

REQUIREMENTS (install into your Python env, not run here):
    pip install "transformers>=4.40" torch pandas scikit-learn accelerate

RUN (this scaffold is not executed automatically), from the finetuning/ folder:
    python finetune_deberta.py
Then evaluate the saved model by pointing MODEL_NAME in
evaluate_deberta_baseline.py at ./finetuned_deberta  (or add it as a second
model there).
"""

import os
import sys
from pathlib import Path

# Load the base model from local cache only; download nothing.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support

# --------------------------------------------------------------------------- #
# HYPERPARAMETERS  (these are the knobs to tune)
# --------------------------------------------------------------------------- #
EPOCHS = 4
LEARNING_RATE = 2e-5
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
MAX_LENGTH = 512            # matches the baseline eval
SEED = 42

# Up-weight rare/important classes in the loss. Set to False for plain CE.
USE_CLASS_WEIGHTS = True
# Extra multiplier applied to the contradiction class on top of the
# inverse-frequency weight (1.0 = inverse-frequency only). Raise to trade
# some precision for higher contradiction recall.
CONTRADICTION_BOOST = 1.5

# Metric used to pick the "best" epoch checkpoint (must be a key returned by
# compute_metrics, higher = better). "macro_f1" balances all three classes;
# "contradiction_f1" would double down on CONTRADICTION_BOOST above by also
# biasing checkpoint choice toward that one class.
METRIC_FOR_BEST_MODEL = "macro_f1"

# --------------------------------------------------------------------------- #
# Constants / paths
# --------------------------------------------------------------------------- #
MODEL_NAME = "cross-encoder/nli-deberta-v3-small"

ID2LABEL = {0: "contradiction", 1: "entailment", 2: "neutral"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}
LABEL_ORDER = [0, 1, 2]

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_MODEL_DIR = BASE_DIR / "finetuned_deberta"
TRAINER_WORK_DIR = BASE_DIR / "outputs" / "finetune_checkpoints"

TRAIN_FILE = DATA_DIR / "nli final clean train v2.csv"
VAL_FILE = DATA_DIR / "nli final clean validation v2.csv"

REQUIRED_COLUMNS = ["premise", "hypothesis", "gold_label"]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def normalize_gold_label(value) -> int:
    """Accept label names (case-insensitive) or the exact whole-number ids
    0/1/2. Fractional numeric values (e.g. "1.5") are rejected rather than
    silently truncated."""
    text = str(value).strip().lower()
    if text in LABEL2ID:
        return LABEL2ID[text]
    try:
        numeric_value = float(text)
    except (ValueError, TypeError):
        raise ValueError(f"Unrecognized gold_label value: {value!r}")
    if not numeric_value.is_integer():
        raise ValueError(
            f"gold_label must be a whole number or label name, got fractional "
            f"value: {value!r}"
        )
    label_id = int(numeric_value)
    if label_id in ID2LABEL:
        return label_id
    raise ValueError(f"Unknown gold_label id: {value!r}")


def load_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        print("       Run make_v2_datasets.py first to create the v2 files.", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        print(f"ERROR: {path.name} missing columns: {missing}", file=sys.stderr)
        sys.exit(1)
    blank = (
        df["premise"].isna() | (df["premise"].astype(str).str.strip() == "")
        | df["hypothesis"].isna() | (df["hypothesis"].astype(str).str.strip() == "")
    )
    if blank.any():
        print(f"ERROR: {path.name} has blank premise/hypothesis rows: "
              f"{df.index[blank].tolist()}", file=sys.stderr)
        sys.exit(1)
    df = df.copy()
    df["label"] = df["gold_label"].apply(normalize_gold_label)
    return df


class NLIDataset(torch.utils.data.Dataset):
    """Tokenises (premise, hypothesis) pairs on the fly."""

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int):
        self.premises = df["premise"].astype(str).tolist()
        self.hypotheses = df["hypothesis"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.premises[idx],
            self.hypotheses[idx],
            truncation=True,
            max_length=self.max_length,
            padding=False,  # dynamic padding via the collator
        )
        enc["labels"] = self.labels[idx]
        return enc


# --------------------------------------------------------------------------- #
# Metrics — model selection targets contradiction F1 (and macro F1)
# --------------------------------------------------------------------------- #
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, labels=LABEL_ORDER, zero_division=0
    )
    by = {ID2LABEL[l]: i for i, l in enumerate(LABEL_ORDER)}
    return {
        "accuracy": float((preds == labels).mean()),
        "macro_f1": float(f1_score(labels, preds, labels=LABEL_ORDER,
                                   average="macro", zero_division=0)),
        "contradiction_recall": float(recall[by["contradiction"]]),
        "contradiction_f1": float(f1[by["contradiction"]]),
        "entailment_recall": float(recall[by["entailment"]]),
        "neutral_recall": float(recall[by["neutral"]]),
    }


def main():
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  (CPU training is slow; a GPU is recommended)")

    train_df = load_split(TRAIN_FILE)
    val_df = load_split(VAL_FILE)
    print(f"train v2: {len(train_df)} rows | validation v2: {len(val_df)} rows")

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, local_files_only=True,
            num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: could not load base model from local cache: {exc}", file=sys.stderr)
        sys.exit(1)

    train_ds = NLIDataset(train_df, tokenizer, MAX_LENGTH)
    val_ds = NLIDataset(val_df, tokenizer, MAX_LENGTH)
    collator = DataCollatorWithPadding(tokenizer)

    # Inverse-frequency class weights (optionally boosting contradiction).
    class_weights = None
    if USE_CLASS_WEIGHTS:
        counts = train_df["label"].value_counts().to_dict()
        total = len(train_df)
        w = []
        for lid in LABEL_ORDER:
            freq = counts.get(lid, 1)
            weight = total / (3.0 * freq)
            if lid == LABEL2ID["contradiction"]:
                weight *= CONTRADICTION_BOOST
            w.append(weight)
        class_weights = torch.tensor(w, dtype=torch.float, device=device)
        print(f"class weights [contra, entail, neutral] = {w}")

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = torch.nn.functional.cross_entropy(
                outputs.logits, labels, weight=class_weights
            )
            return (loss, outputs) if return_outputs else loss

    # Build TrainingArguments, tolerating the eval_strategy/evaluation_strategy
    # rename across transformers versions.
    ta_common = dict(
        output_dir=str(TRAINER_WORK_DIR),
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        logging_steps=50,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model=METRIC_FOR_BEST_MODEL,
        greater_is_better=True,
        seed=SEED,
        report_to="none",
    )
    try:
        args = TrainingArguments(eval_strategy="epoch", save_strategy="epoch", **ta_common)
    except TypeError:
        args = TrainingArguments(evaluation_strategy="epoch", save_strategy="epoch", **ta_common)

    trainer_cls = WeightedTrainer if USE_CLASS_WEIGHTS else Trainer
    trainer = trainer_cls(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # --------------------------------------------------------------- #
    # Per-epoch metrics table. trainer.state.log_history keeps every
    # epoch's eval metrics in memory even though save_total_limit=1 only
    # keeps the best checkpoint's files on disk.
    # --------------------------------------------------------------- #
    metric_cols = ["epoch", "step", "eval_loss", "eval_accuracy", "eval_macro_f1",
                   "eval_contradiction_recall", "eval_contradiction_f1",
                   "eval_entailment_recall", "eval_neutral_recall"]
    epoch_rows = [e for e in trainer.state.log_history if "eval_loss" in e]
    metrics_df = pd.DataFrame(epoch_rows)[metric_cols].round(4)
    metrics_df.insert(0, "epoch_num", range(1, len(metrics_df) + 1))

    print("\nPer-epoch validation metrics:")
    print(metrics_df.to_string(index=False))

    report_dir = BASE_DIR / "outputs"
    report_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(report_dir / "finetune_epoch_metrics.csv", index=False)
    print(f"(saved to {report_dir / 'finetune_epoch_metrics.csv'})")

    selected = metrics_df.loc[metrics_df["step"] == trainer.state.best_global_step]
    print(f"\nSelected checkpoint: epoch {selected['epoch_num'].iloc[0]} "
          f"(step {trainer.state.best_global_step}), "
          f"selected on '{METRIC_FOR_BEST_MODEL}' = {trainer.state.best_metric:.4f}")
    print(selected.to_string(index=False))

    print("\nFinal validation metrics (best checkpoint, re-evaluated as a sanity check):")
    for k, v in trainer.evaluate().items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    OUTPUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(OUTPUT_MODEL_DIR))
    tokenizer.save_pretrained(str(OUTPUT_MODEL_DIR))
    print(f"\nSaved fine-tuned model to: {OUTPUT_MODEL_DIR}")
    print("Next: evaluate it on the frozen test sets with evaluate_deberta_baseline.py")


if __name__ == "__main__":
    main()
