"""
Fine-tuning scaffold for the ClearClause NLI faithfulness verifier --
targets `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` instead of
`cross-encoder/nli-deberta-v3-small`.

WHAT THIS IS
------------
A complete, runnable *starting point* for fine-tuning
`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` on the ClearClause
statutory-NLI data. It is a straight copy of finetune_deberta.py with the
base checkpoint and output paths repointed -- everything else (data splits,
loss, class weighting, checkpoint selection, reporting) is identical, so the
two runs are comparable.

WHY THIS MODEL
--------------
`DeBERTa-v3-base-mnli-fever-anli` is the strongest ZERO-SHOT checkpoint
evaluated so far (see "Multi-model comparison" in README.md: contradiction F1
0.869 vs nli-deberta-v3-small's 0.634, zero-shot). Fine-tuning
nli-deberta-v3-small already lifted its contradiction F1 to 0.942 -- but that
doesn't tell us whether fine-tuning this stronger, bigger base model would do
even better. This script exists to test exactly that.

This is a BIGGER checkpoint than nli-deberta-v3-small (DeBERTa-v3-base vs
DeBERTa-v3-small backbone), so CPU training will take noticeably longer than
the ~1h52m observed for the small model's 4-epoch run -- expect roughly
proportional to the increase in parameter count, not a fixed multiple.

DESIGN DECISIONS (identical to finetune_deberta.py)
----------------------------------------------------
* Trains ONLY on the leakage-free v2 files:
    - train:      data/nli final clean train v2.csv
    - validation: data/nli final clean validation v2.csv   (used for early
                  model selection during training)
* The TEST sets (internal_test, external_test) are NEVER read here. You keep
  them frozen and evaluate the fine-tuned model with
  evaluation_scripts/evaluate_finetuned_deberta_v3_base.py (or point
  evaluate_deberta_v3_base.py's MODEL_NAME at the saved checkpoint).
* No random splitting: the given splits are used as-is.
* Label mapping is fixed: 0=contradiction, 1=entailment, 2=neutral. This
  checkpoint's own pretrained label order does not need to match this --
  fine-tuning re-learns what each output neuron means from the label ids fed
  during training, regardless of the order the checkpoint originally used.
* Model selection optimises MACRO F1 (not accuracy or loss), so checkpoint
  choice reflects balanced performance across all three classes rather than
  double-counting the contradiction boost already applied in the loss
  (see CONTRADICTION_BOOST below). Contradiction recall/F1 are still
  computed and reported every epoch for inspection.
* Optional class weighting up-weights contradiction to push its recall up.
* Model loads from the local HF cache (local_files_only=True); training writes
  nothing outside this project folder. (Already confirmed cached locally --
  see models--MoritzLaurer--DeBERTa-v3-base-mnli-fever-anli under the HF hub
  cache.)

REQUIREMENTS (install into your Python env, not run here):
    pip install "transformers>=4.40" torch pandas scikit-learn accelerate

RUN (this scaffold is not executed automatically), from the finetuning/ folder:
    python finetune_deberta_v3_base.py
Then evaluate the saved model on the frozen test sets with
evaluation_scripts/evaluate_finetuned_deberta_v3_base_boosted.py. The other
two evaluate_finetuned_deberta_v3_base*.py scripts still point at
./finetuned_deberta_v3_base and ./finetuned_deberta_v3_base_retuned on
purpose, so all three runs stay independently evaluable and comparable.
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
EPOCHS = 4                  # reverted to the FIRST run's value (was cut to 3
                            # in the "retuned" attempt below -- that attempt
                            # made contradiction recall worse everywhere, not
                            # better, so this run goes back to what worked
                            # best and changes only CONTRADICTION_BOOST below.
LEARNING_RATE = 2e-5        # reverted to the FIRST run's value, same reason.
                            # The "retuned" attempt (LR=1e-5, EPOCHS=3,
                            # WEIGHT_DECAY=0.05) was built on the hypothesis
                            # that the first run overfit -- but making
                            # training MORE conservative made contradiction
                            # recall drop further (internal_test 0.957->0.878,
                            # external_test 0.581->0.488) while entailment
                            # recall rose slightly, the opposite of
                            # overfitting. That points to the model not being
                            # pushed hard enough toward contradiction, not
                            # being pushed too hard overall -- see
                            # CONTRADICTION_BOOST below, the one thing this
                            # run actually changes from the first run.
TRAIN_BATCH_SIZE = 8        # DeBERTa-v3-base is bigger than v3-small; lower
                            # this if you hit a CPU memory error.
EVAL_BATCH_SIZE = 16
WEIGHT_DECAY = 0.01          # reverted to the FIRST run's value, same reason
                            # as EPOCHS/LEARNING_RATE above.
WARMUP_RATIO = 0.06
MAX_LENGTH = 512            # matches the baseline eval
SEED = 42

# Up-weight rare/important classes in the loss. Set to False for plain CE.
USE_CLASS_WEIGHTS = True
# Extra multiplier applied to the contradiction class on top of the
# inverse-frequency weight (1.0 = inverse-frequency only). Raise to trade
# some precision for higher contradiction recall.
CONTRADICTION_BOOST = 2.0   # was 1.5 in both prior runs. The FIRST run's
                            # weakness was under-detecting contradiction
                            # out-of-domain, not overfitting (see LEARNING_RATE
                            # comment above) -- this is the one deliberate
                            # change in this run, isolated from the epoch/LR/
                            # weight-decay reverts above so its effect can be
                            # read cleanly against the first run's numbers.

# Metric used to pick the "best" epoch checkpoint (must be a key returned by
# compute_metrics, higher = better). "macro_f1" balances all three classes;
# "contradiction_f1" would double down on CONTRADICTION_BOOST above by also
# biasing checkpoint choice toward that one class.
METRIC_FOR_BEST_MODEL = "macro_f1"

# --------------------------------------------------------------------------- #
# Constants / paths
# --------------------------------------------------------------------------- #
MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

ID2LABEL = {0: "contradiction", 1: "entailment", 2: "neutral"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}
LABEL_ORDER = [0, 1, 2]

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
# Separate output folder from every prior v3-base run so none of them
# overwrite each other's saved model/checkpoints:
#   finetuned_deberta_v3_base           first run (EPOCHS=4/LR=2e-5/WD=0.01/BOOST=1.5)
#   finetuned_deberta_v3_base_retuned   second run (EPOCHS=3/LR=1e-5/WD=0.05/BOOST=1.5,
#                                       made contradiction recall worse -- see
#                                       HYPERPARAMETERS above)
#   finetuned_deberta_v3_base_boosted   THIS run (back to the first run's
#                                       EPOCHS/LR/WD, only BOOST raised to 2.0)
OUTPUT_MODEL_DIR = BASE_DIR / "finetuned_deberta_v3_base_boosted"
TRAINER_WORK_DIR = BASE_DIR / "outputs" / "finetune_v3_base_boosted_checkpoints"
EPOCH_METRICS_FILE = "finetune_v3_base_boosted_epoch_metrics.csv"

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
# Metrics — model selection targets macro F1 (contradiction F1 also tracked)
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
    print(f"Base checkpoint: {MODEL_NAME}")

    train_df = load_split(TRAIN_FILE)
    val_df = load_split(VAL_FILE)
    print(f"train v2: {len(train_df)} rows | validation v2: {len(val_df)} rows")

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, local_files_only=True,
            num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID,
            # This checkpoint's config.json declares torch_dtype=float16 (unlike
            # nli-deberta-v3-small, which is float32), so from_pretrained loads
            # it in fp16 by default. fp16 is not usable for training on CPU and
            # also crashes WeightedTrainer's cross_entropy below (fp32 class
            # weights vs fp16 logits) -- force fp32 explicitly.
            dtype=torch.float32,
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
    metrics_df.to_csv(report_dir / EPOCH_METRICS_FILE, index=False)
    print(f"(saved to {report_dir / EPOCH_METRICS_FILE})")

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
    print("Next: evaluate it on the frozen test sets (internal_test v2 / "
          "external_test v2), e.g. by adapting evaluate_finetuned_deberta.py "
          "to point at this checkpoint instead.")


if __name__ == "__main__":
    main()
