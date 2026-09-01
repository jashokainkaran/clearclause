"""
Shared evaluation utilities for the ClearClause multi-model NLI baseline
comparison.

Used by all five model scripts:
  evaluate_deberta_baseline.py
  evaluate_deberta_v3_base.py
  evaluate_roberta_large_mnli.py
  evaluate_distilroberta_nli.py
  evaluate_modernbert_nli.py

Inference only. No training, no fine-tuning, no random splitting, no
modification of gold labels or source data. Every caller script forces
offline/local-cache-only behaviour (HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE /
local_files_only=True) before this module is imported.

Canonical ClearClause label mapping -- used for GOLD labels, and as the fixed
output order for every report, prediction file, and confusion matrix:
  0 = contradiction
  1 = entailment
  2 = neutral

A checkpoint's own output order is read from model.config.id2label at
runtime (see resolve_label_permutation) and remapped into this canonical
order. No model's label order is ever assumed -- each script's checkpoint is
independently verified when it is run.
"""

import csv
import sys
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


# --------------------------------------------------------------------------- #
# Canonical (gold-label) constants
# --------------------------------------------------------------------------- #

CANONICAL_ID2LABEL = {0: "contradiction", 1: "entailment", 2: "neutral"}
CANONICAL_LABEL2ID = {v: k for k, v in CANONICAL_ID2LABEL.items()}
CANONICAL_LABEL_ORDER = [0, 1, 2]
VALID_LABEL_NAMES = set(CANONICAL_LABEL2ID.keys())

REQUIRED_COLUMNS = ["premise", "hypothesis", "gold_label"]


# --------------------------------------------------------------------------- #
# Gold-label normalisation
# --------------------------------------------------------------------------- #

def normalize_gold_label(value) -> int:
    """Accept label names (case-insensitive) or the exact whole-number ids
    0/1/2. Fractional numeric values (e.g. "1.5") are rejected rather than
    silently truncated -- a gold_label must name a class exactly, not round
    to one."""
    text = str(value).strip().lower()
    if text in CANONICAL_LABEL2ID:
        return CANONICAL_LABEL2ID[text]
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
    if label_id in CANONICAL_ID2LABEL:
        return label_id
    raise ValueError(f"Unknown gold_label id: {value!r}")


def _is_blank(series: pd.Series) -> pd.Series:
    """True where a cell is NaN or, once stringified, empty/whitespace-only."""
    return series.isna() | (series.astype(str).str.strip() == "")


# --------------------------------------------------------------------------- #
# Dataset loading & validation (identical rules for every model)
# --------------------------------------------------------------------------- #

def load_dataset(name: str, path: Path) -> pd.DataFrame:
    """Load one CSV and run all per-dataset validation checks. Read-only."""
    if not path.exists():
        print(f"ERROR [{name}]: input file not found: {path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(path)
    validate_dataset(name, df)
    return df


def validate_dataset(name: str, df: pd.DataFrame) -> None:
    """Print/log the required per-dataset validation. Exits on hard failures."""
    print(f"\n--- Validating dataset: {name} ({len(df)} rows) ---")

    # (1) required columns exist
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        print(f"ERROR [{name}]: missing required column(s): {missing}", file=sys.stderr)
        sys.exit(1)
    print(f"  [ok] required columns present: {REQUIRED_COLUMNS}")

    # (2) no missing/whitespace-only premise
    bad_premise = df.index[_is_blank(df["premise"])].tolist()
    if bad_premise:
        print(f"ERROR [{name}]: blank/missing premise at rows: {bad_premise}", file=sys.stderr)
        sys.exit(1)
    print("  [ok] no blank/missing premise")

    # (3) no missing/whitespace-only hypothesis
    bad_hyp = df.index[_is_blank(df["hypothesis"])].tolist()
    if bad_hyp:
        print(f"ERROR [{name}]: blank/missing hypothesis at rows: {bad_hyp}", file=sys.stderr)
        sys.exit(1)
    print("  [ok] no blank/missing hypothesis")

    # (4) gold_label only contains entailment/contradiction/neutral
    normalized_names = []
    bad_labels = []
    for idx, value in df["gold_label"].items():
        try:
            normalized_names.append(CANONICAL_ID2LABEL[normalize_gold_label(value)])
        except ValueError:
            bad_labels.append((idx, value))
    if bad_labels:
        print(f"ERROR [{name}]: invalid gold_label values (row, value): {bad_labels}", file=sys.stderr)
        sys.exit(1)
    print(f"  [ok] gold_label values all within {sorted(VALID_LABEL_NAMES)}")

    # (5) duplicate nli_id count (only if column exists)
    if "nli_id" in df.columns:
        dup_count = int(df["nli_id"].duplicated().sum())
        print(f"  duplicate nli_id count: {dup_count}")
    else:
        print("  nli_id column not present -> duplicate check skipped")

    # (6) row count
    print(f"  row count: {len(df)}")

    # (7) label distribution
    dist = pd.Series(normalized_names).value_counts()
    dist_str = ", ".join(f"{lbl}={int(dist.get(lbl, 0))}" for lbl in sorted(VALID_LABEL_NAMES))
    print(f"  label distribution: {dist_str}")


def duplicate_nli_id_count(df: pd.DataFrame) -> int:
    if "nli_id" in df.columns:
        return int(df["nli_id"].duplicated().sum())
    return 0


def label_distribution(gold_ids) -> dict:
    counts = {CANONICAL_ID2LABEL[i]: 0 for i in CANONICAL_LABEL_ORDER}
    for i in gold_ids:
        counts[CANONICAL_ID2LABEL[i]] += 1
    return counts


# --------------------------------------------------------------------------- #
# Model loading & label-mapping verification
# --------------------------------------------------------------------------- #

def load_model_offline(model_name: str, device: torch.device):
    """Load tokenizer + model from LOCAL CACHE ONLY. Exit clearly if not cached.
    Never downloads anything (offline mode is enforced by the caller script
    before this function runs)."""
    print(f"Loading model '{model_name}' from local cache (local_files_only=True) ...")
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, local_files_only=True
        )
    except Exception as exc:  # noqa: BLE001 - clear, catch-all exit
        print(
            "\nERROR: could not load the model from the local Hugging Face cache.\n"
            f"       Model '{model_name}' does not appear to be fully cached.\n"
            "       This script refuses to download it (offline mode is enforced,\n"
            "       so no legal data or network call leaves this machine).\n"
            "       Cache the model once on a trusted machine/connection, then rerun.\n"
            f"       Underlying error: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    model.to(device)
    model.eval()
    return tokenizer, model


def get_environment_versions() -> tuple:
    """Return (torch_version, transformers_version) for report metadata.
    transformers is imported lazily here (it is always already loaded by the
    time this is called, via load_model_offline)."""
    import transformers

    return torch.__version__, transformers.__version__


def resolve_label_permutation(model_id2label: dict, model_name: str) -> list:
    """
    Verify a checkpoint's own id2label against the canonical ClearClause
    mapping (contradiction/entailment/neutral) and return `permutation` such
    that, for a native-order probability tensor `p`:

        p[:, permutation] == canonical-order probabilities
        (i.e. permutation[canonical_id] = the checkpoint's own output index
        for that canonical label)

    This never assumes any fixed ordering -- it is driven entirely by the
    model's own config at runtime, read fresh for every checkpoint. Exits
    with a clear error (refuses to guess) if the checkpoint does not expose
    exactly one each of contradiction/entailment/neutral.
    """
    print(f"Detected model.config.id2label for '{model_name}': {model_id2label}")

    if len(model_id2label) != 3:
        print(
            f"ERROR: '{model_name}' reports {len(model_id2label)} output "
            "label(s); expected exactly 3 (contradiction/entailment/neutral). "
            "Refusing to guess a label mapping.",
            file=sys.stderr,
        )
        sys.exit(1)

    normalized = {}
    for idx, raw_name in model_id2label.items():
        name = str(raw_name).strip().lower()
        if name not in CANONICAL_LABEL2ID:
            print(
                f"ERROR: could not safely interpret '{model_name}' output "
                f"index {idx} = {raw_name!r} as one of "
                "contradiction/entailment/neutral. Raw id2label: "
                f"{model_id2label}. Refusing to silently apply another "
                "model's label mapping.",
                file=sys.stderr,
            )
            sys.exit(1)
        normalized[int(idx)] = name

    if set(normalized.values()) != VALID_LABEL_NAMES:
        print(
            f"ERROR: '{model_name}' id2label does not contain exactly one "
            f"each of contradiction/entailment/neutral. Resolved: "
            f"{normalized}. Refusing to continue.",
            file=sys.stderr,
        )
        sys.exit(1)

    permutation = [None, None, None]
    for model_idx, name in normalized.items():
        canonical_idx = CANONICAL_LABEL2ID[name]
        permutation[canonical_idx] = model_idx

    print(
        f"Resolved canonical label remap for '{model_name}': "
        f"contradiction<-nativeIdx{permutation[0]}, "
        f"entailment<-nativeIdx{permutation[1]}, "
        f"neutral<-nativeIdx{permutation[2]}"
    )
    return permutation


# --------------------------------------------------------------------------- #
# Token-length pre-flight check (runs before any inference)
# --------------------------------------------------------------------------- #

def check_token_lengths(name: str, df: pd.DataFrame, tokenizer, max_length: int,
                         output_dir: Path) -> None:
    """
    Tokenize every premise-hypothesis pair WITHOUT truncation and check the
    combined token length (special tokens included, exactly as the model
    would see the pair) against `max_length`.

    Never truncates anything itself. If every row fits, prints a summary and
    returns. If any row exceeds `max_length`, this refuses to silently cut
    the premise or hypothesis: it prints a clear error for the first
    offending row, writes every offending row to
    outputs/<model>/<dataset>_overlength_rows.csv, and exits before any
    inference is run -- so reported metrics are always based on the complete
    dataset, never a silently-truncated or silently-dropped subset of it.
    """
    premises = df["premise"].astype(str).tolist()
    hypotheses = df["hypothesis"].astype(str).tolist()
    has_nli_id = "nli_id" in df.columns
    has_source_id = "source_id" in df.columns

    combined_lengths = []
    overlength_rows = []

    for i, (premise, hypothesis) in enumerate(zip(premises, hypotheses)):
        premise_len = len(tokenizer.encode(premise, add_special_tokens=False))
        hypothesis_len = len(tokenizer.encode(hypothesis, add_special_tokens=False))
        # Exact combined length the model would see (with special tokens),
        # computed without any truncation.
        pair_encoding = tokenizer(premise, hypothesis, truncation=False)
        combined_len = len(pair_encoding["input_ids"])
        combined_lengths.append(combined_len)

        if combined_len > max_length:
            overlength_rows.append({
                "row_index": i,
                "nli_id": df.iloc[i]["nli_id"] if has_nli_id else "",
                "source_id": df.iloc[i]["source_id"] if has_source_id else "",
                "premise_token_count": premise_len,
                "hypothesis_token_count": hypothesis_len,
                "combined_token_count": combined_len,
                "max_length": max_length,
                "premise": premise,
                "hypothesis": hypothesis,
            })

    rows_checked = len(df)
    max_combined = max(combined_lengths) if combined_lengths else 0
    avg_combined = (sum(combined_lengths) / rows_checked) if rows_checked else 0.0
    num_exceeding = len(overlength_rows)

    print(f"\n--- Token-length validation for dataset: {name} ---")
    print(f"  rows checked: {rows_checked}")
    print(f"  max observed combined token length: {max_combined}")
    print(f"  average combined token length: {avg_combined:.2f}")
    print(f"  rows exceeding MAX_LENGTH ({max_length}): {num_exceeding}")

    if not overlength_rows:
        return

    overlength_path = output_dir / f"{name}_overlength_rows.csv"
    write_overlength_report(overlength_rows, overlength_path)

    first = overlength_rows[0]
    print(
        f"\nERROR [{name}]: {num_exceeding} row(s) exceed MAX_LENGTH={max_length} "
        "tokens when the premise and hypothesis are tokenized together WITHOUT "
        "truncation. Truncating either field would silently discard part of "
        "the statutory premise or the atomic claim being verified, so this "
        "dataset's evaluation is stopped before any inference is run.\n"
        f"  dataset: {name}\n"
        f"  row index: {first['row_index']}\n"
        f"  nli_id: {first['nli_id']!r}\n"
        f"  actual token length: {first['combined_token_count']}\n"
        f"  configured MAX_LENGTH: {max_length}\n"
        f"  premise token length: {first['premise_token_count']}\n"
        f"  hypothesis token length: {first['hypothesis_token_count']}\n"
        f"  full overlength report written to: {overlength_path}",
        file=sys.stderr,
    )
    sys.exit(1)


def write_overlength_report(overlength_rows: list, path: Path) -> None:
    """Write every over-length row found by check_token_lengths() to its own
    CSV file, so no offending row's full text is ever lost even though
    evaluation stops."""
    fieldnames = ["row_index", "nli_id", "source_id", "premise_token_count",
                  "hypothesis_token_count", "combined_token_count",
                  "max_length", "premise", "hypothesis"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in overlength_rows:
            writer.writerow(row)


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #

def run_inference(df, tokenizer, model, device, batch_size, permutation):
    """
    Forward pass only. No truncation is ever applied here: every pair has
    already been verified by check_token_lengths() to fit within MAX_LENGTH
    (including special tokens) before this function is called. Returns
    (pred_ids, confidences, canonical_probs).

    Probabilities are reordered into canonical (contradiction, entailment,
    neutral) order via `permutation` before argmax/confidence are computed,
    so every downstream metric, report, and prediction file is model-agnostic.
    """
    premises = df["premise"].astype(str).tolist()
    hypotheses = df["hypothesis"].astype(str).tolist()

    pred_ids, confidences, canonical_probs = [], [], []

    with torch.inference_mode():
        for start in range(0, len(premises), batch_size):
            end = start + batch_size
            encoded = tokenizer(
                premises[start:end],
                hypotheses[start:end],
                padding=True,
                truncation=False,
                return_tensors="pt",
            ).to(device)

            logits = model(**encoded).logits
            native_probs = torch.softmax(logits, dim=-1)
            probs = native_probs[:, permutation]  # reorder to canonical order

            batch_pred = torch.argmax(probs, dim=-1).cpu().tolist()
            batch_conf = torch.max(probs, dim=-1).values.cpu().tolist()

            pred_ids.extend(batch_pred)
            confidences.extend(batch_conf)
            canonical_probs.extend(probs.cpu().tolist())

            print(f"    processed {min(end, len(premises))}/{len(premises)}")

    return pred_ids, confidences, canonical_probs


# --------------------------------------------------------------------------- #
# Metrics & reporting
# --------------------------------------------------------------------------- #

def build_predictions_df(df, gold_ids, pred_ids, confidences, canonical_probs) -> pd.DataFrame:
    """Original columns + gold_label_id, gold_label_name, pred_label,
    confidence, correct, and the three canonical-order class probabilities."""
    out = df.copy()
    out["gold_label_id"] = gold_ids
    out["gold_label_name"] = [CANONICAL_ID2LABEL[i] for i in gold_ids]
    out["pred_label"] = [CANONICAL_ID2LABEL[i] for i in pred_ids]
    out["confidence"] = [round(float(c), 6) for c in confidences]
    out["correct"] = [g == p for g, p in zip(gold_ids, pred_ids)]
    out["prob_contradiction"] = [round(float(p[0]), 6) for p in canonical_probs]
    out["prob_entailment"] = [round(float(p[1]), 6) for p in canonical_probs]
    out["prob_neutral"] = [round(float(p[2]), 6) for p in canonical_probs]
    return out


def compute_metrics(gold_ids, pred_ids) -> dict:
    accuracy = sum(g == p for g, p in zip(gold_ids, pred_ids)) / len(gold_ids)
    precision, recall, f1, support = precision_recall_fscore_support(
        gold_ids, pred_ids, labels=CANONICAL_LABEL_ORDER, zero_division=0
    )
    macro_f1 = float(f1.mean())
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        gold_ids, pred_ids, labels=CANONICAL_LABEL_ORDER,
        average="weighted", zero_division=0,
    )
    cm = confusion_matrix(gold_ids, pred_ids, labels=CANONICAL_LABEL_ORDER)

    by = {CANONICAL_ID2LABEL[i]: idx for idx, i in enumerate(CANONICAL_LABEL_ORDER)}
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
        "macro_f1": macro_f1,
        "weighted_f1": float(weighted_f1),
        "confusion_matrix": cm,
        "entailment_recall": float(recall[by["entailment"]]),
        "neutral_recall": float(recall[by["neutral"]]),
        "neutral_precision": float(precision[by["neutral"]]),
        "contradiction_recall": float(recall[by["contradiction"]]),
        "contradiction_f1": float(f1[by["contradiction"]]),
    }


def write_confusion_matrix_csv(cm, path: Path) -> None:
    """Write the gold x predicted confusion matrix as its own CSV file
    (rows = gold, columns = predicted, fixed contradiction/entailment/neutral
    order). This file is always written, separately from the text report."""
    labels = [CANONICAL_ID2LABEL[i] for i in CANONICAL_LABEL_ORDER]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["gold\\pred"] + labels)
        for i, label in enumerate(labels):
            writer.writerow([label] + [int(cm[i][j]) for j in range(len(labels))])


def format_report(model_name: str, name: str, df_len: int, dup_ids: int, dist: dict,
                   metrics: dict, predictions_df: pd.DataFrame,
                   device, batch_size: int, max_length: int,
                   torch_version: str, transformers_version: str) -> str:
    lines = []
    lines.append("NLI Diagnostic Baseline Report")
    lines.append(f"Model: {model_name}")
    lines.append(f"Device: {device}")
    lines.append(f"Batch size: {batch_size}")
    lines.append(f"Max sequence length: {max_length}")
    lines.append(f"PyTorch version: {torch_version}")
    lines.append(f"Transformers version: {transformers_version}")
    lines.append(f"Dataset: {name}")
    lines.append(f"Row count: {df_len}")
    dist_str = ", ".join(f"{lbl}={dist[lbl]}" for lbl in sorted(dist.keys()))
    lines.append(f"Label distribution: {dist_str}")
    lines.append(f"Duplicate nli_id count: {dup_ids}")
    lines.append("")

    lines.append(f"Accuracy: {metrics['accuracy']:.4f}")
    lines.append(f"Macro F1: {metrics['macro_f1']:.4f}")
    lines.append(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    lines.append(f"Entailment recall: {metrics['entailment_recall']:.4f}")
    lines.append(f"Neutral recall: {metrics['neutral_recall']:.4f}")
    lines.append(f"Neutral precision: {metrics['neutral_precision']:.4f}")
    lines.append(f"Contradiction recall: {metrics['contradiction_recall']:.4f}")
    lines.append(f"Contradiction F1: {metrics['contradiction_f1']:.4f}")
    lines.append("")

    lines.append("Per-label metrics:")
    lines.append(f"{'Label':<15}{'Precision':>12}{'Recall':>12}{'F1':>12}{'Support':>12}")
    for idx, label_id in enumerate(CANONICAL_LABEL_ORDER):
        lines.append(
            f"{CANONICAL_ID2LABEL[label_id]:<15}"
            f"{metrics['precision'][idx]:>12.4f}"
            f"{metrics['recall'][idx]:>12.4f}"
            f"{metrics['f1'][idx]:>12.4f}"
            f"{metrics['support'][idx]:>12d}"
        )
    lines.append("")

    lines.append("Confusion matrix (rows = gold, columns = predicted):")
    header = "gold\\pred".ljust(16) + "".join(
        f"{CANONICAL_ID2LABEL[l]:>15}" for l in CANONICAL_LABEL_ORDER
    )
    lines.append(header)
    cm = metrics["confusion_matrix"]
    for i, label_id in enumerate(CANONICAL_LABEL_ORDER):
        row = f"{CANONICAL_ID2LABEL[label_id]:<16}" + "".join(
            f"{cm[i][j]:>15d}" for j in range(len(CANONICAL_LABEL_ORDER))
        )
        lines.append(row)
    lines.append("(Also saved separately as <dataset>_confusion_matrix.csv)")
    lines.append("")

    incorrect = predictions_df[~predictions_df["correct"]]
    lines.append(f"Total incorrect predictions: {len(incorrect)}")
    lines.append("")

    # Structured incorrect-prediction table
    cols = ["nli_id", "source_id", "split", "gold_label",
            "pred_label", "confidence", "error_type", "hypothesis"]
    widths = {"nli_id": 12, "source_id": 12, "split": 14, "gold_label": 14,
              "pred_label": 14, "confidence": 11, "error_type": 20, "hypothesis": 60}
    lines.append("Incorrect predictions:")
    lines.append("".join(f"{c:<{widths[c]}}" for c in cols))
    for _, row in incorrect.iterrows():
        def cell(col):
            if col == "gold_label":
                return str(row.get("gold_label_name", ""))
            if col == "confidence":
                return f"{float(row['confidence']):.4f}"
            val = row.get(col, "")
            return "" if pd.isna(val) else str(val)
        rendered = []
        for c in cols:
            text = cell(c)
            if c == "hypothesis":
                rendered.append(text)  # last column, do not pad/truncate
            else:
                rendered.append(f"{text[:widths[c] - 1]:<{widths[c]}}")
        lines.append("".join(rendered))
    lines.append("")

    return "\n".join(lines)


def summary_row(name: str, df_len: int, metrics: dict,
                batch_size: int, max_length: int) -> dict:
    p, r, f1 = metrics["precision"], metrics["recall"], metrics["f1"]
    by = {CANONICAL_ID2LABEL[i]: idx for idx, i in enumerate(CANONICAL_LABEL_ORDER)}
    e, n, c = by["entailment"], by["neutral"], by["contradiction"]
    return {
        "dataset": name,
        "rows": df_len,
        "batch_size": batch_size,
        "max_length": max_length,
        "accuracy": round(metrics["accuracy"], 6),
        "macro_f1": round(metrics["macro_f1"], 6),
        "weighted_f1": round(metrics["weighted_f1"], 6),
        "entailment_precision": round(float(p[e]), 6),
        "entailment_recall": round(float(r[e]), 6),
        "entailment_f1": round(float(f1[e]), 6),
        "neutral_precision": round(float(p[n]), 6),
        "neutral_recall": round(float(r[n]), 6),
        "neutral_f1": round(float(f1[n]), 6),
        "contradiction_precision": round(float(p[c]), 6),
        "contradiction_recall": round(float(r[c]), 6),
        "contradiction_f1": round(float(f1[c]), 6),
    }


SUMMARY_COLUMNS = [
    "dataset", "rows", "batch_size", "max_length",
    "accuracy", "macro_f1", "weighted_f1",
    "entailment_precision", "entailment_recall", "entailment_f1",
    "neutral_precision", "neutral_recall", "neutral_f1",
    "contradiction_precision", "contradiction_recall", "contradiction_f1",
]


# --------------------------------------------------------------------------- #
# Per-dataset orchestration (shared by every model script's main())
# --------------------------------------------------------------------------- #

def evaluate_set(model_name: str, name: str, df: pd.DataFrame, tokenizer, model,
                  device, permutation: list, batch_size: int, max_length: int,
                  output_dir: Path, torch_version: str,
                  transformers_version: str) -> dict:
    print(f"\n=== Evaluating: {name} ({len(df)} rows) ===")
    gold_ids = df["gold_label"].apply(normalize_gold_label).tolist()

    # Verify every pair fits within MAX_LENGTH (no truncation) before running
    # any inference. Exits, after writing an overlength report, if not --
    # metrics are never computed on a silently truncated or dropped subset.
    check_token_lengths(name, df, tokenizer, max_length, output_dir)

    pred_ids, confidences, canonical_probs = run_inference(
        df, tokenizer, model, device, batch_size, permutation
    )

    predictions_df = build_predictions_df(df, gold_ids, pred_ids, confidences, canonical_probs)
    metrics = compute_metrics(gold_ids, pred_ids)
    dist = label_distribution(gold_ids)
    dup_ids = duplicate_nli_id_count(df)

    report = format_report(model_name, name, len(df), dup_ids, dist, metrics,
                            predictions_df, device, batch_size, max_length,
                            torch_version, transformers_version)

    pred_path = output_dir / f"{name}_predictions.csv"
    report_path = output_dir / f"{name}_report.txt"
    cm_path = output_dir / f"{name}_confusion_matrix.csv"
    predictions_df.to_csv(pred_path, index=False)
    report_path.write_text(report, encoding="utf-8")
    write_confusion_matrix_csv(metrics["confusion_matrix"], cm_path)
    print(f"  wrote {pred_path.name}, {report_path.name}, and {cm_path.name}")

    return summary_row(name, len(df), metrics, batch_size, max_length)
