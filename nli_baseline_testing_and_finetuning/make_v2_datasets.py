"""
Create leakage-free "v2" dataset files for ClearClause NLI fine-tuning.

WHAT THIS DOES
--------------
The original splits share identical (premise, hypothesis) pairs between the
train file and the validation / internal_test files. That is harmless for a
zero-shot baseline, but once the model is fine-tuned on train, those shared
pairs would be *memorised* and would inflate validation / internal_test
scores. This script removes that leakage from the evaluation sets only.

It writes NEW files (suffix " v2") and never modifies or deletes the originals:
  - nli final clean train v2.csv           (identical copy of train; train may
                                             legitimately contain anything)
  - nli final clean validation v2.csv      (rows whose (premise,hypothesis)
                                             pair also appears in train removed)
  - nli final clean internal test v2.csv   (same leakage removal)
  - nli final clean external test v2.csv   (identical copy; already 0 overlap)

Nothing is relabelled, nothing is randomly re-split, and the test difficulty
is preserved. Only exact train-overlap rows are dropped from val / internal.

A plain-text report of exactly what was removed is written to
outputs/dataset_v2_dedup_report.txt.
"""

import re
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

TRAIN = DATA_DIR / "nli final clean train.csv"
VAL = DATA_DIR / "nli final clean validation.csv"
INTERNAL = DATA_DIR / "nli final clean internal test.csv"
EXTERNAL = DATA_DIR / "nli final clean external test.csv"


def norm(text) -> str:
    """Normalise for pair-matching: strip, lowercase, collapse whitespace."""
    return re.sub(r"\s+", " ", str(text).strip().lower())


def pair_key(df: pd.DataFrame) -> pd.Series:
    return df["premise"].map(norm) + " ||| " + df["hypothesis"].map(norm)


def label_counts(df: pd.DataFrame) -> str:
    vc = df["gold_label"].astype(str).str.lower().value_counts()
    return ", ".join(f"{k}={int(v)}" for k, v in vc.items())


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(TRAIN)
    val = pd.read_csv(VAL)
    internal = pd.read_csv(INTERNAL)
    external = pd.read_csv(EXTERNAL)

    train_pairs = set(pair_key(train))

    report = []
    report.append("ClearClause NLI - v2 dataset leakage removal report")
    report.append("=" * 55)
    report.append("Leakage definition: an evaluation row whose normalised")
    report.append("(premise, hypothesis) pair also occurs in the TRAIN file.")
    report.append("Only such rows are removed, and only from validation /")
    report.append("internal_test. Train and external are copied unchanged.")
    report.append("")

    cleaned = {}
    for name, df in [("validation", val), ("internal_test", internal)]:
        keys = pair_key(df)
        leaked_mask = keys.isin(train_pairs)
        leaked = df[leaked_mask]
        kept = df[~leaked_mask].copy()
        cleaned[name] = kept

        report.append(f"[{name}]")
        report.append(f"  original rows:            {len(df)}")
        report.append(f"  rows removed (leaked):    {int(leaked_mask.sum())}")
        report.append(f"  rows kept (v2):           {len(kept)}")
        report.append(f"  label dist BEFORE:        {label_counts(df)}")
        report.append(f"  label dist AFTER (v2):    {label_counts(kept)}")
        if len(leaked):
            report.append(f"  removed-by-label:         {label_counts(leaked)}")
            if "error_type" in leaked.columns:
                et = leaked["error_type"].fillna("(none)").value_counts()
                et_str = ", ".join(f"{k}={int(v)}" for k, v in et.items())
                report.append(f"  removed-by-error_type:    {et_str}")
        report.append("")

    # Write v2 files (originals left untouched).
    train.to_csv(DATA_DIR / "nli final clean train v2.csv", index=False)
    cleaned["validation"].to_csv(DATA_DIR / "nli final clean validation v2.csv", index=False)
    cleaned["internal_test"].to_csv(DATA_DIR / "nli final clean internal test v2.csv", index=False)
    external.to_csv(DATA_DIR / "nli final clean external test v2.csv", index=False)

    # Sanity re-check: confirm zero overlap remains in the v2 eval files.
    report.append("Post-cleaning verification (should be 0):")
    for name in ("validation", "internal_test"):
        remaining = int(pair_key(cleaned[name]).isin(train_pairs).sum())
        report.append(f"  {name} v2 rows still overlapping train: {remaining}")
    report.append("")
    report.append("Files written (originals unchanged):")
    report.append("  data/nli final clean train v2.csv           (copy of train)")
    report.append("  data/nli final clean validation v2.csv      (deduped)")
    report.append("  data/nli final clean internal test v2.csv   (deduped)")
    report.append("  data/nli final clean external test v2.csv   (copy of external)")

    text = "\n".join(report)
    (OUTPUT_DIR / "dataset_v2_dedup_report.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
