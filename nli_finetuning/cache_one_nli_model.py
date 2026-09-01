#!/usr/bin/env python
"""
Download/cache one Hugging Face NLI checkpoint.

Usage:
    python cache_one_nli_model.py <model_name>

This script downloads model/tokenizer files only. It does not read legal CSV
files and does not run inference.
"""

from __future__ import annotations

import sys

from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python cache_one_nli_model.py <model_name>", file=sys.stderr)
        return 2

    model_name = sys.argv[1].strip()
    if not model_name:
        print("ERROR: model_name cannot be empty.", file=sys.stderr)
        return 2

    print("=" * 80)
    print(f"Preparing Hugging Face checkpoint: {model_name}")
    print("This step downloads tokenizer/model files only.")
    print("No legal CSV data is read or uploaded.")

    try:
        print("\nChecking local cache first...")
        AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        AutoModelForSequenceClassification.from_pretrained(
            model_name,
            local_files_only=True,
        )
        print(f"\nSUCCESS: {model_name} is already cached.")
        return 0
    except OSError:
        print("\nCheckpoint is not fully available in the local cache.")
        print("Downloading missing files from Hugging Face...")

    try:
        AutoTokenizer.from_pretrained(model_name)
        AutoModelForSequenceClassification.from_pretrained(model_name)
        print(f"\nSUCCESS: {model_name} downloaded and cached.")
        return 0
    except Exception as exc:
        print(f"\nFAILED: Could not cache {model_name}", file=sys.stderr)
        print(f"Reason: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
