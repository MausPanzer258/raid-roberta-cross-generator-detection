#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prepare leakage-safe IID train/validation/test splits for the RAID project.

Input
-----
The balanced RAID subset created in the previous step:
    raid_balanced_final/raid_balanced.csv

What this script does
---------------------
1. Loads the balanced dataset.
2. Removes exact duplicate texts (keeps the first occurrence).
3. Preserves the original text otherwise: no lowercasing, punctuation removal,
   stemming, stop-word removal, etc.
4. Creates a leakage-safe grouping variable based on source_id.
5. Uses StratifiedGroupKFold to create 10 source-disjoint folds while
   stratifying jointly by generator/model and domain.
6. Assigns:
       folds 0-6 -> train       (~70%)
       fold 7   -> validation  (~10%)
       folds 8-9 -> test       (~20%)
7. Verifies that there is:
       - zero source_id overlap across splits
       - zero exact-text overlap across splits
8. Saves the cleaned data, split files, manifest, duplicate audit,
   distribution summary, and leakage report.

Why source-aware splitting matters
----------------------------------
Different RAID generations can share the same underlying source_id. If texts
derived from the same source appear in both training and test sets, evaluation
can be artificially optimistic. Grouping by source_id prevents this.

Why we do NOT aggressively clean the text
-----------------------------------------
The research question is whether an AI detector learns general AI-writing
signals or generator-specific artifacts. Punctuation, formatting, lexical
choice, discourse style, and other surface signals are therefore part of the
phenomenon we want to study. We only remove exact duplicate texts.

Requirements
------------
    pip install pandas scikit-learn
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError as exc:
    raise SystemExit(
        "\nMissing dependency.\n"
        "Install the required packages first:\n\n"
        "    pip install pandas numpy scikit-learn\n\n"
        f"Original import error: {exc}\n"
    )


RANDOM_STATE = 42
N_FOLDS = 10

DEFAULT_INPUT_CANDIDATES = [
    Path("raid_balanced_final/raid_balanced.csv"),
    Path("raid_balanced.csv"),
]

REQUIRED_COLUMNS = [
    "id",
    "source_id",
    "model",
    "domain",
    "text",
    "label",
    "generator",
]


def find_default_input() -> Path:
    for path in DEFAULT_INPUT_CANDIDATES:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find raid_balanced.csv automatically.\n"
        "Expected one of:\n"
        + "\n".join(f"  - {p}" for p in DEFAULT_INPUT_CANDIDATES)
        + "\n\nYou can also specify the file explicitly:\n"
          '  python prepare_raid_splits_v2.py --input "path/to/raid_balanced.csv"'
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(
        value.encode("utf-8", errors="replace")
    ).hexdigest()


def validate_input(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"Input dataset is missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    if df["id"].isna().any():
        raise RuntimeError("Input contains missing RAID row IDs.")

    if df["id"].duplicated().any():
        n = int(df["id"].duplicated().sum())
        raise RuntimeError(f"Input contains {n} duplicated RAID row IDs.")

    if df["text"].isna().any():
        raise RuntimeError("Input contains missing text.")

    if df["text"].astype(str).str.strip().eq("").any():
        raise RuntimeError("Input contains empty text.")


def deduplicate_exact_text(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove exact duplicate text rows while keeping an audit file.

    We intentionally use exact equality only; near-duplicates are NOT removed.
    """
    work = df.copy()
    work["text"] = work["text"].astype(str)
    work["text_hash"] = work["text"].map(sha256_text)

    duplicate_mask = work.duplicated(subset=["text"], keep="first")

    first_id_for_text = (
        work.loc[~duplicate_mask, ["text", "id"]]
        .drop_duplicates("text")
        .set_index("text")["id"]
    )

    removed = work.loc[duplicate_mask].copy()
    if not removed.empty:
        removed["duplicate_of_id"] = removed["text"].map(first_id_for_text)

    clean = work.loc[~duplicate_mask].copy().reset_index(drop=True)

    return clean, removed.reset_index(drop=True)


def make_group_column(df: pd.DataFrame) -> pd.Series:
    """
    Group by source_id.

    If a source_id is missing, the row gets its own unique group based on id,
    instead of grouping every missing value together.
    """
    source = df["source_id"].astype("string")
    fallback = "missing_source::" + df["id"].astype(str)

    return source.where(
        source.notna() & source.str.strip().ne(""),
        fallback,
    ).astype(str)


def create_folds(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    work["split_group"] = make_group_column(work)

    # Joint model x domain stratification is more informative for this
    # experiment than binary label stratification alone.
    work["stratum"] = (
        work["model"].astype(str)
        + "__"
        + work["domain"].astype(str)
    )

    sgkf = StratifiedGroupKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fold = np.full(len(work), -1, dtype=int)

    for fold_id, (_, test_idx) in enumerate(
        sgkf.split(
            X=np.zeros(len(work)),
            y=work["stratum"],
            groups=work["split_group"],
        )
    ):
        fold[test_idx] = fold_id

    if (fold < 0).any():
        raise RuntimeError("At least one row was not assigned to a fold.")

    work["fold"] = fold

    work["split"] = np.select(
        [
            work["fold"].between(0, 6),
            work["fold"].eq(7),
            work["fold"].isin([8, 9]),
        ],
        [
            "train",
            "validation",
            "test",
        ],
        default="ERROR",
    )

    if (work["split"] == "ERROR").any():
        raise RuntimeError("Unexpected fold-to-split assignment error.")

    return work


def pairwise_overlap(
    df: pd.DataFrame,
    column: str,
    a: str,
    b: str,
) -> int:
    left = set(df.loc[df["split"] == a, column].astype(str))
    right = set(df.loc[df["split"] == b, column].astype(str))
    return len(left & right)


def leakage_report(df: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]

    rows: List[Dict[str, object]] = []

    for a, b in pairs:
        rows.append({
            "split_a": a,
            "split_b": b,
            "source_group_overlap": pairwise_overlap(
                df, "split_group", a, b
            ),
            "exact_text_hash_overlap": pairwise_overlap(
                df, "text_hash", a, b
            ),
        })

    return pd.DataFrame(rows)


def validate_no_leakage(report: pd.DataFrame) -> None:
    bad_source = int(report["source_group_overlap"].sum())
    bad_text = int(report["exact_text_hash_overlap"].sum())

    if bad_source != 0 or bad_text != 0:
        raise RuntimeError(
            "Leakage check FAILED.\n"
            f"Source-group overlaps: {bad_source}\n"
            f"Exact-text overlaps:   {bad_text}"
        )


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tidy split summary by split/model/domain.
    """
    summary = (
        df.groupby(
            ["split", "generator", "model", "domain", "label"],
            dropna=False,
        )
        .agg(
            rows=("id", "count"),
            unique_sources=("split_group", "nunique"),
            mean_chars=("n_chars", "mean")
            if "n_chars" in df.columns
            else ("id", "count"),
            median_chars=("n_chars", "median")
            if "n_chars" in df.columns
            else ("id", "count"),
        )
        .reset_index()
    )

    if "n_chars" in df.columns:
        summary["mean_chars"] = summary["mean_chars"].round(1)
        summary["median_chars"] = summary["median_chars"].round(1)

    return summary


def print_console_summary(
    original: pd.DataFrame,
    clean: pd.DataFrame,
    split_df: pd.DataFrame,
    duplicates: pd.DataFrame,
    leak: pd.DataFrame,
) -> None:

    print("=" * 80)
    print("RAID LEAKAGE-SAFE IID SPLIT PREPARATION")
    print("=" * 80)

    print(f"Original rows              : {len(original):,}")
    print(f"Exact duplicate rows removed: {len(duplicates):,}")
    print(f"Clean rows                 : {len(clean):,}")
    print(
        f"Unique source groups       : "
        f"{split_df['split_group'].nunique():,}"
    )

    print("\nSplit sizes:")
    print(
        split_df["split"]
        .value_counts()
        .reindex(["train", "validation", "test"])
        .to_string()
    )

    print("\nModel distribution:")
    model_table = pd.crosstab(
        split_df["split"],
        split_df["model"],
    ).reindex(["train", "validation", "test"])
    print(model_table.to_string())

    print("\nDomain distribution:")
    domain_table = pd.crosstab(
        split_df["split"],
        split_df["domain"],
    ).reindex(["train", "validation", "test"])
    print(domain_table.to_string())

    print("\nBinary-label distribution:")
    label_table = pd.crosstab(
        split_df["split"],
        split_df["label"],
    ).reindex(["train", "validation", "test"])
    print(label_table.to_string())

    print("\nLeakage checks:")
    print(leak.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Remove exact duplicates and create source_id-aware "
            "70/10/20 IID splits for the balanced RAID subset."
        )
    )

    parser.add_argument(
        "--input",
        default=None,
        help=(
            "Path to raid_balanced.csv. "
            "If omitted, common project paths are detected automatically."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default="raid_experiment_data",
        help=(
            "Directory for cleaned/split files "
            "(default: raid_experiment_data)"
        ),
    )

    args = parser.parse_args()

    input_path = Path(args.input) if args.input else find_default_input()

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading: {input_path}")
    original = pd.read_csv(input_path)

    validate_input(original)

    clean, duplicates = deduplicate_exact_text(original)

    split_df = create_folds(clean)

    leak = leakage_report(split_df)
    validate_no_leakage(leak)

    summary = build_summary(split_df)

    # Preserve a clean file without split-specific helper columns.
    clean_columns_to_drop = ["split_group", "stratum", "fold", "split"]
    clean_export = split_df.drop(
        columns=[
            c for c in clean_columns_to_drop
            if c in split_df.columns
        ]
    )

    train = split_df.loc[split_df["split"] == "train"].copy()
    validation = split_df.loc[
        split_df["split"] == "validation"
    ].copy()
    test = split_df.loc[split_df["split"] == "test"].copy()

    # Output paths
    clean_path = output_dir / "raid_clean.csv"
    train_path = output_dir / "iid_train.csv"
    val_path = output_dir / "iid_validation.csv"
    test_path = output_dir / "iid_test.csv"
    manifest_path = output_dir / "iid_split_manifest.csv"
    duplicate_path = output_dir / "duplicates_removed.csv"
    summary_path = output_dir / "iid_split_summary.csv"
    leakage_path = output_dir / "iid_leakage_check.csv"

    clean_export.to_csv(
        clean_path,
        index=False,
        encoding="utf-8",
        quoting=csv.QUOTE_ALL,
    )

    train.to_csv(
        train_path,
        index=False,
        encoding="utf-8",
        quoting=csv.QUOTE_ALL,
    )

    validation.to_csv(
        val_path,
        index=False,
        encoding="utf-8",
        quoting=csv.QUOTE_ALL,
    )

    test.to_csv(
        test_path,
        index=False,
        encoding="utf-8",
        quoting=csv.QUOTE_ALL,
    )

    manifest_columns = [
        "id",
        "source_id",
        "split_group",
        "text_hash",
        "model",
        "generator",
        "domain",
        "label",
        "fold",
        "split",
    ]

    split_df[manifest_columns].to_csv(
        manifest_path,
        index=False,
        encoding="utf-8",
    )

    duplicates.to_csv(
        duplicate_path,
        index=False,
        encoding="utf-8",
        quoting=csv.QUOTE_ALL,
    )

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8",
    )

    leak.to_csv(
        leakage_path,
        index=False,
        encoding="utf-8",
    )

    print_console_summary(
        original,
        clean,
        split_df,
        duplicates,
        leak,
    )

    print("\n" + "=" * 80)
    print("SUCCESS")
    print("=" * 80)
    print(f"Clean dataset : {clean_path.resolve()}")
    print(f"Train         : {train_path.resolve()}")
    print(f"Validation    : {val_path.resolve()}")
    print(f"Test          : {test_path.resolve()}")
    print(f"Manifest      : {manifest_path.resolve()}")
    print(f"Duplicates    : {duplicate_path.resolve()}")
    print(f"Summary       : {summary_path.resolve()}")
    print(f"Leakage check : {leakage_path.resolve()}")

    print("\nVerified split summary:")
    print(f"  Original rows: {len(original):,}")
    print(f"  Exact duplicate rows removed: {len(duplicates):,}")
    print(f"  Clean rows: {len(clean):,}")
    print("  Approximate split ratio: 70% / 10% / 20%")
    print("  source_id/source-group overlap across splits: 0")
    print("  exact-text overlap across splits: 0")

    print("\nNext research step:")
    print(
        "Use these fixed source-disjoint splits for the RoBERTa IID and "
        "leave-one-generator-out experiments. Evaluate models on "
        "identical per-generator held-out test examples."
    )


if __name__ == "__main__":
    main()
