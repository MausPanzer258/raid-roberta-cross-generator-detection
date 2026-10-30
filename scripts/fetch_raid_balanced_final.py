#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RAID balanced subset builder — adaptive final version

Goal
----
Create a balanced subset for our AI-text detection experiment.

Preferred design (if the data support it):
    Human        4,000 = 800 per domain
    GPT-4        1,000 = 200 per domain
    ChatGPT      1,000 = 200 per domain
    Llama Chat   1,000 = 200 per domain
    Mistral Chat 1,000 = 200 per domain
    TOTAL        8,000

Domains:
    abstracts, news, reddit, reviews, wiki

Binary balance:
    Human = AI

Why this version is adaptive
----------------------------
The official RAID file may contain fewer rows than our preferred quota in
some model x domain strata. Instead of crashing, this script:

1. scans the official local train_none.csv once;
2. counts every relevant stratum;
3. computes the largest SAFE common quota up to our preferred 8,000 design;
4. preserves:
       - equal domain representation,
       - equal representation across AI generators,
       - exact Human-vs-AI binary balance;
5. writes an availability report before producing the final dataset.

If all strata have enough data, the result is exactly 8,000 rows.
If not, it automatically chooses the largest smaller balanced dataset.

Important:
    Keep source_id. Later train/validation/test splits should be grouped by
    source_id to reduce source-content leakage.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import pandas as pd
    import requests
    from tqdm import tqdm
except ImportError as exc:
    raise SystemExit(
        "\nMissing dependency.\n"
        "Install required packages first:\n\n"
        "    pip install pandas requests tqdm\n\n"
        f"Original import error: {exc}\n"
    )

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OFFICIAL_URL = "https://dataset.raid-bench.xyz/train_none.csv"

DOMAINS = ["abstracts", "news", "reddit", "reviews", "wiki"]

MODELS = [
    "human",
    "gpt4",
    "chatgpt",
    "llama-chat",
    "mistral-chat",
]

AI_MODELS = [
    "gpt4",
    "chatgpt",
    "llama-chat",
    "mistral-chat",
]

DISPLAY_NAMES = {
    "human": "Human",
    "gpt4": "GPT-4",
    "chatgpt": "ChatGPT",
    "llama-chat": "Llama",
    "mistral-chat": "Mistral",
}

# Preferred per-domain quota.
# Binary balance requires Human quota = 4 * each AI-generator quota.
PREFERRED_AI_PER_DOMAIN = 200
PREFERRED_HUMAN_PER_DOMAIN = 800

CHUNK_ROWS = 50_000
DOWNLOAD_BLOCK = 1024 * 1024
DOWNLOAD_RETRIES = 8

CORE_COLUMNS = [
    "id",
    "source_id",
    "model",
    "domain",
    "decoding",
    "repetition_penalty",
    "title",
    "prompt",
    "generation",
]

OPTIONAL_COLUMNS = [
    "adv_source_id",
    "attack",
]

# Previous folders from our earlier attempts. If one of these contains the
# already-downloaded official raw file, the script reuses it automatically.
RAW_FILE_CANDIDATES = [
    Path("raid_10000_robust_v2/raw/train_none.csv"),
    Path("raid_10000_robust/raw/train_none.csv"),
    Path("raid_10000_data/raw/train_none.csv"),
    Path("raid_10000_robust_v2/train_none.csv"),
    Path("raid_10000_robust/train_none.csv"),
    Path("train_none.csv"),
]


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def format_bytes(n: int) -> str:
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{n} B"


def looks_like_csv(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 1024:
        return False

    try:
        with path.open("rb") as f:
            head = f.read(4096)
        text = head.decode("utf-8", errors="ignore").lower()
    except OSError:
        return False

    if "<html" in text or "<!doctype" in text or "<?xml" in text:
        return False

    return all(marker in text for marker in ("generation", "model", "domain"))


def find_existing_raw_file() -> Path | None:
    for candidate in RAW_FILE_CANDIDATES:
        if looks_like_csv(candidate):
            return candidate
    return None


def download_official_csv(url: str, destination: Path, redownload: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")

    if redownload:
        destination.unlink(missing_ok=True)
        partial.unlink(missing_ok=True)

    if looks_like_csv(destination):
        print("Official RAID CSV already exists; reusing it.")
        print(f"File : {destination}")
        print(f"Size : {format_bytes(destination.stat().st_size)}\n")
        return

    if destination.exists():
        destination.unlink(missing_ok=True)

    session = requests.Session()
    session.headers.update({
        "User-Agent": "RAID-balanced-academic-project/1.0",
        "Accept": "text/csv,*/*",
    })

    last_error = None

    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        existing = partial.stat().st_size if partial.exists() else 0
        headers = {}

        if existing > 0:
            headers["Range"] = f"bytes={existing}-"

        try:
            with session.get(
                url,
                headers=headers,
                stream=True,
                timeout=(30, 180),
                allow_redirects=True,
            ) as response:

                # Server ignored the requested resume range: restart cleanly.
                if existing > 0 and response.status_code == 200:
                    partial.unlink(missing_ok=True)
                    existing = 0

                if response.status_code not in (200, 206):
                    response.raise_for_status()

                content_length = int(
                    response.headers.get("Content-Length", "0") or 0
                )
                total = existing + content_length if content_length else None
                mode = "ab" if existing > 0 else "wb"

                print("Downloading RAID official train_none.csv")
                print(f"Source : {url}")
                if existing:
                    print(f"Resume : {format_bytes(existing)} already downloaded")

                with partial.open(mode) as f, tqdm(
                    total=total,
                    initial=existing,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc="download",
                ) as bar:
                    for block in response.iter_content(chunk_size=DOWNLOAD_BLOCK):
                        if block:
                            f.write(block)
                            bar.update(len(block))

            partial.replace(destination)

            if not looks_like_csv(destination):
                destination.unlink(missing_ok=True)
                raise RuntimeError(
                    "Downloaded file does not look like a valid RAID CSV."
                )

            print("\nDownload finished.")
            print(f"File : {destination}")
            print(f"Size : {format_bytes(destination.stat().st_size)}\n")
            return

        except (requests.RequestException, OSError, RuntimeError) as exc:
            last_error = exc

            if attempt >= DOWNLOAD_RETRIES:
                break

            wait = min(2 ** attempt, 60)
            print(
                f"\nDownload problem [{attempt}/{DOWNLOAD_RETRIES}]: {exc}",
                file=sys.stderr,
            )
            print(
                f"Retrying in {wait}s; an existing .part file will be reused.",
                file=sys.stderr,
            )
            time.sleep(wait)

    raise RuntimeError(
        f"Could not download RAID after {DOWNLOAD_RETRIES} attempts. "
        f"Last error: {last_error}"
    )


# ---------------------------------------------------------------------------
# Local scan and sampling
# ---------------------------------------------------------------------------

def inspect_columns(csv_path: Path) -> List[str]:
    header = pd.read_csv(csv_path, nrows=0)
    columns = list(header.columns)

    missing = [c for c in CORE_COLUMNS if c not in columns]
    if missing:
        raise RuntimeError(
            "RAID CSV schema differs from what this script expects.\n"
            f"Missing columns: {missing}\n"
            f"Available columns: {columns}"
        )

    return columns


def keep_first_by_uuid(
    existing: pd.DataFrame | None,
    new_rows: pd.DataFrame,
    max_keep: int,
) -> pd.DataFrame:
    """
    Keep up to max_keep rows with lexicographically smallest string UUIDs.

    UUID4 IDs are random identifiers, so this is deterministic and random-like.
    Uses sort_values(), never nsmallest(), so string/object IDs are supported.
    """
    if new_rows.empty:
        return new_rows if existing is None else existing

    if existing is None or existing.empty:
        combined = new_rows.copy()
    else:
        combined = pd.concat([existing, new_rows], ignore_index=True)

    combined = combined.drop_duplicates(subset=["id"], keep="first")
    combined["_id_key"] = combined["id"].astype("string").fillna("")

    combined = (
        combined.sort_values(
            "_id_key",
            ascending=True,
            kind="mergesort",
        )
        .head(max_keep)
        .drop(columns=["_id_key"])
        .reset_index(drop=True)
    )

    return combined


def scan_csv(csv_path: Path) -> Tuple[
    Dict[Tuple[str, str], pd.DataFrame],
    Dict[Tuple[str, str], int],
]:
    """
    Scan the full local CSV once.

    We retain only enough rows for the preferred design:
      human: up to 800 per domain
      each AI model: up to 200 per domain

    We also count all matching rows so we can compute a safe adaptive quota.
    """
    all_columns = inspect_columns(csv_path)
    usecols = CORE_COLUMNS + [c for c in OPTIONAL_COLUMNS if c in all_columns]

    reservoirs: Dict[Tuple[str, str], pd.DataFrame | None] = {}
    availability: Dict[Tuple[str, str], int] = {}

    for model in MODELS:
        for domain in DOMAINS:
            reservoirs[(model, domain)] = None
            availability[(model, domain)] = 0

    print("=" * 80)
    print("Scanning official RAID train_none.csv locally")
    print("=" * 80)
    print(f"Raw file   : {csv_path}")
    print(f"Chunk size : {CHUNK_ROWS:,} rows")
    print("No Hugging Face Dataset Viewer /filter API is used.\n")

    reader = pd.read_csv(
        csv_path,
        usecols=usecols,
        chunksize=CHUNK_ROWS,
        low_memory=False,
    )

    rows_seen = 0

    for chunk in tqdm(reader, desc="scan", unit="chunk"):
        rows_seen += len(chunk)

        model_s = chunk["model"].astype("string").fillna("")
        domain_s = chunk["domain"].astype("string").fillna("")
        decoding_s = chunk["decoding"].astype("string").fillna("")
        rep_s = chunk["repetition_penalty"].astype("string").fillna("")

        for model in MODELS:
            max_keep = (
                PREFERRED_HUMAN_PER_DOMAIN
                if model == "human"
                else PREFERRED_AI_PER_DOMAIN
            )

            for domain in DOMAINS:
                mask = (
                    (model_s == model)
                    & (domain_s == domain)
                )

                if model != "human":
                    mask &= (
                        (decoding_s == "sampling")
                        & (rep_s == "no")
                    )

                sub = chunk.loc[mask].copy()

                sub = sub[
                    sub["id"].notna()
                    & sub["generation"].notna()
                    & (sub["generation"].astype(str).str.strip() != "")
                ]

                availability[(model, domain)] += len(sub)

                reservoirs[(model, domain)] = keep_first_by_uuid(
                    reservoirs[(model, domain)],
                    sub,
                    max_keep,
                )

    print(f"\nRows scanned: {rows_seen:,}\n")

    final_reservoirs: Dict[Tuple[str, str], pd.DataFrame] = {}
    for key, value in reservoirs.items():
        if value is None:
            final_reservoirs[key] = pd.DataFrame()
        else:
            final_reservoirs[key] = value

    return final_reservoirs, availability


def calculate_safe_quota(
    availability: Dict[Tuple[str, str], int]
) -> Tuple[int, int]:
    """
    Exact binary balance requires:
        human_per_domain = 4 * ai_per_generator_per_domain

    Choose the largest common AI quota <= 200 supported by ALL strata.
    """
    min_human_available = min(
        availability[("human", domain)]
        for domain in DOMAINS
    )

    min_ai_available = min(
        availability[(model, domain)]
        for model in AI_MODELS
        for domain in DOMAINS
    )

    safe_ai = min(
        PREFERRED_AI_PER_DOMAIN,
        min_ai_available,
        min_human_available // len(AI_MODELS),
    )

    if safe_ai < 1:
        raise RuntimeError(
            "No nonzero common balanced quota can be constructed from "
            "the selected RAID strata."
        )

    safe_human = len(AI_MODELS) * safe_ai

    return safe_human, safe_ai


def build_availability_report(
    availability: Dict[Tuple[str, str], int],
    human_quota: int,
    ai_quota: int,
) -> pd.DataFrame:

    rows = []

    for model in MODELS:
        target = human_quota if model == "human" else ai_quota

        for domain in DOMAINS:
            rows.append({
                "generator": DISPLAY_NAMES[model],
                "model": model,
                "domain": domain,
                "available_rows": availability[(model, domain)],
                "selected_rows": target,
            })

    return pd.DataFrame(rows)


def assemble_final_dataset(
    reservoirs: Dict[Tuple[str, str], pd.DataFrame],
    human_quota: int,
    ai_quota: int,
) -> pd.DataFrame:

    parts = []

    for model in MODELS:
        quota = human_quota if model == "human" else ai_quota

        for domain in DOMAINS:
            selected = reservoirs[(model, domain)].head(quota).copy()

            if len(selected) < quota:
                raise RuntimeError(
                    f"Internal sampling shortfall for {model}/{domain}: "
                    f"needed {quota}, retained {len(selected)}."
                )

            selected["text"] = selected["generation"]
            selected["label"] = 0 if model == "human" else 1
            selected["generator"] = DISPLAY_NAMES[model]
            selected["n_chars"] = selected["generation"].astype(str).str.len()
            selected["n_words_whitespace"] = (
                selected["generation"].astype(str).str.split().str.len()
            )

            parts.append(selected)

    final_df = pd.concat(parts, ignore_index=True)

    # Shuffle output row order only; this is NOT our ML split.
    final_df = final_df.sample(
        frac=1.0,
        random_state=42,
    ).reset_index(drop=True)

    return final_df


# ---------------------------------------------------------------------------
# Validation and summaries
# ---------------------------------------------------------------------------

def validate_final(
    df: pd.DataFrame,
    human_quota: int,
    ai_quota: int,
) -> None:

    expected_human = human_quota * len(DOMAINS)
    expected_ai = ai_quota * len(DOMAINS) * len(AI_MODELS)
    expected_total = expected_human + expected_ai

    if len(df) != expected_total:
        raise AssertionError(
            f"Expected {expected_total:,} rows, got {len(df):,}."
        )

    duplicate_ids = int(df["id"].duplicated().sum())
    if duplicate_ids:
        raise AssertionError(
            f"Found {duplicate_ids} duplicated RAID row IDs."
        )

    if df["text"].isna().any():
        raise AssertionError("Final dataset contains missing text.")

    if df["text"].astype(str).str.strip().eq("").any():
        raise AssertionError("Final dataset contains empty text.")

    for model in MODELS:
        quota = human_quota if model == "human" else ai_quota

        for domain in DOMAINS:
            actual = int(
                (
                    (df["model"] == model)
                    & (df["domain"] == domain)
                ).sum()
            )

            if actual != quota:
                raise AssertionError(
                    f"{model}/{domain}: expected {quota}, got {actual}."
                )

    human_n = int((df["label"] == 0).sum())
    ai_n = int((df["label"] == 1).sum())

    if human_n != ai_n:
        raise AssertionError(
            f"Binary classes are not balanced: Human={human_n}, AI={ai_n}."
        )


def make_summary(df: pd.DataFrame) -> pd.DataFrame:

    print("\nFinal model x domain distribution:\n")

    pivot = pd.pivot_table(
        df,
        index=["generator", "model"],
        columns="domain",
        values="id",
        aggfunc="count",
        fill_value=0,
        margins=True,
        margins_name="TOTAL",
    )
    print(pivot.to_string())

    print("\nBinary label balance:")
    print(
        df["label"]
        .map({0: "Human", 1: "AI"})
        .value_counts()
        .to_string()
    )

    print("\nText-length summary (characters):")
    lengths = (
        df.groupby(["generator", "model"])["n_chars"]
        .agg(["count", "mean", "median", "min", "max"])
        .round(1)
    )
    print(lengths.to_string())

    return (
        df.groupby(["generator", "model", "domain"])
        .agg(
            n=("id", "count"),
            mean_chars=("n_chars", "mean"),
            median_chars=("n_chars", "median"),
            mean_words=("n_words_whitespace", "mean"),
            median_words=("n_words_whitespace", "median"),
        )
        .reset_index()
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build the largest safe balanced RAID subset up to the preferred "
            "8,000-row design."
        )
    )

    parser.add_argument(
        "--output-dir",
        default="raid_balanced_final",
        help="Output directory (default: raid_balanced_final)",
    )

    parser.add_argument(
        "--raw-csv",
        default=None,
        help=(
            "Optional explicit path to an already-downloaded train_none.csv. "
            "If omitted, the script first searches previous project folders."
        ),
    )

    parser.add_argument(
        "--redownload",
        action="store_true",
        help="Force a fresh download of train_none.csv.",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve raw CSV.
    if args.raw_csv:
        raw_csv = Path(args.raw_csv)

        if not looks_like_csv(raw_csv):
            raise RuntimeError(
                f"The supplied raw CSV is missing or invalid: {raw_csv}"
            )

        print(f"Using explicitly supplied RAID raw file:\n{raw_csv}\n")

    elif not args.redownload:
        existing = find_existing_raw_file()

        if existing is not None:
            raw_csv = existing
            print("Found an existing official RAID raw file automatically:")
            print(f"{raw_csv}")
            print(f"Size: {format_bytes(raw_csv.stat().st_size)}\n")
        else:
            raw_csv = output_dir / "raw" / "train_none.csv"
            download_official_csv(
                OFFICIAL_URL,
                raw_csv,
                redownload=False,
            )

    else:
        raw_csv = output_dir / "raw" / "train_none.csv"
        download_official_csv(
            OFFICIAL_URL,
            raw_csv,
            redownload=True,
        )

    # One full local scan.
    reservoirs, availability = scan_csv(raw_csv)

    # Derive safe balanced quota automatically.
    human_quota, ai_quota = calculate_safe_quota(availability)

    expected_total = (
        human_quota * len(DOMAINS)
        + ai_quota * len(DOMAINS) * len(AI_MODELS)
    )

    print("=" * 80)
    print("ADAPTIVE QUOTA")
    print("=" * 80)
    print(f"Human per domain              : {human_quota:,}")
    print(f"Each AI generator per domain : {ai_quota:,}")
    print(f"Human total                   : {human_quota * len(DOMAINS):,}")
    print(
        f"AI total                      : "
        f"{ai_quota * len(DOMAINS) * len(AI_MODELS):,}"
    )
    print(f"Final total                    : {expected_total:,}")

    if expected_total == 8000:
        print("Result: preferred 8,000-row design is fully supported.\n")
    else:
        print(
            "Result: at least one stratum is smaller than the preferred quota.\n"
            "The script automatically reduced ALL quotas proportionally so "
            "the dataset remains perfectly balanced.\n"
        )

    availability_df = build_availability_report(
        availability,
        human_quota,
        ai_quota,
    )

    availability_path = output_dir / "raid_availability.csv"
    availability_df.to_csv(
        availability_path,
        index=False,
        encoding="utf-8",
    )

    # Assemble and validate final data.
    final_df = assemble_final_dataset(
        reservoirs,
        human_quota,
        ai_quota,
    )

    validate_final(
        final_df,
        human_quota,
        ai_quota,
    )

    summary_df = make_summary(final_df)

    csv_path = output_dir / "raid_balanced.csv"
    jsonl_path = output_dir / "raid_balanced.jsonl"
    summary_path = output_dir / "raid_balanced_summary.csv"

    final_df.to_csv(
        csv_path,
        index=False,
        encoding="utf-8",
        quoting=csv.QUOTE_ALL,
    )

    final_df.to_json(
        jsonl_path,
        orient="records",
        lines=True,
        force_ascii=False,
    )

    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8",
    )

    print("\n" + "=" * 80)
    print("SUCCESS")
    print("=" * 80)
    print(f"Raw RAID CSV : {raw_csv.resolve()}")
    print(f"Dataset CSV  : {csv_path.resolve()}")
    print(f"Dataset JSONL: {jsonl_path.resolve()}")
    print(f"Summary      : {summary_path.resolve()}")
    print(f"Availability : {availability_path.resolve()}")
    print(f"Rows         : {len(final_df):,}")
    print(f"Human        : {(final_df['label'] == 0).sum():,}")
    print(f"AI           : {(final_df['label'] == 1).sum():,}")

    print("\nIMPORTANT NEXT STEP")
    print("-------------------")
    print(
        "Do not create a naive row-wise train/test split. "
        "Run prepare_raid_splits_v2.py next to generate source-aware "
        "train/validation/test splits."
    )


if __name__ == "__main__":
    main()
