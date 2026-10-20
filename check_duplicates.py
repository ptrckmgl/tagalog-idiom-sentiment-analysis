"""
Check the extracted idiom CSVs for duplicate rows.

Each extraction CSV has the columns "idioms" and "text". Rows are
duplicates when they have the same idioms AND the same text, within one
file or across files.

Usage:
    python check_duplicates.py
    python check_duplicates.py culturax_idioms news_ph_idioms
    python check_duplicates.py --remove     # rewrite files without exact duplicates
    python check_duplicates.py --merge      # one row per sentence -> merged_idioms/merged.csv
"""

import argparse
import glob
import os
import sys

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIRS = ["culturax_idioms", "news_ph_idioms"]
MERGED_PATH = os.path.join(SCRIPT_DIR, "merged_idioms", "merged.csv")


def load_dir(directory):
    """Load every CSV in a directory into one DataFrame with a "file" column."""

    frames = []

    for path in sorted(glob.glob(os.path.join(SCRIPT_DIR, directory, "*.csv"))):
        df = pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")
        df["file"] = os.path.basename(path)
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["idioms", "text", "file"])

    return pd.concat(frames, ignore_index=True)


def normalize(series):
    """Lowercase and collapse whitespace so trivial differences still match."""

    return series.str.lower().str.split().str.join(" ")


def report(directory, remove):
    df = load_dir(directory)
    print(f"\n=== {directory}: {len(df)} rows in {df['file'].nunique()} files ===")

    if df.empty:
        return

    df["_key"] = normalize(df["idioms"]) + "\t" + normalize(df["text"])

    exact = df.duplicated(subset="_key", keep=False)
    print(f"Exact duplicates (same idioms + text): {df.duplicated(subset='_key').sum()} extra rows")

    for key, group in df[exact].groupby("_key"):
        print(f"  [{group['idioms'].iloc[0]}] x{len(group)} in {sorted(set(group['file']))}")
        print(f"    {group['text'].iloc[0][:100]}")

    if remove:
        # Only drop duplicates inside the same file, so each idiom file
        # keeps its own copy of a sentence that was split across files.
        removed = 0

        for name, group in df.groupby("file"):
            clean = group.drop_duplicates(subset="_key", keep="first")
            removed += len(group) - len(clean)

            if len(clean) != len(group):
                path = os.path.join(SCRIPT_DIR, directory, name)
                clean[["idioms", "text"]].to_csv(path, index=False, encoding="utf-8-sig")

        print(f"Removed {removed} duplicate rows within files")


def merge(directories, output_path):
    """
    Merge all rows of the given folders into one row per sentence.

    Rows with the same (normalized) text are combined: their idioms are
    joined into one sorted, de-duplicated list. The original CSVs are not
    changed.
    """

    df = pd.concat([load_dir(d) for d in directories], ignore_index=True)
    df["_text"] = normalize(df["text"])

    # One entry per idiom, so "a; b" and "a" merge into "a; b"
    df["idiom_list"] = df["idioms"].str.split(";").apply(
        lambda parts: [p.strip() for p in parts if p.strip()]
    )

    merged = df.groupby("_text", sort=False).agg(
        idioms=("idiom_list", lambda lists: "; ".join(sorted({i for l in lists for i in l}))),
        text=("text", "first"),
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\nMerged {len(df)} rows into {len(merged)} unique sentences")
    print(f"Saved to {os.path.relpath(output_path, SCRIPT_DIR)}")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dirs", nargs="*", default=DEFAULT_DIRS,
                        help="folders of extracted idiom CSVs")
    parser.add_argument("--remove", action="store_true",
                        help="rewrite the CSVs without exact duplicates within each file")
    parser.add_argument("--merge", action="store_true",
                        help="merge all rows into one row per sentence (idioms combined) "
                             "and save to merged_idioms/merged.csv")
    args = parser.parse_args()

    for directory in args.dirs:
        report(directory, args.remove)

    if args.merge:
        merge(args.dirs, MERGED_PATH)


if __name__ == "__main__":
    main()
