"""
Split the merged idiom sentences into train, validation and test files.

Reads data/merged_idioms/merged.csv (made by `python processing/check_duplicates.py --merge`),
where every sentence appears once, so no sentence can land in two splits.

Usage:
    python processing/split_dataset.py
    python processing/split_dataset.py --val 0.1 --test 0.1 --seed 67

Output:
    dataset/train.csv
    dataset/val.csv
    dataset/test.csv
"""

import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(SCRIPT_DIR, "data", "merged_idioms", "merged.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "data", "dataset")


def split(df, val_fraction, test_fraction, seed):
    """Cut the rows into train, validation and test."""

    train, rest = train_test_split(
        df, test_size=val_fraction + test_fraction, random_state=seed
    )
    val, test = train_test_split(
        rest,
        test_size=test_fraction / (val_fraction + test_fraction),
        random_state=seed,
    )

    return train, val, test


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", default=INPUT_PATH, help="merged CSV to split")
    parser.add_argument("--output", default=OUTPUT_DIR, help="folder for the split files")
    parser.add_argument("--val", type=float, default=0.1, help="validation fraction")
    parser.add_argument("--test", type=float, default=0.1, help="test fraction")
    parser.add_argument("--seed", type=int, default=67, help="shuffle seed")
    args = parser.parse_args()

    if args.val <= 0 or args.test <= 0 or args.val + args.test >= 1:
        parser.error("--val and --test must be above 0 and add up to less than 1")

    if not os.path.exists(args.input):
        parser.error(f"{args.input} not found. Run: python processing/check_duplicates.py --merge")

    df = pd.read_csv(args.input, encoding="utf-8-sig", dtype=str).fillna("")
    train, val, test = split(df, args.val, args.test, args.seed)

    os.makedirs(args.output, exist_ok=True)

    for name, part in [("train", train), ("val", val), ("test", test)]:
        path = os.path.join(args.output, f"{name}.csv")
        part.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"{os.path.relpath(path, SCRIPT_DIR)}: {len(part)} rows")


if __name__ == "__main__":
    main()
