"""
Estimate how precise each match_type is, from a small hand-checked sample.

Step 1: draw a random sample of matches per match_type into a review file.
    .venv/Scripts/python.exe extractors/sample_match_types.py sample ^
        --input extracted_idioms/newsph_matches.csv extracted_idioms/culturax_matches.csv ^
        --per-type 30 --output review.csv

Step 2: open review.csv in Excel and fill the is_real_match column:
    1 = the matched_text really is this idiom's words used as the idiom
        (figurative OR literal both count: this checks the MATCH, not the meaning)
    0 = false match (a different phrase, a wrong stem, an unrelated word)
    Leave it blank if you cannot tell. Save as CSV (UTF-8).

Step 3: get the precision of each match_type.
    .venv/Scripts/python.exe extractors/sample_match_types.py score --input review.csv
"""

import argparse
import csv
import math
import random
from collections import defaultdict


def read_rows(paths):
    rows = []

    for path in paths:
        with open(path, encoding="utf-8-sig", newline="") as f:
            rows.extend(csv.DictReader(f))

    return rows


def sample(args):
    rows = read_rows(args.input)
    by_type = defaultdict(list)

    for row in rows:
        by_type[row["match_type"]].append(row)

    rng = random.Random(args.seed)
    chosen = []

    for name, group in by_type.items():
        picked = rng.sample(group, min(args.per_type, len(group)))
        print(f"{name:<8} {len(group):>8,} matches, sampled {len(picked)}")
        chosen.extend(picked)

    fields = ["match_type", "idiom", "matched_text", "text", "is_real_match", "source", "doc_id"]

    with open(args.output, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()

        for row in chosen:
            writer.writerow({**row, "is_real_match": ""})

    print(f"Saved {len(chosen)} rows to {args.output}. Fill in is_real_match (1 / 0).")


def wilson(correct, total, z=1.96):
    """95% Wilson score interval for a proportion."""

    if total == 0:
        return 0.0, 0.0

    p = correct / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return centre - half, centre + half


def score(args):
    counts = defaultdict(lambda: [0, 0])  # match_type -> [correct, labeled]
    skipped = 0

    for row in read_rows([args.input]):
        label = row["is_real_match"].strip()

        if label not in ("0", "1"):
            skipped += 1
            continue

        counts[row["match_type"]][1] += 1
        counts[row["match_type"]][0] += int(label)

    print(f"{'match_type':<10}{'labeled':>8}{'correct':>9}{'precision':>11}   95% interval")

    for name, (correct, total) in counts.items():
        low, high = wilson(correct, total)
        print(f"{name:<10}{total:>8}{correct:>9}{correct / total:>11.2f}   {low:.2f} to {high:.2f}")

    print(f"\n{skipped} rows had no 0/1 label and were left out.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    sample_parser = commands.add_parser("sample", help="draw a review sample")
    sample_parser.add_argument("--input", nargs="+", required=True)
    sample_parser.add_argument("--per-type", type=int, default=30)
    sample_parser.add_argument("--seed", type=int, default=1)
    sample_parser.add_argument("--output", default="review.csv")
    sample_parser.set_defaults(run=sample)

    score_parser = commands.add_parser("score", help="precision per match_type from a filled review file")
    score_parser.add_argument("--input", required=True)
    score_parser.set_defaults(run=score)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
