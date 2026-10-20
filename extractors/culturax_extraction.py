"""
Find Filipino idioms in a dataset (streamed from Hugging Face or a local
copy of NewsPH) and save only the SENTENCES that contain them.

How it works:
    1. Load the idioms from every CSV in the idioms_list folder.
    2. Build one Aho-Corasick automaton from all of them, so the dataset
       is only read once for all letters.
    3. Split each row into sentences and keep only the sentences with a
       whole-word match (so "apa" is not found inside "mapayapa").
    4. Save one results file per idiom CSV, e.g. idioms_list/a_idioms.csv
       -> extracted_idioms/a_extracted_idioms.csv

Usage (all arguments are optional; defaults are in the SETTINGS section):
    python extractors/culturax_extraction.py
    python extractors/culturax_extraction.py --idioms idioms_list/a_idioms.csv
    python extractors/culturax_extraction.py --output-dir results --name "{name}_matches.json"
    python extractors/culturax_extraction.py --no-stream      # use the local NewsPH copy
"""

import argparse
import csv
import glob
import json
import os
import re

import ahocorasick
from datasets import load_dataset


# ============================================================
# SETTINGS
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IDIOMS_PATH = os.path.join(SCRIPT_DIR, "idioms_list")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "culturax_idioms")
OUTPUT_NAME = "{name}_culturax_idioms.csv"

# Local copy of the NewsPH dataset (used only when not streaming).
DATASET_PATH = r"C:\Users\patri\Downloads\newsph\newsph"

# True: stream from the Hugging Face Hub, nothing is downloaded in full.
# CulturaX is gated: accept its terms on the dataset page and run
# `huggingface-cli login` first.
STREAM = True
HF_DATASET = "uonlp/CulturaX"
HF_CONFIG = "tl"  # Tagalog subset

# Print a progress line every this many rows.
PROGRESS_EVERY = 100_000

# A sentence ends at . ! ? followed by whitespace, or at a line break.
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


# ============================================================
# 1. Load idioms
# ============================================================

def load_idioms(csv_path, column="Title"):
    """
    Read the idioms from one column of a CSV file.

    Blank cells are skipped, surrounding spaces are trimmed, and
    duplicates are removed (keeping the first occurrence's order).
    """

    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        idioms = (row[column].strip() for row in csv.DictReader(f))
        return list(dict.fromkeys(idiom for idiom in idioms if idiom))


def load_idiom_files(path):
    """
    Load the idioms from a single CSV or from every CSV in a folder.

    Returns a dict that maps each file's short name to its idioms,
    e.g. {"a": [...], "b": [...]} for a_idioms.csv and b_idioms.csv.
    """

    if os.path.isdir(path):
        csv_paths = sorted(glob.glob(os.path.join(path, "*.csv")))
    else:
        csv_paths = [path]

    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {path}")

    idiom_files = {}

    for csv_path in csv_paths:
        name = os.path.splitext(os.path.basename(csv_path))[0]
        name = name.removesuffix("_idioms")
        idiom_files[name] = load_idioms(csv_path)

    return idiom_files


# ============================================================
# 2. Build the search automaton
# ============================================================

def build_automaton(idioms):
    """
    Build an Aho-Corasick automaton that finds all the given idioms.

    The automaton searches lowercased text, so matching ignores case.
    Each idiom is stored with its length, so a match's start position
    can be worked out from the end position the automaton reports.
    """

    automaton = ahocorasick.Automaton()

    for idiom in idioms:
        idiom_lower = idiom.lower()
        automaton.add_word(idiom_lower, (len(idiom_lower), idiom))

    automaton.make_automaton()
    return automaton


# ============================================================
# 3. Search the text
# ============================================================

def is_whole_word(text, start, end):
    """
    Return True if text[start:end] is not part of a larger word,
    meaning the characters on either side are not letters or digits.
    """

    if start > 0 and text[start - 1].isalnum():
        return False

    if end < len(text) and text[end].isalnum():
        return False

    return True


def find_idioms(text, automaton):
    """
    Return the idioms found in the text, in order of first appearance.
    Each idiom is listed once, even if it appears several times.
    """

    text_lower = text.lower()
    found = {}

    for end, (length, idiom) in automaton.iter(text_lower):
        start = end - length + 1

        if is_whole_word(text_lower, start, end + 1):
            found[idiom] = None

    return list(found)


def split_sentences(text):
    """Split a document into sentences (simple punctuation-based rule)."""
    return [s.strip() for s in SENTENCE_SPLIT.split(text) if s.strip()]


def scan_dataset(dataset, automaton):
    """
    Search every row of a dataset (streamed or not), sentence by sentence,
    and return one entry per sentence that contains at least one idiom,
    as {"text": sentence, "idioms": [...]}.
    """

    matches = []
    i = 0

    for i, row in enumerate(dataset, start=1):
        for sentence in split_sentences(row["text"]):
            found = find_idioms(sentence, automaton)

            if found:
                matches.append({"text": sentence, "idioms": found})

        if i % PROGRESS_EVERY == 0:
            print(f"  {i:,} rows scanned, {len(matches):,} sentences matched")

    print(f"  {i:,} rows scanned in total")
    return matches


# ============================================================
# 4. Split and save the results
# ============================================================

def split_by_file(matches, idiom_files):
    """
    Split the matched sentences by the idiom file each idiom came from.

    A sentence that contains idioms from several files goes into each of
    those files, listing only that file's idioms.
    """

    lookup = {name: set(idioms) for name, idioms in idiom_files.items()}
    results = {name: [] for name in idiom_files}

    for row in matches:
        for name, idiom_set in lookup.items():
            own_idioms = [i for i in row["idioms"] if i in idiom_set]

            if own_idioms:
                results[name].append({"idioms": own_idioms, "text": row["text"]})

    return results


def save_results(rows, output_path):
    """
    Save rows to a CSV or JSON file, chosen by the extension.

    A file is written even when there are no rows, so every idiom
    file has a matching results file.
    """

    extension = os.path.splitext(output_path)[1].lower()

    if extension == ".csv":
        with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["idioms", "text"])

            for row in rows:
                writer.writerow(["; ".join(row["idioms"]), row["text"]])

    elif extension == ".json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)

    else:
        raise ValueError(
            f"Unsupported output format '{extension}'. Use .csv or .json."
        )


# ============================================================
# 5. Command-line arguments
# ============================================================

def parse_args():
    """Read the optional command-line arguments, falling back to SETTINGS."""

    parser = argparse.ArgumentParser(
        description="Find idioms in a dataset and save the matching sentences."
    )
    parser.add_argument(
        "--idioms",
        default=IDIOMS_PATH,
        help="folder of idiom CSVs, or a single idiom CSV",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help="folder to save the results in",
    )
    parser.add_argument(
        "--name",
        default=OUTPUT_NAME,
        help='results file name; "{name}" becomes the idiom file\'s name '
             "and the extension (.csv or .json) picks the format",
    )
    parser.add_argument(
        "--dataset",
        default=DATASET_PATH,
        help="path to the local NewsPH dataset (used with --no-stream)",
    )
    parser.add_argument(
        "--stream",
        action=argparse.BooleanOptionalAction,
        default=STREAM,
        help="stream HF_DATASET from the Hub, or use --no-stream for the "
             "local dataset",
    )
    args = parser.parse_args()

    if "{name}" not in args.name:
        parser.error('--name must contain "{name}", or every file would '
                     "overwrite the previous one")

    if os.path.splitext(args.name)[1].lower() not in (".csv", ".json"):
        parser.error("--name must end in .csv or .json")

    return args


# ============================================================
# 6. Main program
# ============================================================

def main():
    args = parse_args()

    idiom_files = load_idiom_files(args.idioms)
    all_idioms = list(dict.fromkeys(
        idiom for idioms in idiom_files.values() for idiom in idioms
    ))
    automaton = build_automaton(all_idioms)
    print(f"Loaded {len(all_idioms)} idioms from {len(idiom_files)} file(s)")

    if args.stream:
        print(f"Streaming {HF_DATASET} ({HF_CONFIG})...")
        dataset = load_dataset(HF_DATASET, HF_CONFIG, split="train", streaming=True)
    else:
        dataset = load_dataset(args.dataset)["train"]
        print(f"Searching {len(dataset):,} rows...")

    matches = scan_dataset(dataset, automaton)
    print(f"Found {len(matches):,} sentences containing an idiom\n")

    os.makedirs(args.output_dir, exist_ok=True)

    for name, rows in split_by_file(matches, idiom_files).items():
        output_path = os.path.join(args.output_dir, args.name.format(name=name))
        save_results(rows, output_path)
        print(f"{len(rows):>6} sentences -> {output_path}")


if __name__ == "__main__":
    main()