"""
Find Filipino idioms in the NewsPH dataset and save the results.

How it works:
    1. Load the idioms from every CSV in the idioms_list folder
       (the files scraper.py exports, one per letter).
    2. Build one Aho-Corasick automaton from all of them. It finds every
       idiom in a text in a single pass, no matter how many there are,
       so the dataset is only read once for all letters.
    3. Scan every row of NewsPH, keeping only matches that are whole words
       (so "apa" is not found inside "mapayapa").
    4. Save one results file per idiom CSV, e.g. idioms_list/a_idioms.csv
       -> extracted_idioms/a_extracted_idioms.csv

Usage (all arguments are optional; defaults are in the SETTINGS section):
    python extractors/news_ph_extraction.py
    python extractors/news_ph_extraction.py --idioms idioms_list/a_idioms.csv
    python extractors/news_ph_extraction.py --output-dir results --name "{name}_matches.json"
"""

import argparse
import csv
import glob
import json
import os

import ahocorasick
from datasets import load_dataset


# ============================================================
# SETTINGS
# Change these defaults, or override them from the command line.
# ============================================================

# Project root (the folder above extractors/). Default paths are built
# from it so the script works no matter which folder you run it from.
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Idioms to search for: a folder (every .csv inside it is used) or a
# single CSV file. Each CSV needs a "Title" column with one idiom per row.
IDIOMS_PATH = os.path.join(SCRIPT_DIR, "idioms_list")

# Folder the results are saved in. It is created if it does not exist.
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "news_ph_idioms")

# Name of each results file. "{name}" is replaced by the idiom file's
# name without "_idioms.csv", so "a_idioms.csv" gives "a". The extension
# picks the format: ".csv" or ".json".
OUTPUT_NAME = "{name}_news_ph_idioms.csv"

# Local copy of the NewsPH dataset.
DATASET_PATH = r"C:\Users\patri\Downloads\newsph\newsph"

# Rows handed to the search function at a time. Bigger batches mean
# less overhead.
BATCH_SIZE = 10_000

# Number of worker processes. None means a single process, which
# benchmarked fastest on Windows: starting workers costs more than the
# search itself. Try 4 if the search gets noticeably slower.
NUM_PROC = None


# ============================================================
# 1. Load idioms
# ============================================================

def load_idioms(csv_path, column="Title"):
    """
    Read the idioms from one column of a CSV file.

    Blank cells are skipped, surrounding spaces are trimmed, and
    duplicates are removed (keeping the first occurrence's order).
    """

    # "utf-8-sig" also reads files saved with a byte-order mark,
    # which Excel adds when it saves a CSV.
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        idioms = (row[column].strip() for row in csv.DictReader(f))

        # dict.fromkeys drops duplicates while keeping the order
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
        # "a_idioms.csv" -> "a"
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

    Examples, searching for "apa":
        "apa"      -> True
        "apa."     -> True
        "mapayapa" -> False
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

    # A dict keeps insertion order and skips duplicates instantly
    found = {}

    # automaton.iter yields (end index, value) for every match,
    # including matches inside longer words, which are filtered out here.
    for end, (length, idiom) in automaton.iter(text_lower):
        start = end - length + 1

        if is_whole_word(text_lower, start, end + 1):
            found[idiom] = None

    return list(found)


def extract_batch(batch, automaton):
    """
    Search a batch of dataset rows and keep only the rows that
    contain at least one idiom.

    Called by datasets.map, which passes a batch as a dict of columns
    and expects a dict of columns back. The returned batch can be
    shorter than the input, which is how unmatched rows get dropped.
    """

    texts = []
    idioms = []

    for text in batch["text"]:
        found = find_idioms(text, automaton)

        if found:
            texts.append(text)
            idioms.append(found)

    return {"text": texts, "idioms": idioms}


# ============================================================
# 4. Split and save the results
# ============================================================

def split_by_file(matches, idiom_files):
    """
    Split the matched rows by the idiom file each idiom came from.

    Returns {"a": [rows...], "b": [rows...], ...}. A row that contains
    idioms from several files goes into each of those files, listing
    only that file's idioms.
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

    CSV:  columns "idioms" and "text". When a row has several idioms,
          they are joined with "; ".
    JSON: a list of objects like {"idioms": [...], "text": "..."}.

    A file is written even when there are no rows, so every idiom
    file has a matching results file.
    """

    extension = os.path.splitext(output_path)[1].lower()

    if extension == ".csv":
        # "utf-8-sig" makes Excel display characters such as "ñ" correctly
        with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["idioms", "text"])

            for row in rows:
                writer.writerow(["; ".join(row["idioms"]), row["text"]])

    elif extension == ".json":
        # ensure_ascii=False keeps Filipino characters readable
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
        description="Find idioms in the NewsPH dataset and save the matches."
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
        help="path to the NewsPH dataset",
    )
    args = parser.parse_args()

    # Check the name now, so a mistake does not surface only after the
    # whole dataset has been searched.
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

    # Load every idiom file, then search for all of their idioms at once
    idiom_files = load_idiom_files(args.idioms)
    all_idioms = list(dict.fromkeys(
        idiom for idioms in idiom_files.values() for idiom in idioms
    ))
    automaton = build_automaton(all_idioms)
    print(f"Loaded {len(all_idioms)} idioms from {len(idiom_files)} file(s)")

    dataset = load_dataset(args.dataset)["train"]
    print(f"Searching {len(dataset)} rows...")

    # The automaton is passed through fn_kwargs rather than read from a
    # global, so it also reaches worker processes if NUM_PROC is set.
    matches = dataset.map(
        extract_batch,
        batched=True,
        batch_size=BATCH_SIZE,
        num_proc=NUM_PROC,
        remove_columns=dataset.column_names,
        fn_kwargs={"automaton": automaton},
    )
    print(f"Found {len(matches)} rows containing an idiom\n")

    # Write one results file per idiom file
    os.makedirs(args.output_dir, exist_ok=True)

    for name, rows in split_by_file(matches, idiom_files).items():
        output_path = os.path.join(args.output_dir, args.name.format(name=name))
        save_results(rows, output_path)
        print(f"{len(rows):>6} rows -> {output_path}")


if __name__ == "__main__":
    main()
