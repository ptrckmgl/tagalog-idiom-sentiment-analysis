"""
Combine the leksyon.com and Wiktionary idiom lists into one list.

How it works:
    1. Read both lists (made by scrapers/scraper_sitemap.py and
       scrapers/scraper_wiktionary.py).
    2. Split titles that hold two forms ("buwanin / bubuwanin") into two rows.
    3. Treat two titles as the same idiom when they have the same lowercase
       words, using the matcher's word rule (so "balat-sibuyas" = "balat sibuyas").
    4. Keep the leksyon row when an idiom is in both lists, and add the
       Wiktionary meaning next to it.

Usage:
    python processing/combine_idiom_lists.py

Output:
    data/combined_idioms/combined_idioms.csv with the columns
    Title, Part of Speech, Meaning, Source (leksyon / wiktionary / both),
    Wiktionary_Meaning, Wiktionary_POS
"""

import argparse
import os
import re

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEKSYON_PATH = os.path.join(SCRIPT_DIR, "data", "leksyon_idioms", "leksyon_idioms.csv")
WIKTIONARY_PATH = os.path.join(SCRIPT_DIR, "data", "wiktionary_idioms", "wiktionary_idioms.csv")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "data", "combined_idioms", "combined_idioms.csv")

# Same word rule as extractors/extract_idioms.py: runs of letters/digits,
# with notes in parentheses dropped.
WORD = re.compile(r"[^\W_]+")


def idiom_key(title):
    """Return the lowercase words of a title, used to find the same idiom in both lists."""
    title = re.sub(r"\([^)]*\)", " ", title)
    return tuple(w.lower() for w in WORD.findall(title))


def clean_text(series):
    """Drop invisible direction marks and put multi-line meanings on one line."""
    return (series.fillna("").astype(str)
            .str.replace("[‎‏]", "", regex=True)
            .str.replace(r"\s*\r?\n\s*", " | ", regex=True)
            .str.strip())


def load_list(path):
    """Read one idiom list, split ' / ' titles, and add the key column."""

    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    df = df[["Title", "Part of Speech", "Meaning"]].copy()

    for column in df.columns:
        df[column] = clean_text(df[column])

    df["Title"] = df["Title"].str.split(r"\s+/\s+")
    df = df.explode("Title", ignore_index=True)
    df["key"] = df["Title"].map(idiom_key)

    return df[df["key"].map(len) > 0].drop_duplicates("key")


def combine(leksyon, wiktionary):
    """Return the combined list, leksyon rows first."""

    wiki = wiktionary.set_index("key")
    in_both = leksyon["key"].isin(wiki.index)

    leksyon = leksyon.assign(
        Source=in_both.map({True: "both", False: "leksyon"}),
        Wiktionary_Meaning=leksyon["key"].map(wiki["Meaning"]).fillna(""),
        Wiktionary_POS=leksyon["key"].map(wiki["Part of Speech"]).fillna(""),
    )

    extra = wiktionary[~wiktionary["key"].isin(leksyon["key"])]
    extra = extra.assign(
        Source="wiktionary",
        Wiktionary_Meaning=extra["Meaning"],
        Wiktionary_POS=extra["Part of Speech"],
    )

    combined = pd.concat([leksyon, extra], ignore_index=True)
    return combined.drop(columns="key")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--leksyon", default=LEKSYON_PATH)
    parser.add_argument("--wiktionary", default=WIKTIONARY_PATH)
    parser.add_argument("--output", default=OUTPUT_PATH)
    args = parser.parse_args()

    combined = combine(load_list(args.leksyon), load_list(args.wiktionary))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    combined.to_csv(args.output, index=False, encoding="utf-8-sig")

    print(f"{os.path.relpath(args.output, SCRIPT_DIR)}: {len(combined)} idioms")
    print(combined["Source"].value_counts().to_string())


if __name__ == "__main__":
    main()
