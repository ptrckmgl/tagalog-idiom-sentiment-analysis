"""
Add a BIO-tag column to the extracted idiom-sentence CSVs.

For each row (columns: idioms, text) the sentence is tokenized and every
token gets one tag:
    B-IDIOM  first token of the idiom
    I-IDIOM  later tokens of the idiom
    O        everything else
The tokens go in a "tokens" column and the tags in a "bio_tags" column right
after it, both as JSON lists, e.g. ["Matigas","ang","ulo","niya"], with one tag
per token so they line up by position.

Usage:
    python bio_tagging.py
    python bio_tagging.py culturax_idioms news_ph_idioms
"""

import csv
import glob
import json
import os
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIRS = ["culturax_idioms", "news_ph_idioms"]

# A word (keeping inner hyphens/apostrophes, e.g. "abo-aboy", "puno't") or a
# single punctuation mark.
TOKEN = re.compile(r"\w+(?:[-'’]\w+)*|[^\w\s]")


def tokenize(text):
    """Return [(token, start, end)] for the text."""
    return [(m.group(), m.start(), m.end()) for m in TOKEN.finditer(text)]


def idiom_spans(text, idiom):
    """Character spans of whole-word, case-insensitive matches of the idiom."""
    pattern = r"(?<!\w)" + r"\s+".join(map(re.escape, idiom.split())) + r"(?!\w)"
    return [m.span() for m in re.finditer(pattern, text, re.IGNORECASE)]


def bio_tags(text, idiom):
    tokens = tokenize(text)
    tags = ["O"] * len(tokens)

    for span_start, span_end in idiom_spans(text, idiom):
        inside = [i for i, (_, s, e) in enumerate(tokens) if s >= span_start and e <= span_end]

        for n, i in enumerate(inside):
            tags[i] = "B-IDIOM" if n == 0 else "I-IDIOM"

    return tags


def dumps(items):
    """JSON list without spaces, e.g. ["Matigas","ang","ulo","niya"]."""
    return json.dumps(items, ensure_ascii=False, separators=(",", ":"))


def tag_file(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = [c for c in reader.fieldnames if c not in ("tokens", "bio_tags")] + ["tokens", "bio_tags"]
        rows = list(reader)

    untagged = 0

    for row in rows:
        tags = bio_tags(row["text"], row["idioms"])
        row["tokens"] = dumps([token for token, _, _ in tokenize(row["text"])])
        row["bio_tags"] = dumps(tags)

        if "B-IDIOM" not in tags:
            untagged += 1

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows), untagged


def main():
    dirs = sys.argv[1:] or DEFAULT_DIRS

    for d in dirs:
        for path in sorted(glob.glob(os.path.join(SCRIPT_DIR, d, "*.csv"))):
            total, untagged = tag_file(path)
            print(f"{os.path.relpath(path, SCRIPT_DIR)}: {total} rows, {untagged} without a match")


if __name__ == "__main__":
    main()
