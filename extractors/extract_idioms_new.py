"""
Find Filipino idioms in a dataset (CulturaX streamed from Hugging Face, or a
local copy of NewsPH) and save every SENTENCE that contains one, together with
HOW it matched.

This replaces extractors/culturax_extraction.py and extractors/news_ph_extraction.py.
Instead of one exact Aho-Corasick pass, it tries four levels in order and
records the first level that works for each idiom in each sentence:

    exact       the idiom's words appear in order, next to each other
                (case, punctuation and hyphen-vs-space are ignored, so
                "Balat-sibuyas" matches "balat sibuyas")
    linker      same, but the linkers na / ng / -ng may differ or be missing
                ("kaibigang tapat" ~ "kaibigan na tapat"; "abo ang laman ng ulo"
                ~ "abo ang laman ulo"). "ang" is NOT treated as a linker.
    gap         same, plus up to MAX_GAP inserted filler words between the
                idiom's words ("nagbibilang pa ng poste")
    root        same, plus words are compared by root form (needs --stem and
                the tglstemmer package), e.g. "bumigat ang kamay" ~ "mabigat ang kamay"

Idioms made of one word are only ever matched at the "exact" level.
When two matches overlap in a sentence, the longer one is kept.

Output: one row per (sentence, idiom) with the columns
    source, doc_id, idiom, match_type, matched_text, n_words, text
(utf-8-sig CSV). One row per match, not one row per sentence, so the matched
idiom can be paired with its sentence for Model 1 / Model 2.

Usage (run from the project root):
    .venv/Scripts/python.exe extractors/extract_idioms.py --source newsph
    .venv/Scripts/python.exe extractors/extract_idioms.py --source culturax --limit-rows 20000
    .venv/Scripts/python.exe extractors/extract_idioms.py --source newsph --stem
Start with --limit-rows to look at the results before a full run.
"""

import argparse
import csv
import glob
import os
import re
from collections import Counter, defaultdict
from functools import lru_cache


# ============================================================
# SETTINGS
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IDIOMS_PATH = os.path.join(SCRIPT_DIR, "idioms_list")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "extracted_idioms")

# Local copy of NewsPH (used with --source newsph): the folder that holds
# train.txt, or the path of train.txt itself.
NEWSPH_PATH = r"C:\Users\patri\Downloads\newsph\newsph"

# CulturaX is gated: accept its terms on the dataset page and run
# `huggingface-cli login` first.
HF_DATASET = "uonlp/CulturaX"
HF_CONFIG = "tl"

# Longest gap (inserted words) allowed between two words of an idiom.
MAX_GAP = 2

# Sentences with more words than this are skipped (likely sentence-split failures).
MAX_WORDS = 100

# Words that may be inserted inside an idiom at the "gap" and "root" levels.
# Edit freely. Use --gap-any to allow ANY word instead (finds more, but with
# more false matches; check it with sample_match_types.py).
FILLERS = {
    "ba", "daw", "raw", "din", "rin", "kasi", "kaya", "lang", "lamang", "man",
    "muna", "naman", "nga", "pa", "pala", "po", "ho", "sana", "siguro", "yata",
    "na", "ko", "mo", "niya", "natin", "namin", "nila", "ninyo", "nyo", "ka",
}

LINKERS = {"na", "ng"}

PROGRESS_EVERY = 100_000

# A sentence ends at . ! ? followed by whitespace, or at a line break.
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")

# A word is a run of letters/digits (so hyphens, commas and quotes split words).
WORD = re.compile(r"[^\W_]+")

# Match levels, from strictest to loosest: (name, linker, max_gap, stem).
# The "root" level is added only when --stem is given.
BASE_LEVELS = [
    ("exact", False, 0, False),
    ("linker", True, 0, False),
    ("gap", True, None, False),   # None -> MAX_GAP
]
ROOT_LEVEL = ("root", True, None, True)


# ============================================================
# 1. Words and root forms
# ============================================================

def make_stemmer():
    """Return a cached word -> root-form function using tglstemmer."""
    try:
        from tglstemmer.stemmer import get_stem
    except ImportError as error:
        raise SystemExit(
            "--stem needs the tglstemmer package: pip install tglstemmer"
        ) from error

    @lru_cache(maxsize=2_000_000)
    def stem(word):
        try:
            return str(get_stem(word))
        except Exception:
            return word

    return stem


def word_forms(word, linker, stem):
    """
    Return the set of forms a sentence word can count as.

    With linker=True a word ending in -ng also counts without the linker
    ("malaking" -> "malaki", "kaibigang" -> "kaibigan").
    With a stemmer, every form is replaced by its root form.
    """

    bases = {word}

    if linker and word.endswith("ng") and len(word) > 3:
        bases.add(word[:-2])
        bases.add(word[:-1])

    if stem is not None:
        return frozenset(stem(base) for base in bases)

    return frozenset(bases)


def tokenize(text):
    """Return [(lowercase word, start, end), ...] for the words of a text."""
    return [(m.group().lower(), m.start(), m.end()) for m in WORD.finditer(text)]


# ============================================================
# 2. Load idioms
# ============================================================

def clean_title(title):
    """Drop notes in parentheses, e.g. 'balat-sibuyas (sensitive)'."""
    return re.sub(r"\([^)]*\)", " ", title)


def load_idioms(path, column="Title"):
    """
    Load the idioms from one CSV or every CSV in a folder.

    Returns a list of (title, words) where words is the tuple of lowercase
    words. Idioms that give the same words (e.g. 'balat-sibuyas' and
    'balat sibuyas') are merged, keeping the first title.
    """

    if os.path.isdir(path):
        csv_paths = sorted(glob.glob(os.path.join(path, "*.csv")))
    else:
        csv_paths = [path]

    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {path}")

    idioms = {}
    odd_titles = []

    for csv_path in csv_paths:
        with open(csv_path, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                title = (row.get(column) or "").strip()

                if not title:
                    continue

                if "/" in title or "(" in title or "…" in title or "..." in title:
                    odd_titles.append(title)

                words = tuple(w for w, _, _ in tokenize(clean_title(title)))

                if words and words not in idioms:
                    idioms[words] = title

    if odd_titles:
        print(f"Note: {len(odd_titles)} idiom titles contain '/', '(' or '...'; "
              f"check them by hand. First few: {odd_titles[:5]}")

    return [(title, words) for words, title in idioms.items()]


# ============================================================
# 3. Matching
# ============================================================

class Matcher:
    """Finds idioms in a tokenized sentence at the configured levels."""

    def __init__(self, idioms, use_stem, max_gap, gap_any):
        self.idioms = idioms
        self.max_gap = max_gap
        self.gap_any = gap_any
        self.stem = make_stemmer() if use_stem else None
        self.levels = BASE_LEVELS + ([ROOT_LEVEL] if use_stem else [])

        # first-word index per kind of key: {False: {...}, True: {...}} (True = root forms)
        self.first = {False: defaultdict(list), True: defaultdict(list)}
        self.wanted = {}

        for index, (_, words) in enumerate(idioms):
            # indexed under every linker variant of the first word, so that
            # "kaibigan na tapat" is a candidate for "kaibigang tapat"
            for key in word_forms(words[0], True, None):
                self.first[False][key].append(index)

            if use_stem:
                for key in word_forms(words[0], True, self.stem):
                    self.first[True][key].append(index)

    def forms(self, word, linker, stemmed):
        return word_forms(word, linker, self.stem if stemmed else None)

    def match_at(self, tokens, forms, start, index, linker, max_gap, stemmed):
        """Try to match one idiom starting at token `start`; return the end (exclusive) or None."""

        raw = self.idioms[index][1]
        cache_key = (index, linker, stemmed)

        if cache_key not in self.wanted:
            stem = self.stem if stemmed else None
            self.wanted[cache_key] = [word_forms(w, linker, stem) for w in raw]

        wanted = self.wanted[cache_key]
        n, m = len(tokens), len(raw)

        def rec(k, pos):
            if k == m:
                return pos

            # 1. the next idiom word is the next sentence word
            if pos < n and wanted[k] & forms[pos]:
                end = rec(k + 1, pos + 1)

                if end is not None:
                    return end

                # idiom word with an attached linker ("kaibigang") against
                # the detached form in the sentence ("kaibigan na")
                if (linker and raw[k].endswith("ng") and k < m - 1
                        and pos + 1 < n and tokens[pos + 1][0] == "na"):
                    end = rec(k + 1, pos + 2)

                    if end is not None:
                        return end

            if k > 0:
                # 2. linker na/ng swapped for the other one, or left out
                if linker and raw[k] in LINKERS and k < m - 1:
                    if pos < n and tokens[pos][0] in LINKERS:
                        end = rec(k + 1, pos + 1)

                        if end is not None:
                            return end

                    end = rec(k + 1, pos)

                    if end is not None:
                        return end

                # 3. up to max_gap inserted words, then the idiom word
                for gap in range(1, max_gap + 1):
                    if pos + gap >= n:
                        break

                    if not self.gap_any and tokens[pos + gap - 1][0] not in FILLERS:
                        break

                    if wanted[k] & forms[pos + gap]:
                        end = rec(k + 1, pos + gap + 1)

                        if end is not None:
                            return end

            return None

        return rec(0, start)

    def find(self, sentence):
        """
        Return the idioms found in a sentence as a list of dicts with
        idiom, match_type, matched_text, n_words. Overlaps keep the longer match.
        """

        tokens = tokenize(sentence)

        if not tokens or len(tokens) > MAX_WORDS:
            return []

        found = {}  # idiom index -> (match_type, start_token, end_token)

        for name, linker, gap, stemmed in self.levels:
            max_gap = self.max_gap if gap is None else gap
            forms = [self.forms(w, linker, stemmed) for w, _, _ in tokens]
            first = self.first[stemmed]

            for start in range(len(tokens)):
                for key in forms[start]:
                    for index in first.get(key, ()):
                        if index in found:
                            continue

                        # single-word idioms are only matched at the strictest level
                        if len(self.idioms[index][1]) == 1 and name != "exact":
                            continue

                        end = self.match_at(tokens, forms, start, index,
                                            linker, max_gap, stemmed)

                        if end is not None:
                            found[index] = (name, start, end)

        # keep the longer match when one span lies inside another
        spans = list(found.items())
        results = []

        for index, (name, start, end) in spans:
            inside_longer = any(
                other != index
                and s <= start and end <= e
                and (e - s) > (end - start)
                for other, (_, s, e) in spans
            )

            if inside_longer:
                continue

            results.append({
                "idiom": self.idioms[index][0],
                "match_type": name,
                "matched_text": sentence[tokens[start][1]:tokens[end - 1][2]],
                "n_words": len(self.idioms[index][1]),
            })

        return results


# ============================================================
# 4. Scan a dataset
# ============================================================

def read_newsph_articles(path):
    """
    Yield one {"text": ...} per NewsPH article from train.txt.

    train.txt has one paragraph per line, and every article starts with a
    header line "= Title =". Reading it article by article gives each match a
    real article ID (a plain load_dataset on the file would give one row per
    LINE, header and blank lines included).
    """
    if os.path.isdir(path):
        path = os.path.join(path, "train.txt")

    paragraphs = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            is_header = line.startswith("= ") and line.endswith(" =") and not line.startswith("= =")

            if is_header:
                if paragraphs:
                    yield {"text": "\n".join(paragraphs)}
                paragraphs = []
            elif line:
                paragraphs.append(line)

    if paragraphs:
        yield {"text": "\n".join(paragraphs)}


def split_sentences(text):
    """Split a document into sentences (simple punctuation-based rule)."""
    return [s.strip() for s in SENTENCE_SPLIT.split(text) if s.strip()]


def scan_dataset(dataset, matcher, source, limit_rows):
    """Return one row per (sentence, idiom) match."""

    rows = []
    seen = set()
    count = 0

    for count, row in enumerate(dataset, start=1):
        doc_id = f"{source}-{count}"

        for sentence in split_sentences(row["text"]):
            for match in matcher.find(sentence):
                key = (match["idiom"], sentence)

                if key in seen:
                    continue

                seen.add(key)
                rows.append({"source": source, "doc_id": doc_id, "text": sentence, **match})

        if count % PROGRESS_EVERY == 0:
            print(f"  {count:,} rows scanned, {len(rows):,} matches")

        if limit_rows and count >= limit_rows:
            break

    print(f"  {count:,} rows scanned in total")
    return rows


COLUMNS = ["source", "doc_id", "idiom", "match_type", "matched_text", "n_words", "text"]


def save_rows(rows, output_path):
    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# 5. Command line and main program
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Find idioms in a dataset and save the matching sentences.")
    parser.add_argument("--source", choices=["culturax", "newsph"], required=True)
    parser.add_argument("--idioms", default=IDIOMS_PATH, help="folder of idiom CSVs, or one idiom CSV")
    parser.add_argument("--output", help="output CSV (default: extracted_idioms/<source>_matches.csv)")
    parser.add_argument("--dataset", default=NEWSPH_PATH, help="folder with NewsPH train.txt, or train.txt itself")
    parser.add_argument("--stem", action="store_true", help='add the "root" level (needs tglstemmer; slower)')
    parser.add_argument("--max-gap", type=int, default=MAX_GAP)
    parser.add_argument("--gap-any", action="store_true", help="allow any word in a gap, not just FILLERS")
    parser.add_argument("--limit-rows", type=int, default=0, help="stop after this many rows (for trying it out)")
    return parser.parse_args()


def main():
    args = parse_args()

    idioms = load_idioms(args.idioms)
    print(f"Loaded {len(idioms)} idioms "
          f"({sum(len(w) == 1 for _, w in idioms)} are single words)")

    matcher = Matcher(idioms, args.stem, args.max_gap, args.gap_any)

    if args.source == "culturax":
        print(f"Streaming {HF_DATASET} ({HF_CONFIG})...")
        from datasets import load_dataset
        dataset = load_dataset(HF_DATASET, HF_CONFIG, split="train", streaming=True)
    else:
        print(f"Reading NewsPH articles from {args.dataset}...")
        dataset = read_newsph_articles(args.dataset)

    rows = scan_dataset(dataset, matcher, args.source, args.limit_rows)

    output = args.output or os.path.join(OUTPUT_DIR, f"{args.source}_matches.csv")
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    save_rows(rows, output)

    by_type = Counter(r["match_type"] for r in rows)
    found = len({r["idiom"] for r in rows})
    print(f"\nSaved {len(rows):,} matches to {output}")
    print(f"Idioms found: {found} of {len(idioms)}")

    for name, _, _, _ in matcher.levels:
        print(f"  {name:<7} {by_type[name]:>8,}")


if __name__ == "__main__":
    main()
