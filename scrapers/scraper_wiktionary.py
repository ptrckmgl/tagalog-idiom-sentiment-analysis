"""Scrape Tagalog idioms from English Wiktionary (Category:Tagalog idioms).

How it works:
  1. List every page in the category with the MediaWiki API, following
     `cmcontinue` so the "next page" of the category is included (392 titles).
  2. For each title, fetch the rendered page and read the Tagalog section:
     the part of speech heading and the first definition.
  3. Write one CSV with the same columns as the leksyon scraper.

Usage:
    .venv/Scripts/python.exe scrapers/scraper_wiktionary.py
    .venv/Scripts/python.exe scrapers/scraper_wiktionary.py --limit 10
"""

import argparse
import csv
import os
import re
import time

import requests
from bs4 import BeautifulSoup

# ============================================================
# SETTINGS
# ============================================================

API_URL = "https://en.wiktionary.org/w/api.php"
CATEGORY = "Category:Tagalog idioms"
OUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "wiktionary_idioms", "wiktionary_idioms.csv",
)
HEADERS = {"User-Agent": "idiom-thesis-scraper/0.1 (patricksantiaguel@gmail.com)"}
DELAY = 0.5  # seconds between page requests

# Wiktionary uses these as section headings for the part of speech
POS_HEADINGS = {
    "Noun", "Verb", "Adjective", "Adverb", "Pronoun", "Preposition",
    "Conjunction", "Interjection", "Particle", "Phrase", "Proverb",
    "Idiom", "Expression", "Determiner", "Numeral", "Affix", "Prefix",
    "Suffix", "Infix", "Contraction", "Lemma", "Proper noun",
}


# ============================================================
# 1. List the category
# ============================================================

def get_titles(session):
    """All page titles in the category, following the API's continue token."""
    titles = []
    params = {
        "action": "query", "list": "categorymembers", "cmtitle": CATEGORY,
        "cmtype": "page", "cmlimit": "500", "format": "json",
    }
    while True:
        data = session.get(API_URL, params=params, timeout=30).json()
        titles += [m["title"] for m in data["query"]["categorymembers"]]
        if "continue" not in data:
            return titles
        params.update(data["continue"])


# ============================================================
# 2. Parse one page
# ============================================================

def get_html(session, title):
    params = {
        "action": "parse", "page": title, "prop": "text",
        "disablelimitreport": "1", "format": "json",
    }
    data = session.get(API_URL, params=params, timeout=30).json()
    return data["parse"]["text"]["*"]


def parse_page(html, title):
    """Return [title, part of speech, first meaning] from the Tagalog section."""
    soup = BeautifulSoup(html, "html.parser")
    start = soup.find(id="Tagalog")
    if start is None:
        return [title, "N/A", "N/A"]

    # the heading is wrapped in a div on current Wiktionary, bare otherwise
    heading = start.find_parent("div", class_="mw-heading") or start
    pos, mean = "N/A", "N/A"
    for el in heading.find_next_siblings():
        if el.name == "h2" or "mw-heading2" in (el.get("class") or []):
            break  # next language
        h = el if el.name and re.fullmatch(r"h[3-6]", el.name) else el.find(re.compile(r"h[3-6]$"))
        if h is not None and h.get_text(strip=True) in POS_HEADINGS and pos == "N/A":
            pos = h.get_text(strip=True)
        if pos != "N/A" and el.name == "ol":
            li = el.find("li")
            if li:
                for sub in li.find_all(["ul", "ol", "dl"]):
                    sub.decompose()  # drop example sentences and quotes
                mean = re.sub(r"\s+", " ", li.get_text(" ", strip=True))
                mean = re.sub(r"\s+([,.;:!?)])", r"\1", mean)
                mean = re.sub(r"\(\s+", "(", mean)
            break
    return [title, pos, mean]


# ============================================================
# 3. Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=OUT_PATH)
    ap.add_argument("--limit", type=int, default=0, help="only scrape the first N titles (0 = all)")
    args = ap.parse_args()

    session = requests.Session()
    session.headers.update(HEADERS)

    titles = get_titles(session)
    print(f"{len(titles)} titles in {CATEGORY}")
    if args.limit:
        titles = titles[:args.limit]

    rows = []
    for i, title in enumerate(titles, 1):
        try:
            rows.append(parse_page(get_html(session, title), title))
        except (requests.RequestException, KeyError, ValueError) as e:
            print(f"[{i}/{len(titles)}] failed {title}: {e}")
            continue
        print(f"[{i}/{len(titles)}] {title}")
        time.sleep(DELAY)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["Title", "Part of Speech", "Meaning"])
        writer.writerows(rows)
    print(f"wrote {len(rows)} idioms to {args.out}")


if __name__ == "__main__":
    main()
