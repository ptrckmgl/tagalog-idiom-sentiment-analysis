import csv
import os
import re
import time
import unicodedata
from collections import defaultdict
from urllib.parse import unquote

import requests
from bs4 import BeautifulSoup

SITEMAP_URL = "https://www.leksyon.com/sitemap.xml"
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "idioms_list")
HEADERS = {"User-Agent": "Mozilla/5.0"}


def fold(text):
    """Lowercase and strip accents so sorting/letter-grouping ignores diacritics."""
    text = unicodedata.normalize("NFD", text.casefold())
    return "".join(c for c in text if not unicodedata.combining(c))


def get_urls():
    xml = requests.get(SITEMAP_URL, headers=HEADERS).text
    urls = re.findall(r"<loc>([^<]+)</loc>", xml)
    return [u for u in urls if "/filipino/" in u]


def parse_page(html, url):
    soup = BeautifulSoup(html, "html.parser")
    term = soup.select_one(".pronunciation-term")
    # fall back to the url slug if the term isn't on the page
    title = term.text.strip() if term else unquote(url.rsplit("/", 1)[-1]).replace("+", " ")

    pos_el = soup.find(class_="part-speech-field-content")
    pos = pos_el.text.strip() if pos_el else "N/A"

    mean_el = soup.select_one(".views-field-field-definition-translation .field-content")
    if mean_el:
        lis = mean_el.find_all("li")
        mean = "\n".join(li.text.strip() for li in lis) if lis else mean_el.text.strip()
    else:
        mean = "N/A"
    mean = re.sub(r"^English meaning:\s*", "", mean)
    return [title, pos, mean]


def load_existing():
    rows = {}
    if not os.path.isdir(OUT_DIR):
        return rows
    for name in os.listdir(OUT_DIR):
        if not name.endswith("_idioms.csv"):
            continue
        with open(os.path.join(OUT_DIR, name), newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row:
                    rows[fold(row[0]).strip()] = row
    return rows


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = load_existing()
    print(f"{len(rows)} existing idioms")

    urls = get_urls()
    print(f"{len(urls)} urls in sitemap")

    session = requests.Session()
    session.headers.update(HEADERS)
    added = 0
    for i, url in enumerate(urls, 1):
        slug = unquote(url.rsplit("/", 1)[-1]).replace("+", " ")
        if fold(slug).strip() in rows:
            continue  # already scraped
        try:
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"[{i}/{len(urls)}] failed {url}: {e}")
            continue
        row = parse_page(resp.content, url)
        key = fold(row[0]).strip()
        if key not in rows:
            rows[key] = row
            added += 1
        print(f"[{i}/{len(urls)}] {row[0]}")
        time.sleep(0.5)
    print(f"added {added} new idioms")

    by_letter = defaultdict(list)
    for key, row in rows.items():
        letter = next((c for c in key if c.isalpha()), "#")
        by_letter[letter].append(row)

    for letter, letter_rows in by_letter.items():
        letter_rows.sort(key=lambda r: fold(r[0]))
        path = os.path.join(OUT_DIR, f"{letter}_idioms.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Title", "Part of Speech", "Meaning"])
            writer.writerows(letter_rows)


if __name__ == "__main__":
    main()
