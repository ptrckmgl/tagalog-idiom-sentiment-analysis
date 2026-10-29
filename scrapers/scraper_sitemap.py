import csv
import os
import re
import time
import unicodedata
from urllib.parse import unquote

import requests
from bs4 import BeautifulSoup

SITEMAP_URL = "https://www.leksyon.com/sitemap.xml"
OUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "leksyon_idioms", "leksyon_idioms.csv")
HEADERS = {"User-Agent": "Mozilla/5.0"}

# Title is written without accents (that is the form the matcher needs);
# the title as shown on leksyon.com is kept in Accented_Title.
COLUMNS = ["Title", "Part of Speech", "Meaning", "Accented_Title"]


def strip_accents(text):
    text = unicodedata.normalize("NFD", text)
    return "".join(c for c in text if not unicodedata.combining(c))


def fold(text):
    """Lowercase and strip accents so sorting ignores diacritics."""
    return strip_accents(text.casefold())


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
    """Return {folded title: [accented title, pos, meaning]} from OUT_PATH."""
    rows = {}
    if not os.path.exists(OUT_PATH):
        return rows
    with open(OUT_PATH, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            title = row.get("Accented_Title") or row["Title"]
            rows[fold(title).strip()] = [title, row["Part of Speech"], row["Meaning"]]
    return rows


def save_rows(rows):
    """Write [accented title, pos, meaning] rows to OUT_PATH, sorted ignoring accents."""
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        for title, pos, mean in sorted(rows, key=lambda r: fold(r[0])):
            writer.writerow([strip_accents(title), pos, mean, title])


def main():
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

    save_rows(rows.values())
    print(f"saved {len(rows)} idioms to {OUT_PATH}")


if __name__ == "__main__":
    main()
