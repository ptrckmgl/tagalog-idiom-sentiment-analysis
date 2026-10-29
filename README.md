# Tagalog idiom sentiment analysis

Dataset pipeline for Filipino/Tagalog idiom sentences. Run every command from the repository root (`pip install -r requirements.txt` first).

## Layout

```
scrapers/      scraper_sitemap.py -> data/leksyon_idioms/, scraper_wiktionary.py -> data/wiktionary_idioms/
processing/    combine_idiom_lists.py (leksyon + Wiktionary -> data/combined_idioms/),
               check_duplicates.py, split_dataset.py (still expect the old merged.csv; to be updated),
               bio_tagging.py (on hold)
extractors/    extract_idioms.py  find idioms in NewsPH / CulturaX -> extracted_idioms/
               sample_match_types.py  hand-check a sample of matches per match type
experiments/   roberta_test.py, tagasenti_test.py  model demos
data/          leksyon_idioms/, wiktionary_idioms/, combined_idioms/ (list + idiom_review.csv)
extracted_idioms/  extraction outputs and write-ups
docs/          PROJECT_STATE.md  snapshot of the project on 2026-10-20 (older layout)
```

## Pipeline

1. `python scrapers/scraper_sitemap.py` and `python scrapers/scraper_wiktionary.py`
2. `python processing/combine_idiom_lists.py`
3. Review `data/combined_idioms/idiom_review.csv` (fill in `your_decision`)
4. `python extractors/extract_idioms.py --source newsph --idioms data/combined_idioms/combined_idioms.csv`
5. Merge and split: `check_duplicates.py` and `split_dataset.py` need updating for the new extractor output.
