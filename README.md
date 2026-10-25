# Tagalog idiom sentiment analysis

Dataset pipeline for Filipino/Tagalog idiom sentences. Run every command from the repository root (`pip install -r requirements.txt` first).

## Layout

```
scrapers/      scraper_sitemap.py  scrape leksyon.com -> data/idioms_list/
extractors/    culturax_extraction.py, news_ph_extraction.py  find idioms in text -> data/culturax_idioms/, data/news_ph_idioms/
processing/    check_duplicates.py (dedupe/merge), split_dataset.py (train/val/test), bio_tagging.py (on hold)
experiments/   roberta_test.py  fill-mask demo of jcblaise/roberta-tagalog-base
data/          idioms_list/, culturax_idioms/, news_ph_idioms/, merged_idioms/merged.csv, dataset/{train,val,test}.csv
docs/          PROJECT_STATE.md  notes on the pipeline and open questions
```

## Pipeline

1. `python scrapers/scraper_sitemap.py`
2. `python extractors/culturax_extraction.py` and `python extractors/news_ph_extraction.py`
3. `python processing/check_duplicates.py --merge`
4. `python processing/split_dataset.py`

See `docs/PROJECT_STATE.md` for details.
