# Project state: Filipino idiom dataset (learn-nlp)

Context file for Claude when planning next steps. Snapshot as of 2026-10-20. Verify against the code before relying on details.

## Goal
Build a dataset of Filipino/Tagalog sentences containing idioms, to later train an NLP model. The exact training task is **not decided yet** (candidates: sentence-level "contains an idiom" classification, or token-level tagging). BIO tagging was written but is on hold. The only model touched so far is `jcblaise/roberta-tagalog-base` (a fill-mask demo in `experiments/roberta_test.py`).

The user is learning NLP and has been asking conceptual questions (seeds, batch size, shuffling, epochs). Explain briefly and concretely. The user prefers scikit-learn and said to install it rather than avoid it.

## Pipeline (all run from the project root with `.venv/Scripts/python.exe`)

1. **Idiom list**: `scrapers/scraper_sitemap.py` (and the older `scrapers/scraper.py`, one letter at a time) scrape leksyon.com into `idioms_list/<letter>_idioms.csv`. 16 files, **939 idioms** total. Columns: `Title, Part of Speech, Meaning`. Idioms are lowercased phrases (e.g. "abo ang laman ng ulo").
2. **Extraction**: `extractors/culturax_extraction.py` (streams `uonlp/CulturaX`, config `tl`, from Hugging Face; gated, needs `huggingface-cli login`) and `extractors/news_ph_extraction.py` (local NewsPH copy at `C:\Users\patri\Downloads\newsph\newsph`). Both build one Aho-Corasick automaton over all idioms, split rows into sentences, and keep sentences with a whole-word match. Output is one CSV per idiom file:
   - `culturax_idioms/<letter>_culturax_idioms.csv`
   - `news_ph_idioms/<letter>_news_ph_idioms.csv`
   - Columns: `idioms` (matched idioms joined with `; `), `text`. Encoding `utf-8-sig`.
   - A sentence matching idioms from several letter files is written to each of those files, listing only that file's idioms.
3. **Duplicate check and merge**: `check_duplicates.py` (pandas).
   - Default: reports exact duplicates (same idioms + text, case/whitespace-normalized) per folder. Found 23,644 extra rows in culturax and 1,155 in news_ph.
   - `--remove`: rewrites the CSVs without exact duplicates within each file. **Not run.**
   - `--merge`: combines both folders into **`merged_idioms/merged.csv`**, one row per unique normalized sentence with all its idioms combined (sorted, `; ` joined). 237,200 rows became **208,835** unique sentences; 3,718 have more than one idiom. Originals are untouched.
4. **Split**: `split_dataset.py` (scikit-learn `train_test_split` twice) reads `merged_idioms/merged.csv` and writes `dataset/train.csv`, `val.csv`, `test.csv`. Defaults are 80/10/10, **seed 67** (the user changed it from 42; the current files were produced with 67). Random split by sentence, so no sentence text is in two files (verified). Rerunning overwrites the three files. It does not balance idioms across the files.
5. **BIO tagging (on hold)**: `bio_tagging.py` adds `tokens` and `bio_tags` columns (B-IDIOM / I-IDIOM / O, JSON lists) to the per-letter extraction CSVs in place. The user said to disregard it for now. As of the last check the extraction CSVs and the merged/split files have only `idioms,text`, so it has not been applied. If revived, it should run on the merged table, not the per-letter files, so every idiom in a merged sentence gets tagged.

## Data facts
- `dataset/` files: train 167,068 / val 20,883 / test 20,884 rows (sizes shift slightly with the seed).
- Only **294 of the 939 idioms** appear in the merged data; 118 of those have 5 or fewer sentences.
- Sentence length: median 24 words, mean 30, max 2,086 (very long rows exist; needs truncation or filtering).
- **Heavy imbalance**: `babae` (77,568 sentences), `magulang` (40,755), `utak` (14,891), `bagyo` (14,467), `maliwanag` (9,523), `tulay`, `mabait`, `bomba`. These are single everyday words, so many matches are **literal uses, not idiomatic ones** (and "idioms" in the list include single words). This likely makes the data noisy for an idiom-detection task.
- Matching is exact whole-word string matching, so inflected or reordered idiom forms are missed (Filipino affixation/reduplication).
- There are no negative examples (sentences without idioms) yet, so a classifier cannot be trained as-is.

## Repo state
- Git branch `main`, last commit `ee386cc`. Many files are uncommitted: moved `extractors/`, `scrapers/`, `tests/test.py` (old root copies are deleted in the working tree), plus new `bio_tagging.py`, `check_duplicates.py`, `split_dataset.py`, `merged_idioms/`, `dataset/`. Nothing here has been committed yet.
- `.venv` has pandas, scikit-learn 1.9.1, `datasets`, `pyahocorasick`, `transformers`, bs4, requests. No `requirements.txt`; README is only a title. Folders `__pycache__`, `tempCodeRunnerFile.py`, and `download.py` at the root are scratch.
- `tests/` contains experiments, not real tests (`roberta_test.py`, `test.py` scrape-parsing snippet).

## Open questions and likely next steps
1. **Decide the task** (sentence classification vs token tagging vs something else). Everything below depends on it.
2. **Fix label quality**: decide how to handle single-word or common-word "idioms" (`babae`, `magulang`, ...). Options: drop or down-weight them, require multi-word idioms, cap rows per idiom, or manually review a sample.
3. **Handle imbalance**: cap sentences per idiom before splitting, and consider a stratified split (rare idioms may only land in train).
4. **Negatives**: if doing classification, add sentences with no idiom match from CulturaX/NewsPH.
5. **Overlap between sources**: CulturaX and NewsPH may share articles. No document ID is kept, so near-duplicates across train/test are possible.
6. **Filter** extreme-length sentences.
7. **Training setup**: pick a model (tagalog RoBERTa is already available), tokenizer and max length, fine-tune with Hugging Face `Trainer`, evaluate with precision/recall/F1 on val, use the test set once at the end.
8. **Housekeeping**: commit the work, add a `requirements.txt` and README, and decide whether `merged_idioms/` and `dataset/` (~40 MB each) should be git-ignored.
