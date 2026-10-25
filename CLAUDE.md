# CLAUDE.md: Filipino idiom dataset (learn-nlp)

Context for a coding assistant working in this repo. Written 2026-10-20 from the project thread and PROJECT_STATE.md (snapshot 2026-10-20). Things marked **(default)** were picked by Claude and not yet confirmed by Pat. Things marked **(untested)** have not been run on the real data.

## Goal

An undergraduate thesis (CvSU Main Campus, Indang) on Filipino/Tagalog idioms. The framework has three steps:

1. A **rule-based matcher** finds idioms from a list of 939 idioms (scraped from leksyon.com) in a sentence.
2. **Model 1** (sequence classification) decides whether the matched idiom is used **figuratively or literally**.
3. **Model 2** (sequence classification) gives the **sentiment** (positive, neutral, negative) of the figurative uses only.

Input to both models is a sentence pair: the matched idiom text and the original sentence. Models: full fine-tuning (not LoRA or other PEFT) of `jcblaise/roberta-tagalog-base`, compared with `bert-base-multilingual-cased` and `xlm-roberta-base`. Primary metric: macro-F1. Test sets: a seen-idiom test set and an unseen-idiom test set (about 10% of idioms held out completely).

**BIO tagging / token classification is off the table.** The task is sequence classification. `bio_tagging.py` is on hold and should not be extended.

The user is a beginner in NLP: explain briefly, prefer scikit-learn where it fits, and say plainly when code is untested.

## Pipeline and layout

Run everything from the repo root with `.venv/Scripts/python.exe` (Windows).

| Step | File | Output |
|---|---|---|
| 1. Scrape idioms | `scrapers/scraper_sitemap.py` | `idioms_list/<letter>_idioms.csv` (16 files, 939 idioms; columns `Title`, `Part of Speech`, `Meaning`) |
| 2. Extract sentences (old) | `extractors/culturax_extraction.py`, `extractors/news_ph_extraction.py` | `culturax_idioms/`, `news_ph_idioms/` (columns `idioms`, `text`; utf-8-sig) |
| 2. Extract sentences (new) | `extractors/extract_idioms.py` **(untested on real data)** | `extracted_idioms/<source>_matches.csv` |
| 3. Estimate match precision | `extractors/sample_match_types.py` **(untested on real data)** | `review.csv`, then a precision table |
| 4. Merge and de-duplicate | `check_duplicates.py` (`--merge`, `--remove`) | `merged_idioms/merged.csv` |
| 5. Split | `split_dataset.py` (scikit-learn, seed 67) | `dataset/train.csv`, `val.csv`, `test.csv` |
| on hold | `bio_tagging.py` | none |

Data sources: **CulturaX** (config `tl`, streamed from Hugging Face, gated: accept terms and run `huggingface-cli login`) and a local copy of **NewsPH** (`C:\Users\patri\Downloads\newsph\newsph`, column `text`).

Steps 4 and 5 were written for the OLD columns (`idioms`, `text`). The new extractor writes one row per match with different columns, so both need updating (see Open work).

## State of the old data (before re-extraction)

- 237,200 rows became 208,835 unique sentences after merging. The old split was a random 80/10/10 split **by sentence**, which does not fit the design.
- Only 294 of 939 idioms appeared, and 118 of those had 5 or fewer sentences.
- Most sentences came from a few single everyday words: *babae* (77,568), *magulang* (40,755), *utak*, *bagyo*, *maliwanag*. Most of these matches are probably literal, not idiomatic.
- Median sentence length is 24 words, maximum 2,086 (sentence-split failures).
- No negative sentences and no literal/figurative labels exist yet.

## Matcher rules (extract_idioms.py)

Words are runs of letters/digits, lowercased, so punctuation and hyphens split words (*balat-sibuyas* = *balat sibuyas*). Parenthetical notes in idiom titles are dropped, and titles with the same words are merged. For each sentence and idiom the first level that matches is recorded as `match_type`:

1. `exact`: idiom words in order, adjacent.
2. `linker`: `na`/`ng` may be swapped or missing, and a word ending in `-ng` also matches its form without the linker (*kaibigang tapat* ~ *kaibigan na tapat*). `ang` is NOT a linker.
3. `gap`: plus up to 2 inserted words (`MAX_GAP`). By default only particles from `FILLERS` may be inserted; `--gap-any` allows any word.
4. `root`: plus root-form comparison with `tglstemmer` (`from tglstemmer.stemmer import get_stem`). Only with `--stem`. It takes about 6 ms per new word and gives some odd stems (*bilang* becomes *bila*), so results are cached and the level is off by default.

Other rules: single-word idioms match at `exact` only. When one match lies inside a longer one, the longer wins. Word-order changes are not handled. Sentences over `MAX_WORDS` (100) are skipped. Duplicate (idiom, sentence) pairs are kept once. Output columns: `source, doc_id, idiom, match_type, matched_text, n_words, text`, where `doc_id` is `<source>-<row number>`.

Known problem: stemming can create false matches (for example *kapit* inside *kapitbahay*). That is why every match carries a `match_type` and `sample_match_types.py` measures precision per type from a hand-checked sample (about 30 per type). Keep loosened matches only for levels with acceptable precision.

## Decisions already made (by Pat)

- Sequence classification, not BIO tagging. Full fine-tuning, not PEFT.
- Taglish sentences stay in the data and are not analyzed separately (separate reporting for fairness across models was offered and is undecided).
- Three annotators: two hired experts plus the group.
- Sentence splitting uses the regex rule in the extractors (calamanCy was dropped from Chapter 3 on 2026-10-20).
- The thesis text lives in a Google Doc; its Chapter 3 tab describes this pipeline in future tense.

## Defaults Claude picked (not yet confirmed)

- `tglstemmer` for root forms, at most 2 inserted words, longest match wins.
- Hold out about 10% of idioms as an unseen-idiom test set; split the rest 80/10/10 stratified by label and grouped by document. Max token length 128. Learning rates 1e-5, 2e-5, 3e-5; batch size 16; up to 5 epochs; 3 seeds.
- Fleiss' kappa plus pairwise Cohen's kappa on an overlap sample of at least 200 sentences.

## Open work

1. Run `extract_idioms.py --source newsph --limit-rows 20000`, then sample and score match types; decide which levels to keep (and whether to use `--stem`).
2. Update `check_duplicates.py` and `split_dataset.py` for the new columns, and re-split **by idiom** (no idiom in more than one split), grouped by `doc_id`. A sentence with two idioms goes to one partition.
3. Dataset cleanup: handle single-word idioms (drop, cap, or review), cap sentences per idiom (placeholder: 50), set a minimum sentences per idiom, and make sure the unseen-idiom set has idioms with enough sentences.
4. Annotation: valid match, then figurative/literal, then sentiment (figurative only). Not started.
5. Check CulturaX/NewsPH overlap (near-duplicate sentences across splits).
6. Housekeeping: commit (branch `main`, last commit `ee386cc`, many files uncommitted), add `requirements.txt` and a README, decide whether `merged_idioms/` and `dataset/` (about 40 MB each) are git-ignored.

## Code conventions (match the existing scripts)

- Module docstring at the top with a short "how it works" list and usage examples.
- A `SETTINGS` section of constants near the top, then numbered sections (`# 1. ...`) separated by `# ====` banners.
- `argparse` with defaults taken from SETTINGS; plain functions with short docstrings; standard library plus pandas/scikit-learn.
- CSV files are written as `utf-8-sig`.
- Keep comments sparse and in the same plain tone. Do not add dependencies without saying why.

## Working with the user

- Short paragraphs, humble wording ("I believe"), options to choose from, and a clear line between general knowledge and sourced claims.
- Tagalog example sentences written by the assistant need native-speaker review.
- Before changing anything outside the file being discussed, say what else would change.
