# Sentences per idiom: 200,000 NewsPH rows

Run: `extract_idioms.py --source newsph --limit-rows 200000 --idioms data/combined_idioms/combined_idioms.csv` (no `--stem`). It used the first 200,000 rows, about 9% of NewsPH. These are **not a random sample**. Output: `newsph_200k_nostem_combined.csv`, 12,316 matches (exact 11,490, linker 717, gap 109).

Counts are per **idiom group**: titles that share a `variant_group` in the review file at the time (since replaced by `data/combined_idioms/idiom_review.csv`, whose groups also include the earlier cleanup's) are pooled into one group, and every other title is its own group. "Cleaned" means the 7 `drop` and 11 `cap or drop` titles are removed. Nothing else is removed yet, so the `review` titles are still in.

## 1. The numbers

| Set | Groups | Sentences | Mean | Median | ≥5 | ≥10 | ≥20 | ≥50 |
|---|---|---|---|---|---|---|---|---|
| All matches | 208 | 12,316 | 59.2 | 3 | 93 | 56 | 36 | 26 |
| All, cap 50 | 208 | 2,320 | 11.2 | 3 | 93 | 56 | 36 | 26 |
| Cleaned | 197 | 3,043 | 15.4 | 3 | 81 | 44 | 24 | 15 |
| Cleaned, cap 50 | 197 | 1,745 | 8.9 | 3 | 81 | 44 | 24 | 15 |
| Cleaned, multi-word only, cap 50 | 163 | 1,058 | 6.5 | 2 | 60 | 26 | 11 | 7 |

Groups per count (cleaned, cap 50):

```
           1   65 #################################################################
           2   28 ############################
         3-4   23 #######################
         5-9   37 #####################################
       10-19   20 ####################
       20-49    9 #########
 50 (capped)   15 ###############
```

## 2. What this shows

- **This is exactly the case you asked about.** With the cap, the average is about 9 per group, but the **median is 3**. A third of the groups (65 of 197) have exactly one sentence. Only 44 groups reach 10.
- **The 18 titles I suggested removing are three quarters of all matches** (9,273 of 12,316). Without them the data is much smaller but less dominated by a few words.
- **Several `review` titles still rank near the top.** *sa una* (627 sentences), *noong una* (94) and *mga wala* (80) are ordinary phrases I flagged earlier. If you drop them, the counts above go down further.
- **Single-word titles still matter.** Even after the cleanup, 34 of the 197 groups are single words, and they hold a good share of the capped sentences. Compare the last two rows of the table.

## 3. Rough projection to all of NewsPH (estimate only)

Scaling each group's count by 10.95 (2.19M ÷ 200k), cleaned, with cap 50:
- **197 groups:** median about 33, about 130 groups with 20 or more, about 6,200 sentences.
- **Multi-word only:** 163 groups, median about 22, about 4,900 sentences.

Treat this as an upper-bound guess:
- An idiom seen once in 200,000 rows won't reliably show up about 11 times in 2.19M. Counts this small are mostly luck.
- More idioms will appear in the full run (the number of groups went from 67 at 20k to 208 at 200k). Most new ones will have small counts, which pulls the median down again.

## 4. What I'd conclude

- **Don't decide the minimum from this sample.** The median is below 5, and the plan said to run the full NewsPH extraction (and ideally CulturaX) before choosing.
- Do the full NewsPH run next. Without `--stem` it should take about 7 minutes at the speed seen here (200k rows in 39 s), and now it shows a progress bar.
- After the full run, re-make this table and pick the minimum from it (for example 10 or 20). Idioms below the minimum go to training only.

## Checks done

- Variant pooling: for three groups the pooled count equals the sum of their titles' counts (for example *bumagsak / nabagsak / pabagsak* gives 441 = 441).
- The 20k combined run with the new progress bar gives the same totals as before (exact 590, linker 63, gap 3).
