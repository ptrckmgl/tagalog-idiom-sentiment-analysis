# Stem vs no-stem on 20,000 NewsPH rows

Two runs of `extractors/extract_idioms.py` (leksyon list, 939 idioms, 95 of them single words), on the first 20,000 rows of NewsPH. The first 20,000 rows are not a random sample.

| File | Command flag | Matches | Idioms found | Time |
|---|---|---|---|---|
| `newsph_20k_nostem.csv` | (none) | 545 | 56 | 5 s |
| `newsph_20k_stem.csv` | `--stem` | 622 | 76 | 56 s |

The stem run contains exactly the same exact/linker/gap rows as the no-stem run (534 / 10 / 1), plus 77 `root` rows. So `--stem` only adds matches; it never changes the others.

Labels below are **my own judgment, not a native speaker's.** Please check `newsph_20k_root_matches_labeled.csv` yourself.

## 1. Without stemming: mostly single common words

- 90% of the 545 matches are single-word idioms (*makati* 105, *bata* 75, *bagyo* 69, *babae* 64, *magulang* 27, *bumagsak* 27, *bomba* 27). These are probably mostly literal uses.
- Only 65 matches (31 idioms) come from multi-word idioms: 54 exact, 10 linker, 1 gap.
- The linker level mostly caught *mga walang* ~ *mga wala*, plus *kulang na kulang* and *laman na lamang*. The one gap match was *Binawian din ng buhay*.

So the multi-word yield is small: about 3 per 1,000 rows.

## 2. What `--stem` added (77 root matches)

| My label | Rows | Share |
|---|---|---|
| valid form of the idiom | 28 | 36% |
| unsure | 18 | 23% |
| false match | 31 | 40% |

**Valid examples:** *naghuhugas ng kamay* found as *naghugas kamay*, *hugas-kamay*, *maghugas- kamay* (5). *baon sa utang* found as *nabaon sa utang* (4). *masama ang lagay* found as *sumama ang lagay*. *nagdalawang-isip* found as *magdadalawang-isip*. *lakas-loob* found as *naglakas loob*.

**False examples:**
- *kamay ng singkit* matched *may sakit* (11), *may nakitang* (4) and similar: 17 rows, 22% of all root matches, from one idiom. This is the *kamay* → *may* over-stemming I flagged earlier, now seen on real data.
- *kinakalawang na* matched *pangalawa na* (4).
- *walang makapitan* matched *nawawalang kapitan* (2): the *kapit* clash.
- *may-kaya* matched *mamamayan kaya* (2).
- *naging bato* matched *naging pambato* (1): *pambato* means "champion".

**Unsure:**
- *mga wala* matched *mga nawawala*, *mga nawalang* and similar (13). These are forms of *wala*, but I doubt *mga wala* is a real idiom. It looks like a fragment of a longer expression.
- 5 other rows are borderline, for example *nilipasan ng panahon* matched *paglipas ng panahon*, and *walang sinasabi* matched *walang pasabi*.

## 3. What this shows

1. **Stemming does recover real matches.** Some titles are written in one inflected form (*naghuhugas ng kamay*), and without stemming the other forms are lost. 20 more idioms were found, but at least some of those are only false hits.
2. **Precision of the root level is about 36% to 59%.** That is the valid share, and the valid plus unsure share. It is far below what you would want for training data, and one idiom (*kamay ng singkit*) causes a fifth of the errors.
3. **Several idiom titles are not really idioms.** *hindi nakaaabot*, *mga wala*, *mga mayroon* and *may sinasabi* are ordinary phrases. Even a "valid form" match of these will probably not be a figurative use. This is a separate issue from the stemmer and is worth checking in the leksyon list.
4. **Cost.** Stemming is about 11 times slower here (56 s vs 5 s for 20,000 rows). The full 2.19M rows would take roughly 1.5 hours with stemming by simple scaling (the cache should help, so this is probably an upper bound), against several minutes without.

## 4. Options (nothing changed in the code yet)

- **A. Keep `root` off** and accept the lost inflected forms. Simplest, and avoids the false matches.
- **B. Keep `root` on but filter it.** For example, ignore stems shorter than 4 letters (this would block *kamay* → *may*), or add a skip list of over-stemmed words (*kamay*, *kaibigan*, *magulang*). Needs a change to `make_stemmer()` / `word_forms()` in `extract_idioms.py` only.
- **C. Keep `root` on and rely on annotation.** Your first annotation step is already "valid match", so the false ones would be removed by hand. This costs annotator time: about 40% of root rows would be wasted.

I believe B followed by a re-check of the sample is the best balance, but it is your choice.

## Caveats

- 20,000 rows is under 1% of NewsPH, and not random.
- Only 77 root rows, so the percentages are rough.
- I did not look at CulturaX.
- The `extract_idioms_new.py` copy was not run.
