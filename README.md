# Word-order flexibility

Measure word-order flexibility from raw text and validate it against Head
Direction Entropy (HDE), Relation Order Entropy (ROE), and Bakker flexibility
ratings.

## 1. Install

Requires Python 3.11+ and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --group cluster
```

## 2. Prepare the data

Download Universal Dependencies 2.17:

```bash
uv run python download_ud.py --target ud
```

Prepare Wikipedia text files named with language and script codes, such as
`eng_Latn.txt`:

```bash
uv run python process_wikipedia.py \
  --output-dir data/wikipedia-5000 \
  --documents 5000 \
  --seed 0
```

Wikipedia is segmented with the multilingual WTPSplit `sat-3l-sm` model. Cache
the model in the shared cluster environment before submitting jobs:

```bash
uv run python -c 'from word_order_flexibility.text import segment_text; print(segment_text("One. Two."))'
```

The Bible directory must contain whitespace-tokenized, one-sentence-per-line
files using the same `language_script.txt` naming convention.

## 3. Inspect and submit the main experiment

The main plan runs:

- broad UD validation and its Futrell language subset at 1,000 sentences;
- a fixed-language UD study at 1,000, 2,000, 5,000, and 10,000 sentences;
- both case-folded and case-sensitive versions of all UD experiments.

Inspect the plan without submitting:

```bash
uv run --group cluster python experiment.py \
  --ud-path ud/ud-treebanks-v2.17 \
  --output-dir results-full
```

Submit it:

```bash
uv run --group cluster python experiment.py \
  --ud-path ud/ud-treebanks-v2.17 \
  --output-dir results-full \
  --partition batch \
  --cluster wice \
  --account YOUR_ACCOUNT \
  --email YOUR_EMAIL \
  --submit
```

The case-folded paper results are written to `all-1000` and `sentence-sizes`.
The case-sensitive results use the corresponding `-cased` directories. The
Futrell comparison is derived from the matching languages in `all-1000`. Use a
new output directory for each definitive run.

## 4. Submit the disjoint domain study

This plan draws five disjoint 1,000-sentence samples for the fixed 20-language
intersection of UD, Bible, and Wikipedia:

```bash
uv run --group cluster python experiment.py \
  --ud-path ud/ud-treebanks-v2.17 \
  --bible-dir data/bible \
  --wikipedia-dir data/wikipedia-5000 \
  --wikipedia-min-words 5 \
  --wikipedia-cpus 8 \
  --output-dir results-full \
  --disjoint-domain-study \
  --partition batch \
  --cluster wice \
  --account YOUR_ACCOUNT \
  --email YOUR_EMAIL \
  --submit
```

## 5. Generate tables and figures

Copy `results-full` to a machine with LaTeX, then run the analyses in this
order:

```bash
uv run word-flexibility disjoint-domain-analysis results-full/disjoint-domain-study
uv run word-flexibility paper results-full
```

The outputs are written to `results-full/disjoint-domain-study/analysis` and
`results-full/paper`.

## 6. Calculate the metric on raw text

```bash
uv run word-flexibility text corpus.txt --lambda 0.5
```

Raw text is sentence-segmented with WTPSplit, tokenized, stripped of
punctuation, and case-folded. Use `--format lines` for one untokenized sentence
per line or `--format tokenized` for whitespace-tokenized sentences.

## 7. Verify the installation

```bash
uv run ruff check .
uv run pytest
```

## References

- Dik Bakker (1998), [*Flexibility and consistency in word order patterns in
  the languages of Europe*](https://doi.org/10.1515/9783110812206.383).
- Richard Futrell, Kyle Mahowald, and Edward Gibson (2015), [*Quantifying Word
  Order Freedom in Dependency Corpora*](https://aclanthology.org/W15-2112/).
- Anne Chao and Tsung-Jen Shen (2003), [*Nonparametric estimation of Shannon's
  index of diversity when there are unseen species in
  sample*](https://doi.org/10.1023/A:1026096204727).
- Aryaman Arora, Clara Meister, and Ryan Cotterell (2022), [*Estimating the
  Entropy of Linguistic
  Distributions*](https://aclanthology.org/2022.acl-short.20/).
- [Universal Dependencies 2.17](http://hdl.handle.net/11234/1-6036).

## License

[MIT](LICENSE)
