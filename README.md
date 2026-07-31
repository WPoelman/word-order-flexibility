# Word-order flexibility

Measure word-order flexibility from raw text.
The package also implements Head Direction Entropy (HDE) and Relation Order Entropy (ROE) for validation on Universal Dependencies (UD).

## Install

Requires Python 3.11+ and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Raw text

```bash
uv run word-flexibility text corpus.txt
```

Raw `text` input is segmented with WTPSplit's multilingual `sat-3l-sm`
sentence model, which is downloaded on first use. The `lines` and `tokenized`
formats continue to treat each non-empty line as one sentence.

The default is $F(0.5)$ with lowercasing, punctuation removal, and a minimum of
three observations per unordered pair. Input formats are ordinary `text`, one
untokenized sentence per `lines`, or one whitespace-tokenized sentence per
`tokenized`:

```bash
uv run word-flexibility text corpus.tok --format tokenized
```

Calculate several decay values or include pair-level results:

```bash
uv run word-flexibility text corpus.txt \
  --lambda 0.25 --lambda 0.5 --lambda 0.75 --lambda 1.0 \
  --output profile.json

uv run word-flexibility text corpus.txt --lambda 0.5 --pairs
```

## UD metrics

Download UD 2.17:

```bash
uv run python download_ud.py --target ud
```

Calculate HDE, ROE, $F(0.5)$, TTR, and pair coverage from the same sentences:

```bash
uv run word-flexibility ud ud/ud-treebanks-v2.17/UD_English-EWT --sentences 1000
```

The implementation excludes punctuation tokens, `punct` dependencies, and
dependencies headed by punctuation. Both frequency-weighted and unweighted UD
aggregations are returned.
Flexibility and TTR use integer-ID UD words; multiword-token range rows and
empty nodes are not counted as additional observations.

## Analysis

Given a CSV with `language`, `flexibility`, `ttr`,
`head_direction_entropy_weighted`, and
`relation_order_entropy_weighted`:

```bash
uv run word-flexibility analyze language_metrics.csv --output-dir analysis
```

This writes raw and TTR-controlled Pearson and Spearman correlations and
validation plots. A `lambda` column with multiple rows per language also
produces a decay-profile plot.

## Paper experiments

Install the cluster dependencies and inspect the [planit](https://github.com/WPoelman/planit) graph:

Paper plots require a TeX installation because Matplotlib uses LaTeX for their
text. Check for it before submitting:

```bash
command -v latex
latex --version
module spider texlive
```

If TeX is provided as a module, load it before submission:

```bash
module load texlive
```

Raw Wikipedia processing also requires the WTPSplit model. Cache it once in the
shared environment before submission:

```bash
uv run python -c "from word_order_flexibility.text import segment_text; print(segment_text('One. Two.'))"
```

```bash
uv sync --group cluster

uv run --group cluster python experiment.py \
  --ud-path ud/ud-treebanks-v2.17 \
  --output-dir results
```

Submit after setting the Slurm options:

```bash
uv run --group cluster python experiment.py \
  --ud-path ud/ud-treebanks-v2.17 \
  --output-dir results \
  --partition batch \
  --cluster wice \
  --account YOUR_ACCOUNT \
  --email YOUR_EMAIL \
  --submit
```

The plan runs the broad and Futrell 1,000-sentence samples and the fixed-sample
1,000/2,000/5,000/10,000-sentence study. Paper figures, analyses, and tables are
written to `results/paper`.

FLORES, Bible, and Wikipedia inputs can be included as parallel branches:

```bash
uv run --group cluster python experiment.py \
  --corpora-only \
  --flores-dir data/flores \
  --bible-dir data/taxi-1500-pre-train \
  --wikipedia-dir data/wikipedia \
  --corpus-sentences 1000 \
  --corpus-samples 10 \
  --corpus-sample-sentences 1000 \
  --output-dir results \
  --partition batch \
  --cluster wice \
  --account YOUR_ACCOUNT \
  --email YOUR_EMAIL \
  --submit
```

Every included language must contain at least the requested number of
sentences. Each language is segmented once; the same in-memory sentences are
used for the deterministic prefix and ten seeded random samples without
replacement. Raw and separately TTR-controlled correlations, sample-level
tables, rank reliability, and stability plots are written to
`results/corpus-analysis`. FLORES inputs combine its `dev` and `devtest`
splits.


## Tests

```bash
uv run pytest
uv run ruff check .
```

## Metric

For word type $a$ preceding $b$ at distance $d$, its directional count receives
weight $\lambda^d$. For an unordered pair,

```text
F_lambda(a,b) = 1 - |c_lambda(a,b) - c_lambda(b,a)|
                     / (c_lambda(a,b) + c_lambda(b,a)).
```

The corpus score is the arithmetic mean over pairs with at least three
undecayed observations. It ranges from 0 (one observed order) to 1 (balanced
orders). Setting $\lambda=1$ removes distance decay.

## Data and references

The included `data/bakker-flexibility-metrics.csv` contains the language-level
Flexibility, Consistency, and Consequence ratings used for the Bakker
comparison. These values are from:

- Dik Bakker (1998), [*Flexibility and consistency in word order patterns in
  the languages of Europe*](https://doi.org/10.1515/9783110812206.383).

The implementation and experiments additionally build on:

- Richard Futrell, Kyle Mahowald, and Edward Gibson (2015), [*Quantifying Word
  Order Freedom in Dependency Corpora*](https://aclanthology.org/W15-2112/),
  for HDE and ROE;
- Anne Chao and Tsung-Jen Shen (2003), [*Nonparametric estimation of Shannon's
  index of diversity when there are unseen species in
  sample*](https://doi.org/10.1023/A:1026096204727), for the entropy estimator;
- Aryaman Arora, Clara Meister, and Ryan Cotterell (2022), [*Estimating the
  Entropy of Linguistic
  Distributions*](https://aclanthology.org/2022.acl-short.20/), for the
  evaluation of entropy estimators on sparse linguistic data; and
- [Universal Dependencies 2.17](http://hdl.handle.net/11234/1-6036), which is
  downloaded separately and retains the licenses of its constituent
  treebanks.

Please cite the relevant original sources when using their metrics, estimator,
ratings, or data.

## License

[MIT](LICENSE)
