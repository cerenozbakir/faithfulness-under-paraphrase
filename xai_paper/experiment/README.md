# Faithfulness Under Paraphrase

Code for my term paper *"Faithfulness Under Paraphrase: Auditing the
Consistency of AI Explanations in Text Classifiers"*.

## What this measures

For a set of SST-2 sentences the classifier gets right, I generate
meaning-preserving paraphrases (backtranslation via German and French
pivots), keep only the paraphrases where the model's predicted label is
unchanged, and compare SHAP and Integrated Gradients word attributions
between each original/paraphrase pair using two scores:

- **Top-k salience retention** (headline FUP score): of the original's
  top-5 most important words, what fraction are still top-5 important in
  the paraphrase?
- **Common-word rank correlation**: for words present in both sentences,
  Spearman correlation between their attribution ranks.

See `src/metrics.py` for the exact definitions and rationale.

## Setup

```bash
pip install -r requirements.txt
```

**This needs normal internet access** to download from Hugging Face Hub:
the `distilbert-base-uncased-finetuned-sst-2-english` classifier, the
`glue/sst2` dataset, and the `Helsinki-NLP/opus-mt-*` backtranslation
models (~1.5GB total, one-time download, cached after that).

I run this in a [Kaggle Notebook](https://kaggle.com/code) (free, has
internet + optional GPU once enabled in the notebook's Settings panel) —
see `KAGGLE_TUTORIAL.md` for notes on how I set it up and what each part
of the code does. It will NOT run in network-sandboxed environments that
block huggingface.co.

## Running

```bash
python3 src/run_pilot.py
```

`n_examples` in `src/run_pilot.py::main()` controls how many examples get
sampled x 2 paraphrase pivots; the results in `results/pilot_results.csv`
are from a full run (296 correctly-classified sentences survived) on a
Kaggle GPU session — SHAP is the slow part, so a CPU-only run takes
noticeably longer.

## Files

- `src/data.py` — loads SST-2 validation set + the pretrained classifier,
  selects correctly-classified examples.
- `src/paraphrase.py` — backtranslation paraphrasing (DE and FR pivots).
- `src/attribution.py` — SHAP (word-level, via the Partition explainer)
  and Captum Layer Integrated Gradients (subword-level, aggregated to
  words) attribution computation.
- `src/metrics.py` — the FUP scores described above, plus normalized
  edit distance and BLEU (degree-of-rephrasing variables for RQ3).
- `src/run_pilot.py` — ties it all together, writes
  `results/pilot_results.csv`.
- `src/smoke_test.py` — an offline sanity check of the SHAP/Captum calling
  conventions and the metrics logic using a tiny randomly-initialized toy
  model (no internet required). Useful for debugging without waiting on
  downloads; does not produce meaningful sentiment results.

## Status

Ran the full pipeline on Kaggle against the real pretrained models:
296 correctly-classified SST-2 sentences, both backtranslation pivots,
giving 575 original/paraphrase pairs after dropping the pairs where the
paraphrase flipped the model's predicted label (17 of a possible 592,
~3%). Results are in `results/pilot_results.csv`.

Headline numbers (mean over all 575 pairs):

- word overlap between original and paraphrase: 0.79
- SHAP top-5 salience retention: 0.62 (median 0.60)
- IG top-5 salience retention: 0.62 (median 0.60) — essentially tied
  with SHAP; a Wilcoxon signed-rank test finds no significant
  difference (p = 0.81)
- SHAP common-word rank correlation: 0.83 (median 0.89)
- IG common-word rank correlation: 0.80 (median 0.86) — here SHAP is
  significantly *more* stable than IG (Wilcoxon p < 0.001)

Degree of rephrasing (RQ3): I compute normalized edit distance and BLEU
between each original/paraphrase pair (`src/metrics.py`) and correlate
them with the FUP scores. Both agree: more rewording -> lower FUP
(edit distance r ≈ -0.33 to -0.58 across the four FUP scores; BLEU
r ≈ +0.34 to +0.57 — BLEU is a similarity score, so the signs are
consistent). This confirms instability scales with how much the
paraphrase diverges from the original, which I flag as a limitation:
some of the measured "unfaithfulness" is confounded with surface-form
change rather than reflecting explanation instability on its own.

One thing worth noting for anyone re-running a small pilot first: an
earlier 15-example pilot suggested IG was clearly more paraphrase-stable
than SHAP. At full scale that gap disappears on top-5 retention and
reverses on rank correlation — a reminder not to draw conclusions from
n=14 pairs.
