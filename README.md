# Faithfulness Under Paraphrase

### Auditing the Consistency of AI Explanations in Text Classifiers

**Do explanations stay consistent when a sentence is reworded but the model's prediction stays the same?**

This term paper project compares **SHAP** and **Integrated Gradients** explanations for a DistilBERT sentiment classifier. It generates paraphrases through German and French backtranslation, retains pairs with unchanged predicted labels, and measures how the attributions change.

**296 source sentences · 575 retained pairs · 2 explanation methods**

[Read the abstract](paper/abstract.pdf) · [Explore the results](experiment/results/pilot_results.csv) · [Browse the code](experiment/src) · [Run on Kaggle](experiment/KAGGLE_TUTORIAL.md)

## Experiment at a glance

| Component | Approach |
| :--- | :--- |
| Task | Binary sentiment classification |
| Dataset | SST-2 validation split (`glue`, `sst2`) |
| Classifier | `distilbert-base-uncased-finetuned-sst-2-english` |
| Source selection | Correct predictions; 6–25 words; sampling seed 42 |
| Paraphrases | English → German/French → English using MarianMT |
| Pair filtering | Skip unchanged text and paraphrases that change the predicted label |
| Explanations | SHAP via a text classification pipeline; Captum Layer Integrated Gradients |
| Stability measures | Top-5 salience retention and shared-token attribution rank correlation |
| Rewording measures | Lexical overlap, normalized edit distance, and sentence BLEU |

The classifier is pretrained; this repository audits explanations rather than training a new sentiment model.

## Results

The committed CSV contains **575 pairs** from **296 distinct source sentences**: 287 German-pivot pairs and 288 French-pivot pairs. The values below were calculated from that CSV.

| Measure | SHAP mean | IG mean | SHAP median | IG median |
| :--- | ---: | ---: | ---: | ---: |
| Top-5 salience retention | 0.618 | 0.617 | 0.600 | 0.600 |
| Shared-token rank correlation | 0.830 | 0.795 | 0.888 | 0.858 |

Mean lexical overlap is **0.795**. Both methods retain roughly 62% of the original top-ranked token identities on average. SHAP has a higher average shared-token rank correlation in this sample. More extensive rewording is associated with lower stability, as discussed in the [paper sections](paper/paper_sections.tex).

These measures describe **attribution stability under paraphrasing**. Stability alone does not establish that an explanation faithfully captures the model's reasoning.

## Getting started

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/cerenozbakir/faithfulness-under-paraphrase.git
cd faithfulness-under-paraphrase
python -m venv .venv
```

Activate the environment:

```bash
# macOS / Linux
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

Install the dependencies and run the default 15-example pilot:

```bash
python -m pip install -r experiment/requirements.txt
python experiment/src/run_pilot.py
```

The first run needs internet access to download the classifier, dataset, and four translation models from Hugging Face. SHAP can be slow. **The current implementation runs on CPU**; enabling a GPU alone does not move the models onto it.

> **Keep the published results:** running the pipeline overwrites `experiment/results/pilot_results.csv`. Copy it elsewhere first if you want to compare a new run with the committed results. The default pilot is smaller than the published experiment.

To request a larger sample, run this from `experiment/`:

```bash
cd experiment
python -c "from src.run_pilot import main; main(n_examples=300)"
```

The requested sample size is an upper bound; the selection filters may yield fewer examples. Dependencies and model revisions are not pinned, so a new environment may require compatibility adjustments and may not reproduce the historical run exactly.

## Repository guide

| Location | Contents |
| :--- | :--- |
| [`paper/`](paper) | Abstract in PDF/Word, LaTeX body sections, and bibliography |
| [`experiment/src/`](experiment/src) | Data loading, backtranslation, attributions, metrics, and pilot runner |
| [`experiment/results/pilot_results.csv`](experiment/results/pilot_results.csv) | Recorded pair-level results |
| [`experiment/requirements.txt`](experiment/requirements.txt) | Python dependencies |
| [`experiment/KAGGLE_TUTORIAL.md`](experiment/KAGGLE_TUTORIAL.md) | Notebook setup and execution notes |

`paper_sections.tex` contains body sections intended for inclusion in a larger LaTeX document. It is not a standalone, compilable manuscript. `abstract.pdf` contains the abstract, not the full paper.

## Metric definitions and limitations

- **Top-5 retention:** select tokens by absolute attribution, normalize identities by case and whitespace, and compare the sets. The implementation reduces *k* for short inputs and uses unique identities in the denominator.
- **Rank correlation:** compute Spearman correlation of **signed attribution values** for shared identities, using the first occurrence of repeated identities. At least three shared identities and nonconstant values are required. This differs from ranking by absolute importance.
- **Surface-form effects:** synonyms can lower retention even when the underlying reasoning is similar. Backtranslation aims to preserve meaning, but the code does not independently verify semantic equivalence.
- **Method differences:** SHAP and IG use different token handling and attribution targets; the current SHAP extraction does not explicitly merge WordPiece continuations. Interpret comparisons with those differences in mind.
- **Scope:** one classifier, one task, and two translation pivots; pairs from the same source sentence are related observations.

The [offline smoke test](experiment/src/smoke_test.py) checks SHAP/Captum calls and metric execution with a randomly initialized toy model. After installing dependencies, run:

```bash
python experiment/src/smoke_test.py
```

It requires no model downloads and does not validate the scientific findings or produce meaningful sentiment predictions.

## Author

**Ceren Özbakır** · Term paper project in Explainable AI

The bibliography is available in [`paper/references.bib`](paper/references.bib).
