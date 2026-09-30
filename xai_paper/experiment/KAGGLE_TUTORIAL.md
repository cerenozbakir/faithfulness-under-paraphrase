# Notes on Running This on Kaggle

I ran this experiment in a [Kaggle Notebook](https://kaggle.com/code)
rather than locally, since it gives free internet access and an optional
GPU without needing to set anything up on my own machine. These are notes
on how I set it up and what each part of the pipeline does, for anyone
reading this repo.

## Setup

Kaggle notebooks need two settings turned on in the Settings panel before
anything below will work: **Internet** (off by default — needed for every
Hugging Face download here) and, optionally, a **GPU accelerator** (free
weekly quota, speeds up Integrated Gradients and backtranslation; SHAP
runs on CPU regardless).

## Hugging Face resources this fetches

All public, no account or API key needed:

| Resource | Identifier | Used for |
|---|---|---|
| Classifier | `distilbert-base-uncased-finetuned-sst-2-english` | the black-box model I'm explaining |
| Dataset | `glue` / `sst2` | labeled sentences to test on |
| Translation (DE pivot) | `Helsinki-NLP/opus-mt-en-de`, `opus-mt-de-en` | paraphrase generation |
| Translation (FR pivot) | `Helsinki-NLP/opus-mt-en-fr`, `opus-mt-fr-en` | a second, differently-worded paraphrase |

## What each part of the code does

### Loading the classifier

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()
```

I used this off-the-shelf DistilBERT checkpoint instead of fine-tuning my
own, since the paper's contribution is the faithfulness audit, not the
classifier — reusing a citable, well-known checkpoint keeps that part of
the pipeline uncontroversial.

### Selecting examples

```python
sst2 = load_dataset("glue", "sst2", split="validation")
# ... shuffle with a fixed seed, keep sentences of 6-25 words,
# keep only ones the model classifies correctly
```

I only keep predictions the model gets *right*, since "does the
explanation stay faithful" only makes sense to ask about a correct
prediction. The word-count filter avoids sentences too short to
paraphrase meaningfully or too long to process quickly. I fixed the
random seed so the sample is reproducible.

### Generating paraphrases (`src/paraphrase.py`)

```python
def backtranslate(text):
    # English -> German -> English (or -> French -> English)
    ...
```

I used backtranslation (translate out and back) instead of a dedicated
paraphrase model or hand-written rewrites, because it's a cheap,
well-established way to get meaning-preserving rewordings without
introducing my own bias into how the paraphrases are written. Two pivot
languages give two independently-worded paraphrases per sentence, which
supports RQ3 (does instability scale with how different the paraphrase
is).

### SHAP attributions (`src/attribution.py`)

```python
pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, top_k=None)
explainer = shap.Explainer(pipe)
shap_values = explainer([text])
```

`shap.Explainer` auto-selects a Partition explainer for text pipelines,
which is efficient because it recursively splits the sentence rather than
checking every possible word subset. It gives word-level attributions by
default, which is convenient for comparing against IG later.

### Integrated Gradients attributions

```python
lig = LayerIntegratedGradients(forward_fn, model.distilbert.embeddings.word_embeddings)
attributions = lig.attribute(inputs=input_ids, baselines=ref_input_ids,
                              target=target_label, n_steps=50)
```

I attributed against the embedding layer (gradients need continuous
inputs, not token IDs), with an all-`[PAD]` baseline representing "no
information." IG works at the subword level, so `attribution.py` merges
`##`-prefixed continuation tokens back into whole words before comparing
against SHAP's word-level output.

### Measuring faithfulness (`src/metrics.py`)

I can't assume a clean 1:1 word alignment between an original sentence
and its paraphrase, so I use two scores instead of a direct comparison:
top-k salience retention (does a word that mattered most in the original
still matter most in the paraphrase?) and rank correlation restricted to
words shared by both sentences. See the module docstring for the full
reasoning.

### Running it end to end

```bash
python src/run_pilot.py
```

Loops the above over `n_examples` sentences and both pivots, writes
`results/pilot_results.csv`. I attached this repo to the Kaggle notebook
as a Dataset input (Kaggle's way of bringing in outside files), copied it
into the writable `/kaggle/working/` directory, and ran this from there.

## Status

Pipeline mechanics verified offline via `smoke_test.py` (no internet
needed — checks the SHAP/Captum calling conventions and metrics logic
against a tiny toy model). Haven't yet run the real pipeline against the
pretrained models — that's next.
