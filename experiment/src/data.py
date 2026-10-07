"""
Load SST-2 and select a sample of inputs the classifier gets right.

We use the publicly available DistilBERT checkpoint fine-tuned on SST-2
(distilbert-base-uncased-finetuned-sst-2-english), rather than fine-tuning
from scratch, since the paper's contribution is the faithfulness-under-
paraphrase (FUP) audit, not the classifier itself. This is a standard,
citable, off-the-shelf checkpoint built on Sanh et al. (2019)'s DistilBERT.
"""

import random

from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"


def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return model, tokenizer


def load_sst2_validation():
    """Returns the SST-2 validation split (sentence, label)."""
    ds = load_dataset("glue", "sst2", split="validation")
    return ds


def select_correctly_classified(model, tokenizer, ds, n=30, seed=42, min_words=6, max_words=25):
    """
    Sample `n` SST-2 validation examples that:
      - the model classifies correctly,
      - have a moderate length (long enough to paraphrase meaningfully,
        short enough to keep attribution/backtranslation fast on CPU).

    Returns a list of dicts: {idx, text, true_label, pred_label, pred_prob}.
    """
    import torch

    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    selected = []
    for idx in indices:
        if len(selected) >= n:
            break
        example = ds[idx]
        text = example["sentence"].strip()
        n_words = len(text.split())
        if not (min_words <= n_words <= max_words):
            continue

        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        pred_label = int(torch.argmax(probs).item())
        pred_prob = float(probs[pred_label].item())

        if pred_label == example["label"]:
            selected.append(
                {
                    "idx": idx,
                    "text": text,
                    "true_label": int(example["label"]),
                    "pred_label": pred_label,
                    "pred_prob": pred_prob,
                }
            )

    return selected


if __name__ == "__main__":
    model, tokenizer = load_model_and_tokenizer()
    ds = load_sst2_validation()
    examples = select_correctly_classified(model, tokenizer, ds, n=5)
    for ex in examples:
        print(ex)
