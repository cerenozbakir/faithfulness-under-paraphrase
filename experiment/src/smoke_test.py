"""
Offline smoke test for the FUP pipeline mechanics (SHAP calling convention,
Captum LayerIntegratedGradients calling convention, and metrics.py logic),
using a tiny randomly-initialized local model instead of downloading
DistilBERT/MarianMT from Hugging Face Hub (blocked in this sandbox).

This does NOT produce meaningful sentiment results (the model is untrained
and the "paraphraser" is a trivial word-shuffle) -- it only proves the code
paths run end-to-end without crashing, so run_pilot.py (which uses the real
pretrained models) is more likely to work correctly the first time you run
it somewhere with internet access (your machine or Google Colab).
"""

import random

import numpy as np
import torch
import torch.nn as nn

from metrics import common_word_rank_correlation, top_k_salience_retention, word_overlap_ratio

import shap
from captum.attr import LayerIntegratedGradients

torch.manual_seed(0)

SENTENCES = [
    "the movie was surprisingly delightful and fun",
    "the film was a boring and tedious waste of time",
    "a genuinely moving and beautifully acted drama",
    "an awful predictable mess with terrible acting",
]

VOCAB = sorted(set(w for s in SENTENCES for w in s.split()))
WORD2ID = {w: i + 1 for i, w in enumerate(VOCAB)}  # 0 = PAD
PAD_ID = 0
VOCAB_SIZE = len(VOCAB) + 1


class ToyClassifier(nn.Module):
    """Embedding -> mean-pool -> linear. Small enough to init randomly."""

    def __init__(self, vocab_size, emb_dim=16, n_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)
        self.linear = nn.Linear(emb_dim, n_classes)

    def forward(self, input_ids):
        emb = self.embedding(input_ids)  # (batch, seq, emb_dim)
        pooled = emb.mean(dim=1)
        return self.linear(pooled)


def encode(text, max_len=12):
    ids = [WORD2ID.get(w, 0) for w in text.split()][:max_len]
    ids += [PAD_ID] * (max_len - len(ids))
    return torch.tensor([ids])


def predict_proba(texts):
    """Plain python-string -> probability function, for SHAP."""
    model.eval()
    with torch.no_grad():
        batch = torch.cat([encode(t) for t in texts], dim=0)
        logits = model(batch)
        return torch.softmax(logits, dim=-1).numpy()


def toy_paraphrase(text):
    """Trivial stand-in for backtranslation: shuffle a couple of word pairs."""
    words = text.split()
    if len(words) > 3:
        i, j = 1, 2
        words[i], words[j] = words[j], words[i]
    return " ".join(words)


def ig_word_attributions_toy(text, target_label, n_steps=30):
    def forward_fn(input_ids):
        return model(input_ids)

    lig = LayerIntegratedGradients(forward_fn, model.embedding)
    input_ids = encode(text)
    ref_ids = torch.zeros_like(input_ids)

    attributions = lig.attribute(
        inputs=input_ids, baselines=ref_ids, target=target_label, n_steps=n_steps
    )
    attributions = attributions.sum(dim=-1).squeeze(0)
    norm = torch.norm(attributions)
    if norm > 0:
        attributions = attributions / norm
    words = text.split()
    attrs = attributions.detach().numpy()[: len(words)]
    return words, attrs


if __name__ == "__main__":
    print("1. Building toy model (randomly initialized, no downloads)...")
    model = ToyClassifier(VOCAB_SIZE)

    print("2. Testing SHAP with a plain predict_proba function...")
    masker = shap.maskers.Text(r"\W+")
    explainer = shap.Explainer(predict_proba, masker, output_names=["NEG", "POS"])
    text = SENTENCES[0]
    shap_values = explainer([text])
    sv = shap_values[0]
    print("   SHAP OK. words:", list(sv.data), "attrs[class1]:", sv.values[:, 1])

    print("3. Testing Captum LayerIntegratedGradients...")
    ig_words, ig_attrs = ig_word_attributions_toy(text, target_label=1)
    print("   Captum OK. words:", ig_words, "attrs:", ig_attrs)

    print("4. Testing metrics.py on original vs. toy-paraphrase...")
    para = toy_paraphrase(text)
    print("   original :", text)
    print("   paraphrase:", para)

    para_shap_values = explainer([para])
    para_sv = para_shap_values[0]
    para_words = list(para_sv.data)
    para_attrs = para_sv.values[:, 1]
    orig_words = list(sv.data)
    orig_attrs = sv.values[:, 1]

    overlap = word_overlap_ratio(orig_words, para_words)
    retention = top_k_salience_retention(orig_words, orig_attrs, para_words, para_attrs, k=3)
    rho, n_shared = common_word_rank_correlation(orig_words, orig_attrs, para_words, para_attrs)
    print(f"   word_overlap={overlap:.2f} top3_retention={retention} rank_corr={rho} (n_shared={n_shared})")

    print("\nAll pipeline stages ran without error.")
