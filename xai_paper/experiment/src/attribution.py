"""
Compute token/word-level attributions with SHAP (Partition explainer via
the shap Text masker, word-level by default) and Captum's Layer Integrated
Gradients (subword-level, aggregated to words here).
"""

import numpy as np
import torch
from captum.attr import LayerIntegratedGradients

import shap


def make_shap_explainer(model, tokenizer, device="cpu"):
    from transformers import TextClassificationPipeline

    pipe = TextClassificationPipeline(
        model=model, tokenizer=tokenizer, top_k=None, device=-1
    )
    explainer = shap.Explainer(pipe)
    return explainer


def shap_word_attributions(explainer, text, target_label_name):
    """
    Returns (words, attributions) for the target class, word-level
    (shap's default Text masker splits on non-word characters).
    """
    shap_values = explainer([text])
    sv = shap_values[0]
    # sv.output_names holds the class label strings; find target column
    class_idx = list(sv.output_names).index(target_label_name)
    words = [w.strip() for w in sv.data]
    attrs = np.array([row[class_idx] for row in sv.values])
    # drop empty tokens (pure whitespace/punctuation splits) that carry no
    # word identity, keeping words/attrs aligned
    keep = [i for i, w in enumerate(words) if w]
    words = [words[i] for i in keep]
    attrs = attrs[keep]
    return words, attrs


def _aggregate_subwords(tokens, attrs):
    """Merge WordPiece continuation tokens ('##xxx') into the previous word."""
    words, word_attrs = [], []
    for tok, val in zip(tokens, attrs):
        if tok in ("[CLS]", "[SEP]", "[PAD]"):
            continue
        if tok.startswith("##") and words:
            words[-1] = words[-1] + tok[2:]
            word_attrs[-1] += val
        else:
            words.append(tok)
            word_attrs.append(val)
    return words, np.array(word_attrs)


def ig_word_attributions(model, tokenizer, text, target_label, n_steps=50):
    """Layer Integrated Gradients attribution, aggregated to whole words."""

    def forward_fn(input_ids, attention_mask):
        return model(input_ids=input_ids, attention_mask=attention_mask).logits

    lig = LayerIntegratedGradients(forward_fn, model.distilbert.embeddings.word_embeddings)

    encoded = tokenizer(text, return_tensors="pt", truncation=True)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    ref_input_ids = torch.full_like(input_ids, tokenizer.pad_token_id)
    ref_input_ids[0, 0] = input_ids[0, 0]
    ref_input_ids[0, -1] = input_ids[0, -1]

    attributions, _delta = lig.attribute(
        inputs=input_ids,
        baselines=ref_input_ids,
        additional_forward_args=(attention_mask,),
        target=target_label,
        n_steps=n_steps,
        return_convergence_delta=True,
    )
    attributions = attributions.sum(dim=-1).squeeze(0)
    norm = torch.norm(attributions)
    if norm > 0:
        attributions = attributions / norm
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    words, word_attrs = _aggregate_subwords(tokens, attributions.detach().numpy())
    return words, word_attrs
