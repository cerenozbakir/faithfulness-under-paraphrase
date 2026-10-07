"""
End-to-end pilot run of the Faithfulness-Under-Paraphrase (FUP) experiment.

For each sampled SST-2 example that the classifier gets right:
  1. Generate two paraphrases via backtranslation (German pivot, French pivot).
  2. Keep only paraphrases where the model's predicted label is unchanged.
  3. Compute SHAP and Integrated Gradients word attributions for the
     original and each surviving paraphrase.
  4. Compute FUP scores (top-k salience retention, common-word rank
     correlation) for each method, each pivot, plus two "degree of
     rephrasing" variables (normalized edit distance, BLEU) used to test
     whether instability scales with how much the paraphrase diverges
     from the original (RQ3).

Writes results/pilot_results.csv and prints a summary.
"""

import os
import sys
import time

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from attribution import ig_word_attributions, make_shap_explainer, shap_word_attributions
from data import load_model_and_tokenizer, load_sst2_validation, select_correctly_classified
from metrics import (
    bleu_score,
    common_word_rank_correlation,
    normalized_edit_distance,
    top_k_salience_retention,
    word_overlap_ratio,
)
from paraphrase import backtranslate

LABEL_NAMES = {0: "NEGATIVE", 1: "POSITIVE"}
TOP_K = 5


def main(n_examples=15, pivots=("de", "fr")):
    t0 = time.time()
    print("Loading model/tokenizer...")
    model, tokenizer = load_model_and_tokenizer()

    print("Loading SST-2 validation set...")
    ds = load_sst2_validation()

    print(f"Selecting {n_examples} correctly-classified examples...")
    examples = select_correctly_classified(model, tokenizer, ds, n=n_examples)
    print(f"  got {len(examples)} examples")

    print("Building SHAP explainer...")
    explainer = make_shap_explainer(model, tokenizer)

    rows = []
    for i, ex in enumerate(examples):
        text = ex["text"]
        pred_label = ex["pred_label"]
        pred_name = LABEL_NAMES[pred_label]
        print(f"[{i+1}/{len(examples)}] {text!r} (pred={pred_name})")

        # --- original attributions ---
        orig_shap_words, orig_shap_attrs = shap_word_attributions(explainer, text, pred_name)
        orig_ig_words, orig_ig_attrs = ig_word_attributions(model, tokenizer, text, pred_label)

        for pivot in pivots:
            try:
                paraphrase = backtranslate([text], pivot=pivot)[0]
            except Exception as e:
                print(f"  [paraphrase error, pivot={pivot}]: {e}")
                continue

            if paraphrase.strip().lower() == text.strip().lower():
                print(f"  pivot={pivot}: paraphrase identical to original, skipping")
                continue

            # check predicted label is unchanged on the paraphrase
            inputs = tokenizer(paraphrase, return_tensors="pt", truncation=True)
            with torch.no_grad():
                logits = model(**inputs).logits
            para_pred = int(torch.argmax(logits, dim=-1).item())

            if para_pred != pred_label:
                print(f"  pivot={pivot}: label flipped ({LABEL_NAMES[para_pred]}), skipping")
                continue

            para_shap_words, para_shap_attrs = shap_word_attributions(explainer, paraphrase, pred_name)
            para_ig_words, para_ig_attrs = ig_word_attributions(model, tokenizer, paraphrase, pred_label)

            overlap = word_overlap_ratio(orig_shap_words, para_shap_words)
            edit_dist = normalized_edit_distance(text, paraphrase)
            bleu = bleu_score(text, paraphrase)

            shap_fup = top_k_salience_retention(orig_shap_words, orig_shap_attrs, para_shap_words, para_shap_attrs, k=TOP_K)
            shap_rho, shap_n_shared = common_word_rank_correlation(orig_shap_words, orig_shap_attrs, para_shap_words, para_shap_attrs)

            ig_fup = top_k_salience_retention(orig_ig_words, orig_ig_attrs, para_ig_words, para_ig_attrs, k=TOP_K)
            ig_rho, ig_n_shared = common_word_rank_correlation(orig_ig_words, orig_ig_attrs, para_ig_words, para_ig_attrs)

            rows.append({
                "idx": ex["idx"],
                "pivot": pivot,
                "original": text,
                "paraphrase": paraphrase,
                "pred_label": pred_name,
                "pred_prob": ex["pred_prob"],
                "correct": True,
                "word_overlap": overlap,
                "norm_edit_dist": edit_dist,
                "bleu": bleu,
                "shap_top5_retention": shap_fup,
                "shap_rank_corr": shap_rho,
                "shap_n_shared_words": shap_n_shared,
                "ig_top5_retention": ig_fup,
                "ig_rank_corr": ig_rho,
                "ig_n_shared_words": ig_n_shared,
            })
            print(f"  pivot={pivot}: overlap={overlap:.2f} "
                  f"SHAP(top5={shap_fup}, rho={shap_rho}) "
                  f"IG(top5={ig_fup}, rho={ig_rho})")

    df = pd.DataFrame(rows)
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "pilot_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows to {out_path}")
    print(f"Total time: {time.time() - t0:.1f}s")

    if len(df):
        print("\n=== Summary (mean over paraphrase pairs) ===")
        print(df[["word_overlap", "shap_top5_retention", "shap_rank_corr",
                   "ig_top5_retention", "ig_rank_corr"]].mean(numeric_only=True))

    return df


if __name__ == "__main__":
    main(n_examples=15)
