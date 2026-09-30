"""
Faithfulness-Under-Paraphrase (FUP) metrics.

Given word-level attributions for an original sentence and a meaning-
preserving paraphrase, we cannot assume a 1:1 token alignment (paraphrasing
changes wording). We therefore report two complementary scores:

1. Top-k salience retention: of the original's top-k most important words
   (by |attribution|), what fraction still appear among the paraphrase's
   own top-k important words (case-insensitive match)? This is the
   headline FUP score.

2. Common-word rank correlation: for the words shared verbatim (case-
   insensitive) between original and paraphrase, Spearman correlation
   between their attribution ranks in each sentence. This measures
   whether *retained* words keep a consistent relative importance.
"""

import Levenshtein
import sacrebleu
from scipy.stats import spearmanr


def _norm(w):
    return w.strip().lower()


def top_k_words(words, attrs, k):
    order = sorted(range(len(words)), key=lambda i: -abs(attrs[i]))
    return [_norm(words[i]) for i in order[:k]]


def top_k_salience_retention(orig_words, orig_attrs, para_words, para_attrs, k=5):
    k = min(k, len(orig_words), len(para_words))
    if k == 0:
        return None
    orig_top = set(top_k_words(orig_words, orig_attrs, k))
    para_top = set(top_k_words(para_words, para_attrs, k))
    if not orig_top:
        return None
    return len(orig_top & para_top) / len(orig_top)


def common_word_rank_correlation(orig_words, orig_attrs, para_words, para_attrs):
    orig_lower = [_norm(w) for w in orig_words]
    para_lower = [_norm(w) for w in para_words]
    shared = sorted(set(orig_lower) & set(para_lower))
    if len(shared) < 3:
        return None, len(shared)

    orig_rank = {w: i for i, w in enumerate(
        sorted(range(len(orig_lower)), key=lambda i: -abs(orig_attrs[i]))
    )}
    # simpler: map word -> its attribution value (first occurrence) then rank shared words
    orig_val = {}
    for w, a in zip(orig_lower, orig_attrs):
        orig_val.setdefault(w, a)
    para_val = {}
    for w, a in zip(para_lower, para_attrs):
        para_val.setdefault(w, a)

    orig_seq = [orig_val[w] for w in shared]
    para_seq = [para_val[w] for w in shared]

    if len(set(orig_seq)) < 2 or len(set(para_seq)) < 2:
        return None, len(shared)

    rho, _p = spearmanr(orig_seq, para_seq)
    return rho, len(shared)


def word_overlap_ratio(orig_words, para_words):
    """Diagnostic: lexical overlap between original and paraphrase (not an
    attribution metric, but explains low FUP scores driven by heavy
    rewording rather than explanation instability)."""
    o = set(_norm(w) for w in orig_words)
    p = set(_norm(w) for w in para_words)
    if not o:
        return 0.0
    return len(o & p) / len(o)


def normalized_edit_distance(original_text, paraphrase_text):
    """Character-level Levenshtein distance between the raw sentences,
    normalized by the longer sentence's length (0 = identical, larger =
    more surface-form change). Used as the RQ3 "degree of rephrasing"
    variable, independent of word-level attribution."""
    a, b = original_text.lower(), paraphrase_text.lower()
    if not a and not b:
        return 0.0
    return Levenshtein.distance(a, b) / max(len(a), len(b))


def bleu_score(original_text, paraphrase_text):
    """Sentence-level BLEU (0-1, treating the original as the reference)
    of the paraphrase. A second, complementary "degree of rephrasing"
    proxy for RQ3: lower BLEU means the paraphrase diverges more from the
    original's surface form."""
    return sacrebleu.sentence_bleu(paraphrase_text, [original_text]).score / 100
