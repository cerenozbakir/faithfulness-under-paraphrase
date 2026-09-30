"""
Generate meaning-preserving paraphrases via backtranslation.

Two pivot languages (German, French) give two paraphrases per input at
different "distances" from the original, which supports RQ3 (does
instability scale with paraphrase distance?).
"""

from transformers import MarianMTModel, MarianTokenizer

PIVOTS = {
    "de": ("Helsinki-NLP/opus-mt-en-de", "Helsinki-NLP/opus-mt-de-en"),
    "fr": ("Helsinki-NLP/opus-mt-en-fr", "Helsinki-NLP/opus-mt-fr-en"),
}

_model_cache = {}


def _load(name):
    if name not in _model_cache:
        tok = MarianTokenizer.from_pretrained(name)
        model = MarianMTModel.from_pretrained(name)
        model.eval()
        _model_cache[name] = (tok, model)
    return _model_cache[name]


def _translate(texts, model_name):
    tok, model = _load(model_name)
    batch = tok(texts, return_tensors="pt", padding=True, truncation=True)
    import torch

    with torch.no_grad():
        generated = model.generate(**batch, max_new_tokens=64)
    return [tok.decode(g, skip_special_tokens=True) for g in generated]


def backtranslate(texts, pivot="de"):
    """Round-trip texts through a pivot language: en -> pivot -> en."""
    fwd_name, back_name = PIVOTS[pivot]
    pivot_texts = _translate(texts, fwd_name)
    back_texts = _translate(pivot_texts, back_name)
    return back_texts


if __name__ == "__main__":
    samples = ["This movie was a genuinely delightful surprise from start to finish."]
    for pivot in PIVOTS:
        print(pivot, "->", backtranslate(samples, pivot=pivot))
