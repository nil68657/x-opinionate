"""Optional RoBERTa baseline (cardiffnlp/twitter-roberta-base-sentiment-latest).

Lazy-loaded. If ``transformers``/``torch`` are not installed, ``is_available()``
returns ``False`` and the UI hides the column.
"""
from functools import lru_cache

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
LABELS = ["negative", "neutral", "positive"]


def is_available() -> bool:
    try:
        import transformers  # noqa: F401
        import torch  # noqa: F401
        return True
    except Exception:
        return False


@lru_cache(maxsize=1)
def _load():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return tok, model


def score(text: str) -> dict:
    import torch

    tok, model = _load()
    enc = tok(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        logits = model(**enc).logits[0]
    probs = torch.softmax(logits, dim=-1).tolist()
    idx = int(max(range(3), key=lambda i: probs[i]))
    return {
        "label": LABELS[idx],
        "negative": probs[0],
        "neutral": probs[1],
        "positive": probs[2],
    }
