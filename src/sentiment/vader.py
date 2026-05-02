"""VADER lexicon-based sentiment baseline."""
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()


def score(text: str) -> dict:
    s = _analyzer.polarity_scores(text)
    if s["compound"] >= 0.05:
        label = "positive"
    elif s["compound"] <= -0.05:
        label = "negative"
    else:
        label = "neutral"
    return {
        "label": label,
        "compound": s["compound"],
        "pos": s["pos"],
        "neu": s["neu"],
        "neg": s["neg"],
    }
