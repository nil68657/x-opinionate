"""Nuanced LLM-based sentiment that explicitly handles sarcasm and intensity.

Returns a structured JSON object per tweet so the dashboard can render
intensity bars, sarcasm flags, and theme aggregation.
"""
from __future__ import annotations

from typing import Callable

from .. import config
from ..llm_client import call_json

LABELS = config.NUANCED_SENTIMENT_LABELS

SYSTEM = f"""You are an expert in social-media opinion mining and emotion analysis.

For each tweet, you output ONE primary nuanced sentiment from this fixed set:
{', '.join(LABELS)}

Critical rules:
- Detect sarcasm/irony EXPLICITLY. A tweet can be lexically positive ("This is fine.",
  "Truly innovative.", "Cool cool cool.") but actually sarcastic/frustrated. Set
  "sarcasm": true and choose the underlying sentiment, not the surface one.
- "intensity" is 1 (mild) to 5 (extreme).
- "polarity" is the coarse 3-way roll-up (positive | negative | neutral) so we can
  benchmark you against classical models.
- Keep "reason" to ONE short sentence."""

USER_TEMPLATE = """Tweet:
\"\"\"{text}\"\"\"

Return JSON with these exact fields:
{{
  "nuanced": "<one of: {labels}>",
  "polarity": "positive" | "negative" | "neutral",
  "intensity": 1,
  "sarcasm": true,
  "emotions": ["one or two emotions"],
  "reason": "one short sentence"
}}"""


def score(text: str) -> dict:
    try:
        result = call_json(
            SYSTEM,
            USER_TEMPLATE.format(text=text, labels=", ".join(LABELS)),
            max_tokens=400,
        )
    except Exception as e:
        return {
            "nuanced": "neutral",
            "polarity": "neutral",
            "intensity": 1,
            "sarcasm": False,
            "emotions": [],
            "reason": f"LLM error: {e}",
        }
    if result.get("nuanced") not in LABELS:
        result["nuanced"] = "neutral"
    if result.get("polarity") not in {"positive", "negative", "neutral"}:
        result["polarity"] = "neutral"
    try:
        result["intensity"] = max(1, min(5, int(result.get("intensity", 1))))
    except (TypeError, ValueError):
        result["intensity"] = 1
    result["sarcasm"] = bool(result.get("sarcasm", False))
    if not isinstance(result.get("emotions"), list):
        result["emotions"] = []
    return result


def score_many(texts: list[str], progress: Callable[[float], None] | None = None) -> list[dict]:
    out: list[dict] = []
    n = max(1, len(texts))
    for i, t in enumerate(texts):
        out.append(score(t))
        if progress is not None:
            progress((i + 1) / n)
    return out
