"""Few-shot, chain-of-thought stance detection.

Returns ``stance``, ``confidence``, and a short ``reasoning`` trace for each
tweet so the dashboard can display the model's justification.
"""
from __future__ import annotations

from typing import Callable

import pandas as pd

from .. import config
from ..llm_client import call_json

LABELS = config.STANCE_LABELS

SYSTEM = """You are an expert political/social analyst performing stance detection
on social-media posts.

For a given TOPIC and TWEET, decide whether the tweet is FOR, AGAINST, or NEUTRAL
toward the topic. Use this reasoning chain internally before answering:

  1. Identify the precise claim implied by the topic.
  2. Identify what the tweet asserts or implies about that claim.
  3. Detect rhetoric (sarcasm, rhetorical question, dog whistle) that might
     flip the surface sentiment.
  4. Pick the stance best supported by the evidence in the tweet itself —
     not by your own opinion of the topic.

Few-shot examples:

TOPIC: "Remote work should remain the default for tech jobs"
TWEET: "Five days in the office is a productivity tax disguised as culture."
STANCE: for
REASON: Frames mandatory office time negatively, supporting remote-as-default.

TOPIC: "Remote work should remain the default for tech jobs"
TWEET: "Nothing replaces the energy of being in the same room shipping together."
STANCE: against
REASON: Endorses in-person collaboration over remote work.

TOPIC: "Remote work should remain the default for tech jobs"
TWEET: "Both hybrid and remote can work; depends on the team and project phase."
STANCE: neutral
REASON: Acknowledges trade-offs without favoring either side.

TOPIC: "Cities should ban new gas-powered car sales by 2035"
TWEET: "EVs aren't ready and the grid isn't either. Forcing this on commuters is irresponsible."
STANCE: against
REASON: Argues the ban is premature and harmful to commuters."""

USER_TEMPLATE = """TOPIC: "{topic}"
TWEET: \"\"\"{text}\"\"\"

Return JSON:
{{
  "stance": "for" | "against" | "neutral",
  "confidence": 0.0-1.0,
  "reasoning": "step-by-step justification, 2-4 sentences"
}}"""


def classify(topic: str, text: str) -> dict:
    try:
        result = call_json(
            SYSTEM,
            USER_TEMPLATE.format(topic=topic, text=text),
            max_tokens=500,
        )
    except Exception as e:
        return {"stance": "neutral", "confidence": 0.0,
                "reasoning": f"LLM error: {e}"}
    if result.get("stance") not in LABELS:
        result["stance"] = "neutral"
    try:
        result["confidence"] = float(result.get("confidence", 0.0))
    except (TypeError, ValueError):
        result["confidence"] = 0.0
    return result


def classify_many(
    topic: str,
    tweets,
    progress: Callable[[float], None] | None = None,
) -> pd.DataFrame:
    rows = []
    n = max(1, len(tweets))
    for i, t in enumerate(tweets):
        text = getattr(t, "text", str(t))
        author = getattr(t, "author", "anon")
        created_at = getattr(t, "created_at", None)
        result = classify(topic, text)
        rows.append({
            "created_at": created_at,
            "author": author,
            "text": text,
            "stance": result["stance"],
            "confidence": round(result["confidence"], 3),
            "reasoning": result["reasoning"],
        })
        if progress is not None:
            progress((i + 1) / n)
    df = pd.DataFrame(rows)
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"])
    return df


def stance_over_time(df: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    """Aggregate counts of each stance per time bucket for the evolution chart."""
    if df.empty:
        return df
    work = df.copy()
    work["created_at"] = pd.to_datetime(work["created_at"])
    work = work.set_index("created_at")
    counts = (
        work.groupby([pd.Grouper(freq=freq), "stance"]).size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for col in LABELS:
        if col not in counts.columns:
            counts[col] = 0
    return counts[["created_at", *LABELS]]
