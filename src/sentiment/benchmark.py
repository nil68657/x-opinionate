"""Run multiple sentiment models on the same tweets and produce a comparison frame.

The frame is intentionally flat (one row per tweet) so it renders directly in
``st.dataframe`` and feeds the agreement matrix.
"""
from __future__ import annotations

from typing import Callable, Sequence

import pandas as pd

from . import llm, vader

try:
    from . import roberta
    _HAS_ROBERTA = roberta.is_available()
except Exception:
    _HAS_ROBERTA = False


def has_roberta() -> bool:
    return _HAS_ROBERTA


def run_benchmark(
    tweets: Sequence,
    progress: Callable[[float], None] | None = None,
    use_roberta: bool = True,
) -> pd.DataFrame:
    rows = []
    n = max(1, len(tweets))
    use_roberta = use_roberta and _HAS_ROBERTA
    for i, t in enumerate(tweets):
        text = getattr(t, "text", t)
        v = vader.score(text)
        l = llm.score(text)
        row = {
            "text": text,
            "vader_label": v["label"],
            "vader_compound": round(v["compound"], 3),
            "llm_polarity": l["polarity"],
            "llm_nuanced": l["nuanced"],
            "llm_intensity": l["intensity"],
            "llm_sarcasm": l["sarcasm"],
            "llm_reason": l.get("reason", ""),
        }
        if use_roberta:
            r = roberta.score(text)
            row["roberta_label"] = r["label"]
            row["roberta_pos"] = round(r["positive"], 3)
            row["roberta_neg"] = round(r["negative"], 3)
        rows.append(row)
        if progress is not None:
            progress((i + 1) / n)
    return pd.DataFrame(rows)


def agreement_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise % agreement between models on the 3-way polarity."""
    cols: dict[str, str] = {"vader": "vader_label", "llm": "llm_polarity"}
    if "roberta_label" in df.columns:
        cols["roberta"] = "roberta_label"
    keys = list(cols.keys())
    mat = pd.DataFrame(index=keys, columns=keys, dtype=float)
    for a in keys:
        for b in keys:
            mat.loc[a, b] = round((df[cols[a]] == df[cols[b]]).mean() * 100, 1)
    return mat


def disagreement_examples(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Tweets where VADER and the LLM disagree — the most interesting rows."""
    mask = df["vader_label"] != df["llm_polarity"]
    cols = ["text", "vader_label", "llm_polarity", "llm_nuanced",
            "llm_sarcasm", "llm_reason"]
    if "roberta_label" in df.columns:
        cols.insert(3, "roberta_label")
    return df.loc[mask, cols].head(n).reset_index(drop=True)
