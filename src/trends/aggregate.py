"""Day-wise trend rollup.

Pipeline:

  tweets ─► extract_topics() ─► explode (one row per (tweet, topic))
        ─► VADER sentiment per tweet (cheap, fine for aggregate signal)
        ─► group by (day, topic, user_tier) ─► counts + mean compound
        ─► for the treemap we also need a topic-level prevailing sentiment

VADER is intentionally chosen over the LLM for the volume layer — trend
analysis cares about *aggregate* signal across many tweets, where VADER's
per-tweet noise averages out and the API cost stays at $0. The LLM is
reserved for the (optional) "Claude insights" panel that takes the
aggregate frame and produces a 3-bullet narrative.
"""
from __future__ import annotations

import re
from typing import Iterable

import pandas as pd

from ..llm_client import call_json
from ..sentiment import vader

HASHTAG_RE = re.compile(r"#(\w{2,40})")


def extract_topics(text: str) -> list[str]:
    """Return a normalised, de-duplicated list of hashtags from a tweet.

    - Case-folded so ``#ClimateBill`` and ``#climatebill`` collapse.
    - Re-prefixed with ``#`` for display.
    - Length 2-40 chars (matches the regex), Twitter's actual limit is 100.
    """
    raw = HASHTAG_RE.findall(text)
    seen: list[str] = []
    for tag in raw:
        normalised = "#" + tag.lower()
        if normalised not in seen:
            seen.append(normalised)
    return seen


def explode_tweets(tweets) -> pd.DataFrame:
    """Long-format: one row per (tweet, topic). Tweets with zero hashtags are dropped."""
    rows = []
    for t in tweets:
        topics = extract_topics(t.text)
        if not topics:
            continue
        v = vader.score(t.text)
        for topic in topics:
            rows.append({
                "tweet_id": t.id,
                "author": t.author,
                "created_at": t.created_at,
                "day": pd.to_datetime(t.created_at).normalize(),
                "topic": topic,
                "is_premium": t.is_premium,
                "user_tier": t.user_tier,
                "compound": v["compound"],
                "polarity": v["label"],
                "text": t.text,
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["day"] = pd.to_datetime(df["day"])
    return df


def topic_day_rollup(exploded: pd.DataFrame) -> pd.DataFrame:
    """One row per (day, topic). Used for the daily treemap and time series."""
    if exploded.empty:
        return exploded
    g = exploded.groupby(["day", "topic"], as_index=False).agg(
        count=("tweet_id", "count"),
        premium_count=("is_premium", "sum"),
        mean_compound=("compound", "mean"),
        positive=("polarity", lambda s: (s == "positive").sum()),
        neutral=("polarity", lambda s: (s == "neutral").sum()),
        negative=("polarity", lambda s: (s == "negative").sum()),
    )
    g["regular_count"] = g["count"] - g["premium_count"]
    g["premium_share"] = (g["premium_count"] / g["count"]).round(3)
    g["prevailing_sentiment"] = g["mean_compound"].apply(_label_compound)
    return g


def topic_day_tier_rollup(exploded: pd.DataFrame) -> pd.DataFrame:
    """One row per (day, topic, user_tier). Powers the nested treemap."""
    if exploded.empty:
        return exploded
    g = exploded.groupby(["day", "topic", "user_tier"], as_index=False).agg(
        count=("tweet_id", "count"),
        mean_compound=("compound", "mean"),
    )
    return g


def top_topics_for_day(rollup: pd.DataFrame, day, n: int = 10) -> pd.DataFrame:
    day = pd.to_datetime(day).normalize()
    if rollup.empty:
        return rollup
    sub = rollup[rollup["day"] == day].sort_values("count", ascending=False).head(n)
    return sub.reset_index(drop=True)


def _label_compound(c: float) -> str:
    if c >= 0.05:
        return "positive"
    if c <= -0.05:
        return "negative"
    return "neutral"


def daily_volume_by_topic(rollup: pd.DataFrame) -> pd.DataFrame:
    """Wide format suitable for an area chart: index=day, columns=topic, values=count."""
    if rollup.empty:
        return rollup
    return (
        rollup.pivot_table(
            index="day", columns="topic", values="count", aggfunc="sum"
        )
        .fillna(0)
        .astype(int)
    )


# ---------------------------------------------------------------------------
# Optional LLM narrative on top of the aggregate.
# ---------------------------------------------------------------------------

_INSIGHT_SYSTEM = """You are a media-trends analyst writing a short daily brief.
Given a per-topic rollup for one day (volume, premium share, prevailing
sentiment), produce 3-5 short, specific bullets about what's happening.

Rules:
- Be concrete. Reference specific topics by their hashtag.
- Call out asymmetries: 'mostly premium-user driven', 'overwhelmingly
  negative', 'volume spike vs prior days', etc.
- No hedging filler ('it seems', 'perhaps').
- 1 sentence per bullet, max ~25 words."""

_INSIGHT_USER = """Date: {day}

Top topics today:
{rows}

Return JSON:
{{
  "headline": "one-sentence summary of the day",
  "bullets": ["...", "...", "..."]
}}"""


def llm_daily_insights(rollup: pd.DataFrame, day) -> dict:
    sub = top_topics_for_day(rollup, day, n=8)
    if sub.empty:
        return {"headline": "No tweets on this day.", "bullets": []}
    rows = "\n".join(
        f"- {r.topic}: {r['count']} tweets, "
        f"{int(r.premium_count)} premium / {int(r.regular_count)} regular, "
        f"prevailing {r.prevailing_sentiment} (mean compound {r.mean_compound:+.2f})"
        for r in sub.itertuples()
    )
    try:
        result = call_json(
            _INSIGHT_SYSTEM,
            _INSIGHT_USER.format(day=pd.to_datetime(day).date(), rows=rows),
            max_tokens=500,
        )
    except Exception as e:
        return {"headline": f"LLM error: {e}", "bullets": []}
    if not isinstance(result.get("bullets"), list):
        result["bullets"] = []
    return result
