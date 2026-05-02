"""Pluggable tweet sources.

Three sources, all returning a uniform ``list[Tweet]``:

* :func:`demo_brand_tweets` / :func:`demo_stance_tweets` — baked-in JSON fixtures
  so the app is fully runnable offline.
* :func:`from_csv_upload` — accepts a Streamlit ``UploadedFile`` (or any
  file-like) with at least a ``text`` column.
* :func:`from_x_api` — live recent-search via the X API v2 (requires
  ``X_BEARER_TOKEN``).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from .. import config

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass
class Tweet:
    id: str
    text: str
    author: str
    created_at: datetime
    metadata: dict = field(default_factory=dict)


def _parse_dt(value) -> datetime:
    if isinstance(value, datetime):
        return value
    if value is None:
        return datetime.now(timezone.utc)
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return pd.to_datetime(value).to_pydatetime()


def _load_json(path: Path) -> list[Tweet]:
    raw = json.loads(path.read_text())
    out: list[Tweet] = []
    for r in raw:
        out.append(
            Tweet(
                id=str(r.get("id", "")),
                text=r["text"],
                author=r.get("author", "anon"),
                created_at=_parse_dt(r.get("created_at")),
                metadata=r.get("metadata", {}),
            )
        )
    return out


def demo_brand_tweets(brand: str | None = None) -> list[Tweet]:
    tweets = _load_json(DATA_DIR / "sample_brand_tweets.json")
    if brand and brand.lower() != "all":
        q = brand.lower()
        tweets = [
            t for t in tweets
            if q in t.metadata.get("brand", "").lower() or q in t.text.lower()
        ]
    return tweets


def demo_stance_tweets(topic: str | None = None) -> list[Tweet]:
    tweets = _load_json(DATA_DIR / "sample_stance_tweets.json")
    if topic and topic.lower() != "all":
        q = topic.lower()
        tweets = [t for t in tweets if q in t.metadata.get("topic", "").lower()]
    return tweets


def available_brands() -> list[str]:
    return sorted({t.metadata.get("brand", "") for t in demo_brand_tweets()})


def available_topics() -> list[str]:
    return sorted({t.metadata.get("topic", "") for t in demo_stance_tweets()})


def from_csv_upload(file) -> list[Tweet]:
    """Build tweets from any file-like CSV. Required column: ``text``."""
    df = pd.read_csv(file)
    if "text" not in df.columns:
        raise ValueError("CSV must contain a 'text' column.")
    out: list[Tweet] = []
    reserved = {"id", "text", "author", "created_at"}
    for i, row in df.iterrows():
        out.append(
            Tweet(
                id=str(row.get("id", i)),
                text=str(row["text"]),
                author=str(row.get("author", "anon")),
                created_at=_parse_dt(row.get("created_at")),
                metadata={k: row[k] for k in df.columns if k not in reserved},
            )
        )
    return out


def from_x_api(query: str, max_results: int = 50) -> list[Tweet]:
    """Recent search via X API v2. Requires ``X_BEARER_TOKEN``."""
    if not config.has_x_api():
        raise RuntimeError(
            "X_BEARER_TOKEN is not set. Add it to .env to enable live X search."
        )
    import tweepy

    client = tweepy.Client(bearer_token=config.X_BEARER_TOKEN)
    resp = client.search_recent_tweets(
        query=query,
        max_results=min(max(max_results, 10), 100),
        tweet_fields=["created_at", "author_id", "lang", "public_metrics"],
    )
    out: list[Tweet] = []
    for t in resp.data or []:
        out.append(
            Tweet(
                id=str(t.id),
                text=t.text,
                author=str(t.author_id),
                created_at=t.created_at or datetime.now(timezone.utc),
                metadata={"public_metrics": dict(t.public_metrics or {})},
            )
        )
    return out


def to_dataframe(tweets: Iterable[Tweet]) -> pd.DataFrame:
    rows = []
    for t in tweets:
        row = {
            "id": t.id,
            "author": t.author,
            "created_at": t.created_at,
            "text": t.text,
        }
        row.update(t.metadata)
        rows.append(row)
    return pd.DataFrame(rows)
