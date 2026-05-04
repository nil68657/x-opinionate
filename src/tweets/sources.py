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
    is_premium: bool = False
    metadata: dict = field(default_factory=dict)

    @property
    def user_tier(self) -> str:
        return "premium" if self.is_premium else "regular"


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
                is_premium=bool(r.get("is_premium", False)),
                metadata=r.get("metadata", {}),
            )
        )
    return out


def demo_trend_tweets() -> list[Tweet]:
    return _load_json(DATA_DIR / "sample_trend_tweets.json")


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
    """Build tweets from any file-like CSV. Required column: ``text``.

    Optional columns: ``id``, ``author``, ``created_at``, ``is_premium``
    (truthy values: ``1``, ``true``, ``yes``, ``premium``, ``verified``).
    """
    df = pd.read_csv(file)
    if "text" not in df.columns:
        raise ValueError("CSV must contain a 'text' column.")
    out: list[Tweet] = []
    reserved = {"id", "text", "author", "created_at", "is_premium"}
    for i, row in df.iterrows():
        out.append(
            Tweet(
                id=str(row.get("id", i)),
                text=str(row["text"]),
                author=str(row.get("author", "anon")),
                created_at=_parse_dt(row.get("created_at")),
                is_premium=_truthy(row.get("is_premium", False)),
                metadata={k: row[k] for k in df.columns if k not in reserved},
            )
        )
    return out


def _truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "premium", "verified"}


def from_x_api(query: str, max_results: int = 50) -> list[Tweet]:
    """Recent search via X API v2. Requires ``X_BEARER_TOKEN``.

    Expands the ``author_id`` so we can read ``user.verified`` /
    ``user.verified_type`` and populate ``is_premium``.
    """
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
        expansions=["author_id"],
        user_fields=["username", "verified", "verified_type", "public_metrics"],
    )
    users_by_id = {}
    if getattr(resp, "includes", None) and resp.includes.get("users"):
        for u in resp.includes["users"]:
            users_by_id[str(u.id)] = u
    out: list[Tweet] = []
    for t in resp.data or []:
        author_id = str(t.author_id)
        user = users_by_id.get(author_id)
        is_premium = False
        if user is not None:
            verified_type = getattr(user, "verified_type", None) or ""
            is_premium = bool(getattr(user, "verified", False)) or str(verified_type).lower() in {"blue", "business", "government"}
            handle = f"@{user.username}" if getattr(user, "username", None) else author_id
        else:
            handle = author_id
        meta = {"public_metrics": dict(t.public_metrics or {})}
        if user is not None and getattr(user, "public_metrics", None):
            meta["user_metrics"] = dict(user.public_metrics)
        out.append(
            Tweet(
                id=str(t.id),
                text=t.text,
                author=handle,
                created_at=t.created_at or datetime.now(timezone.utc),
                is_premium=is_premium,
                metadata=meta,
            )
        )
    return out


def to_dataframe(tweets: Iterable[Tweet]) -> pd.DataFrame:
    rows = []
    for t in tweets:
        row = {
            "id": t.id,
            "author": t.author,
            "is_premium": t.is_premium,
            "user_tier": t.user_tier,
            "created_at": t.created_at,
            "text": t.text,
        }
        row.update(t.metadata)
        rows.append(row)
    return pd.DataFrame(rows)
