"""Premium-vs-regular word clouds: comparison, single-tier, and layered overlay.

Four views, each answering a slightly different question:

* **comparison** — one canvas, every word coloured by which tier uses it more.
  Shows asymmetries at a glance ("layoffs is gray-side; cloture is blue-side").
* **premium / regular** — single-tier clouds for direct inspection.
* **overlay** — two semi-transparent clouds layered on the same axes for the
  literal "superimposed" view the user asked for.

VADER and the LLM are not used here — this is purely lexical frequency.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

import matplotlib.pyplot as plt
from wordcloud import STOPWORDS, WordCloud

URL_RE = re.compile(r"https?://\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z']{2,}")

# Filler words that drown out signal in tweet-sized text.
EXTRA_STOP = {
    "like", "just", "really", "today", "every", "still", "going", "actually",
    "honestly", "thing", "things", "people", "even", "much", "many", "way",
    "got", "get", "yes", "yeah", "okay", "ok", "one", "two", "amp", "rt",
}
STOPS = set(STOPWORDS) | {s.lower() for s in EXTRA_STOP}

# Premium and regular brand colours used across the dashboard.
PREMIUM_COLOUR = "#1f6feb"
REGULAR_COLOUR = "#5a5a5a"
MIXED_COLOUR = "#a87adb"


def tokenize(text: str) -> list[str]:
    """Strip URLs / mentions / the leading ``#``, lowercase, drop stop-words."""
    text = URL_RE.sub("", text)
    text = MENTION_RE.sub("", text)
    text = HASHTAG_RE.sub(r"\1", text)  # keep word, drop the hash
    tokens = [t.lower() for t in TOKEN_RE.findall(text)]
    return [t for t in tokens if t not in STOPS]


def tier_word_counts(tweets: Iterable) -> tuple[Counter, Counter]:
    """Return ``(premium_counts, regular_counts)`` over the tweet stream."""
    premium: Counter = Counter()
    regular: Counter = Counter()
    for t in tweets:
        bucket = premium if t.is_premium else regular
        bucket.update(tokenize(t.text))
    return premium, regular


def comparison_cloud(
    premium: Counter,
    regular: Counter,
    max_words: int = 100,
    width: int = 1000,
    height: int = 480,
) -> WordCloud | None:
    """One cloud whose colour encodes premium-vs-regular dominance per word."""
    combined: Counter = Counter()
    combined.update(premium)
    combined.update(regular)
    if not combined:
        return None

    def color_func(word, **_kwargs):
        pn = premium.get(word, 0)
        rn = regular.get(word, 0)
        total = pn + rn
        if total == 0:
            return MIXED_COLOUR
        ratio = (pn - rn) / total  # -1 (all regular) to +1 (all premium)
        if ratio > 0.5:
            return PREMIUM_COLOUR
        if ratio > 0.15:
            return "#5a8de8"  # leans premium
        if ratio < -0.5:
            return REGULAR_COLOUR
        if ratio < -0.15:
            return "#888888"  # leans regular
        return MIXED_COLOUR

    return WordCloud(
        width=width, height=height,
        background_color="white",
        max_words=max_words,
        color_func=color_func,
        prefer_horizontal=0.9,
        relative_scaling=0.4,
    ).generate_from_frequencies(dict(combined))


def single_tier_cloud(
    counts: Counter,
    colour: str,
    max_words: int = 80,
    width: int = 1000,
    height: int = 400,
) -> WordCloud | None:
    if not counts:
        return None
    return WordCloud(
        width=width, height=height,
        background_color="white",
        max_words=max_words,
        color_func=lambda *_a, **_kw: colour,
        prefer_horizontal=0.9,
        relative_scaling=0.4,
    ).generate_from_frequencies(dict(counts))


def overlay_figure(
    premium: Counter,
    regular: Counter,
    max_words: int = 80,
) -> plt.Figure:
    """Render both clouds with alpha onto a single matplotlib figure."""
    fig, ax = plt.subplots(figsize=(10, 5), dpi=120)
    ax.set_axis_off()

    if regular:
        rc = WordCloud(
            width=1000, height=500, background_color=None, mode="RGBA",
            max_words=max_words,
            color_func=lambda *_a, **_kw: REGULAR_COLOUR,
            prefer_horizontal=0.9,
            relative_scaling=0.4,
        ).generate_from_frequencies(dict(regular))
        ax.imshow(rc.to_array(), interpolation="bilinear", alpha=0.55)

    if premium:
        pc = WordCloud(
            width=1000, height=500, background_color=None, mode="RGBA",
            max_words=max_words,
            color_func=lambda *_a, **_kw: PREMIUM_COLOUR,
            prefer_horizontal=0.9,
            relative_scaling=0.4,
        ).generate_from_frequencies(dict(premium))
        ax.imshow(pc.to_array(), interpolation="bilinear", alpha=0.55)

    fig.tight_layout(pad=0)
    return fig


def top_distinctive(
    a: Counter, b: Counter, n: int = 10, min_total: int = 2
) -> list[tuple[str, int, int]]:
    """Words most over-represented in ``a`` vs ``b``.

    Score = ``a[word] - b[word]`` (raw lift). Filtered by ``total >= min_total``.
    Returns ``[(word, a_count, b_count), ...]`` sorted by lift descending.
    """
    seen = set(a) | set(b)
    rows = []
    for w in seen:
        ac, bc = a.get(w, 0), b.get(w, 0)
        if ac + bc < min_total:
            continue
        rows.append((w, ac, bc, ac - bc))
    rows.sort(key=lambda r: r[3], reverse=True)
    return [(w, ac, bc) for w, ac, bc, _ in rows[:n]]
