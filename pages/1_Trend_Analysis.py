"""Day-wise trend analysis with sentiment-coloured treemap + multi-day area chart."""
from __future__ import annotations

import io

import pandas as pd
import plotly.express as px
import streamlit as st

from src import config
from src.tweets import sources
from src.trends import aggregate

st.set_page_config(page_title="Trend Analysis · x-opinionate", page_icon="📈", layout="wide")

st.title("📈 Trend Analysis")
st.caption(
    "Day-wise rollup of trending hashtags, coloured by prevailing sentiment, "
    "split by premium vs regular users."
)

# ---------------------------------------------------------------------------
# Sidebar: data source.
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Data source")
    source_options = ["Demo data", "Upload CSV"]
    if config.has_x_api():
        source_options.append("Live X API")
    src = st.radio("Source", source_options, index=0)

    tweets: list = []
    if src == "Demo data":
        tweets = sources.demo_trend_tweets()
    elif src == "Upload CSV":
        upload = st.file_uploader(
            "CSV with `text` column (optional: `created_at`, `is_premium`)",
            type=["csv"],
        )
        if upload is not None:
            tweets = sources.from_csv_upload(upload)
    else:
        query = st.text_input(
            "X recent-search query",
            value='(#AIRegulation OR #ClimateBill OR #TechLayoffs OR #WorldCup) -is:retweet lang:en',
        )
        n = st.slider("Max tweets", 10, 100, 100, step=10)
        if st.button("Fetch from X"):
            with st.spinner("Querying X API…"):
                tweets = sources.from_x_api(query, max_results=n)

    st.divider()
    st.header("Display")
    top_n = st.slider("Top N topics per day (treemap)", 3, 12, 8)
    show_insights = st.checkbox(
        "Generate Claude daily-insights brief",
        value=False,
        disabled=not config.has_anthropic(),
        help=("Requires ANTHROPIC_API_KEY." if not config.has_anthropic()
              else "One LLM call per selected day."),
    )

if not tweets:
    st.info("Pick a data source on the left to get started.")
    st.stop()

# ---------------------------------------------------------------------------
# Build aggregates.
# ---------------------------------------------------------------------------
with st.spinner("Extracting hashtags + scoring sentiment (VADER)…"):
    exploded = aggregate.explode_tweets(tweets)

if exploded.empty:
    st.warning(
        "No hashtags found in any of the loaded tweets. The trend page works "
        "off `#hashtags` — try a different data source or a query that "
        "includes hashtags."
    )
    st.stop()

rollup = aggregate.topic_day_rollup(exploded)
tier_rollup = aggregate.topic_day_tier_rollup(exploded)
days = sorted(exploded["day"].dt.date.unique())

# ---------------------------------------------------------------------------
# Headline numbers.
# ---------------------------------------------------------------------------
m1, m2, m3, m4 = st.columns(4)
m1.metric("Tweets analysed", len(exploded))
m2.metric("Distinct topics", exploded["topic"].nunique())
m3.metric("Days covered", len(days))
m4.metric(
    "Premium share",
    f"{exploded['is_premium'].mean():.0%}",
    help=f"{int(exploded['is_premium'].sum())} of {len(exploded)} topical tweets came from premium / verified accounts.",
)

# ---------------------------------------------------------------------------
# Day picker + treemap.
# ---------------------------------------------------------------------------
st.subheader("Daily snapshot")
selected_day = st.select_slider(
    "Day",
    options=days,
    value=days[-1],
    format_func=lambda d: d.strftime("%a %b %d"),
)

day_top = aggregate.top_topics_for_day(rollup, selected_day, n=top_n)

if day_top.empty:
    st.info("No tweets on this day.")
else:
    # Build the nested treemap dataframe: one row per (topic, tier).
    tier_today = tier_rollup[tier_rollup["day"] == pd.to_datetime(selected_day)].copy()
    tier_today = tier_today[tier_today["topic"].isin(day_top["topic"])]
    # Attach the topic-level prevailing sentiment so colour propagates from parent.
    sentiment_lookup = day_top.set_index("topic")["mean_compound"].to_dict()
    tier_today["topic_mean_compound"] = tier_today["topic"].map(sentiment_lookup)

    fig = px.treemap(
        tier_today,
        path=[px.Constant(selected_day.strftime("%a %b %d, %Y")), "topic", "user_tier"],
        values="count",
        color="topic_mean_compound",
        color_continuous_scale=[
            (0.0, "#d93025"),  # negative
            (0.5, "#dadce0"),  # neutral
            (1.0, "#1f9d55"),  # positive
        ],
        range_color=(-0.6, 0.6),
        hover_data={"mean_compound": ":.2f", "count": True, "topic_mean_compound": False},
    )
    fig.update_traces(
        textinfo="label+value+percent parent",
        hovertemplate=(
            "<b>%{label}</b><br>"
            "Tweets: %{value}<br>"
            "Topic mean sentiment: %{color:+.2f}<br>"
            "<extra></extra>"
        ),
    )
    fig.update_layout(
        height=520, margin=dict(l=10, r=10, t=10, b=10),
        coloraxis_colorbar=dict(title="Mean<br>sentiment", tickformat="+.2f"),
    )
    st.plotly_chart(fig, use_container_width=True)

    cols = st.columns([3, 2])
    with cols[0]:
        st.markdown("**Top topics on this day**")
        display = day_top[[
            "topic", "count", "premium_count", "regular_count",
            "premium_share", "prevailing_sentiment", "mean_compound",
        ]].copy()
        display["mean_compound"] = display["mean_compound"].round(2)
        display["premium_share"] = (display["premium_share"] * 100).round(0).astype(int).astype(str) + "%"
        display.columns = ["topic", "tweets", "premium", "regular", "premium %", "sentiment", "mean compound"]
        st.dataframe(display, hide_index=True, use_container_width=True)

    with cols[1]:
        st.markdown("**Premium vs regular volume**")
        bar_df = day_top.melt(
            id_vars="topic",
            value_vars=["premium_count", "regular_count"],
            var_name="tier",
            value_name="count",
        )
        bar_df["tier"] = bar_df["tier"].map({"premium_count": "premium", "regular_count": "regular"})
        bar_fig = px.bar(
            bar_df, x="count", y="topic", color="tier", orientation="h",
            color_discrete_map={"premium": "#1f6feb", "regular": "#9aa0a6"},
            category_orders={"topic": list(day_top.sort_values("count")["topic"])},
        )
        bar_fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10),
                              legend=dict(orientation="h", y=1.05, x=1, xanchor="right"))
        st.plotly_chart(bar_fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Multi-day evolution.
# ---------------------------------------------------------------------------
st.subheader("Topic volume across the window")
wide = aggregate.daily_volume_by_topic(rollup).reset_index()
long = wide.melt(id_vars="day", var_name="topic", value_name="count")
area = px.area(
    long, x="day", y="count", color="topic", groupnorm=None,
    line_shape="spline",
)
area.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10),
                   legend=dict(orientation="h", y=-0.2))
st.plotly_chart(area, use_container_width=True)

# ---------------------------------------------------------------------------
# Sentiment heatmap (topic × day).
# ---------------------------------------------------------------------------
st.subheader("Sentiment heatmap — topic × day")
heat = rollup.pivot_table(
    index="topic", columns="day", values="mean_compound", aggfunc="mean",
)
heat.columns = [c.strftime("%a %b %d") for c in heat.columns]
heat_fig = px.imshow(
    heat,
    color_continuous_scale=[(0.0, "#d93025"), (0.5, "#dadce0"), (1.0, "#1f9d55")],
    zmin=-0.6, zmax=0.6,
    aspect="auto",
    text_auto=".2f",
)
heat_fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10),
                       coloraxis_colorbar=dict(title="Mean<br>compound"))
st.plotly_chart(heat_fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Optional Claude insights.
# ---------------------------------------------------------------------------
if show_insights and not day_top.empty:
    st.subheader(f"Claude's brief — {selected_day.strftime('%a %b %d')}")
    with st.spinner("Asking Claude to summarise the day…"):
        insight = aggregate.llm_daily_insights(rollup, selected_day)
    st.markdown(f"**{insight.get('headline', '')}**")
    for b in insight.get("bullets", []):
        st.markdown(f"- {b}")

# ---------------------------------------------------------------------------
# Per-tweet detail + export.
# ---------------------------------------------------------------------------
with st.expander("Per-tweet detail (long form: one row per tweet × topic)", expanded=False):
    show = exploded.copy()
    show["day"] = show["day"].dt.date
    st.dataframe(
        show[["day", "topic", "user_tier", "author", "polarity", "compound", "text"]],
        hide_index=True, use_container_width=True,
    )
    buf = io.StringIO()
    show.to_csv(buf, index=False)
    st.download_button(
        "Download as CSV",
        data=buf.getvalue(),
        file_name="trend_analysis.csv",
        mime="text/csv",
    )
