"""Stance detection dashboard."""
from __future__ import annotations

import io

import pandas as pd
import plotly.express as px
import streamlit as st

from src import config
from src.stance import classifier
from src.tweets import sources

st.set_page_config(page_title="Stance Detection · x-opinionate", page_icon="⚖️", layout="wide")

st.title("⚖️ Stance Detection")
st.caption(
    "Few-shot, chain-of-thought prompting classifies each tweet as "
    "**for / against / neutral** toward a target topic."
)

if not config.has_anthropic():
    st.error("ANTHROPIC_API_KEY missing — set it in `.env` and reload.")
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar.
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Topic + data source")

    available_topics = sources.available_topics()
    topic_mode = st.radio(
        "Topic", ["Pick from demo data", "Custom topic"], index=0,
    )
    if topic_mode == "Pick from demo data":
        topic = st.selectbox("Demo topic", available_topics)
    else:
        topic = st.text_input(
            "Topic statement",
            value="Remote work should remain the default for tech jobs",
            help="Phrase as a claim. Stance is measured AGAINST or FOR this claim.",
        )

    source_options = ["Demo data (matching topic)", "Upload CSV"]
    if config.has_x_api():
        source_options.append("Live X API")
    src = st.radio("Source", source_options, index=0)

    tweets: list = []
    if src == "Demo data (matching topic)":
        tweets = sources.demo_stance_tweets(topic if topic in available_topics else None)
        if not tweets:
            tweets = sources.demo_stance_tweets()
    elif src == "Upload CSV":
        upload = st.file_uploader("CSV with at least a `text` column", type=["csv"])
        if upload is not None:
            tweets = sources.from_csv_upload(upload)
    else:
        query = st.text_input("X recent-search query", value='"remote work" -is:retweet lang:en')
        n = st.slider("Max tweets", 10, 100, 50, step=10)
        if st.button("Fetch from X"):
            with st.spinner("Querying X API…"):
                tweets = sources.from_x_api(query, max_results=n)

if not tweets:
    st.info("Pick a topic + data source on the left.")
    st.stop()

st.markdown(f"**Topic:** _{topic}_  \n**{len(tweets)}** tweets loaded from `{src}`.")

# ---------------------------------------------------------------------------
# Run classification.
# ---------------------------------------------------------------------------
if st.button("Classify stance", type="primary") or "stance_payload" not in st.session_state \
        or st.session_state.get("stance_topic") != topic:
    progress = st.progress(0.0, text="Classifying…")
    with st.spinner("Calling Claude with few-shot CoT prompts…"):
        df = classifier.classify_many(
            topic, tweets,
            progress=lambda f: progress.progress(f, text=f"Classifying… {int(f * 100)}%"),
        )
    progress.empty()
    st.session_state["stance_payload"] = df
    st.session_state["stance_topic"] = topic

df: pd.DataFrame = st.session_state["stance_payload"]

# ---------------------------------------------------------------------------
# Headline metrics.
# ---------------------------------------------------------------------------
total = len(df)
counts = df["stance"].value_counts().to_dict()
m1, m2, m3, m4 = st.columns(4)
m1.metric("Tweets classified", total)
m2.metric("For",     counts.get("for", 0),     f"{counts.get('for', 0) / max(total, 1):.0%}")
m3.metric("Against", counts.get("against", 0), f"{counts.get('against', 0) / max(total, 1):.0%}")
m4.metric("Neutral", counts.get("neutral", 0), f"{counts.get('neutral', 0) / max(total, 1):.0%}")

# ---------------------------------------------------------------------------
# Distribution + confidence.
# ---------------------------------------------------------------------------
left, right = st.columns(2)
with left:
    st.subheader("Stance distribution")
    dist = (
        df["stance"].value_counts().reindex(classifier.LABELS, fill_value=0)
        .reset_index()
    )
    dist.columns = ["stance", "count"]
    fig = px.bar(
        dist, x="stance", y="count", color="stance",
        color_discrete_map=config.STANCE_COLOURS,
    )
    fig.update_layout(showlegend=False, height=380, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Model confidence by stance")
    fig = px.box(
        df, x="stance", y="confidence", color="stance", points="all",
        color_discrete_map=config.STANCE_COLOURS,
    )
    fig.update_layout(showlegend=False, height=380, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Stance evolution over time.
# ---------------------------------------------------------------------------
st.subheader("Stance evolution over time")
freq = st.select_slider(
    "Time bucket",
    options=["H", "D", "W"],
    value="D",
    format_func=lambda x: {"H": "hour", "D": "day", "W": "week"}[x],
)
ts = classifier.stance_over_time(df, freq=freq)
if ts.empty or len(ts) < 2:
    st.info("Need tweets across at least two time buckets to plot evolution.")
else:
    long = ts.melt(id_vars="created_at", var_name="stance", value_name="count")
    fig = px.area(
        long, x="created_at", y="count", color="stance",
        color_discrete_map=config.STANCE_COLOURS, groupnorm=None,
    )
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Per-tweet table with reasoning.
# ---------------------------------------------------------------------------
st.subheader("Per-tweet classifications with reasoning")
filt = st.multiselect(
    "Show stances", classifier.LABELS, default=classifier.LABELS,
)
view = df[df["stance"].isin(filt)].copy()
view["created_at"] = pd.to_datetime(view["created_at"])
st.dataframe(
    view[["created_at", "author", "stance", "confidence", "text", "reasoning"]],
    use_container_width=True, hide_index=True,
)

csv_buf = io.StringIO()
df.to_csv(csv_buf, index=False)
st.download_button(
    "Download classifications as CSV",
    data=csv_buf.getvalue(),
    file_name="stance_classifications.csv",
    mime="text/csv",
)
