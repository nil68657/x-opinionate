"""Brand sentiment dashboard."""
from __future__ import annotations

import io
import time

import pandas as pd
import plotly.express as px
import streamlit as st

from src import config
from src.clustering import themes
from src.sentiment import benchmark
from src.tweets import sources

st.set_page_config(page_title="Brand Sentiment · x-opinionate", page_icon="📊", layout="wide")

st.title("📊 Brand Sentiment Dashboard")
st.caption("Nuanced LLM sentiment + classical-model benchmark + complaint themes.")

if not config.has_anthropic():
    st.error("ANTHROPIC_API_KEY missing — set it in `.env` and reload.")
    st.stop()

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
        brands = ["All"] + sources.available_brands()
        brand = st.selectbox("Brand", brands, index=0)
        tweets = sources.demo_brand_tweets(None if brand == "All" else brand)
    elif src == "Upload CSV":
        upload = st.file_uploader("CSV with at least a `text` column", type=["csv"])
        if upload is not None:
            tweets = sources.from_csv_upload(upload)
    else:
        query = st.text_input("X recent-search query", value='"Acme Phone" -is:retweet lang:en')
        n = st.slider("Max tweets", 10, 100, 50, step=10)
        if st.button("Fetch from X"):
            with st.spinner("Querying X API…"):
                tweets = sources.from_x_api(query, max_results=n)

    st.divider()
    st.header("Models")
    use_roberta = st.checkbox(
        "Include RoBERTa baseline",
        value=benchmark.has_roberta(),
        disabled=not benchmark.has_roberta(),
        help=("Install `requirements-classical.txt` (transformers + torch) to enable."
              if not benchmark.has_roberta() else
              "First call will download the cardiffnlp/twitter-roberta-base-sentiment-latest model."),
    )

    streaming = st.checkbox(
        "Simulate live streaming",
        value=False,
        help="Display tweets one-by-one with a short delay before scoring.",
    )

if not tweets:
    st.info("Pick a data source on the left to get started.")
    st.stop()

st.markdown(f"**{len(tweets)}** tweets loaded from `{src}`.")

# ---------------------------------------------------------------------------
# Optional simulated streaming preview.
# ---------------------------------------------------------------------------
if streaming:
    placeholder = st.empty()
    for i, t in enumerate(tweets, start=1):
        with placeholder.container():
            st.write(f"**Streaming tweet {i}/{len(tweets)}** · @{t.author}")
            st.write(t.text)
        time.sleep(0.2)
    placeholder.empty()

# ---------------------------------------------------------------------------
# Run benchmark.
# ---------------------------------------------------------------------------
run = st.button("Run sentiment benchmark", type="primary")
if not run and "brand_df" not in st.session_state:
    st.info("Click **Run sentiment benchmark** to score the tweets.")
    st.stop()

if run:
    progress = st.progress(0.0, text="Scoring tweets…")
    with st.spinner("Calling Claude + classical models…"):
        df = benchmark.run_benchmark(
            tweets,
            progress=lambda f: progress.progress(f, text=f"Scoring… {int(f * 100)}%"),
            use_roberta=use_roberta,
        )
    progress.empty()
    st.session_state["brand_df"] = df
    st.session_state.pop("brand_themes", None)

df: pd.DataFrame = st.session_state["brand_df"]

# ---------------------------------------------------------------------------
# Headline metrics.
# ---------------------------------------------------------------------------
total = len(df)
neg = int((df["llm_polarity"] == "negative").sum())
sarcasm = int(df["llm_sarcasm"].sum())
high_intensity = int((df["llm_intensity"] >= 4).sum())

m1, m2, m3, m4 = st.columns(4)
m1.metric("Tweets scored", total)
m2.metric("Negative (LLM)", neg, f"{neg / max(total, 1):.0%}")
m3.metric("Sarcasm detected", sarcasm)
m4.metric("High-intensity (≥4)", high_intensity)

# ---------------------------------------------------------------------------
# Distribution charts.
# ---------------------------------------------------------------------------
left, right = st.columns(2)
with left:
    st.subheader("Nuanced sentiment mix")
    nuanced_counts = df["llm_nuanced"].value_counts().reset_index()
    nuanced_counts.columns = ["nuanced", "count"]
    fig = px.bar(
        nuanced_counts.sort_values("count", ascending=True),
        x="count", y="nuanced", orientation="h",
        color="nuanced",
    )
    fig.update_layout(showlegend=False, height=380, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Polarity by model")
    polarity_cols = {"vader": "vader_label", "claude": "llm_polarity"}
    if "roberta_label" in df.columns:
        polarity_cols["roberta"] = "roberta_label"
    rows = []
    for model, col in polarity_cols.items():
        for label, n in df[col].value_counts().items():
            rows.append({"model": model, "polarity": label, "count": int(n)})
    pol_df = pd.DataFrame(rows)
    fig = px.bar(
        pol_df, x="model", y="count", color="polarity", barmode="stack",
        color_discrete_map=config.POLARITY_COLOURS,
    )
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Agreement matrix + LLM-vs-classical disagreements.
# ---------------------------------------------------------------------------
st.subheader("Model agreement matrix (% of tweets where two models agreed on polarity)")
mat = benchmark.agreement_matrix(df)
st.dataframe(mat.style.background_gradient(cmap="Blues", vmin=0, vmax=100).format("{:.1f}%"))

st.subheader("Where the LLM disagrees with VADER")
st.caption(
    "These are the rows where the nuanced model is most likely earning its keep — "
    "watch for sarcasm flags and surface-vs-meaning mismatches."
)
disagreements = benchmark.disagreement_examples(df, n=10)
if disagreements.empty:
    st.success("No polarity disagreements between VADER and Claude on this sample.")
else:
    st.dataframe(disagreements, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Per-tweet detail.
# ---------------------------------------------------------------------------
with st.expander("Per-tweet scores", expanded=False):
    st.dataframe(df, use_container_width=True, hide_index=True)
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    st.download_button(
        "Download as CSV",
        data=csv_buf.getvalue(),
        file_name="brand_sentiment_scores.csv",
        mime="text/csv",
    )

# ---------------------------------------------------------------------------
# Why customers are unhappy: thematic clustering of negative tweets.
# ---------------------------------------------------------------------------
st.subheader("Why customers are unhappy")
neg_df = df[df["llm_polarity"] == "negative"].reset_index(drop=True)

if len(neg_df) < 4:
    st.info("Need at least 4 negative tweets to surface themes.")
else:
    k = st.slider(
        "Number of themes", 2, max(2, min(8, len(neg_df) // 2)),
        value=min(4, max(2, len(neg_df) // 3)),
    )
    if st.button("Cluster negative tweets into themes"):
        with st.spinner("Embedding, clustering, and asking Claude to name themes…"):
            labels = themes.cluster(neg_df["text"].tolist(), k=k)
            named = themes.label_clusters(neg_df["text"].tolist(), labels)
        st.session_state["brand_themes"] = (labels, named, neg_df)

    payload = st.session_state.get("brand_themes")
    if payload is not None:
        labels, named, neg_df = payload
        for cid, info in sorted(named.items(), key=lambda kv: -kv[1]["size"]):
            with st.container(border=True):
                cols = st.columns([3, 1])
                cols[0].markdown(f"### {info['theme']}")
                cols[1].metric("Tweets", info["size"])
                st.caption(info["summary"])
                for ex in info["examples"]:
                    st.markdown(f"> {ex}")
