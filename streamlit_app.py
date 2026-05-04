"""x-opinionate: opinion mining on social posts using Claude.

This is the Streamlit Community Cloud entrypoint (Cloud auto-detects this
filename). Run locally with::

    streamlit run streamlit_app.py
"""
import streamlit as st

from src import config

st.set_page_config(
    page_title="x-opinionate",
    page_icon="🗣️",
    layout="wide",
)

st.title("x-opinionate")
st.caption("Nuanced sentiment + stance detection for X-style posts, powered by Claude.")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Brand Sentiment Dashboard")
    st.markdown(
        "- Stream tweets mentioning a brand or product\n"
        "- Nuanced LLM sentiment that catches **sarcasm**, **frustration**, "
        "and **excitement**\n"
        "- Benchmarked against **VADER** and **RoBERTa** baselines\n"
        "- Surface *why* customers are unhappy via thematic clustering"
    )
    st.page_link("pages/1_Brand_Sentiment.py", label="Open dashboard →", icon="📊")

with col2:
    st.subheader("Stance Detection")
    st.markdown(
        "- Pick a controversial topic\n"
        "- Classify each tweet as **for / against / neutral** with **few-shot "
        "chain-of-thought** prompting\n"
        "- Inspect the model's reasoning per post\n"
        "- Track stance evolution over time"
    )
    st.page_link("pages/2_Stance_Detection.py", label="Open dashboard →", icon="⚖️")

st.divider()

with st.sidebar:
    st.header("Setup")
    if config.has_anthropic():
        st.success(f"Anthropic configured · `{config.ANTHROPIC_MODEL}`")
    else:
        st.error("ANTHROPIC_API_KEY missing — copy `.env.example` to `.env`.")
    if config.has_x_api():
        st.success("Live X API enabled")
    else:
        st.info("Live X API off — add `X_BEARER_TOKEN` to enable.")

    st.divider()
    st.caption(
        "**Tweet sources** available on each page: demo data (offline), "
        "CSV upload, or live X recent search."
    )

st.markdown(
    "### Architecture\n"
    "```\n"
    "Tweets ──► nuanced sentiment (Claude)  ─┐\n"
    "       ──► VADER + RoBERTa baselines   ─┼─► benchmark + agreement matrix\n"
    "       ──► negative subset ──► embed ──► KMeans ──► Claude theme labels\n"
    "       ──► stance classifier (Claude)  ─► time-series of for/against/neutral\n"
    "```"
)
