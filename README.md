# x-opinionate

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://x-opinionate.streamlit.app)
[![CI](https://github.com/nil68657/x-opinionate/actions/workflows/ci.yml/badge.svg)](https://github.com/nil68657/x-opinionate/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)

Opinion mining on X-style social posts, powered by Claude.

> Live demo: **<https://x-opinionate.streamlit.app>** *(replace with your real Cloud URL once deployed; the badge above already links there)*

Three dashboards in one Streamlit app:

1. **Trend Analysis** — day-wise rollup of trending hashtags as a Plotly
   **treemap coloured by prevailing sentiment** and split by **premium vs
   regular** users, plus a multi-day area chart, a topic × day sentiment
   heatmap, and an optional Claude daily-brief.
2. **Brand Sentiment Dashboard** — nuanced LLM sentiment that catches *sarcasm*,
   *frustration* and *excitement*, benchmarked against **VADER** and
   **RoBERTa**, with thematic clustering of negative tweets to surface *why*
   customers are unhappy.
3. **Stance Detection** — given a topic, classify each tweet as **for /
   against / neutral** using **few-shot chain-of-thought** prompting, with
   per-tweet reasoning and a stance-over-time chart.

## Quick start

```bash
git clone https://github.com/nil68657/x-opinionate.git
cd x-opinionate

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Optional — adds RoBERTa baseline + sentence-transformer embeddings
pip install -r requirements-classical.txt

cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY

streamlit run streamlit_app.py
```

The app runs fully offline against baked-in demo data (sample brand and
stance tweets, including sarcasm cases) so you can demo it without an X
account. Uploading a CSV with a `text` column works too. If you set
`X_BEARER_TOKEN`, a "Live X API" source appears for recent-search streaming.

## Project layout

```
x-opinionate/
├── streamlit_app.py                Streamlit entrypoint (home + nav)
├── pages/
│   ├── 1_Trend_Analysis.py         Day-wise hashtag trends (treemap + area)
│   ├── 2_Brand_Sentiment.py        Brand sentiment dashboard
│   └── 3_Stance_Detection.py       Stance detection dashboard
├── src/
│   ├── config.py                   Env + label constants
│   ├── llm_client.py               Anthropic JSON wrapper
│   ├── tweets/
│   │   └── sources.py              Demo / CSV / live X tweet sources
│   ├── trends/
│   │   └── aggregate.py            Hashtag extraction + day×topic×tier rollup
│   ├── sentiment/
│   │   ├── vader.py                VADER baseline
│   │   ├── roberta.py              RoBERTa baseline (optional)
│   │   ├── llm.py                  Nuanced Claude sentiment (sarcasm-aware)
│   │   └── benchmark.py            Compare models + agreement matrix
│   ├── stance/
│   │   └── classifier.py           Few-shot CoT stance detection
│   ├── clustering/
│   │   └── themes.py               Embed → cluster → Claude-named themes
│   └── data/
│       ├── sample_brand_tweets.json
│       ├── sample_stance_tweets.json
│       └── sample_trend_tweets.json
├── requirements.txt
├── requirements-classical.txt
└── .env.example
```

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub (any branch — Cloud picks per-app).
2. Go to <https://share.streamlit.io/> → **Create app** → **Deploy a public app from GitHub**.
3. Fill in:
   - **Repository**: `nil68657/x-opinionate`
   - **Branch**: `main`
   - **Main file path**: `streamlit_app.py` (auto-detected by Cloud)
   - **Python version**: pinned by `runtime.txt` (3.12).
4. Open **Advanced settings → Secrets** and paste:

   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   ANTHROPIC_MODEL   = "claude-sonnet-4-5"
   # Optional
   X_BEARER_TOKEN    = ""
   ```
5. Click **Deploy**. First build takes ~2–3 minutes.

Notes:
- `requirements.txt` is intentionally lean so Cloud builds fast. The optional
  RoBERTa baseline (`requirements-classical.txt`) pulls in `torch` (~1 GB) and
  is **not recommended** on Cloud's free tier — leave it off and the dashboard
  will simply hide that model column.
- Secrets set in the Cloud UI are picked up by `src/config.py`'s
  `st.secrets` fallback automatically — no code changes needed.
- `.streamlit/config.toml` configures theme + headless mode in both
  environments.

## Configuration

| Var                  | Required | Notes                                                                 |
| -------------------- | -------- | --------------------------------------------------------------------- |
| `ANTHROPIC_API_KEY`  | yes      | Used for nuanced sentiment, stance, and theme labelling               |
| `ANTHROPIC_MODEL`    | no       | Defaults to `claude-sonnet-4-5`. Try `claude-opus-4-5` for top quality |
| `X_BEARER_TOKEN`     | no       | Enables the "Live X API" tweet source                                 |

## Notes on the model choices

- **Nuanced sentiment** uses a fixed 10-label set (`excited`, `joyful`,
  `satisfied`, `neutral`, `confused`, `disappointed`, `frustrated`, `angry`,
  `sarcastic`, `anxious`) plus a 1–5 intensity score, an explicit `sarcasm`
  flag, and a one-sentence reason. The 3-way polarity roll-up (`positive` /
  `negative` / `neutral`) is what we benchmark against VADER and RoBERTa.
- **VADER** is a lexicon-based baseline — fast, free, and a decent reference
  for surface polarity. It famously misses sarcasm, which is exactly the gap
  the LLM is meant to fill.
- **RoBERTa** uses
  [`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest),
  fine-tuned on tweets. First call downloads ~500 MB of weights.
- **Stance detection** is a separate prompt with a few-shot chain-of-thought
  template covering multiple example topics and a sarcasm-flip case. Each
  classification returns the model's reasoning, which the dashboard
  surfaces.
- **Themes** are produced by clustering negative tweets (sentence-transformer
  embeddings if installed, TF-IDF otherwise) with KMeans, then asking Claude
  to write a short theme label and one-sentence summary per cluster.
