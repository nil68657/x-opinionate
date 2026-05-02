"""Centralised env + constants.

Lookup order for every secret:
  1. Process environment (so ``.env`` via python-dotenv wins locally).
  2. Streamlit ``st.secrets`` (so the Cloud Secrets UI works in production).
"""
import os

from dotenv import load_dotenv

load_dotenv()


def _get(key: str, default: str = "") -> str:
    val = os.getenv(key)
    if val:
        return val.strip()
    try:
        import streamlit as st  # noqa: WPS433 - intentional lazy import

        if key in st.secrets:
            return str(st.secrets[key]).strip()
    except Exception:
        pass
    return default


ANTHROPIC_API_KEY: str = _get("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL: str = _get("ANTHROPIC_MODEL", "claude-sonnet-4-5")
X_BEARER_TOKEN: str = _get("X_BEARER_TOKEN")

# Fixed label set the LLM must pick from. Chosen to cover the cases the brief
# calls out (sarcasm, frustration, excitement) while staying small enough to
# be useful on a dashboard.
NUANCED_SENTIMENT_LABELS: list[str] = [
    "excited",
    "joyful",
    "satisfied",
    "neutral",
    "confused",
    "disappointed",
    "frustrated",
    "angry",
    "sarcastic",
    "anxious",
]

STANCE_LABELS: list[str] = ["for", "against", "neutral"]

POLARITY_COLOURS: dict[str, str] = {
    "positive": "#1f9d55",
    "neutral": "#9aa0a6",
    "negative": "#d93025",
}

STANCE_COLOURS: dict[str, str] = {
    "for": "#1f9d55",
    "neutral": "#9aa0a6",
    "against": "#d93025",
}


def has_anthropic() -> bool:
    return bool(ANTHROPIC_API_KEY)


def has_x_api() -> bool:
    return bool(X_BEARER_TOKEN)
