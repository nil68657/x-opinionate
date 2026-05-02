"""Thin wrapper around the Anthropic client that returns parsed JSON."""
from __future__ import annotations

import json
import re
from typing import Any

from anthropic import Anthropic

from . import config

_client: Anthropic | None = None


def get_client() -> Anthropic:
    """Lazy-create a single Anthropic client per process."""
    global _client
    if _client is None:
        if not config.has_anthropic():
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        _client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
    return _client


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def _strip_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = _FENCE_RE.sub("", text).strip()
    return text


def call_json(system: str, user: str, max_tokens: int = 1024) -> dict[str, Any]:
    """Call Claude and parse the response as JSON.

    The system prompt is augmented with a strict JSON-only directive so we don't
    have to fight preambles or markdown fences in the response.
    """
    client = get_client()
    msg = client.messages.create(
        model=config.ANTHROPIC_MODEL,
        max_tokens=max_tokens,
        system=(
            system
            + "\n\nRespond with ONLY a single valid JSON object. "
            + "No prose, no preamble, no code fences."
        ),
        messages=[{"role": "user", "content": user}],
    )
    text = "".join(getattr(b, "text", "") for b in msg.content).strip()
    text = _strip_fence(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        # Last-ditch: try to grab the first {...} block.
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise ValueError(f"Model did not return valid JSON: {e}\nGot: {text[:300]}")
