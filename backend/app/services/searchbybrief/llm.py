"""
Bifrost-backed OpenAI client and JSON-calling utilities for the searchbybrief pipeline.

Bifrost is an internal OpenAI-compatible proxy, so we use the standard Chat
Completions API (client.chat.completions.create) with a custom base_url and
the internal virtual key in place of an OpenAI key.
"""

import json
import time
from typing import Any

from openai import OpenAI

from app.config import Settings

_client: OpenAI | None = None


def get_client() -> OpenAI:
    """Return a lazily-initialised Bifrost OpenAI client (singleton)."""
    global _client
    if _client is None:
        settings = Settings()
        _client = OpenAI(
            api_key=settings.bifrost_api_key,
            base_url=settings.bifrost_base_url,
        )
    return _client


def call_llm_json(
    system_prompt: str,
    user_prompt: str,
    model: str,
    max_tokens: int = 4000,
    retries: int = 3,
    sleep_seconds: float = 2.0,
) -> dict[str, Any]:
    """
    Call the LLM via Bifrost and return a parsed JSON dict.

    Strips markdown fenced code blocks if the model wraps its response.
    Retries with linear backoff on any exception.
    """
    client = get_client()
    last_err: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            text = response.choices[0].message.content.strip()
            return json.loads(text)

        except Exception as exc:
            last_err = exc
            if attempt < retries:
                time.sleep(sleep_seconds * attempt)

    raise RuntimeError(f"LLM call failed after {retries} attempts: {last_err}")
