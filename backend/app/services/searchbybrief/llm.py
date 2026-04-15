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

def get_client(api_key_override: str | None = None) -> OpenAI:
    """
    Build an OpenAI client for this request.

    Priority:
    1) Configured Bifrost key (uses Bifrost base URL)
    2) Explicit per-request API key override (popup key)
    """
    settings = Settings()
    if settings.bifrost_api_key:
        return OpenAI(
            api_key=settings.bifrost_api_key,
            base_url=settings.bifrost_base_url,
        )
    if api_key_override:
        return OpenAI(api_key=api_key_override)
    raise RuntimeError(
        "SearchByBrief LLM auth is not configured. "
        "Provide a popup API key or set BIFROST_API_KEY."
    )


def call_llm_json(
    system_prompt: str,
    user_prompt: str,
    model: str,
    max_tokens: int = 4000,
    retries: int = 3,
    sleep_seconds: float = 2.0,
    api_key_override: str | None = None,
) -> dict[str, Any]:
    """
    Call the LLM via Bifrost and return a parsed JSON dict.

    Strips markdown fenced code blocks if the model wraps its response.
    Retries with linear backoff on any exception.
    """
    client = get_client(api_key_override=api_key_override)
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


def call_llm_vision_json(
    messages: list[dict[str, Any]],
    model: str,
    max_tokens: int = 2000,
    retries: int = 3,
    sleep_seconds: float = 2.0,
    api_key_override: str | None = None,
) -> dict[str, Any]:
    """
    Call the LLM via Bifrost with a pre-built messages list and return a parsed JSON dict.

    Unlike call_llm_json, this function accepts the full messages list directly so
    callers can include image_url content blocks for multimodal (vision) requests.

    Strips markdown fenced code blocks if the model wraps its response.
    Retries with linear backoff on any exception.
    """
    client = get_client(api_key_override=api_key_override)
    last_err: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            text = response.choices[0].message.content.strip()
            # Strip markdown fences if present
            if text.startswith("```"):
                text = text.split("\n", 1)[-1]
                text = text.rsplit("```", 1)[0]
            return json.loads(text)

        except Exception as exc:
            last_err = exc
            if attempt < retries:
                time.sleep(sleep_seconds * attempt)

    raise RuntimeError(f"Vision LLM call failed after {retries} attempts: {last_err}")
