"""
NVIDIA-backed OpenAI-compatible client utilities for the searchbybrief pipeline.
"""

import json
import time
from typing import Any

from openai import OpenAI

from app.config import settings


def _is_non_retryable_image_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return ("invalid_image_url" in text) or ("error while downloading" in text)

def get_client(api_key_override: str | None = None) -> OpenAI:
    """
    Build an OpenAI client for this request.

    NVIDIA NIM exposes OpenAI-compatible chat completions, so the standard
    OpenAI client is pointed at the NVIDIA base URL with NVIDIA_API_KEY.
    """
    api_key = api_key_override or settings.require_nvidia_api_key()
    return OpenAI(api_key=api_key, base_url=settings.llm_base_url)


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
    Call the LLM via NVIDIA's OpenAI-compatible endpoint and return a parsed JSON dict.

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
    Call the LLM via NVIDIA's OpenAI-compatible endpoint with a pre-built messages list and return a parsed JSON dict.

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
            if _is_non_retryable_image_error(exc):
                # Deterministic URL/download errors won't recover on retry.
                raise RuntimeError(f"Vision LLM call failed (non-retryable): {exc}")
            last_err = exc
            if attempt < retries:
                time.sleep(sleep_seconds * attempt)

    raise RuntimeError(f"Vision LLM call failed after {retries} attempts: {last_err}")
