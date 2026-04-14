"""
Stage 3 visual auditing utilities for Search by Brief.

Purpose
-------
Use a visual LLM (gpt-4o-mini via Bifrost) to:
1. score individual candidate thumbnails against Stage 0 search lanes
2. combine visual scores with Stage 2 reranker scores
3. audit small batches for diversity / coverage
4. generate structured repair requests back to Stage 1/2

Intended integration
--------------------
- Input: state["search_params"] from Stage 0, plus Stage 2 candidate pool
- Output: enriched candidates, optional lane audits, optional repair requests

Assumptions
-----------
- Bifrost is OpenAI-compatible and supports chat.completions.create
- Candidate rows include thumbnail_url or at least asset_id
- search_params is the persisted Stage 0 IntentResult converted to dict

Suggested state contract
------------------------
state = {
    "search_params": {...},              # Stage 0 output
    "stage2_candidates": [...],          # list[dict] or DataFrame-like records
    "stage3_candidates": [...],          # filled by score_candidates_node
    "stage3_lane_audits": [...],         # filled by audit_lanes_node
    "stage3_repair_requests": [...],     # filled by audit_lanes_node
    "final_selection": [...],            # optional downstream
}
"""

from __future__ import annotations

import json
import time
from typing import Any, Optional

import pandas as pd
from openai import OpenAI

from app.config import Settings


# -----------------------------------------------------------------------------
# Bifrost client
# -----------------------------------------------------------------------------

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        settings = Settings()
        _client = OpenAI(
            api_key=settings.bifrost_api_key,
            base_url=settings.bifrost_base_url,
        )
    return _client


def call_bifrost_chat_json(
    messages: list[dict[str, Any]],
    model: str,
    max_tokens: int = 2000,
    retries: int = 3,
    sleep_seconds: float = 2.0,
) -> dict[str, Any]:
    client = get_client()
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
            return json.loads(text)

        except Exception as exc:
            last_err = exc
            if attempt < retries:
                time.sleep(sleep_seconds * attempt)

    raise RuntimeError(f"LLM call failed after {retries} attempts: {last_err}")


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

VISION_MODEL = "gpt-4o-mini"


# -----------------------------------------------------------------------------
# Thumbnail helpers
# -----------------------------------------------------------------------------

def build_thumbnail_url_from_asset_id(asset_id: int | str) -> str:
    """
    Placeholder thumbnail builder based on the user's example.

    Replace this if your system has a canonical thumbnail URL builder.
    """
    return f"https://image.shutterstock.com/image-photo/placeholder-260nw-{asset_id}.jpg"


def ensure_thumbnail_url(candidate: dict[str, Any]) -> str | None:
    url = candidate.get("thumbnail_url")
    if url:
        return str(url)
    asset_id = candidate.get("asset_id")
    if asset_id is None:
        return None
    return build_thumbnail_url_from_asset_id(asset_id)


# -----------------------------------------------------------------------------
# Stage 0 context compaction
# -----------------------------------------------------------------------------

def build_stage3_brief_context(search_params: dict[str, Any]) -> dict[str, Any]:
    brief_diagnostics = search_params.get("brief_diagnostics", {})
    hard_constraints = search_params.get("hard_constraints", {})
    shared_filters = search_params.get("shared_filters", [])
    search_lanes = search_params.get("search_lanes", [])

    compact_lanes = []
    for lane in search_lanes:
        compact_lanes.append(
            {
                "lane_name": lane.get("lane_name"),
                "lane_goal": lane.get("lane_goal"),
                "embedding_query": lane.get("embedding_query"),
                "visual_proxies": lane.get("visual_proxies", []),
                "ranking_hints": lane.get("ranking_hints", []),
            }
        )

    return {
        "brief_diagnostics": {
            "brief_form": brief_diagnostics.get("brief_form"),
            "retrieval_intent": brief_diagnostics.get("retrieval_intent"),
            "search_complexity": brief_diagnostics.get("search_complexity"),
            "is_multi_lane": brief_diagnostics.get("is_multi_lane"),
        },
        "hard_constraints": {
            "subjects_required": hard_constraints.get("subjects_required", []),
            "demographics_required": hard_constraints.get("demographics_required", []),
            "composition_required": hard_constraints.get("composition_required", []),
            "style_required": hard_constraints.get("style_required", []),
            "location_required": hard_constraints.get("location_required", []),
            "media_type_required": hard_constraints.get("media_type_required", []),
            "licensing_required": hard_constraints.get("licensing_required", []),
            "exclusions": hard_constraints.get("exclusions", []),
        },
        "shared_filters": shared_filters,
        "search_lanes": compact_lanes,
    }


def build_lane_lookup(search_params: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        lane["lane_name"]: lane
        for lane in search_params.get("search_lanes", [])
        if lane.get("lane_name")
    }


# -----------------------------------------------------------------------------
# Prompts
# -----------------------------------------------------------------------------

VISUAL_SCORING_SYSTEM_PROMPT = """
You are a visual auditing model for Stage 3 of a creative search workflow.

Your task is to inspect a candidate image thumbnail and score how well it matches:
1. the overall brief,
2. a specific search lane,
3. key hard constraints and visual style requirements.

Important:
- Use only what is visually inferable from the image plus the provided structured brief context.
- Be conservative.
- Thumbnails may not reveal small details perfectly.
- Treat exclusions seriously when they are visibly violated.
- Return JSON only.

Score each dimension from 0.0 to 1.0:
- subject_match
- lane_match
- style_match
- composition_match
- overall_fit

Also return:
- likely_exclusion_violation: boolean
- exclusion_violation_reason: short string or null
- likely_tags: array of short visual tags
- short_rationale: 1-3 sentences
"""

SET_AUDIT_SYSTEM_PROMPT = """
You are performing a set-level visual audit for a creative search workflow.

You are given:
- structured brief context
- one search lane
- a small batch of candidate thumbnails for that lane

Your task:
- judge whether the batch covers the lane well
- identify repetition / near-duplicate tendencies
- identify missing attributes
- suggest whether repair retrieval is needed

Return JSON only.
"""


# -----------------------------------------------------------------------------
# Payload builders
# -----------------------------------------------------------------------------

def build_visual_scoring_payload(
    search_params: dict[str, Any],
    lane: dict[str, Any],
    candidate_metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    return {
        "brief_context": build_stage3_brief_context(search_params),
        "target_lane": {
            "lane_name": lane.get("lane_name"),
            "lane_goal": lane.get("lane_goal"),
            "embedding_query": lane.get("embedding_query"),
            "visual_proxies": lane.get("visual_proxies", []),
            "ranking_hints": lane.get("ranking_hints", []),
            "lane_filters": lane.get("lane_filters", []),
            "literal_terms_preserved": lane.get("literal_terms_preserved", []),
        },
        "candidate_metadata": candidate_metadata or {},
        "required_output_schema": {
            "subject_match": "float 0.0-1.0",
            "lane_match": "float 0.0-1.0",
            "style_match": "float 0.0-1.0",
            "composition_match": "float 0.0-1.0",
            "overall_fit": "float 0.0-1.0",
            "likely_exclusion_violation": "boolean",
            "exclusion_violation_reason": "string or null",
            "likely_tags": ["string"],
            "short_rationale": "string",
        },
    }


def build_set_audit_payload(
    search_params: dict[str, Any],
    lane: dict[str, Any],
    candidate_ids: list[Any],
) -> dict[str, Any]:
    return {
        "brief_context": build_stage3_brief_context(search_params),
        "target_lane": {
            "lane_name": lane.get("lane_name"),
            "lane_goal": lane.get("lane_goal"),
            "embedding_query": lane.get("embedding_query"),
            "visual_proxies": lane.get("visual_proxies", []),
            "ranking_hints": lane.get("ranking_hints", []),
        },
        "candidate_ids_in_order": candidate_ids,
        "required_output_schema": {
            "lane_coverage_quality": "low | medium | high",
            "duplicate_or_redundancy_risk": "low | medium | high",
            "missing_attributes": ["string"],
            "best_candidate_ids": ["candidate id"],
            "drop_candidate_ids": ["candidate id"],
            "repair_needed": "boolean",
            "repair_reason": "string or null",
            "short_rationale": "string",
        },
    }


# -----------------------------------------------------------------------------
# Single-image visual scoring
# -----------------------------------------------------------------------------

def score_candidate_thumbnail_for_lane(
    thumbnail_url: str,
    search_params: dict[str, Any],
    lane: dict[str, Any],
    candidate_metadata: Optional[dict[str, Any]] = None,
    model: str = VISION_MODEL,
) -> dict[str, Any]:
    payload = build_visual_scoring_payload(
        search_params=search_params,
        lane=lane,
        candidate_metadata=candidate_metadata,
    )

    messages = [
        {"role": "system", "content": VISUAL_SCORING_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(payload, ensure_ascii=False, indent=2),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": thumbnail_url},
                },
            ],
        },
    ]

    return call_bifrost_chat_json(
        messages=messages,
        model=model,
        max_tokens=1400,
    )


# -----------------------------------------------------------------------------
# Candidate scoring over a pool
# -----------------------------------------------------------------------------

def score_candidates_with_visual_audit(
    candidates: list[dict[str, Any]],
    search_params: dict[str, Any],
    lane_name_key: str = "origin_lane_name",
    model: str = VISION_MODEL,
    limit: Optional[int] = None,
    sleep_between_calls: float = 0.05,
) -> list[dict[str, Any]]:
    lane_lookup = build_lane_lookup(search_params)
    work = candidates[:limit] if limit is not None else candidates

    output: list[dict[str, Any]] = []

    for candidate in work:
        enriched = dict(candidate)
        lane_name = candidate.get(lane_name_key)
        thumbnail_url = ensure_thumbnail_url(candidate)

        if not thumbnail_url:
            enriched["visual_audit_result"] = None
            enriched["visual_audit_error"] = "Missing thumbnail_url and asset_id"
            output.append(enriched)
            continue

        try:
            candidate_metadata = {
                "asset_id": candidate.get("asset_id"),
                "stage2_score": candidate.get("stage2_score"),
                "origin_lane_name": lane_name,
                "media_type": candidate.get("media_type"),
                "title": candidate.get("title"),
            }

            # Use origin lane if present, otherwise score against all lanes
            if lane_name and lane_name in lane_lookup:
                lanes_to_score = [lane_lookup[lane_name]]
            else:
                lanes_to_score = list(lane_lookup.values())

            lane_results = {}
            best_lane_name = None
            best_lane_result = None
            best_lane_score = -1.0

            for lane in lanes_to_score:
                lane_result = score_candidate_thumbnail_for_lane(
                    thumbnail_url=thumbnail_url,
                    search_params=search_params,
                    lane=lane,
                    candidate_metadata=candidate_metadata,
                    model=model,
                )
                lane_score = float(lane_result.get("overall_fit", 0.0))
                lane_results[lane["lane_name"]] = lane_result

                if lane_score > best_lane_score:
                    best_lane_score = lane_score
                    best_lane_name = lane["lane_name"]
                    best_lane_result = lane_result

            enriched["thumbnail_url"] = thumbnail_url
            enriched["visual_audit_result"] = best_lane_result
            enriched["visual_lane_results"] = lane_results
            enriched["best_lane_name"] = best_lane_name
            enriched["best_lane_score"] = best_lane_score
            enriched["visual_audit_error"] = None

        except Exception as exc:
            enriched["thumbnail_url"] = thumbnail_url
            enriched["visual_audit_result"] = None
            enriched["visual_audit_error"] = str(exc)

        output.append(enriched)
        if sleep_between_calls > 0:
            time.sleep(sleep_between_calls)

    return output


# -----------------------------------------------------------------------------
# Flatten and score aggregation
# -----------------------------------------------------------------------------

def flatten_visual_audit_result(result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {}
    flat: dict[str, Any] = {}
    for k, v in result.items():
        if isinstance(v, list):
            flat[f"visual_{k}"] = " | ".join(map(str, v))
        elif isinstance(v, dict):
            flat[f"visual_{k}"] = json.dumps(v, ensure_ascii=False)
        else:
            flat[f"visual_{k}"] = v
    return flat


def flatten_visual_audit_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for candidate in candidates:
        row = dict(candidate)
        row.update(flatten_visual_audit_result(candidate.get("visual_audit_result")))
        row["best_lane_name"] = candidate.get("best_lane_name")
        row["best_lane_score"] = candidate.get("best_lane_score")
        if "visual_lane_results" in candidate:
            row["visual_lane_results_json"] = json.dumps(candidate["visual_lane_results"], ensure_ascii=False)
        flattened.append(row)
    return flattened


def compute_stage3_scores(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    df = pd.DataFrame(candidates)

    numeric_cols = [
        "stage2_score",
        "visual_subject_match",
        "visual_lane_match",
        "visual_style_match",
        "visual_composition_match",
        "visual_overall_fit",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "visual_likely_exclusion_violation" not in df.columns:
        df["visual_likely_exclusion_violation"] = False

    df["visual_exclusion_penalty"] = df["visual_likely_exclusion_violation"].fillna(False).astype(float) * 0.20

    if "best_lane_score" in df.columns:
        df["best_lane_score"] = pd.to_numeric(df["best_lane_score"], errors="coerce")
    else:
        df["best_lane_score"] = df.get("visual_overall_fit", 0.0)

    global_visual_score = (
        0.35 * df.get("visual_subject_match", 0.0).fillna(0.0) +
        0.35 * df.get("visual_style_match", 0.0).fillna(0.0) +
        0.30 * df.get("visual_composition_match", 0.0).fillna(0.0)
    )

    df["stage3_score"] = (
        0.50 * df.get("stage2_score", 0.0).fillna(0.0) +
        0.25 * global_visual_score +
        0.25 * df["best_lane_score"].fillna(0.0) -
        df["visual_exclusion_penalty"].fillna(0.0)
    )

    return df.to_dict(orient="records")


# -----------------------------------------------------------------------------
# Lane batch audit
# -----------------------------------------------------------------------------

def audit_lane_batch(
    thumbnail_urls: list[str],
    search_params: dict[str, Any],
    lane: dict[str, Any],
    candidate_ids: Optional[list[Any]] = None,
    model: str = VISION_MODEL,
) -> dict[str, Any]:
    if candidate_ids is None:
        candidate_ids = list(range(len(thumbnail_urls)))

    payload = build_set_audit_payload(
        search_params=search_params,
        lane=lane,
        candidate_ids=candidate_ids,
    )

    user_content: list[dict[str, Any]] = [
        {"type": "text", "text": json.dumps(payload, ensure_ascii=False, indent=2)}
    ]
    for url in thumbnail_urls:
        user_content.append({"type": "image_url", "image_url": {"url": url}})

    messages = [
        {"role": "system", "content": SET_AUDIT_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    return call_bifrost_chat_json(
        messages=messages,
        model=model,
        max_tokens=1800,
    )


def build_repair_request_from_audit(
    audit_result: dict[str, Any],
    lane: dict[str, Any],
) -> Optional[dict[str, Any]]:
    if not audit_result.get("repair_needed"):
        return None

    missing_attributes = audit_result.get("missing_attributes", [])
    repair_reason = audit_result.get("repair_reason")

    repair_query = lane.get("embedding_query", "")
    if missing_attributes:
        repair_query = repair_query.rstrip(".") + ". Emphasize: " + ", ".join(missing_attributes) + "."

    return {
        "repair_reason": repair_reason,
        "target_lane_name": lane.get("lane_name"),
        "new_lane": {
            "lane_name": f"{lane.get('lane_name')} Repair",
            "lane_goal": f"Repair lane for {lane.get('lane_name')}",
            "embedding_query": repair_query,
            "literal_terms_preserved": lane.get("literal_terms_preserved", []),
            "visual_proxies": lane.get("visual_proxies", []),
            "lane_filters": lane.get("lane_filters", []),
            "ranking_hints": (lane.get("ranking_hints", []) + missing_attributes)[:10],
        },
    }


# -----------------------------------------------------------------------------
# Shortlist construction
# -----------------------------------------------------------------------------

def build_stage3_shortlist(
    candidates: list[dict[str, Any]],
    top_n: int = 100,
    per_lane_cap: int = 20,
    lane_name_key: str = "origin_lane_name",
) -> list[dict[str, Any]]:
    df = pd.DataFrame(candidates)
    if df.empty:
        return []

    if "stage3_score" not in df.columns:
        raise ValueError("Candidates must include stage3_score before shortlist construction.")

    pieces: list[pd.DataFrame] = []

    if lane_name_key in df.columns:
        for _, group in df.groupby(lane_name_key):
            pieces.append(group.sort_values("stage3_score", ascending=False).head(per_lane_cap))
        base = pd.concat(pieces, axis=0).drop_duplicates(subset=["asset_id"])
    else:
        base = df.copy()

    final = base.sort_values("stage3_score", ascending=False).head(top_n)
    return final.to_dict(orient="records")


# -----------------------------------------------------------------------------
# Lane audit orchestration
# -----------------------------------------------------------------------------

def audit_top_candidates_by_lane(
    candidates: list[dict[str, Any]],
    search_params: dict[str, Any],
    top_per_lane: int = 8,
    lane_name_key: str = "origin_lane_name",
    model: str = VISION_MODEL,
    sleep_between_calls: float = 0.05,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lane_lookup = build_lane_lookup(search_params)
    df = pd.DataFrame(candidates)
    if df.empty or lane_name_key not in df.columns:
        return [], []

    audits: list[dict[str, Any]] = []
    repair_requests: list[dict[str, Any]] = []

    for lane_name, group in df.groupby(lane_name_key):
        lane = lane_lookup.get(lane_name)
        if not lane:
            continue

        group = group.sort_values("stage3_score", ascending=False).head(top_per_lane).copy()
        group["thumbnail_url"] = group.apply(
            lambda r: r.get("thumbnail_url") or build_thumbnail_url_from_asset_id(r.get("asset_id")),
            axis=1,
        )

        thumbnail_urls = [u for u in group["thumbnail_url"].tolist() if u]
        candidate_ids = group["asset_id"].tolist()

        if not thumbnail_urls:
            continue

        try:
            audit = audit_lane_batch(
                thumbnail_urls=thumbnail_urls,
                search_params=search_params,
                lane=lane,
                candidate_ids=candidate_ids,
                model=model,
            )
            audit_record = {
                "lane_name": lane_name,
                "candidate_ids": candidate_ids,
                "audit_result": audit,
            }
            audits.append(audit_record)

            repair = build_repair_request_from_audit(audit, lane)
            if repair:
                repair_requests.append(repair)

        except Exception as exc:
            audits.append(
                {
                    "lane_name": lane_name,
                    "candidate_ids": candidate_ids,
                    "audit_error": str(exc),
                }
            )

        if sleep_between_calls > 0:
            time.sleep(sleep_between_calls)

    return audits, repair_requests


# -----------------------------------------------------------------------------
# LangGraph-friendly node functions
# -----------------------------------------------------------------------------

def score_candidates_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node:
    - reads state["search_params"]
    - reads state["stage2_candidates"]
    - writes state["stage3_candidates"]

    Expected candidate shape:
    {
        "asset_id": ...,
        "origin_lane_name": ...,
        "stage2_score": ...,
        "thumbnail_url": optional,
        ...
    }
    """
    search_params = state["search_params"]
    candidates = state.get("stage2_candidates", [])

    visually_scored = score_candidates_with_visual_audit(
        candidates=candidates,
        search_params=search_params,
        lane_name_key="origin_lane_name",
        model=VISION_MODEL,
    )
    flattened = flatten_visual_audit_candidates(visually_scored)
    stage3_candidates = compute_stage3_scores(flattened)

    return {
        **state,
        "stage3_candidates": stage3_candidates,
    }


def shortlist_candidates_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node:
    - reads state["stage3_candidates"]
    - writes state["stage3_shortlist"]
    """
    candidates = state.get("stage3_candidates", [])
    shortlist = build_stage3_shortlist(
        candidates=candidates,
        top_n=100,
        per_lane_cap=20,
        lane_name_key="origin_lane_name",
    )
    return {
        **state,
        "stage3_shortlist": shortlist,
    }


def audit_lanes_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node:
    - reads state["search_params"]
    - reads state["stage3_candidates"] or state["stage3_shortlist"]
    - writes state["stage3_lane_audits"], state["stage3_repair_requests"]

    Prefers shortlist if present.
    """
    search_params = state["search_params"]
    candidates = state.get("stage3_shortlist") or state.get("stage3_candidates") or []

    audits, repair_requests = audit_top_candidates_by_lane(
        candidates=candidates,
        search_params=search_params,
        top_per_lane=8,
        lane_name_key="origin_lane_name",
        model=VISION_MODEL,
    )

    return {
        **state,
        "stage3_lane_audits": audits,
        "stage3_repair_requests": repair_requests,
    }


def should_repair_router(state: dict[str, Any]) -> str:
    """
    LangGraph conditional router helper.

    Returns:
    - "repair" if repair requests exist
    - "done" otherwise
    """
    repair_requests = state.get("stage3_repair_requests", [])
    return "repair" if repair_requests else "done"


# -----------------------------------------------------------------------------
# Optional utilities for debugging / local testing
# -----------------------------------------------------------------------------

def candidates_to_dataframe(candidates: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(candidates)


def audits_to_dataframe(audits: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in audits:
        row = {"lane_name": record.get("lane_name")}
        if "audit_result" in record and isinstance(record["audit_result"], dict):
            for k, v in record["audit_result"].items():
                if isinstance(v, list):
                    row[k] = " | ".join(map(str, v))
                elif isinstance(v, dict):
                    row[k] = json.dumps(v, ensure_ascii=False)
                else:
                    row[k] = v
        if "audit_error" in record:
            row["audit_error"] = record["audit_error"]
        rows.append(row)
    return pd.DataFrame(rows)