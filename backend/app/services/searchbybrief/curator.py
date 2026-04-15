"""
Stage 3 — Agentic Curation for the searchbybrief pipeline.

Responsibility
--------------
Take the reranked candidate pool from Stage 2 and produce a final curated
shortlist of ≤100 images, using a multimodal LLM to:

  1. Score individual candidate thumbnails against the Stage 0 search lanes
     (score_candidates_node).
  2. Build a per-lane shortlist capped at 20 images per lane, ranked by a
     weighted combination of stage2_score and visual audit scores
     (shortlist_candidates_node).
  3. Audit each lane's batch for diversity, coverage, and exclusion violations.
     If gaps are found, generate a textual repair feedback message for the
     planner and trigger another planning + retrieval loop
     (audit_lanes_node).

Inputs consumed from state
--------------------------
  state["search_params"]    — IntentResult dict from Stage 0
  state["refined_pool"]     — list[CandidateRecord] from Stage 2

Outputs written to state
------------------------
  state["stage3_candidates"]  — visually scored candidates
  state["stage3_shortlist"]   — top 100, per-lane capped
  state["stage3_lane_audits"] — per-lane audit dicts
  state["feedback"]           — "done" or repair description for planner
  state["final_collection"]   — set when feedback == "done"

Dependency notes
----------------
  - Calls call_llm_vision_json from .llm (multimodal Bifrost client)
  - Uses CandidateRecord / RepairRequest contracts from .schemas
  - Does NOT import from stage_3.py
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import statistics
import time
from typing import Any, Optional

import pandas as pd

from app.config import settings
from .llm import call_llm_vision_json, call_llm_json
from .schemas import RepairRequest


# ---------------------------------------------------------------------------
# Vision model
# ---------------------------------------------------------------------------

VISION_MODEL = "gpt-4o-mini"

# Loop back to the planner if the shortlist median stage3_score is below this.
# A few weak lanes are acceptable; the median captures the overall collection quality.
MEDIAN_SCORE_THRESHOLD = 0.70

# Cap expensive visual scoring calls. Candidates are selected in a lane-balanced
# way so each lane contributes top items before we hit this ceiling.
MAX_VISUAL_SCORING_CANDIDATES = 50


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

VISUAL_SCORING_SYSTEM_PROMPT = """
You are a visual auditing model for Stage 3 of a creative search workflow.

Your task is to inspect a candidate image thumbnail and score how well it matches:
1. the overall brief,
2. a specific search lane,
3. key hard constraints and visual style requirements.

Important:
- Use only what is visually inferable from the image plus the provided structured brief context.
- Be conservative. Thumbnails may not reveal small details perfectly.
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

Return JSON only with these fields:
- lane_coverage_quality: "low" | "medium" | "high"
- duplicate_or_redundancy_risk: "low" | "medium" | "high"
- missing_attributes: array of short strings
- best_candidate_ids: array of candidate ids
- drop_candidate_ids: array of candidate ids
- repair_needed: boolean
- repair_reason: string or null
- short_rationale: string
"""


# ---------------------------------------------------------------------------
# Stage 0 context compaction
# ---------------------------------------------------------------------------

def _build_brief_context(search_params: dict[str, Any]) -> dict[str, Any]:
    """Compact IntentResult dict into only the fields the vision model needs."""
    brief_diagnostics = search_params.get("brief_diagnostics", {})
    hard_constraints = search_params.get("hard_constraints", {})

    compact_lanes = [
        {
            "lane_name": lane.get("lane_name"),
            "lane_goal": lane.get("lane_goal"),
            "embedding_query": lane.get("embedding_query"),
            "visual_proxies": lane.get("visual_proxies", []),
            "ranking_hints": lane.get("ranking_hints", []),
        }
        for lane in search_params.get("search_lanes", [])
    ]

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
            "exclusions": hard_constraints.get("exclusions", []),
        },
        "shared_filters": search_params.get("shared_filters", []),
        "search_lanes": compact_lanes,
    }


def _build_lane_lookup(search_params: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        lane["lane_name"]: lane
        for lane in search_params.get("search_lanes", [])
        if lane.get("lane_name")
    }


def _get_thumbnail_url(candidate: dict[str, Any]) -> Optional[str]:
    """Return thumbnail_url if present, else None (no fallback — caller must skip)."""
    url = candidate.get("thumbnail_url")
    return str(url) if url else None


# ---------------------------------------------------------------------------
# Payload builders
# ---------------------------------------------------------------------------

def _build_visual_scoring_payload(
    search_params: dict[str, Any],
    lane: dict[str, Any],
    candidate_metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    return {
        "brief_context": _build_brief_context(search_params),
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


def _build_set_audit_payload(
    search_params: dict[str, Any],
    lane: dict[str, Any],
    candidate_ids: list[Any],
) -> dict[str, Any]:
    return {
        "brief_context": _build_brief_context(search_params),
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


# ---------------------------------------------------------------------------
# Per-candidate visual scoring
# ---------------------------------------------------------------------------

def _score_candidate_for_lane(
    thumbnail_url: str,
    search_params: dict[str, Any],
    lane: dict[str, Any],
    candidate_metadata: Optional[dict[str, Any]] = None,
    model: str = VISION_MODEL,
    api_key_override: Optional[str] = None,
) -> dict[str, Any]:
    payload = _build_visual_scoring_payload(
        search_params=search_params,
        lane=lane,
        candidate_metadata=candidate_metadata,
    )
    messages = [
        {"role": "system", "content": VISUAL_SCORING_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": json.dumps(payload, ensure_ascii=False, indent=2)},
                {"type": "image_url", "image_url": {"url": thumbnail_url}},
            ],
        },
    ]
    return call_llm_vision_json(
        messages=messages,
        model=model,
        max_tokens=1400,
        api_key_override=api_key_override,
    )


def _score_all_candidates(
    candidates: list[dict[str, Any]],
    search_params: dict[str, Any],
    model: str = VISION_MODEL,
    sleep_between_calls: float = 0.05,
    api_key_override: Optional[str] = None,
) -> list[dict[str, Any]]:
    """
    Score each candidate thumbnail against its origin lane (or all lanes if unknown).

    Candidates without a thumbnail_url are passed through with visual_audit_error set.
    """
    lane_lookup = _build_lane_lookup(search_params)

    def _score_one_candidate(idx: int, candidate: dict[str, Any], total_candidates: int) -> dict[str, Any]:
        enriched = dict(candidate)
        thumbnail_url = _get_thumbnail_url(candidate)
        lane_name = candidate.get("origin_lane_name")

        if not thumbnail_url:
            enriched["visual_audit_result"] = None
            enriched["visual_audit_error"] = "Missing thumbnail_url — visual scoring skipped"
            return enriched

        candidate_metadata = {
            "asset_id": candidate.get("asset_id"),
            "stage2_score": candidate.get("stage2_score"),
            "origin_lane_name": lane_name,
            "media_type": candidate.get("media_type"),
            "title": candidate.get("title"),
        }

        lanes_to_score = (
            [lane_lookup[lane_name]]
            if lane_name and lane_name in lane_lookup
            else list(lane_lookup.values())
        )

        print(f"  [stage3 score] ({idx}/{total_candidates}) asset_id={candidate.get('asset_id')}  "
              f"lane={lane_name or 'all'}  scoring {len(lanes_to_score)} lane(s)...", flush=True)
        try:
            t_start = time.perf_counter()
            lane_results: dict[str, Any] = {}
            best_lane_name: Optional[str] = None
            best_lane_result: Optional[dict[str, Any]] = None
            best_lane_score: float = -1.0

            for lane in lanes_to_score:
                result = _score_candidate_for_lane(
                    thumbnail_url=thumbnail_url,
                    search_params=search_params,
                    lane=lane,
                    candidate_metadata=candidate_metadata,
                    model=model,
                    api_key_override=api_key_override,
                )
                score = float(result.get("overall_fit", 0.0))
                lane_results[lane["lane_name"]] = result
                if score > best_lane_score:
                    best_lane_score = score
                    best_lane_name = lane["lane_name"]
                    best_lane_result = result

            latency = round(time.perf_counter() - t_start, 3)
            r = best_lane_result or {}
            s2 = float(candidate.get('stage2_score') or 0.0)
            subj = r.get('subject_match', '?')
            lane_m = r.get('lane_match', '?')
            style = r.get('style_match', '?')
            excl = r.get('likely_exclusion_violation', '?')
            print(f"    → overall_fit={best_lane_score:.2f}  subject={subj}  lane_match={lane_m}  "
                  f"style={style}  stage2={s2:.2f}  exclusion={excl}  ({latency}s)", flush=True)

            enriched["thumbnail_url"] = thumbnail_url
            enriched["visual_audit_result"] = best_lane_result
            enriched["visual_lane_results"] = lane_results
            enriched["best_lane_name"] = best_lane_name
            enriched["best_lane_score"] = best_lane_score
            enriched["visual_audit_error"] = None
            enriched["visual_scoring_latency_s"] = latency

        except Exception as exc:
            print(f"    → ERROR: {exc}", flush=True)
            enriched["thumbnail_url"] = thumbnail_url
            enriched["visual_audit_result"] = None
            enriched["visual_audit_error"] = str(exc)

        if sleep_between_calls > 0:
            time.sleep(sleep_between_calls)
        return enriched

    total_candidates = len(candidates)
    if total_candidates == 0:
        return []

    configured_workers = int(getattr(settings, "searchbybrief_curator_concurrency", 1) or 1)
    max_workers = max(1, configured_workers)
    if max_workers == 1 or total_candidates == 1:
        return [
            _score_one_candidate(idx=i, candidate=candidate, total_candidates=total_candidates)
            for i, candidate in enumerate(candidates, start=1)
        ]

    max_workers = min(max_workers, total_candidates)
    indexed_results: dict[int, dict[str, Any]] = {}
    print(f"[curator] Stage 3 parallel scoring enabled (workers={max_workers})", flush=True)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_score_one_candidate, i, candidate, total_candidates): i
            for i, candidate in enumerate(candidates, start=1)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                indexed_results[idx] = future.result()
            except Exception as exc:
                candidate = candidates[idx - 1]
                indexed_results[idx] = {
                    **candidate,
                    "visual_audit_result": None,
                    "visual_audit_error": f"Parallel scoring worker failed: {exc}",
                }

    return [indexed_results[i] for i in range(1, total_candidates + 1)]


def _select_candidates_for_visual_scoring(
    candidates: list[dict[str, Any]],
    max_total: int = MAX_VISUAL_SCORING_CANDIDATES,
    lane_name_key: str = "origin_lane_name",
) -> list[dict[str, Any]]:
    """
    Select a lane-balanced subset of top candidates for visual scoring.

    Strategy:
    - Bucket by lane.
    - Sort each lane by stage2_score descending (stable with input order).
    - Round-robin pick one from each lane until max_total is reached.
    """
    if max_total <= 0 or not candidates:
        return []
    if len(candidates) <= max_total:
        return candidates

    buckets: dict[str, list[dict[str, Any]]] = {}
    for idx, candidate in enumerate(candidates):
        lane = str(candidate.get(lane_name_key) or "unknown-lane")
        row = dict(candidate)
        row["_input_idx"] = idx
        score = pd.to_numeric(row.get("stage2_score"), errors="coerce")
        row["_stage2_score"] = float(score) if pd.notna(score) else float("-inf")
        buckets.setdefault(lane, []).append(row)

    lane_order = list(buckets.keys())
    for lane in lane_order:
        buckets[lane].sort(key=lambda r: (r["_stage2_score"], -r["_input_idx"]), reverse=True)

    selected: list[dict[str, Any]] = []
    lane_positions = {lane: 0 for lane in lane_order}

    while len(selected) < max_total:
        made_progress = False
        for lane in lane_order:
            pos = lane_positions[lane]
            lane_items = buckets[lane]
            if pos >= len(lane_items):
                continue
            picked = dict(lane_items[pos])
            picked.pop("_input_idx", None)
            picked.pop("_stage2_score", None)
            selected.append(picked)
            lane_positions[lane] = pos + 1
            made_progress = True
            if len(selected) >= max_total:
                break
        if not made_progress:
            break

    return selected


# ---------------------------------------------------------------------------
# Score aggregation
# ---------------------------------------------------------------------------

def _flatten_visual_result(result: Any) -> dict[str, Any]:
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


def _flatten_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for candidate in candidates:
        row = dict(candidate)
        row.update(_flatten_visual_result(candidate.get("visual_audit_result")))
        row["best_lane_name"] = candidate.get("best_lane_name")
        row["best_lane_score"] = candidate.get("best_lane_score")
        if "visual_lane_results" in candidate:
            row["visual_lane_results_json"] = json.dumps(
                candidate["visual_lane_results"], ensure_ascii=False
            )
        flattened.append(row)
    return flattened


def _compute_stage3_scores(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Compute stage3_score as a weighted blend:
      50% stage2_score  (reranker relevance)
      25% global visual score  (subject + style + composition average)
      25% best_lane_score  (lane-specific overall_fit)
      − exclusion penalty (0.20 if likely_exclusion_violation)
    """
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
    exclusion_penalty = (
        df["visual_likely_exclusion_violation"].fillna(False).astype(float) * 0.20
    )

    if "best_lane_score" in df.columns:
        df["best_lane_score"] = pd.to_numeric(df["best_lane_score"], errors="coerce")
    else:
        df["best_lane_score"] = df.get("visual_overall_fit", 0.0)

    global_visual = (
        0.35 * df.get("visual_subject_match", pd.Series(0.0, index=df.index)).fillna(0.0)
        + 0.35 * df.get("visual_style_match", pd.Series(0.0, index=df.index)).fillna(0.0)
        + 0.30 * df.get("visual_composition_match", pd.Series(0.0, index=df.index)).fillna(0.0)
    )

    df["stage3_score"] = (
        0.50 * df.get("stage2_score", pd.Series(0.0, index=df.index)).fillna(0.0)
        + 0.25 * global_visual
        + 0.25 * df["best_lane_score"].fillna(0.0)
        - exclusion_penalty.fillna(0.0)
    )

    return df.to_dict(orient="records")


# ---------------------------------------------------------------------------
# Shortlist construction
# ---------------------------------------------------------------------------

def _build_shortlist(
    candidates: list[dict[str, Any]],
    top_n: int = 100,
    per_lane_cap: int = 20,
    lane_name_key: str = "origin_lane_name",
    min_subject_score: float = 0.5,
) -> list[dict[str, Any]]:
    df = pd.DataFrame(candidates)
    if df.empty:
        return []

    if "stage3_score" not in df.columns:
        raise ValueError("Candidates must include stage3_score before shortlist construction.")

    df["stage3_score"] = pd.to_numeric(df["stage3_score"], errors="coerce")
    df = df.sort_values("stage3_score", ascending=False)

    # Drop clearly off-subject assets before shortlist assembly.
    if "visual_subject_match" in df.columns:
        subject_scores = pd.to_numeric(df["visual_subject_match"], errors="coerce")
        df = df[subject_scores.isna() | (subject_scores >= min_subject_score)]

    # Deduplicate globally by asset_id, keeping the highest stage3_score.
    if "asset_id" in df.columns:
        df = df.drop_duplicates(subset=["asset_id"], keep="first")

    pieces: list[pd.DataFrame] = []
    if lane_name_key in df.columns:
        for _, group in df.groupby(lane_name_key):
            pieces.append(
                group.sort_values("stage3_score", ascending=False).head(per_lane_cap)
            )
        base = pd.concat(pieces, axis=0)
    else:
        base = df.copy()

    if "asset_id" in base.columns:
        base = base.sort_values("stage3_score", ascending=False).drop_duplicates(subset=["asset_id"], keep="first")

    # Interleave lanes while preserving score order within each lane.
    # Use score-aware scheduling so clearly stronger lanes can still dominate,
    # while a soft diversity penalty avoids long contiguous blocks.
    if lane_name_key in base.columns and not base.empty:
        lane_groups: dict[str, list[dict[str, Any]]] = {}
        for lane_name, group in base.groupby(lane_name_key):
            sorted_group = group.sort_values("stage3_score", ascending=False)
            records = sorted_group.to_dict(orient="records")
            lane_groups[str(lane_name)] = records
        lane_order = sorted(
            lane_groups.keys(),
            key=lambda name: lane_groups[name][0].get("stage3_score", 0.0),
            reverse=True,
        )
        lane_idx = {name: 0 for name in lane_order}
        lane_pick_count = {name: 0 for name in lane_order}
        interleaved: list[dict[str, Any]] = []

        diversity_penalty = float(
            getattr(settings, "searchbybrief_curator_diversity_penalty", 0.03) or 0.03
        )
        while len(interleaved) < top_n:
            best_lane: Optional[str] = None
            best_priority = float("-inf")

            for lane_name in lane_order:
                idx = lane_idx[lane_name]
                lane_items = lane_groups[lane_name]
                if idx >= len(lane_items):
                    continue
                next_score = float(lane_items[idx].get("stage3_score") or 0.0)
                # Higher score wins, but repeated picks from the same lane are
                # softly penalized to keep variation in the feed.
                priority = next_score - diversity_penalty * lane_pick_count[lane_name]
                if priority > best_priority:
                    best_priority = priority
                    best_lane = lane_name

            if best_lane is None:
                break

            idx = lane_idx[best_lane]
            interleaved.append(lane_groups[best_lane][idx])
            lane_idx[best_lane] = idx + 1
            lane_pick_count[best_lane] += 1

        return interleaved

    return base.sort_values("stage3_score", ascending=False).head(top_n).to_dict(orient="records")


# ---------------------------------------------------------------------------
# Lane batch audit
# ---------------------------------------------------------------------------

def _audit_lane_batch(
    thumbnail_urls: list[str],
    search_params: dict[str, Any],
    lane: dict[str, Any],
    candidate_ids: Optional[list[Any]] = None,
    model: str = VISION_MODEL,
    api_key_override: Optional[str] = None,
) -> dict[str, Any]:
    if candidate_ids is None:
        candidate_ids = list(range(len(thumbnail_urls)))

    payload = _build_set_audit_payload(
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
    return call_llm_vision_json(
        messages=messages,
        model=model,
        max_tokens=1800,
        api_key_override=api_key_override,
    )


def _rewrite_embedding_query(
    original_query: str,
    missing_attributes: list[str],
    lane_goal: str,
    model: str = VISION_MODEL,
    api_key_override: Optional[str] = None,
) -> str:
    """
    Ask the LLM to write a genuinely different embedding_query for the lane.

    The key constraint: the original query already ran and returned poor results.
    The rewrite must approach the same subject from a meaningfully different visual
    angle — different framing, different scene moment, different POV — so that the
    vector search hits a different region of the embedding space.
    Returns the original query unchanged if the LLM call fails.
    """
    if not missing_attributes:
        return original_query

    system_prompt = (
        "You are a vision-language embedding expert helping fix a failing image search lane.\n\n"
        "The original embedding_query already ran and returned candidates that did NOT match the lane goal. "
        "Your task is to write a NEW query that will retrieve DIFFERENT images by approaching the same "
        "subject from a different visual angle — different scene moment, different framing, different POV, "
        "or different concrete visual details.\n\n"
        "Rules:\n"
        "- Do NOT just rephrase or lightly edit the original. It already failed — write something meaningfully different.\n"
        "- Incorporate the listed missing visual attributes into the new caption.\n"
        "- Output must read as a fluent, descriptive image caption (1-2 sentences), never an instruction.\n"
        "- Never use prefixes like 'Emphasize:', 'Note:', 'Image of:', etc.\n"
        'Return JSON: {"embedding_query": "<new caption>"}'
    )
    user_prompt = (
        f"Lane goal: {lane_goal}\n"
        f"Failed query (do NOT just rephrase this): {original_query}\n"
        f"Missing visual attributes that were absent from retrieved candidates: {', '.join(missing_attributes)}"
    )
    try:
        result = call_llm_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            max_tokens=200,
            api_key_override=api_key_override,
        )
        rewritten = result.get("embedding_query", "").strip()
        return rewritten if rewritten else original_query
    except Exception:
        return original_query


def _build_repair_request(
    audit_result: dict[str, Any],
    lane: dict[str, Any],
    api_key_override: Optional[str] = None,
) -> Optional[RepairRequest]:
    """Convert a lane audit result into a RepairRequest if repair is needed."""
    if not audit_result.get("repair_needed"):
        return None

    missing_attributes: list[str] = audit_result.get("missing_attributes", [])
    repair_reason: str = audit_result.get("repair_reason") or "Coverage gap identified by visual audit"

    # Rewrite the embedding_query as a natural image caption that weaves in the
    # missing attributes — no instruction suffixes, no "Emphasize:" text.
    repair_query = _rewrite_embedding_query(
        original_query=lane.get("embedding_query", ""),
        missing_attributes=missing_attributes,
        lane_goal=lane.get("lane_goal", ""),
        api_key_override=api_key_override,
    )
    print(f"    [repair query] {lane.get('lane_name')!r} → {repair_query!r}", flush=True)

    # Merge missing attributes into visual_proxies without duplicating
    existing_proxies = lane.get("visual_proxies", [])
    extra_proxies = [a for a in missing_attributes if a not in existing_proxies]
    repair_proxies = (existing_proxies + extra_proxies)[:10]

    # Merge into ranking_hints (cap at 10)
    existing_hints = lane.get("ranking_hints", [])
    extra_hints = [a for a in missing_attributes if a not in existing_hints]
    repair_hints = (existing_hints + extra_hints)[:10]

    return RepairRequest(
        repair_reason=repair_reason,
        target_lane_name=lane.get("lane_name", ""),
        new_lane={
            # Same lane_name — the UPDATE replaces this lane in-place in the planner.
            # Stage 1 will re-run the search under the original name with the
            # refined query, so origin_lane_name stays consistent across iterations.
            "lane_name": lane.get("lane_name", ""),
            "lane_goal": lane.get("lane_goal", ""),
            "original_embedding_query": lane.get("embedding_query", ""),
            "embedding_query": repair_query,
            "literal_terms_preserved": lane.get("literal_terms_preserved", []),
            "visual_proxies": repair_proxies,
            "lane_filters": lane.get("lane_filters", []),
            "ranking_hints": repair_hints,
        },
    )


def _audit_top_candidates_by_lane(
    candidates: list[dict[str, Any]],
    search_params: dict[str, Any],
    top_per_lane: int = 8,
    lane_name_key: str = "origin_lane_name",
    model: str = VISION_MODEL,
    sleep_between_calls: float = 0.05,
    api_key_override: Optional[str] = None,
) -> tuple[list[dict[str, Any]], list[RepairRequest]]:
    lane_lookup = _build_lane_lookup(search_params)
    df = pd.DataFrame(candidates)

    if df.empty or lane_name_key not in df.columns:
        return [], []

    audits: list[dict[str, Any]] = []
    repair_requests: list[RepairRequest] = []

    lane_groups = list(df.groupby(lane_name_key))
    for lane_idx, (lane_name, group) in enumerate(lane_groups, 1):
        lane = lane_lookup.get(lane_name)
        if not lane:
            print(f"  [stage3 audit] ({lane_idx}/{len(lane_groups)}) lane={lane_name!r} — not found in search_params, skipping", flush=True)
            continue

        top_group = group.sort_values("stage3_score", ascending=False).head(top_per_lane)
        thumbnail_urls = [
            url for url in top_group["thumbnail_url"].tolist()
            if url
        ]
        candidate_ids = top_group["asset_id"].tolist()

        if not thumbnail_urls:
            print(f"  [stage3 audit] ({lane_idx}/{len(lane_groups)}) lane={lane_name!r} — no thumbnails, skipping", flush=True)
            continue

        print(f"  [stage3 audit] ({lane_idx}/{len(lane_groups)}) lane={lane_name!r}  auditing {len(thumbnail_urls)} thumbnail(s)...", flush=True)
        try:
            audit = _audit_lane_batch(
                thumbnail_urls=thumbnail_urls,
                search_params=search_params,
                lane=lane,
                candidate_ids=candidate_ids,
                model=model,
                api_key_override=api_key_override,
            )
            rn = audit.get("repair_needed", False)
            cov = audit.get("coverage", "?")
            red = audit.get("redundancy", "?")
            missing = audit.get("missing_attributes", [])
            print(f"    → repair_needed={rn}  coverage={cov}  redundancy={red}"
                  f"{('  missing=' + str(missing)) if missing else ''}", flush=True)
            audits.append({"lane_name": lane_name, "candidate_ids": candidate_ids, "audit_result": audit})

            repair = _build_repair_request(audit, lane, api_key_override=api_key_override)
            if repair:
                repair_requests.append(repair)

        except Exception as exc:
            audits.append({"lane_name": lane_name, "candidate_ids": candidate_ids, "audit_error": str(exc)})

        if sleep_between_calls > 0:
            time.sleep(sleep_between_calls)

    return audits, repair_requests


# ---------------------------------------------------------------------------
# Repair feedback formatter
# ---------------------------------------------------------------------------

def _format_repair_feedback(repair_requests: list[RepairRequest]) -> str:
    """
    Convert a list of RepairRequests into a REPAIR LANE directive block for the
    planner.  Critically, we do NOT supply a pre-baked replacement query —
    instead we tell the planner WHAT failed and WHY, so it must derive a fresh
    embedding_query from the lane_goal itself rather than copying our suggestion.
    """
    lines = [
        "CURATOR REPAIR FEEDBACK",
        "=" * 60,
        "Stage 3 visual audit found lanes whose retrieved candidates did NOT match",
        "the lane goal. For each REPAIR LANE directive below:",
        "  - The 'Failed query' is the embedding_query that was used and DID NOT WORK.",
        "  - Do NOT copy, rephrase, or lightly edit the failed query. It already failed.",
        "  - Write a BRAND NEW embedding_query from scratch using the lane_goal and",
        "    missing_attributes. Approach the subject from a different visual angle,",
        "    different scene moment, or different concrete detail set.",
        "  - Update visual_proxies and ranking_hints to match your new query.",
        "  - Preserve all other lane fields (lane_name, lane_goal, lane_filters, etc.) exactly.",
        "  - Preserve every lane NOT listed here exactly as-is.",
        "",
    ]
    for req in repair_requests:
        new_lane = req["new_lane"]
        orig_query = new_lane.get("original_embedding_query", "") or new_lane.get("embedding_query", "")
        missing = new_lane.get("visual_proxies", [])  # proxies carry the missing attrs
        # Pull missing_attributes from the audit result stored in new_lane's extra hints
        missing_attrs = [
            a for a in (new_lane.get("ranking_hints", []))
            if not a.startswith("prefer ")
        ]
        lines += [
            f"REPAIR LANE \"{req['target_lane_name']}\"",
            f"  Lane goal   : {new_lane.get('lane_goal', '')}",
            f"  Failed query: {orig_query}",
            f"  Missing attributes absent from retrieved candidates: {missing_attrs or missing}",
            f"  Problem     : {req['repair_reason']}",
            f"  Action      : Write a new embedding_query from scratch using the lane_goal above.",
            f"                Do NOT copy or rephrase the failed query.",
            "",
        ]

    lines.append("=" * 60)
    lines.append("END CURATOR REPAIR FEEDBACK")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LangGraph node functions  (imported by main.py)
# ---------------------------------------------------------------------------

def curator_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node — Stage 3: Agentic curation (single node for main.py graph).

    Internally runs visual scoring, shortlist construction, and a median-score
    quality check. Only invokes the lane audit (and potentially loops back to the
    planner) when the median stage3_score of the shortlist falls below
    MEDIAN_SCORE_THRESHOLD — a few weak lanes are tolerated so long as the
    overall collection clears the bar.

    Reads:
      state["search_params"]  — IntentResult as dict or Pydantic model
      state["refined_pool"]   — list[CandidateRecord] from Stage 2

    Writes:
      state["stage3_candidates"]  — visually scored candidates
      state["stage3_shortlist"]   — top 100, per-lane capped at 20
      state["stage3_lane_audits"] — per-lane audit dicts (empty if median check passes)
      state["feedback"]           — "done" or repair directives for the planner
      state["final_collection"]   — set when feedback == "done"
    """
    search_params = state["search_params"]
    if hasattr(search_params, "model_dump"):
        search_params = search_params.model_dump()

    candidates = state.get("refined_pool", [])
    api_key_override = state.get("openai_api_key")
    scoring_candidates = _select_candidates_for_visual_scoring(
        candidates=candidates,
        max_total=MAX_VISUAL_SCORING_CANDIDATES,
        lane_name_key="origin_lane_name",
    )
    print(
        f"\n[curator] candidate prefilter — selected {len(scoring_candidates)} / {len(candidates)} "
        f"for visual scoring (lane-balanced cap={MAX_VISUAL_SCORING_CANDIDATES})",
        flush=True,
    )

    # Step 1: visual scoring
    print(f"[curator] Step 1/3 — visual scoring {len(scoring_candidates)} candidate(s)...", flush=True)
    visually_scored = _score_all_candidates(
        candidates=scoring_candidates,
        search_params=search_params,
        api_key_override=api_key_override,
    )
    flattened = _flatten_candidates(visually_scored)
    stage3_candidates = _compute_stage3_scores(flattened)

    # Step 2: shortlist (sorted by stage3_score, per-lane capped)
    print(f"[curator] Step 2/3 — building shortlist from {len(stage3_candidates)} scored candidates...", flush=True)
    shortlist = _build_shortlist(
        candidates=stage3_candidates,
        top_n=100,
        per_lane_cap=20,
        lane_name_key="origin_lane_name",
    )

    # Step 3: median quality check — weak individual lanes are OK; we care about
    # the overall collection clearing the bar, not every lane being perfect.
    print(f"[curator] Step 3/3 — median score check (shortlist={len(shortlist)})...", flush=True)
    scores = [
        c["stage3_score"]
        for c in shortlist
        if isinstance(c.get("stage3_score"), (int, float))
    ]
    median_score = statistics.median(scores) if scores else 0.0
    print(f"[curator] median stage3_score={median_score:.3f}  threshold={MEDIAN_SCORE_THRESHOLD}  "
          f"→ {'PASS (done)' if median_score >= MEDIAN_SCORE_THRESHOLD else 'FAIL — running lane audit'}", flush=True)

    if median_score >= MEDIAN_SCORE_THRESHOLD:
        return {
            **state,
            "stage3_candidates": stage3_candidates,
            "stage3_shortlist": shortlist,
            "stage3_lane_audits": [],
            "feedback": "done",
            "final_collection": shortlist,
        }

    # Step 4: median too low — audit lanes to generate targeted repair directives.
    print(f"[curator] Step 4 — auditing {len(set(c.get('origin_lane_name') for c in shortlist))} lane(s)...", flush=True)
    audits, repair_requests = _audit_top_candidates_by_lane(
        candidates=shortlist,
        search_params=search_params,
        top_per_lane=8,
        lane_name_key="origin_lane_name",
        api_key_override=api_key_override,
    )

    feedback = _format_repair_feedback(repair_requests) if repair_requests else "done"
    result: dict[str, Any] = {
        **state,
        "stage3_candidates": stage3_candidates,
        "stage3_shortlist": shortlist,
        "stage3_lane_audits": audits,
        # Store repair request dicts so callers (e.g. test report) can show query diffs
        "stage3_repair_requests": [dict(r) for r in repair_requests],
        "feedback": feedback,
    }
    if feedback == "done":
        result["final_collection"] = shortlist
    return result


def score_candidates_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node — Stage 3a: Visual scoring.

    Reads:
      state["search_params"]  — IntentResult as dict
      state["refined_pool"]   — list[CandidateRecord] from Stage 2

    Writes:
      state["stage3_candidates"] — candidates enriched with visual_audit_result
                                   and stage3_score
    """
    search_params = state["search_params"]
    candidates = state.get("refined_pool", [])
    api_key_override = state.get("openai_api_key")
    scoring_candidates = _select_candidates_for_visual_scoring(
        candidates=candidates,
        max_total=MAX_VISUAL_SCORING_CANDIDATES,
        lane_name_key="origin_lane_name",
    )

    # If search_params is a Pydantic model, convert to dict for downstream helpers
    if hasattr(search_params, "model_dump"):
        search_params = search_params.model_dump()

    visually_scored = _score_all_candidates(
        candidates=scoring_candidates,
        search_params=search_params,
        api_key_override=api_key_override,
    )
    flattened = _flatten_candidates(visually_scored)
    stage3_candidates = _compute_stage3_scores(flattened)

    return {**state, "stage3_candidates": stage3_candidates}


def shortlist_candidates_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node — Stage 3b: Shortlist construction.

    Reads:
      state["stage3_candidates"] — scored candidates from score_candidates_node

    Writes:
      state["stage3_shortlist"]  — top 100 across all lanes, per-lane capped at 20
    """
    candidates = state.get("stage3_candidates", [])
    shortlist = _build_shortlist(
        candidates=candidates,
        top_n=100,
        per_lane_cap=20,
        lane_name_key="origin_lane_name",
    )
    return {**state, "stage3_shortlist": shortlist}


def audit_lanes_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node — Stage 3c: Lane diversity audit and repair decision.

    Reads:
      state["search_params"]       — IntentResult as dict
      state["stage3_shortlist"]    — preferred input (falls back to stage3_candidates)

    Writes:
      state["stage3_lane_audits"]  — per-lane audit result dicts
      state["feedback"]            — "done" if no repairs needed, or a textual
                                     repair description sent back to the planner
      state["final_collection"]    — set to stage3_shortlist when feedback == "done"
    """
    search_params = state["search_params"]
    if hasattr(search_params, "model_dump"):
        search_params = search_params.model_dump()

    candidates = state.get("stage3_shortlist") or state.get("stage3_candidates") or []
    api_key_override = state.get("openai_api_key")

    audits, repair_requests = _audit_top_candidates_by_lane(
        candidates=candidates,
        search_params=search_params,
        top_per_lane=8,
        lane_name_key="origin_lane_name",
        api_key_override=api_key_override,
    )

    if repair_requests:
        feedback = _format_repair_feedback(repair_requests)
        return {
            **state,
            "stage3_lane_audits": audits,
            "feedback": feedback,
            # final_collection remains unset; planner will refine and loop
        }

    return {
        **state,
        "stage3_lane_audits": audits,
        "feedback": "done",
        "final_collection": state.get("stage3_shortlist", []),
    }
