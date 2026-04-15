"""
Stage 0: Intent Extraction (Planner node)

Converts a customer creative brief into a structured search plan with:
- Brief diagnostics (form, retrieval intent, complexity)
- Hard and operational constraints
- One or more search lanes, each with an embedding-ready query
"""

import json
from typing import Optional

from app.config import settings
from .llm import call_llm_json
from .schemas import IntentResult


# ---------------------------------------------------------------------------
# Schema definitions — passed to the LLM as part of the prompt so it can
# select valid enum values.
# ---------------------------------------------------------------------------

INTENT_NODE_SCHEMA = {
    "brief_form": {
        "type": "single-label",
        "definition": "Primary structural form of the brief as provided.",
        "allowed_values": {
            "keyword_list": "Short or long list of requested subjects/topics.",
            "conceptual_creative": "Narrative or campaign-style request requiring interpretation into visual concepts.",
            "reference_match": "Driven by example assets, attached images, links, or look-alike requests.",
            "history_or_account_based": "Driven by prior licenses, portfolio usage, account history, or customer history.",
            "attachment_or_sheet_driven": "Main instructions live in decks, docs, sheets, or other attachments.",
        },
    },
    "retrieval_intent": {
        "type": "single-label",
        "definition": "Dominant objective of search and curation.",
        "allowed_values": {
            "exact_asset_recovery": "Find the exact image/video or a higher-resolution/versional equivalent.",
            "similar_asset_search": "Find visually similar content to references.",
            "concept_translation": "Translate abstract business or creative needs into search-ready visual concepts.",
            "catalog_population": "Populate collections, folders, content banks, or categories at scale.",
            "account_expansion_or_depletion": "Recommend assets based on past account behavior or spend-down context.",
        },
    },
    "search_complexity": {
        "type": "single-label",
        "definition": "Estimated search difficulty from an execution perspective.",
        "allowed_values": {
            "low": "Straightforward retrieval with limited nuance and few hard filters.",
            "medium": "Some interpretation or constraints but manageable with standard search.",
            "high": "Multiple constraints, style nuance, multiple buckets, or decomposition required.",
            "very_high": "Likely requires iterative review, deep decomposition, or cross-reference handling.",
        },
    },
}

CONSTRAINT_LABELS = {
    "hard_constraint_categories": [
        "subjects_required",
        "demographics_required",
        "composition_required",
        "style_required",
        "location_required",
        "media_type_required",
        "licensing_required",
        "exclusions",
    ],
    "operational_constraint_categories": [
        "output_structure",
        "volume_targets",
        "coverage_requirements",
        "reference_dependency",
        "attachment_dependency",
    ],
}

# Output contract shown verbatim to the LLM.
INTENT_NODE_OUTPUT_SPEC = {
    "brief_diagnostics": {
        "brief_form": "one of schema brief_form allowed_values",
        "retrieval_intent": "one of schema retrieval_intent allowed_values",
        "search_complexity": "one of schema search_complexity allowed_values",
        "is_multi_lane": "boolean",
        "reasoning_summary": "1-3 sentence summary",
    },
    "hard_constraints": {
        "subjects_required": [],
        "demographics_required": [],
        "composition_required": [],
        "style_required": [],
        "location_required": [],
        "media_type_required": [],
        "licensing_required": [],
        "exclusions": [],
    },
    "operational_constraints": {
        "output_structure": [],
        "volume_targets": [],
        "coverage_requirements": [],
        "reference_dependency": [],
        "attachment_dependency": [],
    },
    "shared_filters": [],
    "search_lanes": [
        {
            "lane_name": "short lane name",
            "lane_goal": "what this lane is trying to retrieve",
            "embedding_query": "a natural-language visual retrieval description for embedding search",
            "literal_terms_preserved": [],
            "visual_proxies": [],
            "lane_filters": [],
            "ranking_hints": [],
        }
    ],
}


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are Stage 0: Intent Extraction for an embedding-based image search agent.

Your job is to convert a customer creative brief into a structured search plan that will drive vector similarity search against an image index.

## Core principle: one semantic direction per lane

Each search lane's embedding_query will be encoded as a single vector and compared against image embedding vectors in the index. A query that describes two or more distinct visual subjects simultaneously (e.g. "a vaccine syringe next to a calculator and a thermometer") produces a blended vector that matches none of them well. Every lane must have exactly ONE clear visual subject.

## Writing good embedding_queries

Write embedding_queries as if they are image captions or alt-text — not as creative briefs or business descriptions. Good queries:
- Describe the visual content directly: what is shown, how it looks, what context it is in
- Are 1–2 sentences maximum
- Have a single subject (one animal, one object type, one scene type)
- Include medium/style only when that style is confirmed or strongly implied by the brief
- Avoid business language ("designed to represent", "suitable for", "intended to depict")

Bad: "A digital illustration that visually represents a calculator specifically for temperature stability related to vaccines, in a modern and clear style."
Good: "A flat vector icon of a medical syringe and vaccine vial on a white background."

## Lane decomposition rules

1. **Enumerated subject lists**: group related items into thematic lanes. Keep specific names in literal_terms_preserved. Give a lane to any sub-group with 4+ visually distinct items.
2. **Composite concepts**: when a brief asks for a concept made of multiple distinct visual components (e.g. "temperature stability calculator for vaccines" = syringe/vaccine + calculator/computation + thermometer/temperature), give EACH component its own lane. The retriever merges pools downstream — it is always better to over-decompose than to fuse.
3. **Single clear subjects**: a lane about "dolphins" should not also include whales unless they are visually inseparable.
4. **Activity lists within a lane**: if a brief lists multiple distinct activities within a single theme (e.g. "beach: sunning, sailing, paragliding, water skiing"), each activity is a visually different scene and should get its own lane. Do not bundle them into one — a person lying on a beach and a person paragliding will not appear near each other in any embedding space.
5. **POV and camera perspective**: if the brief specifies a shooting perspective or authorship style (e.g. "first-person POV", "UGC", "social influencer style", "from the POV of X"), this is a strong visual signal. Capture it in shared_filters (e.g. "style:ugc", "style:first_person_pov") AND reflect it in every embedding_query (e.g. "first-person view looking down at...", "handheld-style footage of..."). If a specific sub-group has a different POV (e.g. "band tour from the POV of the band" means backstage/tour-bus/green-room, NOT a stage performance shot), reflect that in the lane's embedding_query accurately.

## Style inference

- If the brief strongly implies a medium (e.g. "coloring book" → line art, isolated on white; "icon set" → flat vector; "campaign shoot" → photography, candid), infer it and add to style_required and shared_filters.
- If the brief is open-ended or style-agnostic (e.g. "open to anything"), do NOT fabricate style constraints. Leave style_required empty and omit style from embedding_queries so the search is not artificially narrowed.
- visual_proxies should always contain 2–3 concrete visual descriptors, but only descriptors that genuinely apply — never invented ones.

## Hard constraints population

- subjects_required: list ALL distinct subjects and themes named in the brief — these become the checklist against which the final collection is verified. Do not leave subjects_required empty if the brief names specific subjects, themes, or scene types.
- style_required: only list styles that are explicitly stated or strongly implied. Do not invent styles.
- exclusions: list anything the brief explicitly says to avoid.

## Filters and hints

- shared_filters: metadata constraints that apply to ALL lanes (e.g. "media_type:illustration" when the brief implies illustrations only; "style:ugc" or "style:first_person_pov" when the brief asks for UGC/influencer/POV content; "has_model_release" for people-featuring briefs).
- lane_filters: constraints for one lane only (e.g. "location:underwater" for a fish lane but not a beach bird lane).
- ranking_hints: soft preferences for Stage 2 reranker, not hard filters. Examples: "prefer isolated on white over complex backgrounds", "prefer horizontal orientation", "deprioritise images with text overlays", "prefer images where the subject fills the frame", "prefer bright and sunny lighting".

Return valid JSON only — no markdown fences, no commentary outside the JSON object.

## Handling CURATOR REPAIR FEEDBACK

When the attachment text contains a block starting with "CURATOR REPAIR FEEDBACK", Stage 3
found lanes whose retrieved candidates did NOT visually match the lane goal.

The attachment will also contain a "CURRENT SEARCH PLAN" block with the full existing
IntentResult JSON. Use that as your base — copy all fields verbatim and apply only the
changes specified in the CURATOR REPAIR FEEDBACK directives.

Rules:
1. Preserve every lane NOT mentioned in the feedback exactly as it appears in the current plan.
2. For each REPAIR LANE directive:
   a. The "Failed query" is the embedding_query that already ran and produced bad results.
      Do NOT copy it, rephrase it, or use it as a starting point. It failed.
   b. Write a BRAND NEW embedding_query from scratch, starting only from the lane_goal
      and the listed missing_attributes. Use a different visual scene, moment, framing,
      or concrete detail set so the vector search hits a different part of the embedding space.
   c. Replace visual_proxies and ranking_hints to match your new query.
   d. Preserve lane_name, lane_goal, lane_filters, and literal_terms_preserved exactly.
3. Do not add or remove lanes.
4. Re-emit the full IntentResult JSON with only those targeted field replacements applied.\
"""

_USER_TEMPLATE = """\
Schema definitions:
{schema_json}

Constraint categories:
{constraint_json}

Required output JSON contract:
{output_spec_json}

Input brief:
{brief_text}

Optional attachment/reference text:
{attachment_text}

Instructions:
1. Check whether the attachment text contains a "CURATOR REPAIR FEEDBACK" block.
   - YES: take the "CURRENT SEARCH PLAN" JSON as the base. For each REPAIR LANE directive,
     find that lane by name and write a BRAND NEW embedding_query from the lane_goal and
     missing_attributes — do NOT copy or rephrase the "Failed query". Update visual_proxies
     and ranking_hints to match. Preserve all other lanes and all other fields exactly.
     Skip steps 2-7.
   - NO: proceed with full plan derivation (steps 2-7 below).
2. Diagnose the brief (brief_form, retrieval_intent, search_complexity, is_multi_lane).
3. Extract hard constraints and operational constraints. Always populate subjects_required with the named subjects, themes, and scene types from the brief.
4. Infer implied style/media type ONLY if the brief strongly implies one — add to style_required and shared_filters. If the brief specifies a shooting style or POV (UGC, first-person, influencer), capture it in shared_filters (e.g. "style:ugc", "style:first_person_pov") and reflect it in embedding_queries.
5. Decompose into search lanes:
   - One lane per distinct visual subject, scene type, or activity — never bundle visually different activities in one lane.
   - For composite concepts, split into one lane per component.
   - For enumerated lists, group into thematic lanes; preserve specific names in literal_terms_preserved.
   - For POV-specified sub-groups, write the embedding_query from the correct visual perspective (e.g. "band tour POV" → backstage/green-room/tour bus, not a stage shot).
6. For each lane, write an embedding_query as an image caption: short, visual, single subject, no business language. Include the implied shooting style/POV if specified.
7. Populate visual_proxies with 2–3 concrete visual descriptors that genuinely apply to this lane.
8. Add ranking_hints where there are soft visual preferences (lighting, orientation, framing, mood, etc.).
9. Add shared_filters for metadata constraints that apply across all lanes.

Return JSON with exactly these top-level keys:
brief_diagnostics, hard_constraints, operational_constraints, shared_filters, search_lanes\
"""

_SYSTEM_PROMPT_V2 = """\
You are Stage 0 Planner (compact mode) for an embedding-based image search workflow.

Goal:
- Produce a SHORT search plan optimized for speed.
- Focus on high-quality embedding queries; avoid verbose diagnostics.

Rules:
1. Return ONLY JSON.
2. Create 2-6 lanes depending on brief complexity.
3. Each lane must represent ONE clear visual subject/scene.
4. embedding_query should be 1 sentence, concrete visual language, no business prose.
5. If brief specifies style/POV (e.g. UGC, first-person), reflect that in the query.
6. Keep shared_filters minimal (only explicit hard requirements).
7. Keep lane_name short (2-6 words).
8. On repair iterations (CURATOR REPAIR FEEDBACK present), preserve lane names and rewrite only failing lane embedding_query entries.
"""

_USER_TEMPLATE_V2 = """\
Input brief:
{brief_text}

Optional attachment/reference text:
{attachment_text}

Return JSON with exactly this structure:
{{
  "shared_filters": ["optional", "global", "filters"],
  "search_lanes": [
    {{
      "lane_name": "short lane name",
      "embedding_query": "single-subject visual caption"
    }}
  ]
}}
"""


def _normalize_compact_output(raw: dict) -> dict:
    """
    Convert compact v2 planner output into full IntentResult shape.
    """
    shared_filters = raw.get("shared_filters")
    if not isinstance(shared_filters, list):
        shared_filters = []
    shared_filters = [str(x).strip() for x in shared_filters if str(x).strip()]

    lanes_raw = raw.get("search_lanes")
    if not isinstance(lanes_raw, list):
        lanes_raw = []

    lanes = []
    seen_names: set[str] = set()
    for idx, lane in enumerate(lanes_raw, start=1):
        if not isinstance(lane, dict):
            continue
        lane_name = str(lane.get("lane_name") or "").strip()
        embedding_query = str(lane.get("embedding_query") or "").strip()
        if not embedding_query:
            continue
        if not lane_name:
            lane_name = f"lane {idx}"
        original_name = lane_name
        suffix = 2
        while lane_name.lower() in seen_names:
            lane_name = f"{original_name} {suffix}"
            suffix += 1
        seen_names.add(lane_name.lower())
        lanes.append(
            {
                "lane_name": lane_name,
                "lane_goal": str(lane.get("lane_goal") or lane_name),
                "embedding_query": embedding_query,
                "literal_terms_preserved": [],
                "visual_proxies": [],
                "lane_filters": [],
                "ranking_hints": [],
            }
        )

    if not lanes:
        fallback_query = "natural, commercially usable lifestyle image matching the brief"
        lanes.append(
            {
                "lane_name": "primary lane",
                "lane_goal": "primary lane",
                "embedding_query": fallback_query,
                "literal_terms_preserved": [],
                "visual_proxies": [],
                "lane_filters": [],
                "ranking_hints": [],
            }
        )

    return {
        "brief_diagnostics": {
            "brief_form": "conceptual_creative",
            "retrieval_intent": "concept_translation",
            "search_complexity": "medium" if len(lanes) <= 3 else "high",
            "is_multi_lane": len(lanes) > 1,
            "reasoning_summary": "Compact planner v2 output focused on embedding queries.",
        },
        "hard_constraints": {
            "subjects_required": [],
            "demographics_required": [],
            "composition_required": [],
            "style_required": [],
            "location_required": [],
            "media_type_required": [],
            "licensing_required": [],
            "exclusions": [],
        },
        "operational_constraints": {
            "output_structure": [],
            "volume_targets": [],
            "coverage_requirements": [],
            "reference_dependency": [],
            "attachment_dependency": [],
        },
        "shared_filters": shared_filters,
        "search_lanes": lanes,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_intent_node(
    brief_text: str,
    attachment_text: Optional[str] = None,
    model: Optional[str] = None,
    api_key_override: Optional[str] = None,
) -> IntentResult:
    """
    Extract a structured search plan from a customer brief.

    Args:
        brief_text: The raw brief text from the user.
        attachment_text: Optional extra context (e.g. curator feedback from a
            previous iteration, or text extracted from an uploaded document).
        model: Override the default Bifrost model for this call.

    Returns:
        Validated IntentResult parsed from the LLM response.
    """
    if model is None:
        model = settings.bifrost_model

    planner_version = (settings.searchbybrief_planner_version or "v1").strip().lower()
    if planner_version == "v2":
        user_prompt = _USER_TEMPLATE_V2.format(
            brief_text=brief_text,
            attachment_text=attachment_text or "None provided",
        )
        raw = call_llm_json(
            system_prompt=_SYSTEM_PROMPT_V2,
            user_prompt=user_prompt,
            model=model,
            max_tokens=settings.searchbybrief_planner_max_tokens_v2,
            api_key_override=api_key_override,
        )
        normalized = _normalize_compact_output(raw)
        return IntentResult.model_validate(normalized)

    user_prompt = _USER_TEMPLATE.format(
        schema_json=json.dumps(INTENT_NODE_SCHEMA, indent=2),
        constraint_json=json.dumps(CONSTRAINT_LABELS, indent=2),
        output_spec_json=json.dumps(INTENT_NODE_OUTPUT_SPEC, indent=2),
        brief_text=brief_text,
        attachment_text=attachment_text or "None provided",
    )

    raw = call_llm_json(
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        model=model,
        max_tokens=settings.searchbybrief_planner_max_tokens_v1,
        api_key_override=api_key_override,
    )
    return IntentResult.model_validate(raw)
