"""
Pydantic schemas for the searchbybrief pipeline.

These define the contract between stages:
  - Stage 0 (planner) produces: IntentResult
  - Stage 1 (retriever) consumes: IntentResult, reads search_lanes + shared_filters + hard_constraints
  - Stage 2 (reranker) consumes: candidate_pool items
  - Stage 3 (curator) consumes: refined_pool items
"""

from typing import Literal, Optional
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Stage 0 output — search plan produced by the planner node
# ---------------------------------------------------------------------------

class BriefDiagnostics(BaseModel):
    brief_form: Literal[
        "keyword_list",
        "conceptual_creative",
        "reference_match",
        "history_or_account_based",
        "attachment_or_sheet_driven",
    ]
    retrieval_intent: Literal[
        "exact_asset_recovery",
        "similar_asset_search",
        "concept_translation",
        "catalog_population",
        "account_expansion_or_depletion",
    ]
    search_complexity: Literal["low", "medium", "high", "very_high"]
    is_multi_lane: bool
    reasoning_summary: str


class HardConstraints(BaseModel):
    subjects_required: list[str] = []
    demographics_required: list[str] = []
    composition_required: list[str] = []
    style_required: list[str] = []
    location_required: list[str] = []
    media_type_required: list[str] = []
    licensing_required: list[str] = []
    exclusions: list[str] = []


class OperationalConstraints(BaseModel):
    output_structure: list[str] = []
    volume_targets: list[str] = []
    coverage_requirements: list[str] = []
    reference_dependency: list[str] = []
    attachment_dependency: list[str] = []


class SearchLane(BaseModel):
    lane_name: str
    lane_goal: str
    # Primary input to Stage 1: embed this string for vector search
    embedding_query: str
    # Specific subject terms to preserve; can be used for keyword boosting
    literal_terms_preserved: list[str] = []
    # Visual style descriptors; can inform reranker scoring in Stage 2
    visual_proxies: list[str] = []
    # Metadata filters applied to this lane only (e.g. "editorial", "location:ocean")
    lane_filters: list[str] = []
    # Soft hints for Stage 2 reranker (not hard filters)
    ranking_hints: list[str] = []


class IntentResult(BaseModel):
    """
    Full output from Stage 0 (planner_node). Stored in state["search_params"].

    Stage 1 (retriever) usage:
      - Iterate over `search_lanes`, embed each `embedding_query`, run one vector search per lane
      - Apply `shared_filters` to every lane query
      - Apply `lane.lane_filters` to each individual lane query
      - Use `hard_constraints.exclusions` and `hard_constraints.licensing_required` as post-filters
      - Merge per-lane results, deduplicate by image ID → candidate_pool
    """
    brief_diagnostics: BriefDiagnostics
    hard_constraints: HardConstraints
    operational_constraints: OperationalConstraints
    # Metadata filters that apply across ALL lanes (e.g. model-released, non-editorial)
    shared_filters: list[str] = []
    search_lanes: list[SearchLane]
