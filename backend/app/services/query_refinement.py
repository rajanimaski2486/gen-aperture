"""
Query refinement: maps brief-analysis attributes to OpenSearch filter clauses.

ATTRIBUTE_FIELD_MAP is the single source of truth that associates normalised
text triggers with their OpenSearch filter specification.  Adding new
mappings never requires touching the filter-building logic — just extend the
dict.

Attribute text comes from the Project Manager's brief analysis, which covers:
  • Visual Requirements  (subjects, scenes, compositions)
  • Themes and Moods     (emotions, atmosphere, style)
  • Technical Constraints (orientation, colour palette, quality)
"""
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Extensible attribute → OpenSearch filter mapping
# ---------------------------------------------------------------------------
# Each key is a normalised trigger phrase (lowercase, single spaces).
# Each value is a filter spec:
#   "field"  – OpenSearch field name
#   "type"   – "term" | "terms" | "range"
#   "value"  – for term / terms filters
#   "gte"    – lower bound for range filters  (use ES date-math or numbers)
#   "lte"    – upper bound for range filters
# ---------------------------------------------------------------------------

ATTRIBUTE_FIELD_MAP: Dict[str, Dict[str, Any]] = {

    # ── Orientation ──────────────────────────────────────────────────────
    "horizontal":            {"field": "orientation", "type": "term", "value": "horizontal"},
    "landscape orientation": {"field": "orientation", "type": "term", "value": "horizontal"},
    "wide format":           {"field": "orientation", "type": "term", "value": "horizontal"},
    "widescreen":            {"field": "orientation", "type": "term", "value": "horizontal"},
    "vertical":              {"field": "orientation", "type": "term", "value": "vertical"},
    "portrait orientation":  {"field": "orientation", "type": "term", "value": "vertical"},
    "square":                {"field": "orientation", "type": "term", "value": "square"},
    "square format":         {"field": "orientation", "type": "term", "value": "square"},

    # ── Popularity / license threshold ───────────────────────────────────
    "highly licensed":       {"field": "total_paid_license_count_all_time", "type": "range", "gte": 500},
    "very popular":          {"field": "total_paid_license_count_all_time", "type": "range", "gte": 500},
    "most popular":          {"field": "total_paid_license_count_all_time", "type": "range", "gte": 500},
    "popular images":        {"field": "total_paid_license_count_all_time", "type": "range", "gte": 5},
    "trending":              {"field": "total_paid_license_count_all_time", "type": "range", "gte": 5},
    "bestselling":           {"field": "total_paid_license_count_all_time", "type": "range", "gte": 5},
    "best selling":          {"field": "total_paid_license_count_all_time", "type": "range", "gte": 5},
    "high quality":          {"field": "total_paid_license_count_all_time", "type": "range", "gte": 100},
    "premium quality":       {"field": "total_paid_license_count_all_time", "type": "range", "gte": 100},
    "premium":               {"field": "total_paid_license_count_all_time", "type": "range", "gte": 100},
    "professional quality":  {"field": "total_paid_license_count_all_time", "type": "range", "gte": 100},
    "professionally shot":   {"field": "total_paid_license_count_all_time", "type": "range", "gte": 100},
    "commercial quality":    {"field": "total_paid_license_count_all_time", "type": "range", "gte": 50},

    # ── Recency ───────────────────────────────────────────────────────────
    "very recent":           {"field": "date_added", "type": "range", "gte": "now-6M"},
    "very new":              {"field": "date_added", "type": "range", "gte": "now-6M"},
    "recent images":         {"field": "date_added", "type": "range", "gte": "now-1y"},
    "recent":                {"field": "date_added", "type": "range", "gte": "now-1y"},
    "latest":                {"field": "date_added", "type": "range", "gte": "now-1y"},
    "up to date":            {"field": "date_added", "type": "range", "gte": "now-2y"},
    "current":               {"field": "date_added", "type": "range", "gte": "now-2y"},
    "contemporary":          {"field": "date_added", "type": "range", "gte": "now-2y"},
    "modern imagery":        {"field": "date_added", "type": "range", "gte": "now-3y"},
    "modern":                {"field": "date_added", "type": "range", "gte": "now-3y"},
}

# Fields where only one filter clause should appear (most-specific wins,
# determined by longest matching trigger evaluated first).
_SINGLE_VALUE_FIELDS = frozenset(
    {"orientation", "date_added", "total_paid_license_count_all_time"}
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_refinement_filters(
    requirements: Optional[Dict[str, Any]],
    extra_text: str = "",
) -> List[Dict[str, Any]]:
    """
    Derive OpenSearch filter clauses from Project Manager structured output.

    Scans ``requirements["analysis"]`` plus any structured sub-fields
    (visual_requirements, themes_moods, technical_constraints) and the
    optional *extra_text* argument for recognised trigger phrases, then
    converts them to OpenSearch filter DSL objects.

    Args:
        requirements: The ``requirements`` dict from AgentState, containing
                      at minimum an ``"analysis"`` key with the full PM text.
        extra_text:   Additional free-form text to scan (e.g. user_query).

    Returns:
        List of OpenSearch filter clause dicts ready to insert into a
        ``bool.filter`` array.  Returns ``[]`` when no triggers are found.
    """
    if not requirements and not extra_text:
        return []

    # Collect all relevant text into one corpus
    texts: List[str] = []
    if requirements:
        analysis = requirements.get("analysis", "")
        if analysis:
            texts.append(analysis)
        for key in ("visual_requirements", "themes_moods", "technical_constraints"):
            val = requirements.get(key)
            if isinstance(val, list):
                texts.extend(str(v) for v in val)
            elif isinstance(val, str):
                texts.append(val)
    if extra_text:
        texts.append(extra_text)

    combined = _normalize(" ".join(texts))

    # Match triggers longest-first so multi-word phrases win over their parts
    seen_fields: Dict[str, Dict[str, Any]] = {}
    triggered: List[Dict[str, Any]] = []

    for trigger, spec in sorted(
        ATTRIBUTE_FIELD_MAP.items(), key=lambda kv: -len(kv[0])
    ):
        pattern = r"\b" + re.escape(_normalize(trigger)) + r"\b"
        if not re.search(pattern, combined):
            continue

        clause = _build_clause(spec)
        if clause is None:
            continue

        field = spec["field"]
        if field in _SINGLE_VALUE_FIELDS:
            # Keep only the first (most-specific) match for exclusive fields
            if field not in seen_fields:
                seen_fields[field] = clause
                triggered.append(clause)
                logger.debug("Refinement: '%s' → %s filter", trigger, field)
        else:
            triggered.append(clause)
            logger.debug("Refinement: '%s' → %s filter", trigger, field)

    logger.info(
        "Refinement: extracted %d filter(s) from brief analysis", len(triggered)
    )
    return triggered


def describe_filters(filters: List[Dict[str, Any]]) -> List[str]:
    """
    Return human-readable descriptions for a list of OpenSearch filter clauses.
    Used for the workflow trace / UI metadata.
    """
    descriptions: List[str] = []
    for f in filters:
        if "term" in f:
            field, value = next(iter(f["term"].items()))
            descriptions.append(f"{field} = {value}")
        elif "terms" in f:
            field, values = next(iter(f["terms"].items()))
            descriptions.append(f"{field} in [{', '.join(str(v) for v in values)}]")
        elif "range" in f:
            field, bounds = next(iter(f["range"].items()))
            parts = [f"{k} {v}" for k, v in bounds.items()]
            descriptions.append(f"{field} {' and '.join(parts)}")
    return descriptions


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    """Lowercase and collapse whitespace (no punctuation stripping — triggers
    rely on word-boundary regex, so punctuation handled by the caller)."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def _build_clause(spec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert a spec dict to an OpenSearch filter clause dict."""
    field = spec["field"]
    ftype = spec["type"]

    if ftype == "term":
        return {"term": {field: spec["value"]}}
    if ftype == "terms":
        return {"terms": {field: spec["value"]}}
    if ftype == "range":
        range_body: Dict[str, Any] = {}
        if "gte" in spec:
            range_body["gte"] = spec["gte"]
        if "lte" in spec:
            range_body["lte"] = spec["lte"]
        return {"range": {field: range_body}} if range_body else None

    logger.warning("Unknown filter type '%s' for field '%s'", ftype, field)
    return None
