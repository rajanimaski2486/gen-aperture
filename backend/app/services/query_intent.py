"""
Query intent detection for text-only search queries.

Performs a comprehensive LLM-based analysis to extract:
  - Entity terms (≤6 cleaned search entities)
  - Boolean query (Lucene AND/OR syntax, ≤6 terms)
  - Semantic query (4–8 terms scaled by complexity)
  - Media type detection (image/video/both)
  - Generated/non-generated intent
  - Named entities (locations, brands/trademarks, celebrities, seasons)
  - Category suggestions → resolved GIDs
  - Exclusion terms (must_not on keywords_en)
  - Refinement filters (orientation, recency, popularity)
"""
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.config import settings
from app.services.query_refinement import extract_refinement_filters

logger = logging.getLogger(__name__)

_STOP_WORDS = frozenset({
    "show", "me", "find", "get", "search", "for",
    "images", "photos", "pictures", "popular",
    "trending", "best", "top", "of", "the", "a",
    "an", "in", "from", "some", "please",
})

_VIDEO_RE = re.compile(r"\b(videos?|footage|clips?|reels?)\b", re.IGNORECASE)
_IMAGE_RE = re.compile(r"\b(images?|photos?|pictures?|pics?)\b", re.IGNORECASE)
_GENERATED_RE = re.compile(r"\b(ai[-\s]?generated|generated(?:\s+(?:images?|photos?|content|results?|pictures?))?)\b", re.IGNORECASE)
_NOT_GENERATED_RE = re.compile(r"\b(real photos?|authentic|no ai|not generated|non[-\s]?generated)\b", re.IGNORECASE)
_EXCLUSION_RE = re.compile(
    r"\b(?:no|without|exclude|minus|not)\s+([^,.]+?)(?=\s+(?:and|with|but|or)\b|[,.]|$)",
    re.IGNORECASE,
)


@dataclass
class QueryIntentResult:
    """Container for the full intent analysis of a text-only query."""

    entity_terms: List[str] = field(default_factory=list)
    boolean_query: str = ""
    semantic_query: str = ""
    media_type: Optional[str] = None          # "image" | "video" | "both" | None
    is_generated: Optional[bool] = None       # True (show AI) | False (explicit real) | None
    named_entities: Dict[str, List[str]] = field(default_factory=lambda: {
        "locations": [], "brands_trademarks": [], "celebrities": [], "seasons": [],
    })
    category_gids: List[int] = field(default_factory=list)
    category_names: List[str] = field(default_factory=list)
    exclusion_terms: List[str] = field(default_factory=list)
    refinement_filters: List[Dict[str, Any]] = field(default_factory=list)
    intent: str = ""
    mood_style: List[str] = field(default_factory=list)


def _fallback_query_intent(raw_query: str, *, intent: str) -> QueryIntentResult:
    """Fast local query intent extraction used by default and as LLM fallback."""
    raw_query = (raw_query or "").strip()
    tokens = re.findall(r"[a-z0-9]+", raw_query.lower())
    terms = [t for t in tokens if t not in _STOP_WORDS][:6]
    if not terms and raw_query:
        terms = raw_query.split()[:6]

    boolean_query = " AND ".join(terms) if terms else raw_query
    semantic_query = " ".join(terms[:8]) if terms else raw_query

    has_video = bool(_VIDEO_RE.search(raw_query))
    has_image = bool(_IMAGE_RE.search(raw_query))
    if has_video and has_image:
        media_type: Optional[str] = "both"
    elif has_video:
        media_type = "video"
    elif has_image:
        media_type = "image"
    else:
        media_type = None

    is_generated: Optional[bool] = None
    if _GENERATED_RE.search(raw_query):
        is_generated = True
    elif _NOT_GENERATED_RE.search(raw_query):
        is_generated = False

    exclusion_terms: List[str] = []
    for match in _EXCLUSION_RE.finditer(raw_query):
        phrase = " ".join(
            t for t in re.findall(r"[a-z0-9]+", match.group(1).lower())
            if t not in _STOP_WORDS
        )
        if phrase:
            exclusion_terms.append(phrase)

    return QueryIntentResult(
        entity_terms=terms,
        boolean_query=boolean_query,
        semantic_query=semantic_query or raw_query,
        media_type=media_type,
        is_generated=is_generated,
        exclusion_terms=exclusion_terms,
        refinement_filters=extract_refinement_filters(None, extra_text=raw_query),
        intent=intent,
    )


def detect_text_query_intent(
    raw_query: str,
    llm: Any,
    category_filter: Any,
) -> QueryIntentResult:
    """
    Perform a full LLM-based intent analysis on a text-only search query.

    Args:
        raw_query:       The user's raw (or follow-up-resolved) search query.
        llm:             A LangChain ChatOpenAI instance for LLM calls.
        category_filter: The global CategoryFilter singleton for GID resolution.

    Returns:
        A populated ``QueryIntentResult``.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    if not settings.text_query_intent_llm_enabled:
        return _fallback_query_intent(raw_query, intent="Direct search (fast path)")

    # Build compact category reference for the prompt
    category_list = "\n".join(
        f"  - {category_filter.label_for_gid(gid)} (gid {gid})"
        for gid in sorted(category_filter._gid_to_value)
    )

    system_prompt = f"""You are a stock-photo search query analyzer. Perform a comprehensive intent analysis:

1. **Entity Recognition (NER)**: Extract the core search entities — subjects, scenes, objects, actions.
   STRIP ALL of these — do NOT include them as entity terms:
   - Intent/navigation words: "show", "me", "find", "get", "see", "display", "can", "you", "please", "search", "for"
   - Reranking/meta words: "best", "match", "matching", "results", "rerank", "top", "ranked", "reviewed"
   - Generic media nouns: "images", "photos", "pictures", "content"
   - Common prepositions: "of", "the", "a", "an", "in", "on", "at", "from", "with"
   - Qualifiers: "popular", "trending", "good", "great", "nice", "amazing"
   Return up to 6 clean entity tokens that are ONLY real-world subjects, scenes, objects, people, or places.

2. **Named Entity Detection**: Identify specific named entities present in the query:
   - locations: Specific geographic places, cities, countries, landmarks (e.g. "Malibu", "Tokyo", "Eiffel Tower")
   - brands_trademarks: Brand names, company names, trademarks (e.g. "Nike", "Apple", "Coca-Cola")
   - celebrities: Famous people's names (e.g. "Taylor Swift", "Elon Musk")
   - seasons: Seasonal references (e.g. "summer", "winter", "spring", "autumn", "fall", "holiday season", "Christmas")
   Return empty lists for categories with no matches.

3. **Media Type**: Detect if the user is requesting a specific media type:
   - "image" if they want photos/images/pictures
   - "video" if they want video/footage/clips
   - "both" if they want mixed media
   - null if not specified (default to image)

4. **Generated Content Intent**: Detect if the user wants AI-generated or explicitly non-generated content:
   - true if they say "AI generated", "generated images", "AI art", "AI-created", etc.
   - false if they say "real photos", "authentic", "no AI", "not generated", "non-generated", etc.
   - null if not specified (no preference)

5. **Filters**: Detect if the user implies any of these filters:
   - orientation: "horizontal" | "vertical" | "square" (only if explicitly stated or clearly implied)
   - recency: if user says "recent", "new", "latest" → provide an ES date-math value like "now-1y"
   - popularity: if user says "popular", "trending", "best-selling" → minimum license threshold (integer)

6. **Exclusion Terms**: Extract ONLY terms the user explicitly wants excluded.
   Look for patterns like "no X", "without X", "not X", "exclude X", "minus X".
   Return as a list of lowercase keyword strings (suitable for `keywords_en` must_not match).
   Return an EMPTY list if no explicit negations are present.

7. **Boolean Query**: Construct a Lucene boolean query string from the entity_terms and named entities.
   - Use AND between distinct concepts (e.g. "surfers AND beach")
   - Use OR between synonyms or alternatives (e.g. "(Nike OR Adidas)")
   - Group alternatives with parentheses
   - Maximum 6 total terms in the boolean expression
   - Example: "(Nike OR Adidas) AND athletes AND running"
   - Example: "beach AND surfers AND sunset"
   - Example: "Tokyo AND cherry AND blossoms AND spring"

8. **Semantic Query**: Build an expanded query for neural vector search.
   - For a very specific single-subject query → 4 terms
   - For a moderately complex query → 5-6 terms
   - For a broad/complex multi-faceted query → 7-8 terms
   Combine entities with closely related synonyms and visual descriptors. No style/quality words.

9. **Category Suggestions**: From this category list, suggest the 1-3 most relevant categories:
{category_list}
   Return category names EXACTLY as shown (e.g. "Animals/Wildlife", "Nature").

Respond in this EXACT JSON format:
{{
  "intent": "Brief description of what the user is looking for",
  "entity_terms": ["entity1", "entity2"],
  "named_entities": {{
    "locations": [],
    "brands_trademarks": [],
    "celebrities": [],
    "seasons": []
  }},
  "media_type": null,
  "is_generated": null,
  "filters": {{
    "orientation": null,
    "recency_gte": null,
    "popularity_gte": null
  }},
  "exclusion_terms": [],
  "boolean_query": "entity1 AND entity2",
  "expanded_semantic_query": "expanded query with synonyms",
  "suggested_categories": ["Category/Name"],
  "mood_style": ["mood/style terms if applicable"]
}}"""

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"User query: {raw_query}"),
        ])

        import json

        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        analysis = json.loads(content)

        # --- Entity terms (≤6) ---
        entity_terms = [
            str(t).strip().lower()
            for t in (analysis.get("entity_terms") or [])
            if str(t).strip()
        ][:6]

        # --- Boolean query ---
        boolean_query = str(analysis.get("boolean_query") or "").strip()
        if not boolean_query and entity_terms:
            boolean_query = " AND ".join(entity_terms)

        # --- Semantic query (4–8 terms) ---
        semantic_query = str(
            analysis.get("expanded_semantic_query") or raw_query
        ).strip()
        # Cap at 8 terms
        sem_words = semantic_query.split()
        if len(sem_words) > 8:
            semantic_query = " ".join(sem_words[:8])

        # --- Media type ---
        raw_media = analysis.get("media_type")
        media_type = (
            str(raw_media).lower()
            if raw_media and str(raw_media).lower() in ("image", "video", "both")
            else None
        )

        # --- is_generated ---
        raw_gen = analysis.get("is_generated")
        is_generated: Optional[bool] = None
        if raw_gen is True:
            is_generated = True
        elif raw_gen is False:
            is_generated = False

        # --- Named entities ---
        raw_ne = analysis.get("named_entities") or {}
        named_entities = {
            "locations": [str(v) for v in (raw_ne.get("locations") or []) if str(v).strip()],
            "brands_trademarks": [str(v) for v in (raw_ne.get("brands_trademarks") or []) if str(v).strip()],
            "celebrities": [str(v) for v in (raw_ne.get("celebrities") or []) if str(v).strip()],
            "seasons": [str(v) for v in (raw_ne.get("seasons") or []) if str(v).strip()],
        }

        # --- Exclusion terms ---
        exclusion_terms = [
            str(t).strip().lower()
            for t in (analysis.get("exclusion_terms") or [])
            if str(t).strip()
        ]

        # --- Category resolution ---
        suggested_cats = analysis.get("suggested_categories") or []
        cat_gids: List[int] = []
        cat_names: List[str] = []
        for name in suggested_cats:
            matches = category_filter.match_categories(name)
            for m in matches:
                if m["gid"] not in cat_gids:
                    cat_gids.append(m["gid"])
                    cat_names.append(m["value"])

        # --- Refinement filters (LLM-detected + keyword-based) ---
        llm_filters = analysis.get("filters") or {}
        refinement_clauses: List[Dict[str, Any]] = []
        if llm_filters.get("orientation"):
            refinement_clauses.append(
                {"term": {"orientation": llm_filters["orientation"]}}
            )
        if llm_filters.get("recency_gte"):
            refinement_clauses.append(
                {"range": {"date_added": {"gte": llm_filters["recency_gte"]}}}
            )
        if llm_filters.get("popularity_gte"):
            refinement_clauses.append(
                {"range": {"total_paid_license_count_all_time": {"gte": int(llm_filters["popularity_gte"])}}}
            )

        # Merge keyword-based refinement filters (avoid duplicates)
        keyword_refinements = extract_refinement_filters(None, extra_text=raw_query)
        seen_fields = {
            next(iter(c.get("term", c.get("range", {})))): True
            for c in refinement_clauses
        }
        for kr in keyword_refinements:
            kr_field = next(iter(kr.get("term", kr.get("range", {}))), None)
            if kr_field and kr_field not in seen_fields:
                refinement_clauses.append(kr)
                seen_fields[kr_field] = True

        logger.info(
            "Query intent: '%s' → entities=%s, boolean='%s', semantic='%s', "
            "media_type=%s, is_generated=%s, named_entities=%s, "
            "exclusions=%s, categories=%s, filters=%d",
            raw_query, entity_terms, boolean_query, semantic_query,
            media_type, is_generated, named_entities,
            exclusion_terms, cat_names, len(refinement_clauses),
        )

        return QueryIntentResult(
            entity_terms=entity_terms,
            boolean_query=boolean_query,
            semantic_query=semantic_query,
            media_type=media_type,
            is_generated=is_generated,
            named_entities=named_entities,
            category_gids=cat_gids,
            category_names=cat_names,
            exclusion_terms=exclusion_terms,
            refinement_filters=refinement_clauses,
            intent=analysis.get("intent", ""),
            mood_style=analysis.get("mood_style") or [],
        )

    except Exception as e:
        logger.warning("Query intent detection failed (%s), using fallback", e)
        return _fallback_query_intent(raw_query, intent="Direct search (fallback)")
