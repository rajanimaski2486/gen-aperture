"""
LangGraph multi-agent system for Gen-Aperture.
Implements Squad Router, Project Manager Strand, and Search Specialist Strand.
"""
import logging
import copy
from typing import Dict, Any, List, TypedDict, Annotated, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from openai import AuthenticationError, OpenAIError
from app.services.photo_search import photo_search_service
from app.services.search_service_mcp import search_service_mcp
from app.services.category_filter import category_filter as _category_filter
from app.services.query_refinement import extract_refinement_filters, describe_filters
from app.services.query_intent import detect_text_query_intent, QueryIntentResult
from app.services.reranker import ReflectionReranker, RerankerConfig, should_rerank, _run_async_from_sync
from app.config import settings
import dataclasses
import re as _re

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Intent helpers (module-level, no LLM needed)
# ---------------------------------------------------------------------------

_SHOW_GENERATED_RE = _re.compile(
    r'\b(?:show|include|find|get|see|display)?\s*'
    r'(?:ai[-\s]?generated|generated(?:\s+(?:images?|photos?|content|results?|pictures?))?)',
    _re.IGNORECASE,
)


def _is_show_generated_query(query: str) -> bool:
    """Return True when the user's message asks for AI/generated images."""
    return bool(_SHOW_GENERATED_RE.search(query))


# ---------------------------------------------------------------------------
# Write-intent guard (module-level, no LLM needed)
# The production OpenSearch domain is READ-ONLY.  Any request that looks like
# a data-mutation operation must be blocked before reaching the agent graph.
# ---------------------------------------------------------------------------

_WRITE_BLOCK_RE = _re.compile(
    r"""
    \b(?:
        # Destructive operations
        (?:delete|remove|drop|wipe|purge|flush|truncate|erase)
        \s+(?:(?:\w+\s+){0,4})
        (?:index(?:es)?|indices|document(?:s)?|record(?:s)?|data|cluster|photos?)
    |
        # Ingestion / mutation into a destination
        (?:add|insert|put|ingest|index|write|commit|push|bulk\s*(?:index|insert|upload|ingest)?)
        \s+(?:(?:\w+\s+){0,4})
        (?:to|into|in)\s+
        (?:opensearch|elastic(?:search)?|the\s+(?:index|cluster|database))
    |
        # Update / modify index artefacts
        (?:update|modify|change|edit|patch|alter)
        \s+(?:(?:\w+\s+){0,4})
        (?:index(?:es)?|mapping(?:s)?|document(?:s)?|record(?:s)?|schema|cluster\s+settings?)
    |
        # Re-indexing
        re(?:-?\s*)index
    |
        # Explicit index creation
        create\s+(?:a\s+)?(?:new\s+)?index
    )
    """,
    _re.IGNORECASE | _re.VERBOSE,
)

_WRITE_BLOCK_RESPONSE = (
    "I'm sorry, but I can't do that. The production OpenSearch domain "
    "(`nelson-v1-prod`) is **read-only** — write operations (create, update, "
    "delete, ingest, re-index, etc.) are not permitted and are blocked at the "
    "infrastructure level.\n\n"
    "If you believe a data change is genuinely required, please raise it with "
    "the platform team directly. I'm here to help you **search** the stock "
    "photo catalogue. How can I assist you with that?"
)


def _is_write_intent_query(query: str) -> bool:
    """Return True when the query appears to request a write/mutation on OpenSearch."""
    return bool(_WRITE_BLOCK_RE.search(query))


class AgentState(TypedDict):
    """State shared across all agents."""
    messages: Annotated[List[Any], add_messages]
    user_query: str
    file_content: str | None
    file_images: List[Dict[str, Any]] | None
    image_analysis: Dict[str, Any] | None
    file_type: str | None
    conversation_history: List[Dict[str, str]]
    
    # Routing decision
    route: Literal["project_manager", "search_specialist"] | None
    
    # Project Manager outputs
    requirements: Dict[str, Any] | None
    extracted_queries: List[str] | None       # legacy / fallback
    lexical_query: str | None                 # 3-5 entity terms for keyword matching
    semantic_query: str | None                # 5-6 terms for neural/vector search

    # Filter state (extracted from brief analysis or user query)
    category_gids: List[int] | None
    refinement_filters: List[Dict[str, Any]] | None
    exclusion_terms: List[str] | None          # keywords_en terms for must_not
    show_generated: bool | None                # inject is_generated=true filter when True
    media_type: str | None                     # "image" | "video" | "both" | None (informational)
    boolean_query: str | None                  # Lucene boolean string with AND/OR (≤6 terms)
    named_entities: Dict[str, List[str]] | None  # locations, brands_trademarks, celebrities, seasons

    # Search intent (determined by LLM)
    search_mode: Literal["relevance", "popular"] | None
    
    # Search Specialist outputs (single modified query — no separate filtered run)
    search_results: List[Dict[str, Any]] | None
    total_results: int

    # Reflection reranker outputs (populated when triggered)
    rerank_applied: bool | None
    rerank_decisions: List[Dict[str, Any]] | None
    rerank_explanation: str | None
    
    # Final response
    response: str | None
    processing_time_ms: int
    
    # Agent workflow trace
    workflow_steps: List[Dict[str, Any]]


class AgentSquad:
    """Multi-agent system for stock photo search."""
    
    def __init__(self, openai_api_key: str, model: str = "gpt-4o-mini"):
        """
        Initialize the agent squad.
        
        Args:
            openai_api_key: User's OpenAI API key
            model: OpenAI model to use (default: gpt-4o-mini)
        """
        # Set environment variable for OpenAI
        import os
        os.environ["OPENAI_API_KEY"] = openai_api_key
        
        self.llm = ChatOpenAI(
            model=model,
            temperature=0.7
        )

        # Reflection reranker — uses the same OPENAI_API_KEY set above
        self._reranker = ReflectionReranker(RerankerConfig.from_settings(settings))

        # Build the agent graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph agent graph."""
        
        # Create graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("router", self._router_node)
        workflow.add_node("project_manager", self._project_manager_node)
        workflow.add_node("search_specialist", self._search_specialist_node)
        workflow.add_node("synthesize", self._synthesize_node)
        
        # Set entry point
        workflow.set_entry_point("router")
        
        # Add conditional edges from router
        workflow.add_conditional_edges(
            "router",
            lambda state: state["route"],
            {
                "project_manager": "project_manager",
                "search_specialist": "search_specialist"
            }
        )
        
        # Project Manager -> Search Specialist
        workflow.add_edge("project_manager", "search_specialist")
        
        # Search Specialist -> Synthesize
        workflow.add_edge("search_specialist", "synthesize")
        
        # Synthesize -> END
        workflow.add_edge("synthesize", END)
        
        return workflow.compile()
    
    def _router_node(self, state: AgentState) -> AgentState:
        """
        Squad Router: Routes requests based on input type AND search intent.
        - If file uploaded -> Project Manager Strand
        - If text only -> Search Specialist Strand (direct)
        
        Also uses LLM to determine search intent:
        - "relevance" -> user wants best-matching images
        - "popular"   -> user wants trending/popular/best-selling images
        """
        logger.info("Router: Analyzing input...")
        
        steps = state.get("workflow_steps", [])
        
        # --- Determine search intent via LLM ---
        intent_prompt = """You are a search intent classifier. Given a user's query about stock photos, determine the search mode.

Respond with EXACTLY one word — either "relevance" or "popular".

Use "popular" if the user's query indicates they want:
- Popular, trending, best-selling, most downloaded, or top images
- Images that are commercially successful or frequently licensed
- "What's hot", "trending now", "most popular" type queries

Use "relevance" for everything else — when the user wants images that best match a specific subject, scene, concept, or description.

If there is conversation history, consider the full context when classifying intent.
"""
        
        # Build context with conversation history
        intent_messages = [SystemMessage(content=intent_prompt)]
        for hist_msg in state.get("conversation_history", []):
            if hist_msg["role"] == "user":
                intent_messages.append(HumanMessage(content=hist_msg["content"]))
            else:
                intent_messages.append(AIMessage(content=hist_msg["content"]))
        intent_messages.append(HumanMessage(content=state["user_query"]))
        
        is_first_search = len(state.get("conversation_history", [])) == 0

        try:
            intent_response = self.llm.invoke(intent_messages)
            raw_intent = intent_response.content.strip().lower()
            search_mode = "popular" if "popular" in raw_intent else ("popular" if is_first_search else "relevance")
        except Exception as e:
            default_mode = "popular" if is_first_search else "relevance"
            logger.warning(f"Router: Intent detection failed ({e}), defaulting to {default_mode}")
            search_mode = default_mode
        
        state["search_mode"] = search_mode
        logger.info(f"Router: Detected search intent -> {search_mode}")
        
        # --- Determine agent routing ---
        if state.get("file_content"):
            logger.info("Router: File detected -> routing to Project Manager")
            state["route"] = "project_manager"
            steps.append({
                "agent": "Squad Router",
                "action": "Route to Project Manager",
                "reasoning": (
                    f"File uploaded ({state.get('file_type', 'unknown')} type) along with query: \"{state['user_query']}\". "
                    f"A file is present, so routing to the Project Manager Strand to analyze the brief. "
                    f"Search intent detected as '{search_mode}' — will use MCP tool 'search_{search_mode}' to get the production OpenSearch query."
                ),
                "input": {"query": state["user_query"], "has_file": True, "file_type": state.get("file_type"), "search_mode": search_mode},
                "decision": f"project_manager → search_specialist (MCP: search_{search_mode}) → synthesize"
            })
        else:
            logger.info("Router: Text only -> routing to Search Specialist")
            state["route"] = "search_specialist"
            steps.append({
                "agent": "Squad Router",
                "action": "Route to Search Specialist",
                "reasoning": (
                    f"Text-only query: \"{state['user_query']}\". No file uploaded, so routing directly to Search Specialist. "
                    f"Search intent detected as '{search_mode}' — will use MCP tool 'search_{search_mode}' to get the production OpenSearch query from Search Service."
                ),
                "input": {"query": state["user_query"], "has_file": False, "search_mode": search_mode},
                "decision": f"search_specialist (MCP: search_{search_mode}) → synthesize"
            })
        
        state["workflow_steps"] = steps
        return state
    
    def _project_manager_node(self, state: AgentState) -> AgentState:
        """
        Project Manager Strand: Analyzes uploaded briefs.
        Extracts visual requirements, themes, moods, constraints,
        exclusion terms, and category suggestions.
        Does NOT search - only requirements extraction.
        """
        logger.info("Project Manager: Analyzing brief...")

        # Build a compact category reference for the LLM
        category_list = "\n".join(
            f"  - {_category_filter.label_for_gid(gid)} (gid {gid})"
            for gid in sorted(_category_filter._gid_to_value)
        )

        system_prompt = f"""You are a Project Manager AI analyzing creative briefs for stock photo searches.

Your #1 job is to understand WHAT SUBJECTS the user needs images of, then generate precise search queries.

IMPORTANT CONTEXT:
- Briefs are often brand style guidelines — they describe HOW images should look (style, mood, quality).
- Style/quality/mood words are image PROPERTIES for filtering, NOT subjects to search for.
- Stock photos are tagged with keywords like: subjects, objects, scenes, activities, locations, people.
- You MUST identify the brand's INDUSTRY/DOMAIN and the user's stated subject to generate useful queries.

Example 1 — User: "find images for J&J healthcare campaign" + J&J brand guidelines brief:
  → Brand domain: healthcare, pharmaceutical, medical
  → lexical_query: "healthcare medical professionals laboratory"
  → semantic_query: "healthcare workers medical research laboratory science innovation"
  → Visual properties (filters): documentary style, authentic, emotional

Example 2 — User: "shipping and logistics" + Maersk brand guidelines brief:
  → Brand domain: shipping, logistics, maritime
  → lexical_query: "shipping containers port workers logistics"
  → semantic_query: "shipping port cargo containers logistics dock workers maritime"
  → Visual properties (filters): candid, authentic, natural lighting, H2H style

Example 3 — User: "travel and leisure" + JetBlue loyalty program brief:
  → Brand domain: airline, travel, loyalty program
  → lexical_query: "travel airport airplane destination luxury"
  → semantic_query: "luxury travel experience airport airplane vacation destination escape"
  → Visual properties (filters): immersive, spontaneous, minimalist, no posed portraits

Your task:

1. **Identify the domain**: What industry/sector does the brand operate in? What concrete subjects would they need images of?

2. **lexical_query**: A single string of 3-6 key entity terms (nouns) separated by spaces.
   These are the CORE subjects for keyword matching. No style/quality words.
   If the user typed a query, its subject matter takes HIGHEST priority.

3. **semantic_query**: A single string of 5-7 terms for semantic/neural vector search.
   Includes the lexical entities PLUS closely related synonyms or contextual terms.
   Scale term count by brief complexity: 5 for focused briefs, 7 for complex multi-faceted ones.
   Richer context helps vector search — but still only subject matter, no style words.

4. **Visual requirements** (style properties: candid, authentic, natural lighting, etc.)
5. **Quality requirements** (resolution, production value, retouching preferences)
6. **Themes and moods** (emotions, atmosphere — for understanding, not for search queries)
7. **Technical constraints** (orientation, color palette, format)
8. **Exclusion terms**: keywords that should NOT appear in search results.
   Look for phrases like "avoid X", "no X", "do not include X", "exclude X".
   Return as a list of lowercase keyword strings. Return EMPTY list if none found.
9. **Category suggestions**: From this category list, pick the 1-3 most relevant:
{category_list}
   Return category names EXACTLY as shown.

10. **Media type**: What type of media does the brief need?
    - "image" for photos/images, "video" for video/footage, "both" for mixed, null if not specified.

11. **Generated content**: Does the brief explicitly request AI-generated or explicitly authentic/real photos?
    - true if brief requests AI-generated content, false if brief explicitly requires authentic/real photos, null if not specified.

12. **Boolean query**: Construct a Lucene boolean query string from the lexical_query terms.
    - Use AND between distinct concepts, OR between synonyms/alternatives.
    - Group alternatives with parentheses. Maximum 6 total terms.
    - Example: "(shipping OR logistics) AND containers AND port AND workers"

13. **Named entities**: Extract specific named entities found in the brief or user query:
    - locations: Geographic places, cities, countries (e.g. "New York", "Japan")
    - brands_trademarks: Brand names, company names mentioned (e.g. "Maersk", "JetBlue")
    - celebrities: Famous people's names
    - seasons: Seasonal references (e.g. "summer", "winter", "holiday season")

Respond in this EXACT format — structured analysis followed by a JSON block:

**Structured Analysis:**
(your detailed analysis text here — include the identified domain/industry)

```json
{{
  "brand_domain": "industry or sector the brand operates in",
  "lexical_query": "3-6 key entity terms",
  "semantic_query": "5-7 terms with synonyms for neural search",
  "boolean_query": "Lucene boolean string with AND/OR, max 6 terms",
  "media_type": null,
  "is_generated": null,
  "named_entities": {{"locations": [], "brands_trademarks": [], "celebrities": [], "seasons": []}},
  "visual_requirements": ["requirement1", "requirement2"],
  "quality_requirements": ["requirement1"],
  "themes_moods": ["theme1", "mood1"],
  "technical_constraints": ["constraint1"],
  "exclusion_terms": ["term_to_exclude1"],
  "suggested_categories": ["Category/Name"]
}}
```"""
        
        # Prepare context
        user_context = f"User query: {state['user_query']}\n\n"
        if state.get('file_content'):
            user_context += f"Brief content ({state.get('file_type', 'unknown')} file):\n{state['file_content']}"
        
        # Include image analysis if available
        if state.get('image_analysis') and state['image_analysis'].get('summary'):
            user_context += f"\n\nImage analysis from the uploaded document:\n{state['image_analysis']['summary']}"
            palette = state['image_analysis'].get('global_palette', [])
            if palette:
                palette_str = ", ".join(f"{c['name']} ({c['hex']})" for c in palette)
                user_context += f"\nDominant colors: {palette_str}"
            mood_tags = state['image_analysis'].get('mood_tags', [])
            if mood_tags:
                user_context += f"\nInferred mood/tone: {', '.join(mood_tags)}"
        
        # Call LLM
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_context)
        ]
        
        response = self.llm.invoke(messages)
        print("Project Manager LLM response:", response);  # Debug log for PM response
        analysis = response.content

        # Try to parse the JSON block from the response
        import json as _json
        structured_data = {}
        try:
            # Find JSON block in the response
            json_match = None
            if "```json" in analysis:
                json_match = analysis.split("```json", 1)[1].split("```", 1)[0].strip()
            elif "```" in analysis:
                for block in analysis.split("```")[1::2]:
                    block = block.strip()
                    if block.startswith("{"):
                        json_match = block
                        break
            if json_match:
                structured_data = _json.loads(json_match)
        except Exception as e:
            logger.warning(f"Project Manager: Could not parse JSON block: {e}")

        # Extract dual queries — lexical (3-6 terms) and semantic (5-7 terms)
        lexical_q = structured_data.get("lexical_query", "").strip()
        semantic_q = structured_data.get("semantic_query", "").strip()

        print(f"Extracted lexical query: '{lexical_q}'")  # Debug log for lexical query
        print(f"Extracted semantic query: '{semantic_q}'")  # Debug log for semantic query

        # Fallback: if structured parse failed, try legacy extraction
        if not lexical_q and not semantic_q:
            fallback_queries = structured_data.get("search_queries") or self._extract_search_queries(analysis)
            lexical_q = " ".join(fallback_queries[:1]) if fallback_queries else state["user_query"]
            semantic_q = " ".join(fallback_queries[:2]) if fallback_queries else state["user_query"]

        # Ensure lexical is capped at 6 terms
        lexical_terms = lexical_q.split()[:6]
        lexical_q = " ".join(lexical_terms)

        # Ensure semantic is capped at 7 terms
        semantic_terms = semantic_q.split()[:7]
        semantic_q = " ".join(semantic_terms)

        state["lexical_query"] = lexical_q
        state["semantic_query"] = semantic_q
        state["extracted_queries"] = [semantic_q, lexical_q]  # keep for backward compat

        logger.info(
            f"Project Manager: lexical_query='{lexical_q}', semantic_query='{semantic_q}'"
        )

        # Extract is_generated early — needed by requirements dict and state
        pm_is_generated = structured_data.get("is_generated")
        
        state["requirements"] = {
            "analysis": analysis,
            "file_type": state.get("file_type"),
            "brand_domain": structured_data.get("brand_domain", ""),
            "is_generated": pm_is_generated,
            "visual_requirements": structured_data.get("visual_requirements", []),
            "quality_requirements": structured_data.get("quality_requirements", []),
            "themes_moods": structured_data.get("themes_moods", []),
            "technical_constraints": structured_data.get("technical_constraints", []),
        }
        state["messages"] = add_messages(state["messages"], [response])

        # ── Exclusion terms from brief ───────────────────────────────────────
        brief_exclusions = [
            str(t).strip().lower()
            for t in (structured_data.get("exclusion_terms") or [])
            if str(t).strip()
        ]
        state["exclusion_terms"] = brief_exclusions

        # ── New PM fields: boolean_query, media_type, named_entities, is_generated ──
        pm_boolean_query = structured_data.get("boolean_query", "").strip()
        if not pm_boolean_query and lexical_q:
            pm_boolean_query = " AND ".join(lexical_terms)
        state["boolean_query"] = pm_boolean_query
        state["media_type"] = (
            str(structured_data.get("media_type")).lower()
            if structured_data.get("media_type")
            and str(structured_data.get("media_type")).lower() in ("image", "video", "both")
            else None
        )
        raw_ne = structured_data.get("named_entities") or {}
        state["named_entities"] = {
            "locations": [str(v) for v in (raw_ne.get("locations") or []) if str(v).strip()],
            "brands_trademarks": [str(v) for v in (raw_ne.get("brands_trademarks") or []) if str(v).strip()],
            "celebrities": [str(v) for v in (raw_ne.get("celebrities") or []) if str(v).strip()],
            "seasons": [str(v) for v in (raw_ne.get("seasons") or []) if str(v).strip()],
        }

        # Derive show_generated from PM analysis + regex on user query
        state["show_generated"] = (pm_is_generated is True) or _is_show_generated_query(state.get("user_query", ""))

        # ── Category extraction — LLM suggestions + text matching ────────────
        # First try LLM-suggested categories
        suggested_cats = structured_data.get("suggested_categories") or []
        cat_gids: List[int] = []
        cat_names: List[str] = []
        for name in suggested_cats:
            matches = _category_filter.match_categories(name)
            for m in matches:
                if m["gid"] not in cat_gids:
                    cat_gids.append(m["gid"])
                    cat_names.append(m["value"])

        # Also try text-matching from the full analysis (existing behaviour)
        text_matches = _category_filter.match_categories(analysis)
        for m in text_matches:
            if m["gid"] not in cat_gids:
                cat_gids.append(m["gid"])
                cat_names.append(m["value"])

        state["category_gids"] = cat_gids
        logger.info(
            "Project Manager: Matched %d categories: %s",
            len(cat_gids), cat_names,
        )

        # ── Refinement filter extraction ─────────────────────────────────────
        state["refinement_filters"] = extract_refinement_filters(
            state["requirements"],
            extra_text=state.get("user_query", ""),
        )
        logger.info(
            "Project Manager: Extracted %d refinement filter(s)",
            len(state["refinement_filters"]),
        )
        
        # Record workflow step
        steps = state.get("workflow_steps", [])
        steps.append({
            "agent": "Project Manager",
            "action": "Analyze Brief & Extract Requirements",
            "prompt": system_prompt,
            "reasoning": (
                f"Analyzed the uploaded {state.get('file_type', 'unknown')} file and user query. "
                f"Identified brand domain: '{structured_data.get('brand_domain', 'unknown')}'. "
                f"Generated lexical query (3-6 terms): '{lexical_q}'. "
                f"Generated semantic query (5-7 terms): '{semantic_q}'. "
                f"Generated boolean query: '{pm_boolean_query}'. "
                f"Matched {len(cat_gids)} categories: {cat_names}. "
                f"Found {len(brief_exclusions)} exclusion term(s): {brief_exclusions}. "
                f"Extracted {len(state['refinement_filters'])} refinement filter(s). "
                f"Named entities: {state.get('named_entities', {})}. "
                f"Media type: {state.get('media_type')}. Is generated: {pm_is_generated}."
            ),
            "output": {
                "brand_domain": structured_data.get("brand_domain", ""),
                "lexical_query": lexical_q,
                "semantic_query": semantic_q,
                "boolean_query": pm_boolean_query,
                "named_entities": state.get("named_entities", {}),
                "media_type": state.get("media_type"),
                "is_generated": pm_is_generated,
                "matched_categories": cat_names,
                "category_gids": cat_gids,
                "exclusion_terms": brief_exclusions,
                "refinement_filters": describe_filters(state["refinement_filters"]),
                "visual_requirements": structured_data.get("visual_requirements", []),
                "quality_requirements": structured_data.get("quality_requirements", []),
                "themes_moods": structured_data.get("themes_moods", []),
                "technical_constraints": structured_data.get("technical_constraints", []),
                "analysis_summary": analysis[:500] + "..." if len(analysis) > 500 else analysis
            }
        })
        state["workflow_steps"] = steps
        
        return state
    
    def _search_specialist_node(self, state: AgentState) -> AgentState:
        """
        Search Specialist Strand: Two distinct workflows depending on input type.

        **Text-only workflow** (no file):
        1. LLM intent analysis → entities (max 5), exclusions, filters, categories, semantic query
        2. MCP call with semantic query → get OpenSearch payload
        3. Modify payload: patch lexical with entities (AND), inject category + refinement filters,
           add must_not for exclusion terms on keywords_en
        4. Single execute

        **Brief workflow** (file present, PM ran first):
        1. Use top 2-3 PM-extracted queries combined for MCP call
        2. MCP call → get OpenSearch payload
        3. LLM entity extraction on the combined query for lexical patching
        4. Modify payload: patch lexical with entities (OR, limit 5), inject category + refinement
           filters from PM, add must_not for PM-extracted exclusions
        5. Single execute
        """
        logger.info("Search Specialist: Executing search via MCP...")

        steps = state.get("workflow_steps", [])
        has_brief = bool(state.get("file_content") or state.get("requirements"))

        if has_brief:
            return self._search_with_brief(state, steps)
        else:
            return self._search_text_only(state, steps)

    # ------------------------------------------------------------------
    # TEXT-ONLY search workflow
    # ------------------------------------------------------------------

    def _search_text_only(self, state: AgentState, steps: List[Dict]) -> AgentState:
        """Search workflow for text-only queries (no file uploaded)."""
        logger.info("Search Specialist [TEXT-ONLY]: Starting text-only workflow...")

        # --- Step 1: Resolve follow-ups if needed ---
        conversation_history = state.get("conversation_history", [])
        if conversation_history:
            raw_query = self._resolve_followup_query(state["user_query"], conversation_history)
            query_source = "contextual_followup"
            logger.info(f"Search Specialist: Resolved follow-up to: {raw_query}")
        else:
            raw_query = state["user_query"]
            query_source = "user_direct"

        # --- Step 2: Full LLM intent analysis ---
        intent_result = detect_text_query_intent(raw_query, self.llm, _category_filter)

        entity_terms = intent_result.entity_terms
        expanded_semantic_query = intent_result.semantic_query
        lexical_query_string = intent_result.boolean_query or (
            " AND ".join(entity_terms) if entity_terms else expanded_semantic_query
        )

        # Derive is_generated intent — LLM result + regex fallback
        show_generated = (intent_result.is_generated is True) or _is_show_generated_query(raw_query)
        explicit_not_generated = intent_result.is_generated is False
        if show_generated:
            logger.info("Search Specialist [TEXT-ONLY]: 'show generated' intent detected")

        category_gids = intent_result.category_gids
        category_names = intent_result.category_names
        exclusion_terms = intent_result.exclusion_terms
        refinement_clauses = intent_result.refinement_filters

        # Store in state
        state["category_gids"] = category_gids
        state["exclusion_terms"] = exclusion_terms
        state["refinement_filters"] = refinement_clauses
        state["boolean_query"] = intent_result.boolean_query
        state["media_type"] = intent_result.media_type
        state["named_entities"] = intent_result.named_entities

        logger.info(
            f"Search Specialist [TEXT-ONLY]: entities={entity_terms}, "
            f"semantic='{expanded_semantic_query}', boolean='{lexical_query_string}', "
            f"exclusions={exclusion_terms}, categories={category_names}, "
            f"filters={len(refinement_clauses)}, media_type={intent_result.media_type}, "
            f"is_generated={intent_result.is_generated}"
        )

        # Record intent analysis step
        steps.append({
            "agent": "Search Specialist",
            "action": "Query Intent Analysis (Text-Only)",
            "reasoning": (
                f"Performed full intent analysis on \"{raw_query}\". "
                f"Extracted {len(entity_terms)} entity terms: {entity_terms} (limit 6). "
                f"Generated boolean query: \"{lexical_query_string}\". "
                f"Generated semantic query: \"{expanded_semantic_query}\". "
                f"Detected {len(exclusion_terms)} exclusion term(s): {exclusion_terms}. "
                f"Matched {len(category_gids)} categories: {category_names} (gids: {category_gids}). "
                f"Extracted {len(refinement_clauses)} filter(s): {describe_filters(refinement_clauses)}. "
                f"Named entities: {intent_result.named_entities}. "
                f"Media type: {intent_result.media_type}. Is generated: {intent_result.is_generated}."
            ),
            "input": {"raw_query": raw_query, "query_source": query_source},
            "output": {
                "intent": intent_result.intent,
                "entity_terms": entity_terms,
                "boolean_query": lexical_query_string,
                "expanded_semantic_query": expanded_semantic_query,
                "named_entities": intent_result.named_entities,
                "media_type": intent_result.media_type,
                "is_generated": intent_result.is_generated,
                "exclusion_terms": exclusion_terms,
                "suggested_categories": category_names,
                "category_gids": category_gids,
                "filters_extracted": describe_filters(refinement_clauses),
                "lexical_query": lexical_query_string,
                "lexical_operator": "AND",
                "mood_style": intent_result.mood_style,
            },
        })

        # --- Step 3: MCP call ---
        search_mode = state.get("search_mode", "relevance")
        mcp_tool_name = "search_popular" if search_mode == "popular" else "search_relevant"
        mcp_result = search_service_mcp.call_tool(mcp_tool_name, expanded_semantic_query)
        mcp_metadata = mcp_result.get("search_service_metadata", {})

        steps.append({
            "agent": "Search Specialist",
            "action": f"MCP Tool: {mcp_tool_name}",
            "reasoning": (
                f"Called Search Service MCP tool '{mcp_tool_name}' with semantic query \"{expanded_semantic_query}\". "
                f"Search Service found {mcp_metadata.get('num_found', 'N/A')} matches using "
                f"'{mcp_metadata.get('ranker', 'unknown')}' ranker."
            ),
            "input": {
                "mcp_tool": mcp_tool_name,
                "expanded_semantic_query": expanded_semantic_query,
                "sort_order": search_mode,
                "search_service_url": (
                    f"http://search.shuttercorp.net/v2/shutterstock/image/search"
                    f"?q={expanded_semantic_query}&sort_order={search_mode}"
                    f"&debug_modes=request&source=enterprise"
                ),
            },
            "output": {
                "ranker": mcp_metadata.get("ranker", "unknown"),
                "ranker_settings": mcp_metadata.get("ranker_settings", "unknown"),
                "num_found_by_search_service": mcp_metadata.get("num_found", 0),
            },
            "search_service_endpoint": mcp_result.get("search_service_endpoint"),
            "search_service_response": mcp_result.get("search_service_response_payload"),
        })

        # --- Step 4: Modify payload & execute ---
        opensearch_query = mcp_result.get("opensearch_query")
        if opensearch_query:
            # Apply all modifications to a single payload
            modifications = self._modify_opensearch_payload(
                opensearch_query=opensearch_query,
                entity_terms=entity_terms,
                lexical_operator="and",
                category_gids=category_gids,
                exclusion_terms=exclusion_terms,
                refinement_filters=refinement_clauses,
                show_generated=show_generated,
                boolean_query_string=intent_result.boolean_query,
                is_not_generated=explicit_not_generated,
            )

            # Record payload modification step
            steps.append({
                "agent": "Search Specialist",
                "action": "OpenSearch Payload Modification (Text-Only)",
                "reasoning": (
                    f"Modified the MCP-returned OpenSearch payload before execution. "
                    f"Modifications applied: {'; '.join(modifications['descriptions'])}."
                ),
                "input": {
                    "modifications_planned": {
                        "lexical_patch": f"AND-joined entity terms: \"{lexical_query_string}\"",
                        "category_filter": f"global_category_ids in {category_gids}" if category_gids else "none",
                        "must_not_exclusions": f"keywords_en must_not: {exclusion_terms}" if exclusion_terms else "none",
                        "refinement_filters": describe_filters(refinement_clauses) if refinement_clauses else "none",
                    },
                },
                "output": {
                    "modifications_applied": modifications["descriptions"],
                    "total_modifications": len(modifications["descriptions"]),
                },
                "opensearch_payload": opensearch_query,
            })

            # Execute — text-only path uses the hybrid_10_90 pipeline
            search_result = photo_search_service.execute_raw_query(
                opensearch_query=opensearch_query,
                search_pipeline="hybrid_10_90",
            )

            # Fallback: if zero results and categories were applied, retry once
            # with only category filters removed.
            if search_result.get("total", 0) == 0 and category_gids:
                fallback_query = copy.deepcopy(opensearch_query)
                removed_count = self._remove_category_filters(fallback_query)
                if removed_count > 0:
                    fallback_result = photo_search_service.execute_raw_query(
                        opensearch_query=fallback_query,
                        search_pipeline="hybrid_10_90",
                    )

                    steps.append({
                        "agent": "Search Specialist",
                        "action": "Fallback Retry (Suppress Category Filters)",
                        "reasoning": (
                            f"Initial query returned 0 results with category filters {category_gids}. "
                            f"Retried by removing only category filter clause(s) "
                            f"(removed {removed_count}); all other constraints remained unchanged."
                        ),
                        "input": {
                            "removed_filter": "terms.global_category_ids",
                            "removed_count": removed_count,
                        },
                        "output": {
                            "total_results": fallback_result.get("total", 0),
                            "returned_results": len(fallback_result.get("results", [])),
                            "took_ms": fallback_result.get("took_ms", 0),
                            "fallback_used": fallback_result.get("total", 0) > 0,
                        },
                        "opensearch_payload": fallback_query,
                    })

                    if fallback_result.get("total", 0) > 0:
                        search_result = fallback_result
                        logger.info(
                            "Search Specialist [TEXT-ONLY]: Fallback succeeded after removing category filters"
                        )
                    else:
                        logger.info(
                            "Search Specialist [TEXT-ONLY]: Fallback returned 0 results; keeping original result"
                        )
        else:
            logger.warning("Search Specialist: MCP query failed, falling back to direct search")
            search_result = photo_search_service.search_photos(
                query=lexical_query_string, size=50, min_score=1.0
            )

        state["search_results"] = search_result.get("results", [])
        state["total_results"] = search_result.get("total", 0)
        state["processing_time_ms"] = search_result.get("took_ms", 0)

        # Record execution step
        steps.append({
            "agent": "Search Specialist",
            "action": "Execute OpenSearch Query (Text-Only)",
            "reasoning": (
                f"Executed the modified OpenSearch query against web-index-v9. "
                f"Found {state['total_results']} total matches in {state['processing_time_ms']}ms, "
                f"returning top {len(state['search_results'])} results."
            ),
            "input": {"index": "web-index-v9", "query_source": "search_service_mcp_modified"},
            "output": {
                "total_results": state["total_results"],
                "returned_results": len(state["search_results"]),
                "took_ms": state["processing_time_ms"],
            },
            "opensearch_payload": opensearch_query,
            "opensearch_url": (
                f"{settings.opensearch_endpoint}/_search"
                f"?search_pipeline=hybrid_10_90"
            ),
        })

        # ── Reflection Reranker (text-only workflow) ──────────────────────────
        # Triggered only when the user's message contains a rerank trigger phrase
        # (e.g. "best", "top ranked", "rerank", "reflect and respond", "reviewed")
        # Use raw_query (the resolved search subject) — NOT the reranking request
        # itself — so the LLM scores candidates against the actual search topic.
        if should_rerank(state["user_query"]):
            rerank_out = _run_async_from_sync(
                self._reranker.rerank(
                    user_query=raw_query,
                    candidates=state["search_results"],
                    search_criteria={
                        "user_query": raw_query,
                        "requirements": state.get("requirements"),
                        "refinement_filters": state.get("refinement_filters"),
                        "exclusion_terms": state.get("exclusion_terms"),
                        "category_gids": state.get("category_gids"),
                    },
                )
            )
            if rerank_out.triggered:
                state["search_results"] = rerank_out.final_results
                state["rerank_applied"] = True
                state["rerank_decisions"] = [
                    dataclasses.asdict(d) for d in rerank_out.decisions
                ]
                state["rerank_explanation"] = rerank_out.explanation
                steps.append({
                    "agent": "Reflection Reranker",
                    "action": "Rerank & Filter Results",
                    "reasoning": (
                        f"Reflection reranking applied. "
                        f"{rerank_out.total_candidates} candidates evaluated → "
                        f"{len(rerank_out.final_results)} selected."
                        + (f" Note: {rerank_out.explanation}" if rerank_out.explanation else "")
                    ),
                    "output": {
                        "kept": len(rerank_out.final_results),
                        "total_evaluated": rerank_out.total_candidates,
                        "explanation": rerank_out.explanation,
                        **rerank_out.pass_summaries,
                    },
                })
                logger.info(
                    "Reranker (text-only): %d → %d results",
                    rerank_out.total_candidates,
                    len(rerank_out.final_results),
                )

        state["workflow_steps"] = steps
        return state

    # ------------------------------------------------------------------
    # BRIEF search workflow
    # ------------------------------------------------------------------

    def _search_with_brief(self, state: AgentState, steps: List[Dict]) -> AgentState:
        """Search workflow when a creative brief was uploaded (PM ran first).

        Uses the PM-generated dual queries directly:
        - semantic_query → sent to MCP to fetch OpenSearch payload (neural embedding)
        - lexical_query  → patched into the lexical sub-query of the payload (OR operator)
        """
        logger.info("Search Specialist [BRIEF]: Starting brief workflow...")

        # Detect "show AI-generated" intent from the user's original message
        # PM node already sets state["show_generated"] — use that if available
        show_generated = state.get("show_generated") or _is_show_generated_query(state.get("user_query", ""))
        if show_generated:
            logger.info("Search Specialist [BRIEF]: 'show generated' intent detected")

        # Determine explicit not-generated flag from PM's is_generated analysis
        pm_is_generated = (state.get("requirements") or {}).get("is_generated")
        explicit_not_generated = pm_is_generated is False

        # --- Step 1: Get PM-generated queries directly ---
        semantic_query = state.get("semantic_query") or ""
        lexical_query = state.get("lexical_query") or ""

        # Fallback if PM didn't produce proper queries
        if not semantic_query.strip():
            semantic_query = state["user_query"]
        if not lexical_query.strip():
            lexical_query = state["user_query"]

        # Parse lexical into terms (already capped at 6 by PM)
        entity_terms = lexical_query.split()[:6]
        lexical_query_string = " OR ".join(entity_terms)

        # Use PM-generated boolean_query if available, otherwise fall back to OR-joined terms
        boolean_query_string = state.get("boolean_query") or lexical_query_string

        # Gather PM-extracted filters, categories, exclusions from state
        category_gids = state.get("category_gids") or []
        exclusion_terms = state.get("exclusion_terms") or []
        refinement_filters = state.get("refinement_filters") or []
        category_names = [_category_filter.label_for_gid(g) for g in category_gids]

        logger.info(
            f"Search Specialist [BRIEF]: semantic='{semantic_query}', "
            f"lexical='{lexical_query_string}', categories={category_names}"
        )

        steps.append({
            "agent": "Search Specialist",
            "action": "Query Preparation (Brief)",
            "reasoning": (
                f"Using PM-generated queries directly. "
                f"Semantic query for MCP/neural search: \"{semantic_query}\". "
                f"Lexical query for keyword matching: \"{lexical_query_string}\" (OR operator, {len(entity_terms)} terms). "
                f"Carrying forward PM-extracted filters: {len(category_gids)} categories {category_names}, "
                f"{len(exclusion_terms)} exclusions {exclusion_terms}, "
                f"{len(refinement_filters)} refinement filter(s)."
            ),
            "input": {
                "pm_semantic_query": semantic_query,
                "pm_lexical_query": lexical_query,
            },
            "output": {
                "entity_terms": entity_terms,
                "lexical_query": lexical_query_string,
                "lexical_operator": "OR",
                "semantic_query": semantic_query,
                "category_gids": category_gids,
                "category_names": category_names,
                "exclusion_terms": exclusion_terms,
                "refinement_filters": describe_filters(refinement_filters),
            },
        })

        # --- Step 2: MCP call with the semantic query ---
        search_mode = state.get("search_mode", "relevance")
        mcp_tool_name = "search_popular" if search_mode == "popular" else "search_relevant"
        mcp_result = search_service_mcp.call_tool(mcp_tool_name, semantic_query)
        mcp_metadata = mcp_result.get("search_service_metadata", {})

        steps.append({
            "agent": "Search Specialist",
            "action": f"MCP Tool: {mcp_tool_name}",
            "reasoning": (
                f"Called Search Service MCP tool '{mcp_tool_name}' with semantic query \"{semantic_query}\". "
                f"Search Service found {mcp_metadata.get('num_found', 'N/A')} matches using "
                f"'{mcp_metadata.get('ranker', 'unknown')}' ranker."
            ),
            "input": {
                "mcp_tool": mcp_tool_name,
                "semantic_query": semantic_query,
                "sort_order": search_mode,
                "search_service_url": (
                    f"http://search.shuttercorp.net/v2/shutterstock/image/search"
                    f"?q={semantic_query}&sort_order={search_mode}"
                    f"&debug_modes=request&source=enterprise"
                ),
            },
            "output": {
                "ranker": mcp_metadata.get("ranker", "unknown"),
                "ranker_settings": mcp_metadata.get("ranker_settings", "unknown"),
                "num_found_by_search_service": mcp_metadata.get("num_found", 0),
            },
            "search_service_endpoint": mcp_result.get("search_service_endpoint"),
            "search_service_response": mcp_result.get("search_service_response_payload"),
        })

        # --- Step 3: Modify payload & execute ---
        opensearch_query = mcp_result.get("opensearch_query")
        if opensearch_query:
            modifications = self._modify_opensearch_payload(
                opensearch_query=opensearch_query,
                entity_terms=entity_terms,
                lexical_operator="or",   # Brief uses OR
                category_gids=category_gids,
                exclusion_terms=exclusion_terms,
                refinement_filters=refinement_filters,
                show_generated=show_generated,
                boolean_query_string=boolean_query_string,
                is_not_generated=explicit_not_generated,
            )

            steps.append({
                "agent": "Search Specialist",
                "action": "OpenSearch Payload Modification (Brief)",
                "reasoning": (
                    f"Modified the MCP-returned OpenSearch payload for brief workflow. "
                    f"Modifications applied: {'; '.join(modifications['descriptions'])}."
                ),
                "input": {
                    "modifications_planned": {
                        "lexical_patch": f"OR-joined entity terms (limit 5): \"{lexical_query_string}\"",
                        "category_filter": f"global_category_ids in {category_gids}" if category_gids else "none",
                        "must_not_exclusions": f"keywords_en must_not: {exclusion_terms}" if exclusion_terms else "none",
                        "refinement_filters": describe_filters(refinement_filters) if refinement_filters else "none",
                    },
                },
                "output": {
                    "modifications_applied": modifications["descriptions"],
                    "total_modifications": len(modifications["descriptions"]),
                },
                "opensearch_payload": opensearch_query,
            })

            search_result = photo_search_service.execute_raw_query(
                opensearch_query=opensearch_query,
                search_pipeline="hybrid_10_90",
            )

            # Fallback: if zero results and categories were applied, retry once
            # with only category filters removed.
            if search_result.get("total", 0) == 0 and category_gids:
                fallback_query = copy.deepcopy(opensearch_query)
                removed_count = self._remove_category_filters(fallback_query)
                if removed_count > 0:
                    fallback_result = photo_search_service.execute_raw_query(
                        opensearch_query=fallback_query,
                        search_pipeline="hybrid_10_90",
                    )

                    steps.append({
                        "agent": "Search Specialist",
                        "action": "Fallback Retry (Suppress Category Filters)",
                        "reasoning": (
                            f"Initial query returned 0 results with category filters {category_gids}. "
                            f"Retried by removing only category filter clause(s) "
                            f"(removed {removed_count}); all other constraints remained unchanged."
                        ),
                        "input": {
                            "removed_filter": "terms.global_category_ids",
                            "removed_count": removed_count,
                        },
                        "output": {
                            "total_results": fallback_result.get("total", 0),
                            "returned_results": len(fallback_result.get("results", [])),
                            "took_ms": fallback_result.get("took_ms", 0),
                            "fallback_used": fallback_result.get("total", 0) > 0,
                        },
                        "opensearch_payload": fallback_query,
                    })

                    if fallback_result.get("total", 0) > 0:
                        search_result = fallback_result
                        logger.info(
                            "Search Specialist [BRIEF]: Fallback succeeded after removing category filters"
                        )
                    else:
                        logger.info(
                            "Search Specialist [BRIEF]: Fallback returned 0 results; keeping original result"
                        )
        else:
            logger.warning("Search Specialist: MCP query failed, falling back to direct search")
            search_result = photo_search_service.search_photos(
                query=lexical_query_string, size=50, min_score=1.0
            )

        state["search_results"] = search_result.get("results", [])
        state["total_results"] = search_result.get("total", 0)
        state["processing_time_ms"] = search_result.get("took_ms", 0)

        # Record execution step
        steps.append({
            "agent": "Search Specialist",
            "action": "Execute OpenSearch Query (Brief)",
            "reasoning": (
                f"Executed the modified OpenSearch query against web-index-v9. "
                f"Found {state['total_results']} total matches in {state['processing_time_ms']}ms, "
                f"returning top {len(state['search_results'])} results."
            ),
            "input": {"index": "web-index-v9", "query_source": "search_service_mcp_modified"},
            "output": {
                "total_results": state["total_results"],
                "returned_results": len(state["search_results"]),
                "took_ms": state["processing_time_ms"],
            },
            "opensearch_payload": opensearch_query,
            "opensearch_url": (
                f"{settings.opensearch_endpoint}/{settings.opensearch_photo_index}/_search"
                f"?search_pipeline=hybrid_10_90"
            ),
        })

        # ── Reflection Reranker (brief workflow) ──────────────────────────────
        # Same trigger logic — reranks results from brief-based searches too
        # Use the semantic_query (actual search subject from PM) — NOT the
        # reranking request — so the LLM scores against the real search topic.
        if should_rerank(state["user_query"]):
            rerank_out = _run_async_from_sync(
                self._reranker.rerank(
                    user_query=semantic_query,
                    candidates=state["search_results"],
                    search_criteria={
                        "user_query": semantic_query,
                        "requirements": state.get("requirements"),
                        "refinement_filters": state.get("refinement_filters"),
                        "exclusion_terms": state.get("exclusion_terms"),
                        "category_gids": state.get("category_gids"),
                    },
                )
            )
            if rerank_out.triggered:
                state["search_results"] = rerank_out.final_results
                state["rerank_applied"] = True
                state["rerank_decisions"] = [
                    dataclasses.asdict(d) for d in rerank_out.decisions
                ]
                state["rerank_explanation"] = rerank_out.explanation
                steps.append({
                    "agent": "Reflection Reranker",
                    "action": "Rerank & Filter Results",
                    "reasoning": (
                        f"Reflection reranking applied. "
                        f"{rerank_out.total_candidates} candidates evaluated → "
                        f"{len(rerank_out.final_results)} selected."
                        + (f" Note: {rerank_out.explanation}" if rerank_out.explanation else "")
                    ),
                    "output": {
                        "kept": len(rerank_out.final_results),
                        "total_evaluated": rerank_out.total_candidates,
                        "explanation": rerank_out.explanation,
                        **rerank_out.pass_summaries,
                    },
                })
                logger.info(
                    "Reranker (brief): %d → %d results",
                    rerank_out.total_candidates,
                    len(rerank_out.final_results),
                )

        state["workflow_steps"] = steps
        return state

    # ------------------------------------------------------------------
    # Unified OpenSearch payload modifier
    # ------------------------------------------------------------------

    def _modify_opensearch_payload(
        self,
        opensearch_query: Dict[str, Any],
        entity_terms: List[str],
        lexical_operator: str,
        category_gids: List[int],
        exclusion_terms: List[str],
        refinement_filters: List[Dict[str, Any]],
        show_generated: bool = False,
        boolean_query_string: str | None = None,
        is_not_generated: bool = False,
    ) -> Dict[str, str]:
        """
        Apply ALL modifications to the OpenSearch payload in-place:
        1. Patch lexical sub-query text (boolean_query_string or entity terms + operator)
        2. Inject category_gids filter (terms on global_category_ids)
        3. Inject refinement filters (orientation, recency, popularity)
        4. Add must_not clause for exclusion terms on keywords_en
        5. Inject is_generated=true filter (and strip from must_not) when show_generated
        6. Set minimum_should_match on lexical query_string
        7. Inject is_generated=false filter when explicitly requested (is_not_generated)
        8. Add collapse on cluster_id_5

        Args:
            opensearch_query: The MCP-returned OpenSearch query (mutated in-place)
            entity_terms:     Clean entity tokens, e.g. ["beach", "sunset"]
            lexical_operator: "and" for text-only, "or" for brief workflow
            category_gids:    List of category GID ints for filtering
            exclusion_terms:  Keywords to exclude via must_not on keywords_en
            refinement_filters: Additional filter clauses (orientation, date, etc.)
            show_generated:   When True, inject is_generated=true and strip from must_not
            boolean_query_string: Pre-built Lucene boolean string; when provided, used
                                  directly as the lexical sub-query text instead of
                                  constructing from entity_terms + operator.
            is_not_generated: When True, inject explicit is_generated=false filter

        Returns:
            dict with "descriptions": list of human-readable modification descriptions
        """
        descriptions: List[str] = []

        # 1. Patch lexical sub-query
        if boolean_query_string:
            # Use the pre-built boolean query string directly
            self._patch_match_nodes(opensearch_query, boolean_query_string, "and")
            descriptions.append(
                f"Patched lexical sub-query → \"{boolean_query_string}\" "
                f"(pre-built boolean query)"
            )
            logger.info(f"Payload mod: Patched lexical → \"{boolean_query_string}\" (boolean)")
        elif entity_terms:
            join_str = f" {lexical_operator.upper()} ".join(entity_terms)
            self._patch_lexical_subquery_with_operator(
                opensearch_query, entity_terms, lexical_operator
            )
            descriptions.append(
                f"Patched lexical sub-query → \"{join_str}\" "
                f"(operator: {lexical_operator.upper()})"
            )
            logger.info(f"Payload mod: Patched lexical → \"{join_str}\"")

        # 2. Inject category filter
        if category_gids:
            cat_filter = {"terms": {"global_category_ids": category_gids}}
            self._inject_filter(opensearch_query, cat_filter)
            cat_labels = [_category_filter.label_for_gid(g) for g in category_gids]
            descriptions.append(
                f"Added category filter: global_category_ids in {category_gids} "
                f"({', '.join(cat_labels)})"
            )
            logger.info(f"Payload mod: Injected category filter gids={category_gids}")

        # 3. Inject refinement filters
        for rf in refinement_filters:
            self._inject_filter(opensearch_query, rf)
        if refinement_filters:
            rf_descs = describe_filters(refinement_filters)
            descriptions.append(
                f"Added {len(refinement_filters)} refinement filter(s): {'; '.join(rf_descs)}"
            )
            logger.info(f"Payload mod: Injected {len(refinement_filters)} refinement filters")

        # 4. Add must_not exclusions on keywords_en
        if exclusion_terms:
            self._inject_must_not_keywords(opensearch_query, exclusion_terms)
            descriptions.append(
                f"Added must_not exclusion on keywords_en: {exclusion_terms}"
            )
            logger.info(f"Payload mod: Added must_not keywords_en={exclusion_terms}")

        # 5. Inject is_generated=true filter (and strip it from must_not) when requested
        if show_generated:
            self._inject_is_generated_filter(opensearch_query)
            descriptions.append(
                "Added is_generated=true filter to all hybrid sub-query filters; "
                "removed is_generated from must_not"
            )
            logger.info("Payload mod: Injected is_generated=true filter (show_generated mode)")

        # 6. Set minimum_should_match on the lexical query_string inside the hybrid
        #    function-score sub-query (queries[1] → bool.should[0].query_string)
        try:
            qs_node = (
                opensearch_query
                ["query"]["hybrid"]["queries"][1]
                ["function_score"]["query"]["bool"]["must"][0]
                ["function_score"]["query"]["function_score"]["query"]
                ["bool"]["should"][0]["query_string"]
            )
            qs_node["minimum_should_match"] = "75%"
            descriptions.append(
                "Set minimum_should_match=75% on hybrid.queries[1] lexical query_string"
            )
            logger.info("Payload mod: Set minimum_should_match=75% on hybrid.queries[1] query_string")
        except (KeyError, IndexError, TypeError):
            logger.debug("Payload mod: Could not set minimum_should_match — path not found in payload")

        # 7. Inject explicit is_generated=false filter when user wants non-generated only
        if is_not_generated and not show_generated:
            self._inject_filter(opensearch_query, {"term": {"is_generated": False}})
            descriptions.append(
                "Added explicit is_generated=false filter (user requested non-generated content)"
            )
            logger.info("Payload mod: Injected is_generated=false filter (explicit non-generated)")

        # 8. (temporarily disabled) collapse + exists filter on cluster_id_5
        # exists_filter = {"exists": {"field": "cluster_id_5"}}
        # self._inject_filter_all_bools(opensearch_query, exists_filter)
        # opensearch_query["collapse"] = {"field": "cluster_id_5"}
        # descriptions.append(
        #     "Added exists filter on cluster_id_5 to both hybrid sub-queries; "
        #     "added collapse on cluster_id_5 for visual deduplication"
        # )
        # logger.info("Payload mod: Added exists+collapse on cluster_id_5")

        if not descriptions:
            descriptions.append("No modifications needed")

        return {"descriptions": descriptions}

    def _inject_filter(self, query_body: Dict[str, Any], filter_clause: Dict[str, Any]) -> None:
        """
        Inject a filter clause into the EXISTING bool.filter in the query tree.

        The Search Service returns a nested function_score structure with a bool
        clause that already has filter/must/must_not arrays.  The hybrid search
        pipeline requires the hybrid query to remain the top-level query, so we
        CANNOT wrap it in an outer bool.  Instead, we walk the tree to find the
        first bool clause that has a 'filter' key and append to it.
        """
        target_bool = self._find_primary_bool(query_body)
        if target_bool is None:
            logger.warning("Payload mod: No existing bool clause found; cannot inject filter")
            return

        existing = target_bool.get("filter", [])
        if not isinstance(existing, list):
            existing = [existing]
        existing.append(filter_clause)
        target_bool["filter"] = existing

    def _inject_must_not_keywords(
        self, query_body: Dict[str, Any], exclusion_terms: List[str]
    ) -> None:
        """
        Inject must_not term clauses into the EXISTING bool.must_not in the query tree.

        Same constraint as _inject_filter: we must find and reuse the existing
        bool clause rather than wrapping the query in a new one.
        """
        target_bool = self._find_primary_bool(query_body)
        if target_bool is None:
            logger.warning("Payload mod: No existing bool clause found; cannot inject must_not")
            return

        must_not = target_bool.get("must_not", [])
        if not isinstance(must_not, list):
            must_not = [must_not]

        for term in exclusion_terms:
            must_not.append({"term": {"keywords_en": term}})

        target_bool["must_not"] = must_not

    def _inject_is_generated_filter(self, query_body: Dict[str, Any]) -> None:
        """
        Inject {"term": {"is_generated": {"value": true}}} into the filter array of
        EVERY bool clause found in the query tree (covers both hybrid sub-queries),
        and remove any existing is_generated term from every bool.must_not array.
        """
        filter_clause: Dict[str, Any] = {"term": {"is_generated": {"value": True}}}
        self._walk_bools_for_is_generated(query_body, filter_clause)

    def _remove_category_filters(self, query_body: Dict[str, Any]) -> int:
        """
        Remove category filter clauses from all bool.filter arrays.

        Specifically removes:
          - {"terms": {"global_category_ids": [...]}}
          - {"term": {"global_category_ids": ...}}

        Returns:
            Number of removed filter clauses.
        """
        return self._walk_and_remove_category_filters(query_body)

    def _walk_and_remove_category_filters(self, node: Any) -> int:
        removed = 0
        if isinstance(node, list):
            for item in node:
                removed += self._walk_and_remove_category_filters(item)
            return removed

        if not isinstance(node, dict):
            return 0

        if "bool" in node:
            b = node["bool"]
            if isinstance(b, dict):
                filters = b.get("filter", [])
                if isinstance(filters, list):
                    kept_filters = []
                    for clause in filters:
                        if self._is_global_category_filter(clause):
                            removed += 1
                        else:
                            kept_filters.append(clause)
                    if len(kept_filters) != len(filters):
                        b["filter"] = kept_filters

        for value in node.values():
            removed += self._walk_and_remove_category_filters(value)

        return removed

    @staticmethod
    def _is_global_category_filter(clause: Any) -> bool:
        if not isinstance(clause, dict):
            return False

        terms = clause.get("terms")
        if isinstance(terms, dict) and "global_category_ids" in terms:
            return True

        term = clause.get("term")
        if isinstance(term, dict) and "global_category_ids" in term:
            return True

        return False

    def _inject_filter_all_bools(
        self, node: Any, filter_clause: Dict[str, Any]
    ) -> None:
        """Inject a filter clause into EVERY bool.filter array in the query tree.

        Unlike ``_inject_filter`` (which targets only the primary bool), this
        walks all nested bool clauses — covering both sub-queries inside a
        hybrid query.  Skips injection if an identical clause is already present.
        """
        if isinstance(node, list):
            for item in node:
                self._inject_filter_all_bools(item, filter_clause)
            return
        if not isinstance(node, dict):
            return
        if "bool" in node:
            b = node["bool"]
            if isinstance(b, dict):
                existing = b.get("filter", [])
                if not isinstance(existing, list):
                    existing = [existing]
                if filter_clause not in existing:
                    existing.append(filter_clause)
                    b["filter"] = existing
        for value in node.values():
            self._inject_filter_all_bools(value, filter_clause)

    def _walk_bools_for_is_generated(
        self, node: Any, filter_clause: Dict[str, Any]
    ) -> None:
        """
        Recursively walk the query tree.  For every bool dict encountered:
          - Append filter_clause to bool.filter (only if not already present).
          - Strip any is_generated term clauses from bool.must_not.
        Covers both sub-queries inside a hybrid query.
        """
        if isinstance(node, list):
            for item in node:
                self._walk_bools_for_is_generated(item, filter_clause)
            return
        if not isinstance(node, dict):
            return

        if "bool" in node:
            b = node["bool"]
            if isinstance(b, dict):
                # --- inject into filter ---
                existing_filter = b.get("filter", [])
                if isinstance(existing_filter, list):
                    if not any(self._is_is_generated_term(c) for c in existing_filter):
                        existing_filter.append(filter_clause)
                        b["filter"] = existing_filter
                        logger.debug("Payload mod: injected is_generated filter into bool.filter")
                # --- remove from must_not ---
                existing_must_not = b.get("must_not", [])
                if isinstance(existing_must_not, list):
                    cleaned = [
                        c for c in existing_must_not
                        if not self._is_is_generated_term(c)
                    ]
                    if len(cleaned) != len(existing_must_not):
                        b["must_not"] = cleaned
                        logger.debug("Payload mod: removed is_generated from bool.must_not")

        # Recurse into all child values
        for value in node.values():
            self._walk_bools_for_is_generated(value, filter_clause)

    @staticmethod
    def _is_is_generated_term(clause: Any) -> bool:
        """Return True if clause is a {"term": {"is_generated": ...}} filter."""
        if isinstance(clause, dict) and "term" in clause:
            term = clause["term"]
            if isinstance(term, dict) and "is_generated" in term:
                return True
        return False

    @staticmethod
    def _find_primary_bool(node: Any, _depth: int = 0) -> Any:
        """
        Walk the query tree and return the first ``bool`` dict that contains
        a ``filter`` or ``must`` key.

        Traversal covers three patterns emitted by the Search Service:
        - Standard: query → function_score → query → bool
        - Hybrid:   query → hybrid → queries[n] → function_score → query → bool

        This is the primary bool clause where production filters (is_active,
        is_shutterstock, etc.) already live.  We inject our additional filters
        and must_not clauses here so the hybrid search pipeline is not broken.

        Returns None if no suitable bool is found.
        """
        if _depth > 15 or not isinstance(node, dict):
            return None

        # If this node is a bool with filter/must arrays, it's our target
        if "bool" in node:
            b = node["bool"]
            if isinstance(b, dict) and ("filter" in b or "must" in b):
                return b

        # Dive into function_score → query chains
        for key in ("query", "function_score"):
            child = node.get(key)
            if isinstance(child, dict):
                result = AgentSquad._find_primary_bool(child, _depth + 1)
                if result is not None:
                    return result

        # Handle hybrid queries: hybrid.queries is a list of sub-queries,
        # one of which (typically the function_score sub-query) contains
        # the primary bool with the production filter/must clauses.
        if "hybrid" in node:
            hybrid = node["hybrid"]
            if isinstance(hybrid, dict):
                for sub_query in hybrid.get("queries", []):
                    result = AgentSquad._find_primary_bool(sub_query, _depth + 1)
                    if result is not None:
                        return result

        return None
    
    def _synthesize_node(self, state: AgentState) -> AgentState:
        """
        Synthesizer: Creates final response for the user.
        Combines requirements analysis (if any) with search results.
        """
        logger.info("Synthesizer: Creating final response...")
        
        # Build response
        response_parts = []
        
        # Add requirements analysis if from Project Manager
        if state.get("requirements"):
            response_parts.append("**Brief Analysis:**")
            response_parts.append(state["requirements"]["analysis"])
            response_parts.append("")
        
        # Add search results summary
        total = state.get("total_results", 0)
        results_count = len(state.get("search_results", []))
        search_mode = state.get("search_mode", "relevance")
        mode_label = "popularity" if search_mode == "popular" else "relevance"
        
        if total > 0:
            response_parts.append(f"**Search Results ({mode_label}):** Found {total} matching photos, showing top {results_count}.")
            
            if state.get("extracted_queries"):
                response_parts.append(f"\n**Search queries used:** {', '.join(state['extracted_queries'])}")
        else:
            response_parts.append("No matching photos found. Try refining your search terms or adjusting filters.")
        
        state["response"] = "\n\n".join(response_parts)
        
        # Record workflow step
        steps = state.get("workflow_steps", [])
        steps.append({
            "agent": "Synthesizer",
            "action": "Compose Final Response",
            "reasoning": f"Combined {'brief analysis + ' if state.get('requirements') else ''}search results into a formatted response. Presenting {len(state.get('search_results', []))} photo results to the user."
        })
        state["workflow_steps"] = steps
        
        logger.info("Synthesizer: Response ready")
        
        return state
    
    def _extract_search_queries(self, analysis: str) -> List[str]:
        """
        Extract search queries from Project Manager analysis.
        Looks for patterns indicating search terms.
        """
        queries = []
        
        # Simple extraction: look for lines starting with numbers or bullets
        lines = analysis.split('\n')
        
        in_query_section = False
        for line in lines:
            line = line.strip()
            
            # Detect query section
            if any(keyword in line.lower() for keyword in ['search quer', 'search term', 'suggested search']):
                in_query_section = True
                continue
            
            # Extract queries in query section
            if in_query_section and line:
                # Remove leading numbers, bullets, dashes
                query = line.lstrip('0123456789.-•* ').strip('"\'')
                if query and len(query.split()) <= 10:  # Reasonable query length
                    queries.append(query)
                
                # Stop after empty line
                if not line:
                    in_query_section = False
        
        # Fallback: if no queries found, use key phrases from analysis
        if not queries:
            # Extract quoted phrases
            import re
            quoted = re.findall(r'"([^"]+)"', analysis)
            queries = [q for q in quoted if len(q.split()) <= 8][:5]
        
        # Limit to 5 queries
        return queries[:5] if queries else [analysis.split('\n')[0][:100]]
    
    def _understand_and_expand_query(self, raw_query: str) -> tuple:
        """
        DEPRECATED: Use ``detect_text_query_intent()`` from ``query_intent.py`` instead.
        Kept for backward compatibility. Will be removed in a future release.

        Use LLM to perform full query intent analysis for TEXT-ONLY searches:

        1a. Extract entities / keywords (max 5)
        1b. Identify applicable filters (orientation, recency, popularity)
        1c. Extract explicit exclusion terms (keywords that must NOT appear)
        1d. Generate expanded semantic query
        1e. Suggest category names that match the user intent

        Returns:
            tuple: (entity_terms, expanded_semantic_query, analysis_dict)
                   analysis_dict contains all extracted fields for workflow trace
        """
        # Build a compact category reference for the LLM
        category_list = "\n".join(
            f"  - {_category_filter.label_for_gid(gid)} (gid {gid})"
            for gid in sorted(_category_filter._gid_to_value)
        )

        system_prompt = f"""You are a stock-photo search query analyzer. Perform a full intent analysis:

1. **Entity Recognition (NER)**: Extract the core search entities — subjects, scenes, objects, actions.
   STRIP ALL of these — do NOT include them as entity terms:
   - Intent/navigation words: "show", "me", "find", "get", "see", "display", "can", "you", "please", "search", "for"
   - Reranking/meta words: "best", "match", "matching", "results", "rerank", "top", "ranked", "reviewed"
   - Generic media nouns: "images", "photos", "photos", "pictures", "content"
   - Common prepositions: "of", "the", "a", "an", "in", "on", "at", "from", "with"
   - Qualifiers: "popular", "trending", "good", "great", "nice", "amazing"
   Return up to 5 clean entity tokens that are ONLY real-world subjects, scenes, objects, people, or places.

2. **Filters**: Detect if the user implies any of these filters:
   - orientation: "horizontal" | "vertical" | "square" (only if explicitly stated or clearly implied)
   - recency: if user says "recent", "new", "latest" → provide an ES date-math value like "now-1y"
   - popularity: if user says "popular", "trending", "best-selling" → minimum license threshold (integer)

3. **Exclusion Terms**: Extract ONLY terms the user explicitly wants excluded.
   Look for patterns like "no X", "without X", "not X", "exclude X", "minus X".
   Return as a list of lowercase keyword strings (suitable for `keywords_en` must_not match).
   Return an EMPTY list if no explicit negations are present.

4. **Semantic Query**: Build an expanded query (max 8 words) combining entities with synonyms and visual descriptors for neural vector search.

5. **Category Suggestions**: From this category list, suggest the 1-3 most relevant categories:
{category_list}
   Return category names EXACTLY as shown (e.g. "Animals/Wildlife", "Nature").

Respond in this EXACT JSON format:
{{
  "intent": "Brief description of what the user is looking for",
  "entity_terms": ["entity1", "entity2"],
  "filters": {{
    "orientation": null,
    "recency_gte": null,
    "popularity_gte": null
  }},
  "exclusion_terms": [],
  "expanded_semantic_query": "expanded query with synonyms",
  "suggested_categories": ["Category/Name"],
  "mood_style": ["mood/style terms if applicable"]
}}"""

        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"User query: {raw_query}")
            ])

            import json
            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            analysis = json.loads(content)
            entity_terms = analysis.get("entity_terms") or []
            expanded_semantic_query = analysis.get("expanded_semantic_query") or raw_query

            # Normalise: lowercase, limit to 5
            entity_terms = [str(t).strip().lower() for t in entity_terms if str(t).strip()][:5]

            # Extract explicit exclusions
            exclusion_terms = [
                str(t).strip().lower()
                for t in (analysis.get("exclusion_terms") or [])
                if str(t).strip()
            ]

            # Resolve suggested categories → GIDs via fuzzy matching
            suggested_cats = analysis.get("suggested_categories") or []
            cat_gids: List[int] = []
            cat_names: List[str] = []
            for name in suggested_cats:
                matches = _category_filter.match_categories(name)
                for m in matches:
                    if m["gid"] not in cat_gids:
                        cat_gids.append(m["gid"])
                        cat_names.append(m["value"])

            # Parse inline filters
            llm_filters = analysis.get("filters") or {}
            refinement_clauses: List[Dict[str, Any]] = []
            if llm_filters.get("orientation"):
                refinement_clauses.append({"term": {"orientation": llm_filters["orientation"]}})
            if llm_filters.get("recency_gte"):
                refinement_clauses.append({"range": {"date_added": {"gte": llm_filters["recency_gte"]}}})
            if llm_filters.get("popularity_gte"):
                refinement_clauses.append({"range": {"total_paid_license_count_all_time": {"gte": int(llm_filters["popularity_gte"])}}})

            # Also pick up keyword-based refinement filters from the raw query
            keyword_refinements = extract_refinement_filters(None, extra_text=raw_query)
            # Merge without duplicating fields
            seen_fields = {next(iter(c.get("term", c.get("range", {})))):True for c in refinement_clauses}
            for kr in keyword_refinements:
                field = next(iter(kr.get("term", kr.get("range", {}))), None)
                if field and field not in seen_fields:
                    refinement_clauses.append(kr)
                    seen_fields[field] = True

            # Attach resolved data to analysis dict for workflow trace
            analysis["_resolved_category_gids"] = cat_gids
            analysis["_resolved_category_names"] = cat_names
            analysis["_refinement_clauses"] = refinement_clauses
            analysis["_exclusion_terms"] = exclusion_terms

            logger.info(
                f"Query understanding: '{raw_query}' → "
                f"entities={entity_terms}, semantic='{expanded_semantic_query}', "
                f"exclusions={exclusion_terms}, categories={cat_names}, "
                f"filters={len(refinement_clauses)}"
            )

            return entity_terms, expanded_semantic_query, analysis

        except Exception as e:
            logger.warning(f"Query expansion failed ({e}), using original query")
            fallback_terms = [t for t in raw_query.lower().split()
                              if t not in {"show", "me", "find", "get", "search", "for",
                                           "images", "photos", "pictures", "popular",
                                           "trending", "best", "top", "of", "the", "a",
                                           "an", "in", "from", "some", "please"}][:5]
            return fallback_terms or raw_query.split()[:5], raw_query, {
                "intent": "Direct search",
                "entity_terms": fallback_terms,
                "expanded_semantic_query": raw_query,
                "exclusion_terms": [],
                "suggested_categories": [],
                "filters": {},
                "mood_style": [],
                "_resolved_category_gids": [],
                "_resolved_category_names": [],
                "_refinement_clauses": [],
                "_exclusion_terms": [],
            }

    def _patch_lexical_subquery_with_operator(
        self, query_body: Dict[str, Any], entity_terms: List[str], operator: str = "and"
    ) -> None:
        """
        Walk the OpenSearch query tree and replace the text in every lexical
        (match / multi_match) sub-query with the joined entity terms.

        Args:
            query_body:   The OpenSearch query DSL dict (mutated in-place).
            entity_terms: Clean entity tokens, e.g. ["beach", "sunset"].
            operator:     "and" for text-only workflow, "or" for brief workflow.
        """
        join_str = f" {operator.upper()} ".join(entity_terms)
        self._patch_match_nodes(query_body, join_str, operator)

    def _patch_match_nodes(
        self, node: Any, query_string: str, operator: str = "and"
    ) -> None:
        """Recursively patch lexical query text; skip neural sub-trees.

        Handles three query node types found in Search Service payloads:
        - query_string:  { "query_string": { "query": "...", ... } }
        - match:         { "match": { "field": { "query": "..." } } }
        - multi_match:   { "multi_match": { "query": "...", "fields": [...] } }
        """
        if isinstance(node, dict):
            # Skip everything inside a "neural" clause
            if "neural" in node:
                return

            # Patch query_string: { "query_string": { "query": "...", ... } }
            if "query_string" in node and isinstance(node["query_string"], dict):
                qs = node["query_string"]
                if isinstance(qs.get("query"), str):
                    logger.info(
                        "Patching lexical query_string: '%s' → '%s' (operator=%s)",
                        qs["query"], query_string, operator,
                    )
                    qs["query"] = query_string
                    qs["default_operator"] = operator

            # Patch match: { "match": { "field": { "query": "...", ... } } }
            if "match" in node and isinstance(node["match"], dict):
                for field_val in node["match"].values():
                    if isinstance(field_val, dict) and isinstance(field_val.get("query"), str):
                        logger.info(
                            "Patching lexical match: '%s' → '%s' (operator=%s)",
                            field_val["query"], query_string, operator,
                        )
                        field_val["query"] = query_string
                        field_val["operator"] = operator
                    elif isinstance(field_val, str):
                        field_name = next(iter(node["match"]))
                        node["match"][field_name] = {"query": query_string, "operator": operator}

            # Patch multi_match: { "query": "...", "fields": [...] }
            if "multi_match" in node and isinstance(node["multi_match"], dict):
                mm = node["multi_match"]
                if isinstance(mm.get("query"), str):
                    logger.info(
                        "Patching lexical multi_match: '%s' → '%s' (operator=%s)",
                        mm["query"], query_string, operator,
                    )
                    mm["query"] = query_string
                    mm["operator"] = operator

            for v in node.values():
                if isinstance(v, (dict, list)):
                    self._patch_match_nodes(v, query_string, operator)

        elif isinstance(node, list):
            for item in node:
                if isinstance(item, (dict, list)):
                    self._patch_match_nodes(item, query_string, operator)

    def _resolve_followup_query(self, user_query: str, conversation_history: List[Dict[str, str]]) -> str:
        """
        Use LLM to resolve a follow-up message into a standalone search query
        by considering the conversation history.
        """
        system_prompt = """You are a search query resolver for a stock photo search engine.

You will be given a conversation history and the user's latest message. The latest message may be a follow-up
that references previous context (e.g. "show me more like that", "but with dogs instead", "make it darker",
"now show popular ones", "similar but outdoor").

Your job is to produce a single, clear, standalone search query that captures the user's full intent
by combining the context from previous messages with the current request.

Rules:
- Output ONLY the search query text, nothing else
- Do NOT include explanations, prefixes, or quotes
- Keep it concise (under 10 words ideally)
- If the follow-up is already a standalone query, return it as-is
- Incorporate relevant context from previous messages (subjects, themes, moods, constraints)
- IMPORTANT: If the user's message is a reranking/refinement request — asking to see
  "best", "best match", "best matching results", "top results", "rerank", "better results",
  "reviewed picks", etc. — WITHOUT changing the search subject, then return ONLY the
  original search subject from the conversation history. Do NOT include meta-words like
  "best", "match", "matching", "results", "rerank", "top ranked" in the output.
"""
        messages = [SystemMessage(content=system_prompt)]
        
        # Add conversation history
        for hist_msg in conversation_history:
            if hist_msg["role"] == "user":
                messages.append(HumanMessage(content=hist_msg["content"]))
            else:
                messages.append(AIMessage(content=hist_msg["content"]))
        
        messages.append(HumanMessage(content=user_query))
        
        try:
            response = self.llm.invoke(messages)
            resolved = response.content.strip().strip('"\'')
            logger.info(f"Resolved follow-up '{user_query}' -> '{resolved}'")
            return resolved
        except Exception as e:
            logger.warning(f"Follow-up resolution failed ({e}), using original query")
            return user_query

    def run(self, user_query: str, file_content: str | None = None, file_images: List[Dict[str, Any]] | None = None, image_analysis: Dict[str, Any] | None = None, file_type: str | None = None, conversation_history: List[Dict[str, str]] | None = None) -> Dict[str, Any]:
        """
        Run the agent squad.
        
        Args:
            user_query: User's message/query
            file_content: Extracted text from uploaded file (optional)
            file_type: Type of uploaded file (optional)
            
        Returns:
            dict with agent outputs including response and search results
        """
        try:
            # --- Write-intent guard ---
            # Block any query that looks like a data-mutation request before
            # it reaches the agent graph.  The transport-level guardrails in
            # opensearch_guardrails.py provide a second safety net, but
            # catching this early gives users a clear, actionable message.
            if _is_write_intent_query(user_query):
                logger.warning(
                    "Write-intent guard triggered for query: %s",
                    user_query[:120],
                )
                return {
                    "response": _WRITE_BLOCK_RESPONSE,
                    "search_results": [],
                    "total_results": 0,
                    "requirements": None,
                    "processing_time_ms": 0,
                    "workflow_steps": [
                        {
                            "agent": "Write-Intent Guard",
                            "action": "Blocked write request",
                            "reasoning": (
                                "The query contains a data-mutation pattern. "
                                "The production OpenSearch domain is read-only; "
                                "write operations are prohibited."
                            ),
                            "input": {"query": user_query},
                            "decision": "refused",
                        }
                    ],
                    "error": "write_operation_denied",
                }

            # Initialize state
            initial_state = AgentState(
                messages=[],
                user_query=user_query,
                file_content=file_content,
                file_images=file_images,
                image_analysis=image_analysis,
                file_type=file_type,
                conversation_history=conversation_history or [],
                route=None,
                search_mode=None,
                requirements=None,
                extracted_queries=None,
                lexical_query=None,
                semantic_query=None,
                category_gids=None,
                refinement_filters=None,
                exclusion_terms=None,
                show_generated=None,
                media_type=None,
                boolean_query=None,
                named_entities=None,
                search_results=None,
                total_results=0,
                # Reranker state — defaults to None/False until triggered
                rerank_applied=None,
                rerank_decisions=None,
                rerank_explanation=None,
                response=None,
                processing_time_ms=0,
                workflow_steps=[]
            )
            
            # Run graph
            logger.info(f"Starting agent execution for query: {user_query[:100]}...")
            final_state = self.graph.invoke(initial_state)
            
            category_gids = final_state.get("category_gids") or []
            refinement_filters = final_state.get("refinement_filters") or []
            exclusion_terms = final_state.get("exclusion_terms") or []
            return {
                "response": final_state.get("response", "No response generated"),
                "search_results": final_state.get("search_results", []),
                "total_results": final_state.get("total_results", 0),
                "requirements": final_state.get("requirements"),
                "processing_time_ms": final_state.get("processing_time_ms", 0),
                "workflow_steps": final_state.get("workflow_steps", []),
                "search_mode": final_state.get("search_mode", "relevance"),
                # Reranker outputs
                "rerank_applied": final_state.get("rerank_applied") or False,
                "rerank_decisions": final_state.get("rerank_decisions"),
                "rerank_explanation": final_state.get("rerank_explanation"),
                "filter_metadata": {
                    "category_gids": category_gids,
                    "category_values": [
                        _category_filter.label_for_gid(gid)
                        for gid in category_gids
                    ],
                    "exclusion_terms": exclusion_terms,
                    "refinement_filter_descriptions": describe_filters(refinement_filters),
                    "filters_applied": bool(category_gids or refinement_filters or exclusion_terms),
                },
            }
            
        except AuthenticationError as e:
            logger.error(f"OpenAI authentication failed: {str(e)}")
            return {
                "response": "Invalid OpenAI API key. Please check your API key and try again.",
                "search_results": [],
                "total_results": 0,
                "requirements": None,
                "processing_time_ms": 0,
                "error": "authentication_error"
            }
        except OpenAIError as e:
            logger.error(f"OpenAI API error: {str(e)}")
            return {
                "response": f"OpenAI API error: {str(e)}. Please try again.",
                "search_results": [],
                "total_results": 0,
                "requirements": None,
                "processing_time_ms": 0,
                "error": "openai_error"
            }
        except Exception as e:
            logger.error(f"Agent execution failed: {str(e)}", exc_info=True)
            return {
                "response": f"I encountered an error processing your request: {str(e)}",
                "search_results": [],
                "total_results": 0,
                "requirements": None,
                "processing_time_ms": 0,
                "error": str(e)
            }
