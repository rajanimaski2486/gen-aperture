"""
LangGraph multi-agent system for Gen-Aperture.
Implements Squad Router, Project Manager Strand, and Search Specialist Strand.
"""
import logging
from typing import Dict, Any, List, TypedDict, Annotated, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from openai import AuthenticationError, OpenAIError
from app.services.photo_search import photo_search_service
from app.services.search_service_mcp import search_service_mcp

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State shared across all agents."""
    messages: Annotated[List[Any], add_messages]
    user_query: str
    file_content: str | None
    file_type: str | None
    conversation_history: List[Dict[str, str]]
    
    # Routing decision
    route: Literal["project_manager", "search_specialist"] | None
    
    # Project Manager outputs
    requirements: Dict[str, Any] | None
    extracted_queries: List[str] | None
    
    # Search intent (determined by LLM)
    search_mode: Literal["relevance", "popular"] | None
    
    # Search Specialist outputs
    search_results: List[Dict[str, Any]] | None
    total_results: int
    
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
        
        try:
            intent_response = self.llm.invoke(intent_messages)
            raw_intent = intent_response.content.strip().lower()
            search_mode = "popular" if "popular" in raw_intent else "relevance"
        except Exception as e:
            logger.warning(f"Router: Intent detection failed ({e}), defaulting to relevance")
            search_mode = "relevance"
        
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
        Extracts visual requirements, themes, moods, constraints.
        Does NOT search - only requirements extraction.
        """
        logger.info("Project Manager: Analyzing brief...")
        
        system_prompt = """You are a Project Manager AI analyzing creative briefs for stock photo searches.

Your task is to extract:
1. Visual requirements (subjects, scenes, compositions)
2. Themes and moods (emotions, atmosphere, style)
3. Technical constraints (orientation, color palette, quality)
4. Search queries (3-5 optimized search terms for stock photo databases)

Analyze the brief and extract actionable search requirements. Be specific and detailed.

Format your response as a structured analysis followed by 3-5 search queries."""
        
        # Prepare context
        user_context = f"User query: {state['user_query']}\n\n"
        if state.get('file_content'):
            user_context += f"Brief content ({state.get('file_type', 'unknown')} file):\n{state['file_content']}"
        
        # Call LLM
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_context)
        ]
        
        response = self.llm.invoke(messages)
        analysis = response.content
        
        # Extract search queries from the analysis
        # Look for lines that appear to be search queries
        queries = self._extract_search_queries(analysis)
        
        logger.info(f"Project Manager: Extracted {len(queries)} search queries")
        
        state["requirements"] = {
            "analysis": analysis,
            "file_type": state.get("file_type")
        }
        state["extracted_queries"] = queries
        state["messages"] = add_messages(state["messages"], [response])
        
        # Record workflow step
        steps = state.get("workflow_steps", [])
        steps.append({
            "agent": "Project Manager",
            "action": "Analyze Brief & Extract Requirements",
            "prompt": system_prompt,
            "reasoning": f"Analyzed the uploaded {state.get('file_type', 'unknown')} file and user query. Extracted visual requirements, themes, moods, and constraints. Generated {len(queries)} optimized search queries for the Search Specialist.",
            "output": {
                "extracted_queries": queries,
                "analysis_summary": analysis[:300] + "..." if len(analysis) > 300 else analysis
            }
        })
        state["workflow_steps"] = steps
        
        return state
    
    def _search_specialist_node(self, state: AgentState) -> AgentState:
        """
        Search Specialist Strand: Executes searches via MCP tools.
        
        Flow:
        1. Determine the search query text
        2. Use LLM to understand intent, extract key terms, and expand query
        3. Call Search Service MCP tool (search_relevant or search_popular)
           to get the production OpenSearch query from debug.request
        4. Execute that query against OpenSearch
        5. Return formatted results
        """
        logger.info("Search Specialist: Executing search via MCP...")
        
        steps = state.get("workflow_steps", [])
        
        # --- Step 1: Determine raw search query text ---
        query_source = "project_manager"
        if state.get("extracted_queries"):
            raw_query = " ".join(state["extracted_queries"][:3])
            logger.info(f"Search Specialist: Using {len(state['extracted_queries'])} extracted queries")
        else:
            conversation_history = state.get("conversation_history", [])
            if conversation_history:
                raw_query = self._resolve_followup_query(state["user_query"], conversation_history)
                query_source = "contextual_followup"
                logger.info(f"Search Specialist: Resolved follow-up to: {raw_query}")
            else:
                raw_query = state["user_query"]
                query_source = "user_direct"
                logger.info("Search Specialist: Using direct user query")
        
        # --- Step 2: LLM-powered query understanding & expansion ---
        search_query, query_analysis = self._understand_and_expand_query(raw_query)
        logger.info(f"Search Specialist: Expanded query: \"{raw_query}\" → \"{search_query}\"")
        
        # Record query understanding step
        steps.append({
            "agent": "Search Specialist",
            "action": "Query Understanding & Expansion",
            "reasoning": (
                f"Analyzed user intent from \"{raw_query}\". "
                f"Extracted core concepts and expanded with synonyms/related terms "
                f"for hybrid semantic+lexical search. "
                f"Optimized query: \"{search_query}\"."
            ),
            "input": {
                "raw_query": raw_query,
                "query_source": query_source,
            },
            "output": query_analysis,
        })
        
        # --- Step 3: Call MCP tool to get OpenSearch query ---
        search_mode = state.get("search_mode", "relevance")
        mcp_tool_map = {"relevance": "search_relevant", "popular": "search_popular"}
        mcp_tool_name = mcp_tool_map.get(search_mode, "search_relevant")
        
        logger.info(f"Search Specialist: Calling MCP tool '{mcp_tool_name}' with expanded query: {search_query}")
        
        mcp_result = search_service_mcp.call_tool(mcp_tool_name, search_query)
        
        # Record MCP call step
        mcp_metadata = mcp_result.get("search_service_metadata", {})
        steps.append({
            "agent": "Search Specialist",
            "action": f"MCP Tool: {mcp_tool_name}",
            "reasoning": (
                f"Called Search Service MCP tool '{mcp_tool_name}' with query \"{search_query}\". "
                f"The Search Service returned a production-grade OpenSearch query using the "
                f"'{mcp_metadata.get('ranker', 'unknown')}' ranker with '{mcp_metadata.get('ranker_settings', 'unknown')}' settings. "
                f"Search Service found {mcp_metadata.get('num_found', 'N/A')} total matches. "
                f"Extracted the debug.request OpenSearch DSL query for direct execution."
            ),
            "input": {
                "mcp_tool": mcp_tool_name,
                "search_query": search_query,
                "query_source": query_source,
                "sort_order": search_mode,
                "search_service_url": f"http://search.shuttercorp.net/v2/shutterstock/image/search?q={search_query}&sort_order={search_mode}&debug_modes=request&source=enterprise"
            },
            "output": {
                "ranker": mcp_metadata.get("ranker", "unknown"),
                "ranker_settings": mcp_metadata.get("ranker_settings", "unknown"),
                "search_type": mcp_metadata.get("search_type", "unknown"),
                "num_found_by_search_service": mcp_metadata.get("num_found", 0)
            }
        })
        
        # --- Step 4: Execute raw query against OpenSearch ---
        opensearch_query = mcp_result.get("opensearch_query")
        
        if opensearch_query:
            logger.info("Search Specialist: Executing MCP-provided OpenSearch query...")
            search_result = photo_search_service.execute_raw_query(
                opensearch_query=opensearch_query
            )
        else:
            # Fallback to direct search if MCP failed
            logger.warning("Search Specialist: MCP query failed, falling back to direct search")
            search_result = photo_search_service.search_photos(
                query=search_query,
                size=20,
                min_score=1.0
            )
        
        state["search_results"] = search_result.get("results", [])
        state["total_results"] = search_result.get("total", 0)
        state["processing_time_ms"] = search_result.get("took_ms", 0)
        
        logger.info(f"Search Specialist: Found {state['total_results']} results")
        
        # Record OpenSearch execution step
        steps.append({
            "agent": "Search Specialist",
            "action": "Execute OpenSearch Query",
            "reasoning": (
                f"Executed the {'MCP-provided production' if opensearch_query else 'fallback'} OpenSearch query against web-index-v9. "
                f"Found {state['total_results']} total matches in {state['processing_time_ms']}ms, "
                f"returning top {len(state['search_results'])} results."
            ),
            "input": {
                "query_source": "search_service_mcp" if opensearch_query else "direct_fallback",
                "index": "web-index-v9"
            },
            "output": {
                "total_results": state["total_results"],
                "returned_results": len(state["search_results"]),
                "took_ms": state["processing_time_ms"]
            },
            "opensearch_payload": mcp_result.get("original_opensearch_query", opensearch_query) if opensearch_query else None
        })
        state["workflow_steps"] = steps
        
        return state
    
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
        Use LLM to understand the user's search intent, extract key terms,
        and expand the query with synonyms and related concepts for optimal
        hybrid (semantic + lexical) search performance.
        
        Returns:
            tuple: (expanded_query_string, analysis_dict)
        """
        system_prompt = """You are a stock photo search query optimizer. Given a user's natural language request, you must:

1. **Understand Intent**: What is the user truly looking for? Identify the core visual concept.
2. **Extract Key Terms**: Pull out the essential subject, action, setting, mood, and style terms.
3. **Expand Query**: Add synonyms, related visual terms, and alternate phrasings that stock photo databases use in their keyword metadata. Think about how stock photo contributors tag their images.

IMPORTANT RULES:
- Stock photo keywords are typically single words or short phrases, not sentences
- Include both specific and broader terms to cast a wider net
- Add visual descriptors (composition, lighting, mood) when implied by the query
- Consider related concepts that photographers would tag (e.g. "happy family" → also "togetherness", "bonding", "lifestyle")
- Keep the expanded query concise — aim for 5-15 well-chosen terms, not a paragraph
- Do NOT add unrelated terms just to pad the query
- The output query will be sent to a search engine that uses BOTH keyword matching and semantic similarity

Respond in this EXACT JSON format:
{
  "intent": "Brief description of what the user is looking for",
  "core_terms": ["term1", "term2"],
  "expanded_terms": ["synonym1", "related1", "visual_descriptor1"],
  "mood_style": ["mood/style terms if applicable"],
  "optimized_query": "The final optimized search query string combining core + expanded terms"
}"""

        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"User query: {raw_query}")
            ])
            
            # Parse the JSON response
            import json
            content = response.content.strip()
            # Handle markdown code blocks
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            
            analysis = json.loads(content)
            optimized_query = analysis.get("optimized_query", raw_query)
            
            logger.info(
                f"Query expansion: '{raw_query}' → '{optimized_query}' "
                f"(core: {analysis.get('core_terms', [])}, "
                f"expanded: {analysis.get('expanded_terms', [])})"
            )
            
            return optimized_query, analysis
            
        except Exception as e:
            logger.warning(f"Query expansion failed ({e}), using original query")
            return raw_query, {
                "intent": "Direct search",
                "core_terms": raw_query.split(),
                "expanded_terms": [],
                "mood_style": [],
                "optimized_query": raw_query
            }

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

    def run(self, user_query: str, file_content: str | None = None, file_type: str | None = None, conversation_history: List[Dict[str, str]] | None = None) -> Dict[str, Any]:
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
            # Initialize state
            initial_state = AgentState(
                messages=[],
                user_query=user_query,
                file_content=file_content,
                file_type=file_type,
                conversation_history=conversation_history or [],
                route=None,
                search_mode=None,
                requirements=None,
                extracted_queries=None,
                search_results=None,
                total_results=0,
                response=None,
                processing_time_ms=0,
                workflow_steps=[]
            )
            
            # Run graph
            logger.info(f"Starting agent execution for query: {user_query[:100]}...")
            final_state = self.graph.invoke(initial_state)
            
            return {
                "response": final_state.get("response", "No response generated"),
                "search_results": final_state.get("search_results", []),
                "total_results": final_state.get("total_results", 0),
                "requirements": final_state.get("requirements"),
                "processing_time_ms": final_state.get("processing_time_ms", 0),
                "workflow_steps": final_state.get("workflow_steps", []),
                "search_mode": final_state.get("search_mode", "relevance")
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
