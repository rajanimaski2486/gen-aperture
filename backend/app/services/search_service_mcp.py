"""
Search Service MCP (Model Context Protocol) Server for Gen-Aperture.

Acts as an MCP tool provider that wraps the Shutterstock Search Service API.
Exposes two tools:
  1. search_relevant - Get OpenSearch query for relevance-ranked results
  2. search_popular  - Get OpenSearch query for popularity-ranked results

Each tool calls the Search Service, extracts the `debug.request` (the raw
OpenSearch query DSL), adapts it for the local OpenSearch cluster, and returns
it for execution.

NOTE: The local cluster (mmr-test-v1-prod) only contains video content, so
the `media_type: image` filter from the production query is removed. The query
structure, ranking, and all other filters are preserved.
"""
import copy
import logging
import json
import urllib.parse
import httpx
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

SEARCH_SERVICE_BASE_URL = "http://search.shuttercorp.net/v2/shutterstock/image/search"


class SearchServiceMCPTool:
    """
    A single MCP tool definition for the Search Service.
    """
    def __init__(self, name: str, description: str, sort_order: str):
        self.name = name
        self.description = description
        self.sort_order = sort_order


class SearchServiceMCP:
    """
    MCP Server that provides Search Service tools to the agent squad.
    
    Tools:
        search_relevant: Fetches the production-grade OpenSearch query for
                         relevance-ordered image search.
        search_popular:  Fetches the production-grade OpenSearch query for
                         popularity/trending image search.
    """

    def __init__(self):
        self.base_url = SEARCH_SERVICE_BASE_URL
        self.client = httpx.Client(timeout=15.0)

        # Define available MCP tools
        self.tools = {
            "search_relevant": SearchServiceMCPTool(
                name="search_relevant",
                description=(
                    "Search for relevant stock images. Use when the user wants "
                    "images that best match their query by visual/textual relevance. "
                    "Returns a production-grade OpenSearch query optimized for relevance ranking."
                ),
                sort_order="relevance",
            ),
            "search_popular": SearchServiceMCPTool(
                name="search_popular",
                description=(
                    "Search for popular/trending stock images. Use when the user wants "
                    "images that are popular, trending, best-selling, or most downloaded. "
                    "Returns a production-grade OpenSearch query optimized for popularity ranking."
                ),
                sort_order="popular",
            ),
        }

    def list_tools(self) -> list[Dict[str, str]]:
        """List available MCP tools (MCP protocol: tools/list)."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query text (e.g. 'sunset beach', 'cat playing')",
                        }
                    },
                    "required": ["query"],
                },
            }
            for tool in self.tools.values()
        ]

    def call_tool(self, tool_name: str, query: str) -> Dict[str, Any]:
        """
        Execute an MCP tool (MCP protocol: tools/call).

        Args:
            tool_name: Either 'search_relevant' or 'search_popular'
            query: The user's search query text

        Returns:
            dict with:
                - opensearch_query: The raw OpenSearch query DSL from debug.request
                - index: The target OpenSearch index/collection
                - host: The target OpenSearch host
                - search_service_response: Metadata from the search service
                - tool_name: Which tool was used
                - sort_order: 'relevance' or 'popular'
        """
        tool = self.tools.get(tool_name)
        if not tool:
            raise ValueError(f"Unknown MCP tool: {tool_name}. Available: {list(self.tools.keys())}")

        return self._fetch_opensearch_query(query, tool)

    def _fetch_opensearch_query(
        self, query: str, tool: SearchServiceMCPTool
    ) -> Dict[str, Any]:
        """
        Call the Search Service API and extract the OpenSearch query from debug.request.
        """
        params = {
            "q": query,
            "sort_order": tool.sort_order,
            "debug_modes": "request",
            "source": "enterprise",
        }

        # Build the full URL for logging / UI display
        endpoint_url = f"{self.base_url}?{urllib.parse.urlencode(params)}"

        logger.info(
            f"MCP [{tool.name}]: Calling Search Service — q={query}, sort_order={tool.sort_order}"
        )

        try:
            response = self.client.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as e:
            logger.error(f"MCP [{tool.name}]: Search Service request failed: {e}")
            return {
                "error": str(e),
                "opensearch_query": None,
                "tool_name": tool.name,
                "sort_order": tool.sort_order,
                "search_service_endpoint": endpoint_url,
                "search_service_response_payload": None,
            }

        # Extract the debug.request — this is the raw OpenSearch query DSL
        debug_request = data.get("debug", {}).get("request", {})

        if not debug_request:
            logger.warning(f"MCP [{tool.name}]: No debug.request in response")
            return {
                "error": "No debug.request found in Search Service response",
                "opensearch_query": None,
                "tool_name": tool.name,
                "sort_order": tool.sort_order,
                "search_service_endpoint": endpoint_url,
                "search_service_response_payload": {k: v for k, v in data.items() if k != "debug"},
            }

        # Extract index and host metadata from the request
        collection = debug_request.pop("collection", "web-index")
        host = debug_request.pop("host", "unknown")
        client_name = debug_request.pop("client", "unknown")

        # Save the original production query before any adaptation (for UI display)
        original_query = copy.deepcopy(debug_request)

        # Adapt the production query for the local cluster
        debug_request = self._adapt_query_for_local_cluster(debug_request)

        # Enhance the query: add _source fields we need for display
        # (field names that actually exist in web-index-v9 on mmr-test-v1-prod)
        debug_request["_source"] = [
            "hadron_id",
            "ext_id",
            "description_en",
            "date_added",
            "total_paid_license_count_all_time",
            "keywords_en",
            "global_category_ids",
            "orientation",
            "media_type",
        ]

        # Increase size to get enough results for reranking
        debug_request["size"] = 50

        # Remove script_fields we don't need (uid is computed client-side)
        debug_request.pop("script_fields", None)

        # Get search service metadata
        ss_response = data.get("response", {})
        num_found = (
            ss_response.get("content", {})
            .get("response", {})
            .get("numFound", 0)
        )
        ranker = ss_response.get("ranker_implementation", {})

        logger.info(
            f"MCP [{tool.name}]: Got OpenSearch query — "
            f"collection={collection}, host={host}, numFound={num_found}, "
            f"ranker={ranker.get('rankerImplementation', 'unknown')}"
        )

        return {
            "opensearch_query": debug_request,
            "original_opensearch_query": original_query,
            "index": collection,
            "host": host,
            "tool_name": tool.name,
            "sort_order": tool.sort_order,
            "search_service_endpoint": endpoint_url,
            "search_service_response_payload": {k: v for k, v in data.items() if k != "debug"},
            "search_service_metadata": {
                "num_found": num_found,
                "ranker": ranker.get("rankerImplementation", "unknown"),
                "ranker_settings": ranker.get("settings", "unknown"),
                "search_type": ss_response.get("search_type", "unknown"),
                "effective_language": ss_response.get("effective_language", "unknown"),
            },
        }

    # ------------------------------------------------------------------
    # Query adaptation for local cluster
    # ------------------------------------------------------------------

    def _adapt_query_for_local_cluster(self, query_body: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adapt a production Search Service query for the local mmr-test-v1-prod cluster.
        
        The local cluster only contains video content, so the `media_type: image`
        filter from the production image-search query must be removed.  All other
        filters (is_active, is_shutterstock, etc.) and the scoring/ranking logic
        are preserved.
        """
        self._remove_media_type_filter(query_body)
        return query_body

    def _remove_media_type_filter(self, node: Any) -> None:
        """
        Recursively walk the query tree and remove any
        `{"term": {"media_type": ...}}` clause from `filter` arrays.
        """
        if isinstance(node, dict):
            # If this dict has a "filter" list, prune media_type terms from it
            if "filter" in node and isinstance(node["filter"], list):
                node["filter"] = [
                    clause for clause in node["filter"]
                    if not self._is_media_type_term(clause)
                ]
                logger.info(
                    f"MCP: Removed media_type filter — "
                    f"{len(node['filter'])} filter clauses remaining"
                )
            # Recurse into all values
            for value in node.values():
                self._remove_media_type_filter(value)
        elif isinstance(node, list):
            for item in node:
                self._remove_media_type_filter(item)

    @staticmethod
    def _is_media_type_term(clause: Any) -> bool:
        """Check if a clause is a `{"term": {"media_type": ...}}` filter."""
        if isinstance(clause, dict) and "term" in clause:
            term = clause["term"]
            if isinstance(term, dict) and "media_type" in term:
                return True
        return False


# Singleton instance
search_service_mcp = SearchServiceMCP()
