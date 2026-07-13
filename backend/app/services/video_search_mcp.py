"""
Video Search MCP (Model Context Protocol) for Gen-Aperture.

Wraps the Video Search Service API.
Exposes two tools:
  1. search_relevant - Get OpenSearch query for relevance-ranked video results
  2. search_popular  - Get OpenSearch query for popular/trending video results

Each tool calls the Video Search Service, extracts the `debug.request` (the raw
OpenSearch query DSL), adapts it for execution on the local read-only domain,
and returns it for execution by PhotoSearchService.execute_raw_query().

The video search service returns queries that include a media_type: video filter —
this is kept (not stripped) to ensure only video assets are returned.
"""
import copy
import logging
import urllib.parse
import httpx
from typing import Dict, Any

logger = logging.getLogger(__name__)

VIDEO_SEARCH_SERVICE_BASE_URL = "http://localhost:8083/video/search"


class VideoSearchServiceMCPTool:
    """A single tool definition for the Video Search Service."""

    def __init__(self, name: str, description: str, sort_order: str):
        self.name = name
        self.description = description
        self.sort_order = sort_order


class VideoSearchMCP:
    """
    MCP provider that wraps the Video Search Service.

    Tools:
        search_relevant: Fetches the production-grade OpenSearch query for
                         relevance-ordered video search.
        search_popular:  Fetches the production-grade OpenSearch query for
                         popularity/trending video search.
    """

    def __init__(self):
        self.base_url = VIDEO_SEARCH_SERVICE_BASE_URL
        self.client = httpx.Client(timeout=15.0)

        self.tools = {
            "search_relevant": VideoSearchServiceMCPTool(
                name="search_relevant",
                description=(
                    "Search for relevant stock videos. Use when the user wants "
                    "video clips that best match their query. Returns a production-grade "
                    "OpenSearch query optimized for relevance ranking."
                ),
                sort_order="relevance",
            ),
            "search_popular": VideoSearchServiceMCPTool(
                name="search_popular",
                description=(
                    "Search for popular/trending stock videos. Use when the user wants "
                    "popular, trending, or best-selling video clips. Returns a production-grade "
                    "OpenSearch query optimized for popularity ranking."
                ),
                sort_order="popular",
            ),
        }

    def call_tool(self, tool_name: str, query: str) -> Dict[str, Any]:
        """
        Execute a video search MCP tool.

        Args:
            tool_name: Either 'search_relevant' or 'search_popular'
            query: The user's search query text

        Returns:
            dict with:
                - opensearch_query: The raw OpenSearch query DSL from debug.request
                - index: The target OpenSearch index
                - host: The target OpenSearch host
                - search_service_response: Metadata from the search service
                - tool_name: Which tool was used
                - sort_order: 'relevance' or 'popular'
        """
        tool = self.tools.get(tool_name)
        if not tool:
            raise ValueError(
                f"Unknown video MCP tool: {tool_name}. Available: {list(self.tools.keys())}"
            )
        return self._fetch_opensearch_query(query, tool)

    def _fetch_opensearch_query(
        self, query: str, tool: VideoSearchServiceMCPTool
    ) -> Dict[str, Any]:
        """Call the Video Search Service API and extract the OpenSearch query."""
        params = {
            "q": query,
            "sort": tool.sort_order,
            "debug_modes": "request",
        }

        endpoint_url = f"{self.base_url}?{urllib.parse.urlencode(params)}"

        logger.info(
            f"VideoMCP [{tool.name}]: Calling Video Search Service — "
            f"q={query}, sort={tool.sort_order}"
        )

        try:
            response = self.client.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as e:
            logger.error(f"VideoMCP [{tool.name}]: Video Search Service request failed: {e}")
            return {
                "error": str(e),
                "opensearch_query": None,
                "tool_name": tool.name,
                "sort_order": tool.sort_order,
                "search_service_endpoint": endpoint_url,
                "search_service_response_payload": None,
            }

        debug_request = data.get("debug", {}).get("request", {})

        if not debug_request:
            logger.warning(f"VideoMCP [{tool.name}]: No debug.request in response")
            return {
                "error": "No debug.request found in Video Search Service response",
                "opensearch_query": None,
                "tool_name": tool.name,
                "sort_order": tool.sort_order,
                "search_service_endpoint": endpoint_url,
                "search_service_response_payload": {k: v for k, v in data.items() if k != "debug"},
            }

        # Extract and discard routing metadata (not needed for direct OpenSearch calls)
        collection = debug_request.pop("collection", "web-index-v9")
        host = debug_request.pop("host", "unknown")
        debug_request.pop("client", None)

        # Save original for UI display
        original_query = copy.deepcopy(debug_request)

        # Adapt for local cluster execution
        debug_request = self._adapt_query_for_local_cluster(debug_request)

        # Enhance _source fields for display
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

        debug_request["size"] = 50

        # Ensure is_generated and date_added are in docvalue_fields
        if "docvalue_fields" not in debug_request:
            debug_request["docvalue_fields"] = []
        for required_field in ["is_generated", "date_added"]:
            if not any(
                (f.get("field") == required_field if isinstance(f, dict) else f == required_field)
                for f in debug_request["docvalue_fields"]
            ):
                debug_request["docvalue_fields"].append({"field": required_field})

        debug_request.pop("script_fields", None)

        # Gather metadata
        ss_response = data.get("response", {})
        num_found = (
            ss_response.get("content", {})
            .get("response", {})
            .get("numFound", 0)
        )
        ranker = ss_response.get("ranker_implementation", {})

        logger.info(
            f"VideoMCP [{tool.name}]: Got OpenSearch query — "
            f"collection={collection}, host={host}, numFound={num_found}"
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

    def _adapt_query_for_local_cluster(self, query_body: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adapt a video search service query for execution on the local cluster.

        Unlike the image MCP adapter, we do NOT strip the media_type filter —
        the video search service already sets media_type: video and we want to
        keep it so the local cluster returns only video assets.

        We do ensure a media_type: video filter is present in the outermost
        bool.filter so results are scoped correctly even if the service omitted it.
        """
        self._ensure_media_type_video_filter(query_body)
        return query_body

    def _ensure_media_type_video_filter(self, node: Any) -> None:
        """
        Walk the query tree and ensure exactly one
        {"term": {"media_type": "video"}} filter exists in the outermost
        bool.filter array. This is idempotent — it will not duplicate if
        already present from the video search service response.
        """
        if isinstance(node, dict):
            if "bool" in node and isinstance(node["bool"], dict):
                bool_clause = node["bool"]
                if "filter" not in bool_clause:
                    bool_clause["filter"] = []
                filters = bool_clause["filter"]
                if isinstance(filters, list):
                    # Remove any existing media_type terms (image or video) to avoid duplication
                    cleaned = [
                        clause for clause in filters
                        if not self._is_media_type_term(clause)
                    ]
                    # Inject the correct video filter
                    cleaned.append({"term": {"media_type": "video"}})
                    bool_clause["filter"] = cleaned
                    logger.info("VideoMCP: Ensured media_type=video filter in bool.filter")
                    return  # Only apply to the outermost bool
            # Recurse into children only if we haven't applied it yet
            for value in node.values():
                if isinstance(value, (dict, list)):
                    self._ensure_media_type_video_filter(value)
        elif isinstance(node, list):
            for item in node:
                self._ensure_media_type_video_filter(item)

    def _is_media_type_term(self, clause: Any) -> bool:
        """Return True if clause is a {"term": {"media_type": ...}} filter."""
        if isinstance(clause, dict) and "term" in clause:
            term = clause["term"]
            if isinstance(term, dict) and "media_type" in term:
                return True
        return False


# Singleton instance
video_search_mcp = VideoSearchMCP()
