#!/usr/bin/env python3
"""Trace the full query flow to see the actual hybrid structure."""
import json
from app.services.search_service_mcp import search_service_mcp

# Try both search modes
for mode in ["search_relevant", "search_popular"]:
    print(f"\n{'='*60}")
    print(f"MODE: {mode}")
    print(f"{'='*60}")
    result = search_service_mcp.call_tool(mode, "healthcare workers")
    q = result.get("opensearch_query", {})
    query = q.get("query", {})
    
    # Check for hybrid at any depth
    def find_hybrid(node, path=""):
        if isinstance(node, dict):
            if "hybrid" in node:
                return path + ".hybrid"
            for k, v in node.items():
                r = find_hybrid(v, path + "." + k)
                if r: return r
        elif isinstance(node, list):
            for i, item in enumerate(node):
                r = find_hybrid(item, path + f"[{i}]")
                if r: return r
        return None

    hybrid_path = find_hybrid(q)
    print(f"Hybrid found at: {hybrid_path}")
    print(f"Top-level query keys: {list(query.keys())}")
    
    # Dump full structure
    print(json.dumps(q, indent=2, default=str))
