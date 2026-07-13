#!/usr/bin/env python3
"""Find deployed neural models on the cluster (including remote connectors).

IMPORTANT: This script uses the application's guarded OpenSearch client so that
the production read-only guardrails (opensearch_guardrails.py) are active.
Raw HTTP libraries (httpx, requests) that bypass those guardrails must NOT be
used against localhost or any other read-only cluster endpoint.
"""
import json
import sys
import os

# Ensure the backend package is importable when run from the backend/ directory.
sys.path.insert(0, os.path.dirname(__file__))

from app.config import settings
from app.services.opensearch_guardrails import create_opensearch_client, is_readonly_endpoint

# Build a guarded client — guardrails are applied automatically because
# localhost is listed in settings.opensearch_readonly_hosts.
_readonly = is_readonly_endpoint(
    endpoint=settings.opensearch_endpoint,
    forced_readonly=settings.opensearch_readonly,
    readonly_hosts=settings.opensearch_readonly_hosts,
)
client = create_opensearch_client(
    endpoint=settings.opensearch_endpoint,
    readonly=_readonly,
    timeout_seconds=5.0,
)
print(f"[info] read-only guardrails active: {_readonly}")

# Check connectors and models via the ML plugin's _search endpoints.
# These are read-only POST /<path>/_search calls and are allowed by the guardrails.
for index_path in [
    "_plugins/_ml/connectors/_search",
    "_plugins/_ml/models/_search",
]:
    print(f"\n=== /{index_path} ===")
    try:
        data = client.transport.perform_request(
            "POST",
            f"/{index_path}",
            body={"query": {"match_all": {}}, "size": 50},
        )
        hits = data.get("hits", {}).get("hits", [])
        total = data.get("hits", {}).get("total", {}).get("value", 0)
        print(f"Total: {total}, returned: {len(hits)}")
        for hit in hits[:10]:
            src = hit.get("_source", {})
            print(f"  id={hit['_id']}, name={src.get('name', '?')}, state={src.get('model_state', '?')}")
    except Exception as e:
        print(f"Error: {e}")

# Check for neural search request processors in any pipeline.
print("\n=== All search pipelines ===")
try:
    data = client.transport.perform_request("GET", "/_search/pipeline")
    for name, pipeline in data.items():
        has_neural = any("neural" in json.dumps(p).lower() for p in pipeline.get("request_processors", []))
        print(
            f"  {name}: request_processors={len(pipeline.get('request_processors', []))}, "
            f"phase={len(pipeline.get('phase_results_processors', []))}, neural_req={has_neural}"
        )
        if pipeline.get("request_processors"):
            for rp in pipeline["request_processors"]:
                print(f"    req_processor: {json.dumps(rp)[:200]}")
except Exception as e:
    print(f"Error: {e}")


