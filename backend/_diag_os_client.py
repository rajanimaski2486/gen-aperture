"""Test if the opensearch-py client correctly passes search_pipeline as a URL param."""
import json
import logging
logging.basicConfig(level=logging.DEBUG)

from opensearchpy import OpenSearch
from app.services.opensearch_guardrails import create_opensearch_client, is_readonly_endpoint
from app.config import settings

with open("test_query.txt") as f:
    payload = json.load(f)

# Use the same client the app uses
readonly = is_readonly_endpoint(
    endpoint=settings.opensearch_endpoint,
    forced_readonly=settings.opensearch_readonly,
    readonly_hosts=settings.opensearch_readonly_hosts,
)
client = create_opensearch_client(
    endpoint=settings.opensearch_endpoint,
    readonly=readonly,
    timeout_seconds=30.0,
)

print(f"Endpoint: {settings.opensearch_endpoint}")
print(f"Index: {settings.opensearch_photo_index}")
print(f"Readonly: {readonly}")
print()

try:
    response = client.search(
        index=settings.opensearch_photo_index,
        body=payload,
        params={"search_pipeline": "hybrid_10_90"},
    )
    total = response.get("hits", {}).get("total", {})
    hits = response.get("hits", {}).get("hits", [])
    print(f"SUCCESS: total={total}, hits_returned={len(hits)}")
except Exception as e:
    print(f"EXCEPTION: {type(e).__name__}: {e}")
