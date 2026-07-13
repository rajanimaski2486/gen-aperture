"""Test the actual test_query.txt payload against the endpoint."""
import json
import httpx

END = "http://localhost:9200"
PIPELINE = "hybrid_10_90"
INDEX = "web-index-v9"

with open("test_query.txt") as f:
    payload = json.load(f)

# Test 1: exact payload as-is
r1 = httpx.post(f"{END}/{INDEX}/_search?search_pipeline={PIPELINE}", json=payload, timeout=15)
total1 = r1.json().get("hits", {}).get("total") if r1.status_code == 200 else f"HTTP {r1.status_code}: {r1.text[:300]}"
hits1 = len(r1.json().get("hits", {}).get("hits", [])) if r1.status_code == 200 else 0
print(f"1. Exact payload (with collapse+exists)   -> total={total1}, hits={hits1}")

# Test 2: remove collapse and the exists filters we added, see if that fixes it
import copy
payload2 = copy.deepcopy(payload)
payload2.pop("collapse", None)

def remove_exists_cluster(node):
    if isinstance(node, dict):
        for key in ("filter", "must_not", "must", "should"):
            if isinstance(node.get(key), list):
                node[key] = [
                    c for c in node[key]
                    if not (isinstance(c, dict) and c.get("exists", {}).get("field") == "cluster_id_5")
                ]
        for v in node.values():
            remove_exists_cluster(v)
    elif isinstance(node, list):
        for item in node:
            remove_exists_cluster(item)

remove_exists_cluster(payload2)
r2 = httpx.post(f"{END}/{INDEX}/_search?search_pipeline={PIPELINE}", json=payload2, timeout=15)
total2 = r2.json().get("hits", {}).get("total") if r2.status_code == 200 else f"HTTP {r2.status_code}: {r2.text[:300]}"
hits2 = len(r2.json().get("hits", {}).get("hits", [])) if r2.status_code == 200 else 0
print(f"2. Payload without collapse+exists filter -> total={total2}, hits={hits2}")

# Test 3: with collapse only (no exists filter)
payload3 = copy.deepcopy(payload2)
payload3["collapse"] = {"field": "cluster_id_5"}
r3 = httpx.post(f"{END}/{INDEX}/_search?search_pipeline={PIPELINE}", json=payload3, timeout=15)
total3 = r3.json().get("hits", {}).get("total") if r3.status_code == 200 else f"HTTP {r3.status_code}: {r3.text[:300]}"
hits3 = len(r3.json().get("hits", {}).get("hits", [])) if r3.status_code == 200 else 0
print(f"3. Payload with collapse only (no exists) -> total={total3}, hits={hits3}")
