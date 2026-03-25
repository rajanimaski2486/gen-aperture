"""Diagnose: do generated images have cluster_id_5 in the index?"""
import json
import httpx

END = "http://nelson-v1-prod.sstk-search-prod.ct.shuttercloud.org"
PIPELINE = "hybrid_10_90"
INDEX = "web-index-v9"
URL = f"{END}/{INDEX}/_search?search_pipeline={PIPELINE}"

base_filters = [
    {"term": {"is_active": {"value": True}}},
    {"term": {"is_generated": {"value": True}}},
]

def run(label, extra_filters=None, collapse=None):
    payload = {
        "_source": ["hadron_id", "ext_id", "is_generated"],
        "size": 3,
        "track_total_hits": True,
        "query": {
            "bool": {
                "filter": base_filters + (extra_filters or []),
                "must": [{"match": {"description_en": "cat"}}],
            }
        },
    }
    if collapse:
        payload["collapse"] = collapse
    r = httpx.post(URL, json=payload, timeout=15)
    if r.status_code != 200:
        print(f"{label:50s} -> HTTP {r.status_code}: {r.text[:200]}")
        return
    total = r.json().get("hits", {}).get("total", {})
    hits = r.json().get("hits", {}).get("hits", [])
    print(f"{label:50s} -> total={total}, hits returned={len(hits)}")

print("=== Diagnosing cluster_id_5 + is_generated ===\n")
run("1. generated cats (no cluster_id_5 filter, no collapse)")
run("2. generated cats + exists:cluster_id_5",
    extra_filters=[{"exists": {"field": "cluster_id_5"}}])
run("3. generated cats + collapse:cluster_id_5 (no exists filter)",
    collapse={"field": "cluster_id_5"})
run("4. generated cats + exists:cluster_id_5 + collapse",
    extra_filters=[{"exists": {"field": "cluster_id_5"}}],
    collapse={"field": "cluster_id_5"})

# Check if any generated images at all lack cluster_id_5
print()
payload_check = {
    "_source": ["hadron_id", "is_generated"],
    "size": 1,
    "track_total_hits": True,
    "query": {
        "bool": {
            "filter": [
                {"term": {"is_active": {"value": True}}},
                {"term": {"is_generated": {"value": True}}},
                {"bool": {"must_not": [{"exists": {"field": "cluster_id_5"}}]}},
            ]
        }
    },
}
r5 = httpx.post(URL, json=payload_check, timeout=15)
if r5.status_code == 200:
    total5 = r5.json().get("hits", {}).get("total", {})
    print(f"5. Generated images WITHOUT cluster_id_5    -> total={total5}")
else:
    print(f"5. HTTP {r5.status_code}: {r5.text[:200]}")
