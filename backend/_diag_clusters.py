"""Diagnose the dev cluster (mmr-test-v1-prod) specifically."""
import json
import httpx

DEV = "http://mmr-test-v1-prod.sstk-search-prod.ct.shuttercloud.org"
PROD = "http://nelson-v1-prod.sstk-search-prod.ct.shuttercloud.org"
INDEX = "web-index-v9"

def check(label, end, pipeline, extra_body=None):
    body = {"size": 1, "track_total_hits": True, "query": {"bool": {
        "filter": [{"term": {"is_active": {"value": True}}}]
    }}}
    if extra_body:
        body.update(extra_body)
    url = f"{end}/{INDEX}/_search"
    if pipeline:
        url += f"?search_pipeline={pipeline}"
    r = httpx.post(url, json=body, timeout=10)
    if r.status_code == 200:
        total = r.json().get("hits", {}).get("total", {})
        print(f"  {label}: total={total}")
    else:
        print(f"  {label}: HTTP {r.status_code} — {r.text[:200]}")

print("=== Dev cluster (mmr-test-v1-prod) ===")
check("no pipeline, any doc", DEV, None)
check("pipeline=hybrid_10_90, any doc", DEV, "hybrid_10_90")
check("pipeline=hybrid_10_90, is_generated=true", DEV, "hybrid_10_90",
      {"query": {"bool": {"filter": [
          {"term": {"is_active": {"value": True}}},
          {"term": {"is_generated": {"value": True}}}
      ]}}})
check("pipeline=hybrid_10_90, is_generated=true + exists:cluster_id_5", DEV, "hybrid_10_90",
      {"query": {"bool": {"filter": [
          {"term": {"is_active": {"value": True}}},
          {"term": {"is_generated": {"value": True}}},
          {"exists": {"field": "cluster_id_5"}}
      ]}}})
# Does hybrid_10_90 pipeline exist on dev?
r_pipe = httpx.get(f"{DEV}/_search/pipeline/hybrid_10_90", timeout=10)
print(f"  pipeline hybrid_10_90 exists: HTTP {r_pipe.status_code}")
r_pipe2 = httpx.get(f"{DEV}/_search/pipeline", timeout=10)
if r_pipe2.status_code == 200:
    pipes = list(r_pipe2.json().keys())
    print(f"  available pipelines: {pipes}")

print("\n=== Prod cluster (nelson-v1-prod) ===")
check("pipeline=hybrid_10_90, is_generated=true", PROD, "hybrid_10_90",
      {"query": {"bool": {"filter": [
          {"term": {"is_active": {"value": True}}},
          {"term": {"is_generated": {"value": True}}}
      ]}}})
check("pipeline=hybrid_10_90, is_generated=true + exists:cluster_id_5", PROD, "hybrid_10_90",
      {"query": {"bool": {"filter": [
          {"term": {"is_active": {"value": True}}},
          {"term": {"is_generated": {"value": True}}},
          {"exists": {"field": "cluster_id_5"}}
      ]}}})
