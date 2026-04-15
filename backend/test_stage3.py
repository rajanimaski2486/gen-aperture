"""
Smoke test for Stage 3 (Agentic Curation) nodes.

Uses the example Stage 0 output (example_output_stage0.json) as search_params
and a fixed set of 10 asset IDs as mock Stage 2 output (refined_pool).

Run from the backend/ directory:
    BIFROST_API_KEY=<key> python test_stage3.py

What this tests
---------------
- score_candidates_node  : visually scores each candidate thumbnail per lane
- shortlist_candidates_node : builds a top-100, per-lane-capped shortlist
- audit_lanes_node       : audits lane coverage and decides done vs. repair

Thumbnail URLs
--------------
Shutterstock thumbnail URLs are constructed from asset_id using the standard
260nw CDN pattern.  These resolve to real thumbnails for active assets.

Stage 2 mock data
-----------------
Ten real Shutterstock asset IDs are distributed across five representative
lanes from the travel brief.  stage2_score is a realistic mock value.
"""

import json
import os
import sys
import urllib.request

# Make backend package importable when run from backend/
sys.path.insert(0, os.path.dirname(__file__))

# ---------------------------------------------------------------------------
# Fixed test data
# ---------------------------------------------------------------------------

ASSET_IDS = [
    2692876601,
    2700576551,
    2646906151,
    2727606613,
    2761480297,
    2716435553,
    2337360303,
    2440563263,
    1344272849,
    2285423829,
    # confirmed road trip images
    2477984575,
    2614981181,
    2503767067,
]

# Distribute assets across lanes that appear in the example Stage 0 output.
# Two assets per lane across five representative lanes.
# stage2_score mimics a realistic Qwen3-VL-Reranker-8B cross-encoder output.
LANE_ASSIGNMENTS = [
    {"lane": "Band Tour POV",          "asset_id": 2692876601, "stage2_score": 0.91},
    {"lane": "Band Tour POV",          "asset_id": 2700576551, "stage2_score": 0.85},
    {"lane": "Concert/Festival Goers", "asset_id": 2646906151, "stage2_score": 0.88},
    {"lane": "Concert/Festival Goers", "asset_id": 2727606613, "stage2_score": 0.82},
    {"lane": "Road Trips",             "asset_id": 2761480297, "stage2_score": 0.79},
    {"lane": "Road Trips",             "asset_id": 2716435553, "stage2_score": 0.74},
    # confirmed road trip images — ground-truth positives for lane scoring validation
    {"lane": "Road Trips",             "asset_id": 2477984575, "stage2_score": 0.90},
    {"lane": "Road Trips",             "asset_id": 2614981181, "stage2_score": 0.90},
    {"lane": "Road Trips",             "asset_id": 2503767067, "stage2_score": 0.90},
    {"lane": "Beach Sunning",          "asset_id": 2337360303, "stage2_score": 0.93},
    {"lane": "Beach Sunning",          "asset_id": 2440563263, "stage2_score": 0.77},
    {"lane": "City Walks",             "asset_id": 1344272849, "stage2_score": 0.86},
    {"lane": "City Walks",             "asset_id": 2285423829, "stage2_score": 0.80},
]


def build_thumbnail_url(asset_id: int) -> str:
    """Standard Shutterstock 260nw CDN thumbnail pattern."""
    return f"https://image.shutterstock.com/image-photo/words-words-words-words-260nw-{asset_id}.jpg"


def build_mock_refined_pool() -> list[dict]:
    return [
        {
            "asset_id": row["asset_id"],
            "origin_lane_name": row["lane"],
            "thumbnail_url": build_thumbnail_url(row["asset_id"]),
            "stage2_score": row["stage2_score"],
            "media_type": "photo",
            "title": f"Mock asset {row['asset_id']}",
        }
        for row in LANE_ASSIGNMENTS
    ]


def validate_thumbnail_urls(refined_pool: list[dict]) -> None:
    """
    HEAD-check every thumbnail URL before sending to the vision model.
    Exits with a non-zero status code if any URL is unreachable or returns
    a non-200 response, so failures are caught immediately rather than
    surfacing as cryptic LLM errors.
    """
    print("Validating thumbnail URLs ...")
    errors: list[str] = []
    for candidate in refined_pool:
        url = candidate["thumbnail_url"]
        asset_id = candidate["asset_id"]
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=10) as resp:
                status = resp.status
        except urllib.error.HTTPError as e:
            status = e.code
        except Exception as e:
            errors.append(f"  asset_id={asset_id}: request failed — {e}")
            continue

        if status != 200:
            errors.append(f"  asset_id={asset_id}: HTTP {status} — {url}")
        else:
            print(f"  OK  {asset_id}")

    if errors:
        print("\nERROR: the following thumbnail URLs could not be resolved:")
        for msg in errors:
            print(msg)
        sys.exit(1)
    print(f"All {len(refined_pool)} thumbnail URLs resolved.\n")


def load_search_params() -> dict:
    json_path = os.path.join(
        os.path.dirname(__file__),
        "app", "services", "searchbybrief", "example_output_stage0.json",
    )
    with open(json_path) as f:
        raw = json.load(f)
    # The example file nests the actual IntentResult under "output"
    return raw["output"]


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def main():
    key = os.environ.get("BIFROST_API_KEY")
    if not key:
        print("ERROR: set BIFROST_API_KEY before running this script.")
        sys.exit(1)

    os.environ.setdefault("BIFROST_API_KEY", key)

    from app.services.searchbybrief.curator import (
        audit_lanes_node,
        score_candidates_node,
        shortlist_candidates_node,
    )

    search_params = load_search_params()
    refined_pool = build_mock_refined_pool()

    print(f"Loaded search_params with {len(search_params['search_lanes'])} lanes.")
    print(f"Mock refined_pool: {len(refined_pool)} candidates across "
          f"{len({r['origin_lane_name'] for r in refined_pool})} lanes.\n")

    validate_thumbnail_urls(refined_pool)

    # Build initial state (mirrors AgentState shape expected by nodes)
    state: dict = {
        "user_request": "Travel clips and stills — UGC, first-person, bright and sunny.",
        "search_params": search_params,
        "refined_pool": refined_pool,
        "candidate_pool": [],
        "stage3_candidates": [],
        "stage3_shortlist": [],
        "stage3_lane_audits": [],
        "final_collection": [],
        "feedback": "",
        "iterations": 1,
    }

    # ------------------------------------------------------------------
    # Node 1: Visual scoring
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Running score_candidates_node ...")
    print("=" * 60)
    state = score_candidates_node(state)

    candidates = state["stage3_candidates"]
    print(f"\n→ {len(candidates)} candidates returned.")

    scored = [c for c in candidates if c.get("stage3_score") is not None]
    skipped = [c for c in candidates if c.get("visual_audit_error")]
    print(f"  Scored:  {len(scored)}")
    print(f"  Skipped (no thumbnail / error): {len(skipped)}")

    if skipped:
        print("\n  Skipped candidates:")
        for c in skipped:
            print(f"    asset_id={c['asset_id']}  error={c.get('visual_audit_error')}")

    latencies = [c["visual_scoring_latency_s"] for c in scored if "visual_scoring_latency_s" in c]
    if latencies:
        print(f"  Total scoring time: {sum(latencies):.1f}s  "
              f"avg={sum(latencies)/len(latencies):.1f}s  "
              f"min={min(latencies):.1f}s  max={max(latencies):.1f}s")

    print("\n  Per-candidate stage3_score:")
    for c in sorted(scored, key=lambda x: x.get("stage3_score", 0), reverse=True):
        print(
            f"    asset_id={c['asset_id']:>12}  lane={c.get('origin_lane_name', '?'):<30}"
            f"  stage2={c.get('stage2_score', 0):.2f}"
            f"  visual_fit={c.get('best_lane_score', 0):.2f}"
            f"  stage3={c.get('stage3_score', 0):.2f}"
            f"  latency={c.get('visual_scoring_latency_s', '?')}s"
        )

    # ------------------------------------------------------------------
    # Node 2: Shortlist construction
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Running shortlist_candidates_node ...")
    print("=" * 60)
    state = shortlist_candidates_node(state)

    shortlist = state["stage3_shortlist"]
    print(f"\n→ Shortlist: {len(shortlist)} candidates (cap: 100 total, 20 per lane).")
    for item in shortlist:
        print(
            f"    asset_id={item['asset_id']:>12}  lane={item.get('origin_lane_name', '?'):<30}"
            f"  stage3={item.get('stage3_score', 0):.2f}"
        )

    # ------------------------------------------------------------------
    # Node 3: Lane audit
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Running audit_lanes_node ...")
    print("=" * 60)
    state = audit_lanes_node(state)

    audits = state["stage3_lane_audits"]
    feedback = state["feedback"]

    print(f"\n→ {len(audits)} lane(s) audited.")
    for audit in audits:
        lane = audit.get("lane_name", "?")
        if "audit_error" in audit:
            print(f"  [{lane}] ERROR: {audit['audit_error']}")
            continue
        result = audit.get("audit_result", {})
        print(
            f"  [{lane}]"
            f"  coverage={result.get('lane_coverage_quality', '?')}"
            f"  redundancy={result.get('duplicate_or_redundancy_risk', '?')}"
            f"  repair_needed={result.get('repair_needed', '?')}"
        )
        if result.get("missing_attributes"):
            print(f"    missing: {result['missing_attributes']}")

    print(f"\n→ feedback = {repr(feedback)}")

    if feedback == "done":
        final = state.get("final_collection", [])
        print(f"✓ Pipeline complete. final_collection has {len(final)} images.")
    else:
        print("↻ Repair requested — planner would receive this feedback on the next iteration.")

    # ------------------------------------------------------------------
    # Dump full stage3_candidates to JSON for inspection
    # ------------------------------------------------------------------
    out_path = os.path.join(os.path.dirname(__file__), "test_stage3_output.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                "stage3_candidates": state.get("stage3_candidates", []),
                "stage3_shortlist": state.get("stage3_shortlist", []),
                "stage3_lane_audits": state.get("stage3_lane_audits", []),
                "feedback": feedback,
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\nFull output written to: test_stage3_output.json")


if __name__ == "__main__":
    main()
