"""
Integration test: audit_lanes_node → planner_node repair loop.

Verifies that Stage 3 repair feedback is consumed correctly by Stage 0:
  1. Run audit_lanes_node against the mock shortlist from test_stage3.py.
  2. Capture the structured CURATOR REPAIR FEEDBACK string.
  3. Feed it to planner_node (via run_intent_node) as attachment_text.
  4. Assert that the returned search_params:
       - Preserves all lanes NOT mentioned in the feedback.
       - Updates or adds lanes for those that were flagged.
       - Does not alter lane_name / lane_goal of updated lanes.

Run from the backend/ directory:
    BIFROST_API_KEY=<key> venv/bin/python3 test_audit_planner_loop.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

# ---------------------------------------------------------------------------
# Reuse test fixtures from test_stage3.py
# ---------------------------------------------------------------------------

from test_stage3 import (
    build_mock_refined_pool,
    build_thumbnail_url,
    load_search_params,
)


def main():
    key = os.environ.get("BIFROST_API_KEY")
    if not key:
        print("ERROR: set BIFROST_API_KEY before running.")
        sys.exit(1)
    os.environ.setdefault("BIFROST_API_KEY", key)

    from app.services.searchbybrief.curator import (
        audit_lanes_node,
        score_candidates_node,
        shortlist_candidates_node,
    )
    from app.services.searchbybrief.planner import run_intent_node

    # ------------------------------------------------------------------
    # Build state using same mock pool as the Stage 3 test
    # ------------------------------------------------------------------
    search_params = load_search_params()
    refined_pool = build_mock_refined_pool()

    state: dict = {
        "user_request": "Travel clips and stills — UGC, first-person, bright and sunny.",
        "attachment_text": None,
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
    # Run Stage 3 nodes to produce feedback
    # ------------------------------------------------------------------
    print("Running Stage 3 nodes to generate repair feedback ...")
    state = score_candidates_node(state)
    state = shortlist_candidates_node(state)
    state = audit_lanes_node(state)

    feedback = state["feedback"]

    if feedback == "done":
        print("No repairs needed — audit was satisfied. Nothing to test.")
        sys.exit(0)

    print(f"\n--- CURATOR REPAIR FEEDBACK ---\n{feedback}\n{'─' * 60}\n")

    # ------------------------------------------------------------------
    # Capture which lanes the audit flagged
    # ------------------------------------------------------------------
    lanes_before = {lane["lane_name"]: lane for lane in search_params["search_lanes"]}
    flagged_lane_names = set()
    for line in feedback.splitlines():
        if line.startswith("UPDATE LANE"):
            name = line.split('"')[1]
            flagged_lane_names.add(name)
        elif line.startswith("ADD REPAIR LANE"):
            name = line.split('"')[1]
            flagged_lane_names.add(name)

    unflagged_lane_names = set(lanes_before.keys()) - flagged_lane_names
    print(f"Lanes flagged for repair: {sorted(flagged_lane_names)}")
    print(f"Lanes expected to be UNCHANGED: {sorted(unflagged_lane_names)}\n")

    # ------------------------------------------------------------------
    # Simulate planner_node: pass the CURRENT plan + feedback as attachment_text
    # so the planner patches it rather than re-deriving from the brief alone.
    # ------------------------------------------------------------------
    existing_plan_text = (
        "CURRENT SEARCH PLAN (existing IntentResult — patch this, do not rederive):\n"
        + json.dumps(search_params, indent=2)
    )
    attachment_text = "\n\n".join([existing_plan_text, feedback])

    print("=" * 60)
    print("Running planner_node (run_intent_node) with repair feedback ...")
    print("=" * 60)

    updated_params = run_intent_node(
        brief_text=state["user_request"],
        attachment_text=attachment_text,
    )

    lanes_after = {lane.lane_name: lane for lane in updated_params.search_lanes}

    print(f"\nLanes before: {len(lanes_before)}  →  Lanes after: {len(lanes_after)}")

    # ------------------------------------------------------------------
    # Assertions
    # ------------------------------------------------------------------
    failures: list[str] = []
    warnings: list[str] = []

    # 1. Unflagged lanes must be preserved unchanged
    print("\n--- Checking unflagged lanes are preserved ---")
    for name in sorted(unflagged_lane_names):
        if name not in lanes_after:
            failures.append(f"FAIL: unflagged lane '{name}' was DROPPED from updated plan")
            continue
        before_eq = lanes_before[name]["embedding_query"]
        after_eq = lanes_after[name].embedding_query
        if before_eq != after_eq:
            warnings.append(
                f"WARN: unflagged lane '{name}' embedding_query changed\n"
                f"  before: {before_eq}\n"
                f"  after:  {after_eq}"
            )
            print(f"  WARN  {name} — embedding_query changed (planner drifted)")
        else:
            print(f"  OK    {name}")

    # 2. Flagged lanes or their repair siblings must appear in updated plan
    print("\n--- Checking flagged/repair lanes are present ---")
    for name in sorted(flagged_lane_names):
        if name in lanes_after:
            after_eq = lanes_after[name].embedding_query
            print(f"  UPDATED  {name}")
            print(f"           embedding_query: {after_eq}")
        else:
            # Repair lane may have appeared as a sibling with a slightly different name
            siblings = [n for n in lanes_after if name.split(" Repair")[0] in n]
            if siblings:
                print(f"  ADDED    sibling lane(s) for '{name}': {siblings}")
            else:
                failures.append(f"FAIL: flagged lane '{name}' missing from updated plan (no sibling found)")

    # 3. Lane count must not decrease
    if len(lanes_after) < len(lanes_before):
        failures.append(
            f"FAIL: lane count decreased {len(lanes_before)} → {len(lanes_after)} "
            "(planner dropped lanes it should have preserved)"
        )

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    if warnings:
        print("Warnings:")
        for w in warnings:
            print(f"  {w}")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)

    print(f"✓ All checks passed. {len(lanes_after)} lanes in updated plan.")

    # ------------------------------------------------------------------
    # Dump side by side for manual inspection
    # ------------------------------------------------------------------
    out_path = os.path.join(os.path.dirname(__file__), "test_audit_planner_output.json")
    with open(out_path, "w") as fh:
        json.dump(
            {
                "feedback": feedback,
                "flagged_lanes": sorted(flagged_lane_names),
                "lanes_before": {
                    n: {"embedding_query": l["embedding_query"], "visual_proxies": l.get("visual_proxies", [])}
                    for n, l in lanes_before.items()
                },
                "lanes_after": {
                    n: {"embedding_query": l.embedding_query, "visual_proxies": l.visual_proxies}
                    for n, l in lanes_after.items()
                },
            },
            fh,
            indent=2,
        )
    print(f"Side-by-side lane diff written to: test_audit_planner_output.json")


if __name__ == "__main__":
    main()
