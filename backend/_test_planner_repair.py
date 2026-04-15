"""
Isolated test: feed saved repair feedback directly into run_intent_node and
print the exact before-vs-after query for every lane.
Run from backend/:
    BIFROST_API_KEY=<key> venv/bin/python3 _test_planner_repair.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from app.services.searchbybrief.planner import run_intent_node
from app.services.searchbybrief.curator import _format_repair_feedback

with open("test_full_workflow_output.json") as f:
    d = json.load(f)

feedback = _format_repair_feedback(d["stage3_repair_requests"])
print("--- Feedback being sent to planner ---")
print(feedback)
print("--------------------------------------\n")
snaps = d.get("search_params_snapshots", [])
initial_lanes = {l["lane_name"]: l["embedding_query"] for l in snaps[0]["lanes"]} if snaps else {}
current_plan = {l["lane_name"]: l for l in d["search_lanes"]}

# The search_params_snapshots contain the full lane list from iteration 1.
# We need the other top-level IntentResult fields — borrow them from the snapshot
# since the JSON output doesn't persist them. Use safe defaults that won't mislead.
full_intent = {
    "brief_diagnostics": {
        "brief_form": "keyword_list",
        "retrieval_intent": "catalog_population",
        "search_complexity": "high",
        "is_multi_lane": True,
        "reasoning_summary": "Multi-lane travel brief covering UGC/influencer POV content across band tours, concerts, road trips, family vacations, hotel stays, outdoor activities, and city travel.",
    },
    "hard_constraints": {
        "subjects_required": ["travel", "ugc", "first-person pov", "bright", "sunny"],
        "demographics_required": [],
        "composition_required": [],
        "style_required": ["ugc", "first_person_pov", "bright", "sunny"],
        "location_required": [],
        "exclusions": [],
    },
    "operational_constraints": {},
    "shared_filters": [],
    "search_lanes": list(current_plan.values()),
}

# Build the attachment the planner would receive on a repair iteration
existing_plan_text = (
    "CURRENT SEARCH PLAN (existing IntentResult — patch this, do not rederive):\n"
    + json.dumps(full_intent, indent=2)
)
attachment_text = "\n\n".join([existing_plan_text, feedback])
brief_text = "[Original brief suppressed — apply the UPDATE LANE directives from CURATOR REPAIR FEEDBACK to the CURRENT SEARCH PLAN in the attachment. Do not re-derive the plan from scratch.]"

print("Calling planner (repair mode)...")
t0 = time.time()
result = run_intent_node(brief_text=brief_text, attachment_text=attachment_text)
elapsed = time.time() - t0
print(f"Done in {elapsed:.1f}s  ({len(result.search_lanes)} lanes returned)\n")

# Diff
flagged = {r["target_lane_name"]: r["new_lane"]["embedding_query"]
           for r in d["stage3_repair_requests"]}

print("=" * 70)
print("FLAGGED LANES — did the planner apply the requested query?")
print("=" * 70)
result_lanes = {l.lane_name: l.embedding_query for l in result.search_lanes}

for lane_name, requested_q in sorted(flagged.items()):
    orig = initial_lanes.get(lane_name, "(not in snapshot)")
    final = result_lanes.get(lane_name, "(lane missing in output!)")
    applied = final == requested_q
    changed_from_orig = final != orig
    print(f"\nLane: {lane_name}")
    print(f"  ORIGINAL  : {orig}")
    print(f"  REQUESTED : {requested_q}")
    print(f"  PLANNER   : {final}")
    print(f"  → applied={'YES ✓' if applied else 'NO ✗'}   changed_from_original={'YES' if changed_from_orig else 'NO (same as before)'}")

print("\n" + "=" * 70)
print("UNFLAGGED LANES — should be identical to current plan")
print("=" * 70)
for lane_name, lane in sorted(current_plan.items()):
    if lane_name in flagged:
        continue
    orig_q = lane["embedding_query"]
    new_q = result_lanes.get(lane_name, "(missing)")
    stable = orig_q == new_q
    print(f"  {'[stable]  ' if stable else '[CHANGED] '} {lane_name}")
    if not stable:
        print(f"    was: {orig_q}")
        print(f"    now: {new_q}")
