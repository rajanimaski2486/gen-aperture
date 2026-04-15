import json

with open("test_full_workflow_output.json") as f:
    d = json.load(f)

snaps = d.get("search_params_snapshots", [])
initial_lanes = {l["lane_name"]: l for l in snaps[0]["lanes"]} if snaps else {}
final_lanes = {l["lane_name"]: l for l in d.get("search_lanes", [])}
repair_requests = {r["target_lane_name"]: r for r in d.get("stage3_repair_requests", [])}

flagged = set(repair_requests.keys())

print("=== Flagged lanes: initial → repair-requested → final ===\n")
for name in sorted(flagged):
    orig = initial_lanes.get(name, {}).get("embedding_query", "(not found)")
    req_lane = repair_requests[name]["new_lane"]
    requested = req_lane.get("embedding_query", "(not found)")
    final = final_lanes.get(name, {}).get("embedding_query", "(not found)")
    print(f"Lane: {name}")
    print(f"  ORIGINAL : {orig}")
    print(f"  REQUESTED: {requested}  {'(same as original)' if orig == requested else '[DIFFERENT from original]'}")
    print(f"  FINAL    : {final}  {'✓ planner applied the request' if requested == final else '[planner diverged from request]'  if final != orig else '(unchanged from original)'}")
    print()

print("=== Unflagged lanes (should be unchanged) ===\n")
for name, lane in sorted(initial_lanes.items()):
    if name in flagged:
        continue
    orig = lane.get("embedding_query", "")
    final = final_lanes.get(name, {}).get("embedding_query", "")
    if orig != final:
        print(f"  [UNEXPECTEDLY CHANGED] {name}")
        print(f"    BEFORE: {orig}")
        print(f"    AFTER:  {final}")
    else:
        print(f"  [stable] {name}")
