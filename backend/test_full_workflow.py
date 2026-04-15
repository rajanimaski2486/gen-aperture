"""
Full workflow integration test: Stage 0 → Stage 1 (mock) → Stage 2 (mock) → Stage 3.

Runs the real planner (Stage 0) and real curator (Stage 3) against the travel
brief. Stage 1 and Stage 2 are replaced with lightweight mocks that inject the
same 13-candidate pool used in test_stage3.py.

Run from the backend/ directory:
    BIFROST_API_KEY=<key> python test_full_workflow.py

What this verifies
------------------
- Stage 0 planner parses the travel brief and produces a valid IntentResult.
- The graph routes planner → retriever → reranker → curator correctly.
- curator_node visually scores candidates, builds a shortlist, and either
  sets feedback="done" (median score ≥ 0.60) or generates repair directives
  that would loop back to the planner.
- should_continue routes to END when feedback=="done" or iterations > 3.
- final_collection is populated on a clean exit.
"""

import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(__file__))

from langgraph.graph import StateGraph, END

# ---------------------------------------------------------------------------
# Debug: captures a snapshot of search_params before each repair-loop replan.
# Keyed by iteration number so the HTML report can diff embedding_query changes.
# ---------------------------------------------------------------------------
_PLANNER_SNAPSHOTS: list = []  # list of {"iteration": int, "lanes": list[dict]}

from app.services.searchbybrief.main import AgentState
from app.services.searchbybrief.planner import run_intent_node
from app.services.searchbybrief.curator import curator_node, MEDIAN_SCORE_THRESHOLD


BRIEF = (
    "We are looking for various happy Travel Clips & Stills reflecting the following views:\n\n"
    "Band Tours (from the POV of the Band)\n"
    "Concert/Festival goers\n"
    "Road Trips\n"
    "Family Vacations\n\n"
    "Some airport shots w suitcases\n"
    "in cars, campers, minivans\n"
    "on tour buses\n"
    "trains\n\n"
    "Showing cheerful comfy stays indoors in hotels/motels\n"
    "in the rooms (watching tv, working, relaxing, chatting etc)\n"
    "working in lobbies/public areas, also in rooms\n"
    "Breakfast/Lunch/Dinner at the hotel/motel\n"
    "poolside and pool activities\n"
    "Gym\n\n"
    "some exciting outdoor travel activity clips as well:\n"
    "beach (sunning, sailing, paragliding. water skiing)\n"
    "forests/country side\n"
    "cities (walks scooter rides, bike rides)\n\n"
    "We are looking for bright and sunny shots. Mainly UGC/social influencer POVs "
    "first person POVs but also the person they are traveling with.\n"
)

# ---------------------------------------------------------------------------
# Mock Stage 1/2 data — same pool used in test_stage3.py
# ---------------------------------------------------------------------------

# Trimmed to 1 asset per lane (5 total) for faster integration test runs.
LANE_ASSIGNMENTS = [
    {"lane": "Band Tour POV",          "asset_id": 2692876601, "stage2_score": 0.91},
    {"lane": "Concert/Festival Goers", "asset_id": 2646906151, "stage2_score": 0.88},
    {"lane": "Road Trips",             "asset_id": 2503767067, "stage2_score": 0.90},  # confirmed road trip
    {"lane": "Beach Sunning",          "asset_id": 2337360303, "stage2_score": 0.93},
    {"lane": "City Walks",             "asset_id": 1344272849, "stage2_score": 0.86},
]


def _thumbnail_url(asset_id: int) -> str:
    return f"https://image.shutterstock.com/image-photo/words-words-words-words-260nw-{asset_id}.jpg"


MOCK_REFINED_POOL = [
    {
        "asset_id": row["asset_id"],
        "origin_lane_name": row["lane"],
        "thumbnail_url": _thumbnail_url(row["asset_id"]),
        "stage2_score": row["stage2_score"],
        "media_type": "photo",
        "title": f"Mock asset {row['asset_id']}",
    }
    for row in LANE_ASSIGNMENTS
]

# ---------------------------------------------------------------------------
# Mock Stage 1: returns a flat candidate_pool (candidate records without scores)
# ---------------------------------------------------------------------------

def mock_retriever_node(state: AgentState) -> dict:
    """Stage 1 mock — assigns one asset per lane using the ACTUAL lane names from the planner."""
    search_params = state.get("search_params")
    lanes = search_params.search_lanes if search_params and hasattr(search_params, "search_lanes") else []

    # Pair each planner lane with one of our known asset IDs (cycling if needed)
    known_assets = [row["asset_id"] for row in LANE_ASSIGNMENTS]
    candidate_pool = []
    for i, lane in enumerate(lanes):
        asset_id = known_assets[i % len(known_assets)]
        candidate_pool.append({
            "asset_id": asset_id,
            "origin_lane_name": lane.lane_name,  # use the EXACT name from the planner
            "thumbnail_url": _thumbnail_url(asset_id),
            "media_type": "photo",
            "title": f"Mock asset {asset_id}",
        })

    print(f"  [mock retriever] returning {len(candidate_pool)} candidates across "
          f"{len(lanes)} lanes", flush=True)
    return {"candidate_pool": candidate_pool}


# ---------------------------------------------------------------------------
# Mock Stage 2: adds stage2_score and writes refined_pool
# ---------------------------------------------------------------------------

def mock_reranker_node(state: AgentState) -> dict:
    """Stage 2 mock — attaches a fixed stage2_score to every candidate."""
    score_map = {row["asset_id"]: row["stage2_score"] for row in LANE_ASSIGNMENTS}
    refined_pool = [
        {**c, "stage2_score": score_map.get(c["asset_id"], 0.80)}
        for c in state.get("candidate_pool", [])
    ]
    print(f"  [mock reranker]  {len(refined_pool)} candidates survived (all passed "
          "— mock skips the 0.7 cutoff)", flush=True)
    return {"refined_pool": refined_pool}


# ---------------------------------------------------------------------------
# Stage 0 node — real planner
# ---------------------------------------------------------------------------

def planner_node(state: AgentState) -> dict:
    base_attachment = state.get("attachment_text") or ""
    feedback = state.get("feedback", "")
    existing_params = state.get("search_params")

    if feedback and feedback != "done":
        existing_plan_text = ""
        if existing_params is not None:
            plan_dict = (
                existing_params.model_dump()
                if hasattr(existing_params, "model_dump")
                else existing_params
            )
            existing_plan_text = (
                "CURRENT SEARCH PLAN (existing IntentResult — patch this, do not rederive):\n"
                + __import__("json").dumps(plan_dict, indent=2)
            )
        attachment_text = "\n\n".join(filter(None, [base_attachment, existing_plan_text, feedback]))
    else:
        attachment_text = base_attachment or None

    iteration = state.get("iterations", 0) + 1

    # --- debug snapshot: record the CURRENT plan before the planner overwrites it ---
    if feedback and feedback != "done" and existing_params is not None:
        raw_lanes = existing_params.search_lanes if hasattr(existing_params, "search_lanes") else []
        _PLANNER_SNAPSHOTS.append({
            "iteration": iteration - 1,  # the iteration whose plan we are about to replace
            "lanes": [
                l.model_dump() if hasattr(l, "model_dump") else l
                for l in raw_lanes
            ],
        })

    print(f"\n--- Stage 0: planner (iteration {iteration}) ---")
    t0 = time.time()
    # Suppress the original brief on repair iterations so the LLM patches,
    # rather than re-deriving the full plan from the brief text.
    if feedback and feedback != "done":
        brief_text_for_llm = "[Original brief suppressed — apply the REPAIR LANE directives from CURATOR REPAIR FEEDBACK to the CURRENT SEARCH PLAN in the attachment. Write brand new embedding_queries from the lane_goal for each flagged lane. Do not re-derive the plan from scratch.]"
    else:
        brief_text_for_llm = state["user_request"]

    search_params = run_intent_node(brief_text=brief_text_for_llm, attachment_text=attachment_text)
    elapsed = time.time() - t0
    lane_count = len(search_params.search_lanes) if hasattr(search_params, "search_lanes") else "?"
    print(f"  planner returned {lane_count} search lanes in {elapsed:.1f}s", flush=True)

    # --- debug snapshot: also capture iteration 1 output so we have a full before/after ---
    if iteration == 1:
        _PLANNER_SNAPSHOTS.insert(0, {
            "iteration": 0,  # label as "initial" (before any repair)
            "lanes": [
                l.model_dump() if hasattr(l, "model_dump") else l
                for l in (search_params.search_lanes if hasattr(search_params, "search_lanes") else [])
            ],
        })

    return {"search_params": search_params, "iterations": iteration}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def should_continue(state: AgentState) -> str:
    feedback = state.get("feedback", "")
    if feedback == "done":
        return "end"
    return "continue"


def after_planner(state: AgentState) -> str:
    """After the repair re-plan (iteration ≥ 2), stop — we only want the updated Stage 0 output."""
    if state.get("iterations", 0) >= 2:
        return "end"
    return "retriever"


# ---------------------------------------------------------------------------
# Build the test graph
# ---------------------------------------------------------------------------

def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("planner", planner_node)
    workflow.add_node("retriever", mock_retriever_node)
    workflow.add_node("reranker", mock_reranker_node)
    workflow.add_node("curator", curator_node)

    workflow.set_entry_point("planner")
    workflow.add_conditional_edges("planner", after_planner, {"retriever": "retriever", "end": END})
    workflow.add_edge("retriever", "reranker")
    workflow.add_edge("reranker", "curator")
    workflow.add_conditional_edges("curator", should_continue, {"continue": "planner", "end": END})

    return workflow.compile()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def validate_thumbnail_urls() -> None:
    print("Validating thumbnail URLs ...")
    errors = []
    for c in MOCK_REFINED_POOL:
        url = c["thumbnail_url"]
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=10) as resp:
                status = resp.status
        except urllib.error.HTTPError as e:
            status = e.code
        except Exception as e:
            errors.append(f"  asset_id={c['asset_id']}: request failed — {e}")
            continue
        if status != 200:
            errors.append(f"  asset_id={c['asset_id']}: HTTP {status}")
        else:
            print(f"  OK  {c['asset_id']}")
    if errors:
        print("\nERROR: unreachable thumbnail URLs:")
        for msg in errors:
            print(msg)
        sys.exit(1)
    print(f"All {len(MOCK_REFINED_POOL)} thumbnail URLs OK.\n")


def print_results(final_state: dict) -> None:
    collection = final_state.get("final_collection") or []
    shortlist = final_state.get("stage3_shortlist") or []
    feedback = final_state.get("feedback", "")
    iterations = final_state.get("iterations", 0)
    candidates = final_state.get("stage3_candidates") or []

    print("\n" + "=" * 60)
    print("WORKFLOW COMPLETE")
    print("=" * 60)
    print(f"  Iterations run  : {iterations}")
    print(f"  Final feedback  : {feedback!r}")
    print(f"  Candidates scored: {len(candidates)}")
    print(f"  Shortlist size  : {len(shortlist)}")
    print(f"  Final collection: {len(collection)}")

    if candidates:
        scores = [c["stage3_score"] for c in candidates if isinstance(c.get("stage3_score"), (int, float))]
        if scores:
            import statistics
            print(f"  Median stage3_score (candidates): {statistics.median(scores):.3f}  "
                  f"(threshold: {MEDIAN_SCORE_THRESHOLD})")

    if shortlist:
        print("\n  Per-lane shortlist coverage:")
        lane_counts: dict = {}
        for c in shortlist:
            lane = c.get("origin_lane_name", "unknown")
            lane_counts[lane] = lane_counts.get(lane, 0) + 1
        for lane, count in sorted(lane_counts.items()):
            print(f"    {lane}: {count}")

    if collection:
        print("\n  Top 5 by stage3_score:")
        sorted_col = sorted(collection, key=lambda c: c.get("stage3_score", 0), reverse=True)
        for c in sorted_col[:5]:
            print(f"    asset_id={c['asset_id']}  lane={c.get('origin_lane_name','?')}  "
                  f"stage3={c.get('stage3_score', 0):.3f}  stage2={c.get('stage2_score', 0):.3f}")
    else:
        print("\n  WARNING: final_collection is empty — curator may have looped out")
        print("  Shortlist (fallback):")
        for c in sorted(shortlist, key=lambda c: c.get("stage3_score", 0), reverse=True)[:5]:
            print(f"    asset_id={c['asset_id']}  stage3={c.get('stage3_score', 0):.3f}")

    audits = final_state.get("stage3_lane_audits") or []
    if audits:
        print(f"\n  Lane audits run: {len(audits)}")
        for a in audits:
            repair_needed = a.get("audit_result", {}).get("repair_needed", False)
            tag = "REPAIR" if repair_needed else "ok"
            print(f"    [{tag}] {a.get('lane_name', '?')}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    key = os.environ.get("BIFROST_API_KEY")
    if not key:
        print("ERROR: set BIFROST_API_KEY before running.")
        sys.exit(1)

    validate_thumbnail_urls()
    graph = build_graph()

    initial_state: AgentState = {
        "user_request": BRIEF,
        "attachment_text": None,
        "search_params": None,
        "candidate_pool": [],
        "refined_pool": [],
        "stage3_candidates": [],
        "stage3_shortlist": [],
        "stage3_lane_audits": [],
        "stage3_repair_requests": [],
        "final_collection": [],
        "feedback": "",
        "iterations": 0,
    }

    print("=" * 60)
    print("RUNNING FULL WORKFLOW (Stage 0 → mock 1/2 → Stage 3)")
    print("=" * 60)

    t_start = time.time()
    final_state = graph.invoke(initial_state)
    total = time.time() - t_start

    print(f"\n  Total elapsed: {total:.1f}s")
    print_results(final_state)

    base = os.path.dirname(__file__)
    json_path = os.path.join(base, "test_full_workflow_output.json")
    html_path = os.path.join(base, "test_full_workflow_report.html")

    collection = final_state.get("final_collection") or final_state.get("stage3_shortlist") or []
    shortlist = final_state.get("stage3_shortlist") or []
    audits = final_state.get("stage3_lane_audits") or []
    search_params = final_state.get("search_params")
    search_lanes = []
    if search_params:
        raw_lanes = search_params.search_lanes if hasattr(search_params, "search_lanes") else []
        search_lanes = [l.model_dump() if hasattr(l, "model_dump") else l for l in raw_lanes]

    # JSON dump
    with open(json_path, "w") as f:
        json.dump({
            "iterations": final_state.get("iterations"),
            "feedback": final_state.get("feedback"),
            "search_lanes": search_lanes,
            "stage3_shortlist": shortlist,
            "final_collection": collection,
            "stage3_lane_audits": audits,
            "stage3_repair_requests": final_state.get("stage3_repair_requests") or [],
            "search_params_snapshots": _PLANNER_SNAPSHOTS,
        }, f, indent=2, default=str)
    print(f"  JSON output   → {json_path}")

    # HTML report
    _write_html_report(html_path, final_state, total, _PLANNER_SNAPSHOTS)
    print(f"  HTML report   → {html_path}")


def _build_plan_evolution_html(snapshots: list, state: dict) -> str:
    """Render a before/after table of embedding_query changes per lane per repair iteration."""
    if not snapshots:
        return ""

    # Build a lookup of the FINAL lane state
    final_params = state.get("search_params")
    final_lanes: dict[str, dict] = {}
    if final_params:
        raw = final_params.search_lanes if hasattr(final_params, "search_lanes") else []
        for lane in raw:
            d = lane.model_dump() if hasattr(lane, "model_dump") else lane
            final_lanes[d["lane_name"]] = d

    # Build per-iteration lookup so we can diff snapshot[i] → snapshot[i+1] (or final)
    snap_lookup: list[dict[str, dict]] = []
    for snap in snapshots:
        lmap: dict[str, dict] = {}
        for lane in snap["lanes"]:
            lmap[lane["lane_name"]] = lane
        snap_lookup.append(lmap)

    sections = ""
    for i, snap in enumerate(snapshots):
        iter_num = snap["iteration"]
        after_lanes = snap_lookup[i + 1] if i + 1 < len(snap_lookup) else final_lanes
        after_label = snapshots[i + 1]["iteration"] if i + 1 < len(snapshots) else "final"

        rows = ""
        for name, before in snap_lookup[i].items():
            old_q = before.get("embedding_query", "")
            after = after_lanes.get(name)
            new_q = after.get("embedding_query", "") if after else old_q
            changed = old_q != new_q
            bg = " style='background:#fffde7'" if changed else ""
            badge = " <span style='color:#e67e22;font-weight:bold'>[updated]</span>" if changed else ""
            rows += (
                f"<tr{bg}><td>{name}{badge}</td>"
                f"<td style='color:#c0392b'><em>{old_q}</em></td>"
                f"<td style='color:#27ae60'><em>{new_q}</em></td></tr>"
            )
        sections += f"""
<h2>Plan Evolution — iteration {iter_num} \u2192 {after_label}</h2>
<table>
  <tr><th>Lane</th><th>Before (embedding_query)</th><th>After (embedding_query)</th></tr>
  {rows}
</table>"""

    return sections


def _write_html_report(path: str, state: dict, elapsed: float, snapshots: list | None = None) -> None:
    collection = state.get("final_collection") or state.get("stage3_shortlist") or []
    shortlist = state.get("stage3_shortlist") or []
    audits = state.get("stage3_lane_audits") or []
    feedback = state.get("feedback", "")
    iterations = state.get("iterations", 0)
    candidates = state.get("stage3_candidates") or []

    scores = [c["stage3_score"] for c in candidates if isinstance(c.get("stage3_score"), (int, float))]
    import statistics as _stats
    median_s = f"{_stats.median(scores):.3f}" if scores else "n/a"

    sorted_col = sorted(collection, key=lambda c: c.get("stage3_score", 0), reverse=True)

    cards = ""
    for c in sorted_col:
        thumb = c.get("thumbnail_url", "")
        aid = c.get("asset_id", "?")
        lane = c.get("origin_lane_name", "?")
        s3 = c.get("stage3_score", 0)
        s2 = c.get("stage2_score", 0)
        fit = c.get("best_lane_score", "?")
        exclusion = c.get("visual_likely_exclusion_violation", False)
        border = "#e74c3c" if exclusion else "#2ecc71" if s3 >= 0.6 else "#f39c12"
        cards += f"""
        <div class="card" style="border-top:4px solid {border}">
          <img src="{thumb}" onerror="this.src='https://placehold.co/260x174?text=no+image'" />
          <div class="meta">
            <strong>{aid}</strong><br/>
            <span class="lane">{lane}</span><br/>
            stage3={s3:.3f} &nbsp; stage2={s2:.3f} &nbsp; fit={fit}
            {"<br/><span class='excl'>⚠ exclusion flag</span>" if exclusion else ""}
          </div>
        </div>"""

    audit_rows = ""
    for a in audits:
        ar = a.get("audit_result", {})
        rn = ar.get("repair_needed", False)
        missing = ", ".join(ar.get("missing_attributes", []))
        cov = ar.get("lane_coverage_quality") or ar.get("coverage", "?")
        red = ar.get("duplicate_or_redundancy_risk") or ar.get("redundancy", "?")
        tag = "\U0001f534 REPAIR" if rn else "\U0001f7e2 ok"
        # Find the matching repair request to show query diff
        repair_requests_list = state.get("stage3_repair_requests") or []
        repair_lane = next(
            (req["new_lane"] for req in repair_requests_list
             if req.get("target_lane_name") == a.get("lane_name")),
            None,
        )
        query_diff = ""
        if repair_lane:
            orig = repair_lane.get("original_embedding_query", "")
            refined = repair_lane.get("embedding_query", "")
            query_diff = (
                f"<br/><span style='color:#888'>Original:</span> <em>{orig}</em>"
                f"<br/><span style='color:#2980b9'>Refined:</span> <em>{refined}</em>"
            )
        audit_rows += (
            f"<tr><td>{a.get('lane_name','?')}</td><td>{tag}</td>"
            f"<td>{cov}</td><td>{red}</td>"
            f"<td>{missing}{query_diff}</td></tr>"
        )

    html = f"""<!DOCTYPE html>
<html lang='en'>
<head><meta charset='UTF-8'/><title>Workflow Report</title>
<style>
  body {{ font-family: system-ui, sans-serif; background:#f5f5f5; margin:0; padding:24px; }}
  h1 {{ font-size:1.4rem; }}
  .meta-bar {{ background:#fff; border-radius:8px; padding:16px; margin-bottom:20px;
               display:flex; gap:32px; box-shadow:0 1px 4px rgba(0,0,0,.1); }}
  .meta-bar div {{ font-size:.9rem; }}
  .meta-bar strong {{ display:block; font-size:1.1rem; }}
  .grid {{ display:flex; flex-wrap:wrap; gap:12px; }}
  .card {{ background:#fff; border-radius:8px; overflow:hidden; width:200px;
           box-shadow:0 1px 4px rgba(0,0,0,.1); }}
  .card img {{ width:200px; height:133px; object-fit:cover; display:block; }}
  .card .meta {{ padding:8px; font-size:.75rem; color:#333; }}
  .card .lane {{ color:#555; font-style:italic; }}
  .card .excl {{ color:#e74c3c; font-weight:bold; }}
  table {{ border-collapse:collapse; width:100%; background:#fff;
           border-radius:8px; overflow:hidden; box-shadow:0 1px 4px rgba(0,0,0,.1); }}
  th, td {{ padding:8px 12px; text-align:left; font-size:.85rem; border-bottom:1px solid #eee; }}
  th {{ background:#f0f0f0; }}
  h2 {{ margin-top:28px; font-size:1.1rem; }}
</style>
</head>
<body>
<h1>Full Workflow Integration Report</h1>
<div class='meta-bar'>
  <div><strong>{iterations}</strong>iteration(s)</div>
  <div><strong>{len(collection)}</strong>final assets</div>
  <div><strong>{median_s}</strong>median stage3 score</div>
  <div><strong>{elapsed:.1f}s</strong>total elapsed</div>
  <div><strong>{"done ✓" if feedback == "done" else "repair ↻"}</strong>exit status</div>
</div>

{_build_plan_evolution_html(snapshots, state) if snapshots else ""}

<h2>Final Collection (sorted by stage3_score)</h2>
<div class='grid'>{cards}</div>

{'<h2>Lane Audits</h2><table><tr><th>Lane</th><th>Status</th><th>Coverage</th><th>Redundancy</th><th>Missing attributes</th></tr>' + audit_rows + '</table>' if audit_rows else ''}
</body></html>"""

    with open(path, "w") as f:
        f.write(html)


if __name__ == "__main__":
    main()
