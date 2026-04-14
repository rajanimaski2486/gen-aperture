"""
Quick smoke test for Stage 0 (Intent Extraction / planner node).

Run from the backend/ directory:
    BIFROST_API_KEY=<key> python test_stage0.py
"""
import json
import os
import sys

# Ensure the backend package is importable when run from backend/
sys.path.insert(0, os.path.dirname(__file__))

BRIEF = """\
We are looking for various happy Travel Clips & Stills reflecting the following views:

Band Tours (from the POV of the Band)
Concert/Festival goers
Road Trips
Family Vacations

Some airport shots w suitcases
in cars, campers, minivans
on tour buses
trains

Showing cheerful comfy stays indoors in hotels/motels
in the rooms (watching tv, working, relaxing, chatting etc)
working in lobbies/public areas, also in rooms
Breakfast/Lunch/Dinner at the hotel/motel
poolside and pool activities
Gym

some exciting outdoor travel activity clips as well:
beach (sunning, sailing, paragliding. water skiing)
forests/country side
cities (walks scooter rides, bike rides)

We are looking for bright and sunny shots. Mainly UGC/social influencer POVs first person POVs but also the person they are traveling with.
"""


def main():
    key = os.environ.get("BIFROST_API_KEY")
    if not key:
        print("ERROR: set BIFROST_API_KEY before running this script.")
        sys.exit(1)

    # Set env so Settings() picks it up without a .env file
    os.environ.setdefault("BIFROST_API_KEY", key)

    from app.services.searchbybrief.planner import run_intent_node

    print("Running Stage 0 intent extraction...")
    print(f"Brief: {BRIEF[:80].strip()}...\n")

    result = run_intent_node(brief_text=BRIEF)

    print("=== Result ===")
    print(result.model_dump_json(indent=2))

    # Pydantic validation already ran; these are belt-and-braces checks
    assert len(result.search_lanes) >= 1, "Expected at least one search lane"
    for lane in result.search_lanes:
        assert lane.embedding_query, f"Lane missing embedding_query: {lane.lane_name}"
        assert lane.visual_proxies, f"Lane missing visual_proxies: {lane.lane_name}"

    print(f"\n✓ {len(result.search_lanes)} search lane(s) returned — all assertions passed.")


if __name__ == "__main__":
    main()
