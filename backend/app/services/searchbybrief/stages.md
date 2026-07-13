# Search by Brief: Pipeline Stages

---

## Stage 0: Intent Extraction ✓ IMPLEMENTED

**Model:** GPT-4.1 via Bifrost (http://bifrost.localhost/openai)
**Input:** request brief (+ optional attachment text from uploaded PDF/DOCX, or curator feedback on loop iterations)

**Plan:**
- action: identify hard constraints ("low depth of field", "female", etc.)
- output: multiple semantic searches, one per distinct visual subject/activity/scene: ["portrait of a woman", "low depth of field", etc.]
- output: optional metadata constraints: "has model release", etc.

**Implementation notes:**
- Output is a validated IntentResult Pydantic model stored in state["search_params"]
- Each search is a lane with an embedding_query (written as an image caption, not a business description), visual_proxies, literal_terms_preserved, lane_filters, and ranking_hints
- Implied style/POV (e.g. "coloring book" -> line art; "UGC" -> style:ugc) is inferred and added to shared_filters
- Key files: planner.py, llm.py, schemas.py

### Stage 0 -> Stage 1 Handover

Stage 1 receives state["search_params"] as an IntentResult. Run one vector search per lane:

```python
for lane in search_params.search_lanes:
    results = vector_search(embed(lane.embedding_query), top_k=500)
```

Merge per-lane results and deduplicate by image ID -> state["candidate_pool"].

**Filters:**

| Source | Apply as |
|---|---|
| search_params.shared_filters | Pre-filter on every lane (e.g. media_type:illustration, style:ugc) |
| lane.lane_filters | Pre-filter on that lane only |
| hard_constraints.exclusions | Post-filter: remove matching results |
| hard_constraints.licensing_required | Post-filter or pre-filter depending on index support |

**Boosting:** lane.literal_terms_preserved has specific subject names from the brief. Use for lexical boosting alongside vector score if the index supports it.

**Pass through to Stage 2:** lane.ranking_hints and lane.visual_proxies -- not used at Stage 1.

**Notes:**
- Expect many lanes for detailed briefs (coloring book brief produced 28 lanes). Run concurrently.
- brief_diagnostics.search_complexity can inform top_k per lane (very_high -> retrieve more candidates).
- attachment_text on AgentState is for the planner only -- Stage 1 does not need it.

---

## Stage 1: Multi-modeal Recall (generate candidates)

**Model:** Qwen3-VL-Embedding-8B (dual-tower) + vector DB
**Target:** collect several thousand candidate images (recall over precision)

**Note:**
If time is tight, this can also be swapped with whatever vector embedding we already have...

---

## Stage 2: Precision Reranking (cross-encoder)

**Model:** Qwen3-VL-Reranker-8B
**Target:** Top 250-500 (or more?) candidates

**Notes:**
Unlike the embedding model, the reranker will process the query and the image pixels simultaneously through attention layers.
The output from this will be a calibrated relevance score (0.0 to 1.0). Discard everything below 0.7

---

## Stage 3: Agentic Curation

**Model:** Qwen 3.5 VL (Full / 397B-A17B)
**Target:** Final 100 images

**Notes:**
This is the main "thinking loop"
- Diversity check: the agent will ensure that the top 100 images are not copies of the same photo. Automatic pruning if necessary.
- Attribute verification: ensure that the users request has been fulfilled.
- Refinement: If the collection of images is missing a specific attribute (e.g., "none of these are in an outdoor setting"), a "repairer" action will be taken back to Stage-1 with a new specific query.


