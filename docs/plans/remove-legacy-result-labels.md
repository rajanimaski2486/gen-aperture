# Remove Legacy Result Labels

## Goal
Remove stale previous-implementation labels from the visible search results area.

## Non-goals
- Do not change search ranking, filtering, or OpenSearch query behavior.
- Do not remove response fields that may still be useful for debugging or API compatibility.
- Do not change reranker logic.

## Constraints
- Keep the results section focused on the returned assets.
- Preserve rerank badges/logs and generation timing.

## Acceptance criteria
- The results header no longer shows `Popular` or `Relevant` badges.
- The filter metadata banner no longer shows category labels such as `Nature, Objects`.
- The filter metadata banner no longer shows old filters such as `total_paid_license_count_all_time gte 1`.
- The assistant text no longer says `Search Results (popularity)` or `Search Results (relevance)`.

## Approach
- Remove the `search-mode-badge` rendering from the frontend results header.
- Remove the `filter-metadata-banner` rendering from the frontend results area.
- Simplify backend synthesizer copy to `Search Results`.

## Files / areas affected
- `frontend/src/App.jsx`
- `backend/app/services/agent_squad.py`
- `docs/plans/remove-legacy-result-labels.md`

## Verification plan
- Build the frontend.
- Compile changed backend Python files.
- Run backend unit tests.
- Run diff whitespace checks.

## Test plan
- Before: results area can show `Popular`, category labels, and old license-count filter text.
- Happy path: search results still render, but the stale labels are absent.
- Proof: build and tests pass; diff removes the render branches and heading suffix.

## Monitoring plan
- Retest locally in the browser with a query that previously displayed those labels.

## Risks / open questions
- None expected; this is display-only cleanup.

## Status
- Implemented and locally verified.
