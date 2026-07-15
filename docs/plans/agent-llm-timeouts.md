# Agent LLM Timeout Cleanup

## Goal
Prevent non-reranker NVIDIA calls in the Agent Squad path from making a chat request run for many minutes.

## Non-goals
- Do not change OpenSearch retrieval behavior.
- Do not lower the 120-second reranker timeout requested by the user.
- Do not rewrite chat to use streaming/job polling in this change.
- Do not clean up SearchByBrief debug output.

## Constraints
- Use the existing `NVIDIA_API_KEY` and NVIDIA OpenAI-compatible endpoint.
- Keep fallback behavior graceful: if an LLM intent call times out, use deterministic/fallback query handling and still search.
- Keep backend terminal noise at warning/error level.

## Acceptance criteria
- Text-only router no longer spends an LLM call just to classify popular/relevance.
- Agent Squad ChatOpenAI calls have a bounded timeout and no automatic retry stretch.
- The known text-only debug `print` statements are removed.
- Existing backend tests and frontend build pass.

## Approach
- Add `agent_llm_timeout_seconds` and `agent_llm_max_retries` settings.
- Pass those settings into Agent Squad `ChatOpenAI`.
- Replace router LLM classification with deterministic search-mode detection.
- Fix the old bug where router ignored the LLM output and always set `popular`.
- Remove text-only debug prints that bypass logging configuration.

## Files / areas affected
- `backend/app/config.py`
- `backend/app/services/agent_squad.py`
- `docs/plans/agent-llm-timeouts.md`

## Verification plan
- Unit test deterministic search-mode detection.
- Compile changed backend modules.
- Run full backend tests.
- Build frontend to guard against accumulated UI changes.
- Confirm local backend health after restart.

## Test plan
- Before: latest saved request took 860.733s while reranker timed out at 120s.
- Happy path: router classification is instantaneous and simple queries still route to search.
- Sad path: slow query intent LLM times out and falls back through existing fallback logic.
- Proof: tests and local health checks pass.

## Monitoring plan
- Inspect future saved `processing_time_ms`; a rerank timeout request should no longer drift toward 14 minutes due to uncapped pre-rerank LLM calls.

## Risks / open questions
- Brief uploads still depend on the main agent LLM and may need separate tuning if slow.

## Status
- Implemented and locally verified.
