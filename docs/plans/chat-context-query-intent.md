# Chat Context in Query Intent Analysis

## Goal
Ensure text-only follow-up messages in an existing chat window use previous conversation context during the Query Intent Analysis step, so related/refinement queries are generated from the latest message plus relevant prior user context.

## Non-goals
- Do not change SearchByBrief behavior.
- Do not add or upgrade dependencies.
- Do not change OpenSearch conversation write guardrails.
- Do not enable extra LLM intent calls by default beyond the existing follow-up resolver path.

## Constraints
- `TEXT_QUERY_INTENT_LLM_ENABLED` defaults to `false`, so fast local intent extraction must remain the default for direct standalone searches.
- The frontend already sends `conversation_id` for the active chat window; backend behavior should own context loading and intent analysis.
- Existing uncommitted changes in unrelated search/deploy files must not be reverted.

## Acceptance criteria
- For text-only Agent Squad searches with prior conversation history, Query Intent Analysis receives a context-aware query derived from the current message and previous user messages.
- The workflow step exposes enough input metadata to verify that prior chat context was considered.
- If follow-up resolution fails, intent analysis still receives a deterministic contextual fallback instead of only the latest short follow-up.
- Standalone/new chat searches keep the current fast local intent behavior.
- Focused tests cover direct fast path, contextual prompt/input construction, and resolver failure fallback.

## Approach
- Add context helpers in `query_intent.py` for compact prior-user-message formatting and deterministic contextual fallback construction.
- Extend `detect_text_query_intent` to accept optional conversation history and an already resolved query while preserving the default fast path.
- Update `AgentSquad._search_text_only` to pass conversation history into Query Intent Analysis and record context metadata in workflow step input.
- Add unit tests around context-aware intent inputs without making real network calls.

## Files / areas affected
- `backend/app/services/query_intent.py`
- `backend/app/services/agent_squad.py`
- `backend/tests/test_agent_squad_latency.py`
- `docs/plans/chat-context-query-intent.md`

## Verification plan
- Run the focused backend unit tests that cover query intent and Agent Squad latency behavior.
- Compile changed backend Python files.
- Review `git diff` to confirm only intended files changed and unrelated dirty files were not modified.

## Test plan
- Before/proof: current Query Intent Analysis step input contains only `raw_query` and `query_source`, so reviewers cannot verify prior chat context was considered.
- Happy path: prior user query plus latest refinement produces a context-aware query passed into intent analysis, and workflow input reports prior context.
- Sad path: resolver/LLM failure falls back to a deterministic combination of the prior user query and latest message.
- After/proof: focused tests assert the exact prompt/input behavior and fallback query.

## Monitoring plan
- Use workflow step input fields (`user_query`, `intent_query`, `query_source`, and `prior_user_context`) to inspect context usage in the existing UI workflow panel.

## Risks / open questions
- A deterministic fallback can only approximate follow-up semantics, but it is safer than dropping prior context entirely when resolver LLM calls fail.
- Very long histories should stay compact; use only recent user messages for intent context.

## Status
- Implemented and locally verified with focused context tests, backend test discovery, and compile checks.
