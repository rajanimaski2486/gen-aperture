# NVIDIA LLM and Conversation OpenSearch Guardrails

## Goal

Use server-side `NVIDIA_API_KEY` for LLM calls, store conversations in `gen-aperture-conversations`, and keep production OpenSearch writes constrained to that single conversation index.

## Non-goals

- Do not add or upgrade dependencies.
- Do not read or commit `backend/.env` secret values.
- Do not allow writes to the photo/search index.
- Do not implement index rollover or archival beyond rejecting writes at configured limits.

## Constraints

- `OPENSEARCH_CONVERSATION_ENDPOINT` should behave as `OPENSEARCH_ENDPOINT` when unset or unchanged from the default.
- Only `gen-aperture-conversations` may be created by the writable guardrail path.
- Conversation writes must be rejected when the index has at least 5000 records or is at least 5 GB.
- Existing read-only guardrails must continue to block photo index mutation attempts.
- NVIDIA NIM is OpenAI-compatible, so reuse the existing OpenAI SDK/LangChain OpenAI client with an NVIDIA base URL and `NVIDIA_API_KEY`.

## Acceptance Criteria

- New conversations no longer require a user-supplied OpenAI API key.
- AgentSquad, reranker, image analysis, and SearchByBrief LLM calls use `NVIDIA_API_KEY` by default.
- Conversation store creates only `gen-aperture-conversations` and writes only to that index.
- Create/update/delete/index requests to any other index remain blocked when guardrails are enabled.
- Conversation create/update/delete writes are rejected when record count or index store size limits are reached.

## Approach

- Add NVIDIA LLM settings and helper methods in app config.
- Update chat/session flow and frontend copy to stop requiring OpenAI keys.
- Extend OpenSearch guardrails with a conversation-write mode that allow-lists the conversation index and blocks all other mutations.
- Add conversation store limit checks before writes and index creation.
- Add focused smoke tests/scripts using mocked clients instead of requiring live OpenSearch or NVIDIA credentials.

## Files / Areas Affected

- `backend/app/config.py`
- `backend/app/services/agent_squad.py`
- `backend/app/services/reranker.py`
- `backend/app/services/image_analyzer.py`
- `backend/app/services/searchbybrief/llm.py`
- `backend/app/services/opensearch_guardrails.py`
- `backend/app/services/conversation_store.py`
- `backend/app/routers/chat.py`
- `backend/app/models/schemas.py`
- `frontend/src/App.jsx`
- `frontend/src/services/api.js`
- documentation files that mention OpenAI keys

## Verification Plan

- Compile changed Python modules.
- Run unit-style smoke checks for guardrail allow/deny behavior.
- Run unit-style smoke checks for conversation write limit rejection using a fake client.
- Run frontend build without dependency install.
- Review final diff for accidental secret exposure and unrelated files.

## Test Plan

- Before/proof: inspect current code paths showing OpenAI key requirement and conversation store writes without guardrail limits.
- Happy path: mocked conversation index under 5000 docs and under 5 GB permits create/update/delete to `gen-aperture-conversations`.
- Sad path: mocked conversation index at 5000 docs or 5 GB rejects writes.
- Sad path: guarded OpenSearch client rejects write attempts to any non-conversation index.
- After/proof: Python compile and frontend build pass.

## Monitoring Plan

- Runtime logs should surface rejected writes with the configured limit reason.
- Health check remains backed by conversation OpenSearch connectivity.

## Risks / Open Questions

- NVIDIA model IDs may need environment-specific overrides; keep defaults configurable.
- Live OpenSearch size/count checks depend on `count` and stats APIs being available.
- This does not migrate any existing conversations from a prior index.

## Status

- Implemented and locally verified.
