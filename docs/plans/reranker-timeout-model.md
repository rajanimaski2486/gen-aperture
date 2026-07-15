# Reranker Model and Timeout

## Goal
Use a faster NVIDIA free-endpoint model for reflection reranking and cap the reranking step at 120 seconds so chat requests do not hang indefinitely.

## Non-goals
- Do not change the main agent model.
- Do not add a new API key or provider.
- Do not add dependencies.
- Do not rewrite chat to use streaming/job polling in this change.

## Constraints
- Reranking uses the same OpenAI-compatible NVIDIA endpoint and `NVIDIA_API_KEY`.
- The timeout must apply to the full reranking pipeline, not only one LLM pass.
- On timeout, preserve original search results and complete the chat response.

## Acceptance criteria
- Default reranker model is `meta/llama-3.2-3b-instruct`.
- Default reranker timeout is 120 seconds and can be overridden with `RERANK_TIMEOUT_SECONDS`.
- Reranker timeout returns a graceful fallback instead of blocking indefinitely.
- The user sees a short timeout note in the response if reranking is skipped after the cap.

## Approach
- Add `rerank_timeout_seconds` to settings and `RerankerConfig`.
- Wrap the two LLM reranker passes plus final selection in `asyncio.wait_for`.
- Set the reranker `AsyncOpenAI` client timeout and disable reranker-specific retries.
- Update agent orchestration to keep timeout explanation visible when rerank falls back.
- Add focused unit tests for config defaults and timeout fallback.

## Files / areas affected
- `backend/app/config.py`
- `backend/app/services/reranker.py`
- `backend/app/services/agent_squad.py`
- `backend/tests/test_reranker_icc_fields.py`
- `frontend/src/App.jsx`
- `docs/plans/reranker-timeout-model.md`

## Verification plan
- Run focused reranker unit tests.
- Run full backend unit test discovery.
- Compile changed backend modules.
- Build the frontend.
- Restart local backend and confirm health.

## Test plan
- Before: reranker LLM call can hang for many minutes.
- Happy path: normal reranking still runs with the smaller configured model.
- Sad path: a simulated slow reranker pass times out and returns original candidates.
- After/proof: tests verify default config and timeout fallback.

## Monitoring plan
- UI progress shows the reranker is capped at two minutes.
- Backend emits warnings/errors only for exceptional conditions such as timeout.

## Risks / open questions
- The smaller model may reduce ranking quality; if so, `meta/llama-3.1-8b-instruct` is the next fallback.
- Exact per-pass progress still requires a future streaming/job-status API.

## Status
- Implemented and locally verified.
