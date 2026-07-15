# UI Rerank Progress

## Goal
Show useful reranking progress in the chat UI while a rerank-triggered request is in flight, and keep backend/terminal logging quiet at warning level.

## Non-goals
- Do not add a streaming/SSE protocol in this change.
- Do not change reranker scoring, ranking, or OpenSearch query behavior.
- Do not add dependencies.

## Constraints
- Current chat API is a single POST, so the browser cannot receive exact backend pass events before the response returns.
- Keep the diff small and reviewable.
- Preserve the existing rerank response/log panel shown after completion.

## Acceptance criteria
- Rerank-triggered pending assistant messages show changing UI status and elapsed time.
- Backend console output defaults to WARNING.
- Extra backend INFO reranker logging added for local diagnosis is removed.
- Existing reranker tests still pass.

## Approach
- Add a frontend-only rerank progress stage model driven by elapsed time while the request is pending.
- Render a compact progress panel in the existing loading assistant bubble.
- Reset progress when the request completes or fails.
- Set backend logging to WARNING and remove the added reranker INFO logs/timing-only imports.

## Files / areas affected
- `frontend/src/App.jsx`
- `frontend/src/index.css`
- `backend/app/main.py`
- `backend/app/services/agent_squad.py`
- `backend/app/services/reranker.py`
- `docs/plans/ui-rerank-progress.md`

## Verification plan
- Build the frontend to catch React/CSS syntax issues.
- Compile changed backend Python service files.
- Run reranker-focused backend tests and full backend test discovery.
- Confirm local frontend/backend routes respond after reload/restart.

## Test plan
- Before: rerank-triggered request shows one static "Applying reflection reranking..." line.
- Happy path: rerank-triggered pending state advances through staged UI statuses with elapsed time.
- Sad path: failed request clears loading/progress state and still shows the existing error toast.
- After/proof: frontend build and backend tests pass.

## Monitoring plan
- During local testing, observe the chat bubble progress instead of backend INFO logs.
- Backend terminal should show warnings/errors only unless uvicorn itself emits startup lines.

## Risks / open questions
- Progress stages are client-side approximations because the endpoint is not streaming. Exact backend pass status would require a streaming/job-status API.

## Status
- Implemented and locally verified.
