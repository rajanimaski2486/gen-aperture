# NVIDIA Query Embeddings and Backfill

## Goal
Add a configurable NVIDIA embedding path for direct `icc_images_ext` kNN queries and provide a dry-run-first script to backfill a matching NVIDIA vector field into OpenSearch.

## Non-goals
- Do not remove the existing OpenAI `text-embedding-3-small` query path.
- Do not run the production backfill from Codex.
- Do not change the LLM, reranker, SearchByBrief, or conversation-store behavior.
- Do not require new Python dependencies.

## Constraints
- Existing OpenAI 256d vectors are not compatible with NVIDIA embedding models.
- Query-time vectors and backfilled document vectors must use the same model, dimensions, vector field, and input-type policy.
- Backfill writes to the photo index are risky and must be explicit; script defaults must not mutate OpenSearch.
- Keep existing OpenSearch read guardrails for app search behavior intact.

## Acceptance Criteria
- Runtime query embedding generation can send NVIDIA-compatible `/v1/embeddings` requests with `input_type=query`.
- Query embedding client timeout is capped at 60 seconds even if env config is higher.
- OpenAI embedding behavior remains backward compatible by default.
- The kNN clause targets the configured vector field, so a new NVIDIA vector field can be used without code edits.
- Agent workflow trace records the embedding provider, embedding model, dimensions, input type, vector field, and timeout used by the direct hybrid query.
- A backfill script can scroll source docs, generate NVIDIA passage embeddings, optionally add a vector mapping, and bulk update records.
- Unit tests cover the OpenAI request, NVIDIA request, dimension validation, and script text/mapping helpers.
- README documents the NVIDIA env settings and safe backfill invocation.

## Approach
- Add provider-aware embedding helpers in `PhotoSearchService`.
- Add settings for embedding provider, optional generic API key/base URL, `input_type`, `truncate`, and whether to send the dimensions parameter.
- Add query embedding metadata to `PhotoSearchService` results and surface it in Agent Squad workflow steps.
- Cap the OpenAI-compatible embedding client timeout at 60 seconds.
- Add `backend/scripts/backfill_nvidia_embeddings.py` with dry-run default, `--create-mapping`, and `--execute` flags.
- Keep the backfill source text simple and deterministic from `title`, `description`, `tags`, and `photographer`.
- Add focused tests instead of live OpenSearch/NVIDIA calls.

## Files / Areas Affected
- `backend/app/config.py`
- `backend/app/services/photo_search.py`
- `backend/app/services/agent_squad.py`
- `backend/scripts/backfill_nvidia_embeddings.py`
- `backend/tests/test_photo_search_direct.py`
- `backend/tests/test_nvidia_embedding_backfill.py`
- `README.md`
- `design.md`
- `docs/plans/nvidia-query-embeddings-backfill.md`

## Verification Plan
- Run focused unit tests for photo search and backfill helpers.
- Compile the new script to catch syntax/import problems.
- Inspect the diff to ensure no unrelated worktree changes are included.

## Test Plan
- Before/proof: existing tests assert OpenAI sends `dimensions=256` and kNN uses `OPENSEARCH_VECTOR_FIELD`.
- Happy path: NVIDIA config sends `input_type=query`, optional `truncate`, and configured `dimensions`; generated kNN uses the configured NVIDIA vector field.
- Happy path: workflow log includes NVIDIA embedding metadata for direct hybrid image search.
- Sad path: embedding timeout config above 60 is capped at 60 for the client.
- Sad path: wrong embedding dimension still fails before querying OpenSearch.
- Backfill helper proof: source-text construction and mapping body generation are deterministic and dimension-aware.
- After/proof: focused tests pass locally without live credentials.

## Monitoring Plan
- Backfill script logs processed/updated/skipped/failed counts.
- Runtime fallback continues to expose embedding failures as lexical-only fallback errors.
- Operators can validate progress by counting docs with the new vector field before switching `OPENSEARCH_VECTOR_FIELD`.

## Risks / Open Questions
- Actual hosted NVIDIA latency and credit limits require a live benchmark with `NVIDIA_API_KEY`.
- The target OpenSearch field mapping must match the cluster's installed kNN plugin/version.
- Existing `OPENSEARCH_KNN_MIN_SCORE` may need retuning for the new embedding distribution.

## Status
- Implemented and verified with focused tests, full backend tests, and a live Agent Squad direct-hybrid smoke check.
