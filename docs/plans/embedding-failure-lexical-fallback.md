# Lexical Fallback for Query Embedding Failures

## Goal
Prevent simple image searches such as `red roses` from returning zero results solely because OpenAI query embedding generation is unavailable or misconfigured.

## Non-goals
- Do not remove OpenAI `text-embedding-3-small` query embeddings for the normal direct search path.
- Do not change `icc_images_ext` mappings or reindex data.
- Do not add dependencies.
- Do not hide embedding failures from workflow/debug metadata.

## Constraints
- `OPENAI_API_KEY` may be absent in local/dev environments.
- The photo index still supports lexical search over `title`, `description`, `tags`, and `photographer`.
- Keep OpenSearch read-only guardrails unchanged.
- Keep the fallback inside `PhotoSearchService` so text and brief image searches both benefit.

## Acceptance criteria
- Direct image search tries OpenAI-vector hybrid search first.
- If query embedding generation fails, direct image search executes a lexical-only query instead of returning an empty result immediately.
- Fallback results include `opensearch_query`, `opensearch_index`, and an explanatory `error` for workflow visibility.
- Focused tests cover missing embedding failure fallback and normal hybrid behavior.

## Approach
- Factor the lexical bool construction out of the hybrid query builder.
- Add a lexical-only query builder that reuses the same `_source`, text fields, exclusions, and supported filters.
- In `execute_direct_hybrid_search`, catch embedding-generation failures before the OpenSearch request and run lexical fallback.
- Update tests with a fake embedding client that raises and assert OpenSearch is still called with lexical-only query.

## Files / areas affected
- `backend/app/services/photo_search.py`
- `backend/tests/test_photo_search_direct.py`
- `docs/plans/embedding-failure-lexical-fallback.md`

## Verification plan
- Run focused `PhotoSearchService` unit tests.
- Run full backend test discovery.
- Compile changed backend files.
- Review diff and staged files before commit.

## Test plan
- Before/proof: local settings show `OPENAI_API_KEY` absent, and current code returns empty before lexical search when embedding fails.
- Happy path: embedding succeeds and hybrid query still combines kNN plus lexical subqueries.
- Sad path: embedding fails and lexical-only search still executes against OpenSearch.
- After/proof: tests assert the lexical fallback query is sent and results are mapped.

## Monitoring plan
- Workflow steps should show the fallback query and the embedding failure reason in the search step output.

## Risks / open questions
- Lexical-only fallback may be less semantically rich than vector hybrid search, but it is preferable to returning zero results when embedding credentials are missing.

## Status
- Implemented and locally verified with focused tests, full backend test discovery, compile checks, Excalidraw JSON parse, and a live read-only `red roses` fallback smoke test.
