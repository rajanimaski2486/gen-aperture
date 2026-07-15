# Reranker Fields for `icc_images_ext`

## Goal

Update reflection reranking so candidate evaluation uses only fields available and useful in `icc_images_ext`.

## Non-goals

- Do not add a new visual reranker model.
- Do not add dependencies or download model assets.
- Do not reintroduce `web-index-v9` fields such as categories, license count, or date.
- Do not change the OpenSearch retrieval query.

## Constraints

- Candidate reranking fields are limited to title, description, tags, image identity, dimensions/inferred orientation, and retrieval score.
- Pexels URL is for linking/debugging, not relevance scoring.
- Photographer is not used for relevance scoring.
- Existing reranker API shape and frontend rerank log should remain compatible.

## Acceptance Criteria

- LLM rerank prompts no longer include category IDs, license count, or legacy keyword labels.
- Candidate summaries label `icc_images_ext` tags as `tags`.
- Python dedupe uses approved text fields instead of old keyword-only overlap.
- Focused tests cover the summary fields, inferred orientation, and dedupe behavior.

## Approach

- Add helper functions for title/description/tag normalization and orientation inference.
- Refactor candidate and critique summaries to use the approved fields.
- Update prompts to describe the approved fields explicitly.
- Add `title` to internal formatted photo results for reranker use.
- Add backend unit tests for the new helper behavior.

## Files / Areas Affected

- `backend/app/services/reranker.py`
- `backend/app/services/photo_search.py`
- `backend/tests/`

## Verification Plan

- Run focused reranker tests.
- Run the full backend unit test suite.
- Compile changed Python modules.
- Run diff whitespace checks.

## Test Plan

- Happy path: candidate summary includes approved `icc_images_ext` fields.
- Sad path: unsupported old fields are absent from reranker summaries.
- Proof: tests assert no license/category/legacy keyword fields are emitted.

## Monitoring Plan

- Reranker workflow output continues reporting pass counts, excludes, borderlines, and final kept results.

## Risks / Open Questions

- Text-only reranking still cannot inspect the actual image pixels.
- A future visual rerank pass could use `medium_url` or `thumbnail_url`, but this change does not add that pass.

## Status

- Implemented and locally verified with focused reranker tests, full backend unit tests, compile, and diff checks.
