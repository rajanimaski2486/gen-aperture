# OpenAI Query Embeddings for `icc_images_ext`

## Goal
Switch direct `icc_images_ext` image search query-vector generation from local CLIP/PCA embeddings to OpenAI `text-embedding-3-small` embeddings with 256 dimensions, matching the vectors already stored in `dense_vector`.

## Non-goals
- Do not add, remove, or upgrade dependencies.
- Do not reindex `icc_images_ext`.
- Do not change NVIDIA NIM chat, image-analysis, SearchByBrief LLM, or reflection-reranker behavior.
- Do not add a second CLIP kNN clause in this change.

## Constraints
- Use the existing `openai` Python package already present in `backend/requirements.txt`.
- Keep OpenAI credentials server-side through `OPENAI_API_KEY`.
- Keep query-vector dimension configurable but defaulted to `256`.
- Keep the OpenSearch hybrid query shape as one kNN subquery plus one lexical subquery.
- Preserve graceful failure when embedding generation fails.

## Acceptance criteria
- `PhotoSearchService` generates query vectors with OpenAI `text-embedding-3-small` using `dimensions=256`.
- The kNN clause continues to target `OPENSEARCH_VECTOR_FIELD=dense_vector`.
- CLIP/PCA is no longer required for the direct `icc_images_ext` image search path.
- SearchByBrief can still use its existing CLIP retriever settings independently.
- Focused tests cover the OpenAI embedding request and generated query path.
- README, `design.md`, `QUICKSTART.md`, and the Excalidraw architecture JSON describe the OpenAI embedding path.

## Approach
- Add OpenAI embedding settings and a `require_openai_api_key` helper to app config.
- Refactor `PhotoSearchService._embed_query_text` to call the OpenAI embeddings endpoint with the configured model and dimensions.
- Cache the OpenAI client on the service instance, similar to the old local embedder cache.
- Update unit tests with a fake embeddings client so no live OpenAI call is made.
- Update docs and diagram text that currently mention CLIP/PCA for direct image search.

## Files / areas affected
- `backend/app/config.py`
- `backend/app/services/photo_search.py`
- `backend/tests/test_photo_search_direct.py`
- `README.md`
- `design.md`
- `QUICKSTART.md`
- `docs/gen-aperture-architecture.excalidraw.json`
- `docs/plans/openai-query-embeddings.md`

## Verification plan
- Run focused `PhotoSearchService` unit tests.
- Compile changed backend modules.
- Parse the Excalidraw JSON.
- Run stale-doc reference scans for direct-search CLIP/PCA wording.
- Review staged diff and git status before committing.

## Test plan
- Before/proof: current `PhotoSearchService` imports `LocalClipTextEmbedder` and uses CLIP/PCA for `_embed_query_text`.
- Happy path: fake OpenAI embeddings client returns a 256d vector and direct hybrid search places that vector in the kNN clause.
- Sad path: malformed embedding dimensions raise a clear error before querying OpenSearch.
- After/proof: focused unit tests, compile, JSON parse, and diff checks pass.

## Monitoring plan
- In local or deployed runs, failed query embedding calls should appear as direct hybrid query failures with a clear `OPENAI_API_KEY` or embedding-dimension error.

## Risks / open questions
- End-to-end live search still depends on a valid server-side `OPENAI_API_KEY`.
- If `dense_vector` was normalized at ingest, the query embedding should match the same ingest normalization policy; this change assumes the stored OpenAI 256d vectors use the API's 256d output directly.

## Status
- Implemented and locally verified with focused direct-search tests, full backend test discovery, Python compile, Excalidraw JSON parse, stale-doc scan, and diff checks.
