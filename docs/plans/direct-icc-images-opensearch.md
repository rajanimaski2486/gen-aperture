# Direct `icc_images_ext` OpenSearch Hybrid Search

## Goal

Replace the image Search Service payload dependency with app-generated OpenSearch hybrid queries against `icc_images_ext` on the configured Aiven OpenSearch endpoint.

## Non-goals

- Do not add or upgrade dependencies.
- Do not bypass OpenSearch read-only guardrails for search.
- Do not change video search behavior unless required by image search.
- Do not alter conversation write guardrails from the NVIDIA/OpenSearch work.

## Constraints

- Use the same `OPENSEARCH_ENDPOINT` and credentials as the conversation/search domain.
- Query `icc_images_ext` by default.
- Generate a hybrid query in the app that combines lexical BM25 over mapped text fields and kNN over the mapped `dense_vector` field.
- Use the Aiven `reveal-hybrid` search pipeline by default.
- The index has `title`, `description`, `tags`, URL fields, and `dense_vector`; it does not expose OpenSearch ML text models.

## Acceptance Criteria

- AgentSquad image search no longer calls Search Service MCP to get the base OpenSearch payload.
- Text and brief image searches execute a locally generated hybrid query against `icc_images_ext`.
- Results are mapped from `icc_images_ext` fields into the existing API `PhotoResult` shape.
- Tests cover the generated hybrid payload and response mapping.

## Approach

- Add direct hybrid search settings for index, vector field, pipeline, and PCA path.
- Add lazy CLIP+PCA text embedding generation in `PhotoSearchService`.
- Add a direct hybrid query builder and execution method in `PhotoSearchService`.
- Replace AgentSquad image Search Service calls/modification flow with direct query generation/execution.
- Leave video MCP behavior as-is for explicit video/mixed search.

## Files / Areas Affected

- `backend/app/config.py`
- `backend/app/services/photo_search.py`
- `backend/app/services/agent_squad.py`
- `backend/tests/`
- `README.md`
- `QUICKSTART.md`

## Verification Plan

- Compile changed Python modules.
- Run existing and new backend unit tests.
- Run frontend build to ensure API shape remains compatible.
- Inspect diff for accidental Search Service image dependency in AgentSquad.

## Test Plan

- Happy path: generated query contains `hybrid` with `knn` on `dense_vector` and lexical query over `title`, `description`, and `tags`.
- Happy path: `icc_images_ext` hit maps URL/title/tag fields into API result fields.
- Sad path: vector generation failure returns a clean search error/fallback rather than calling Search Service.
- Proof: unit tests and compile/build checks.

## Monitoring Plan

- Workflow steps include the generated OpenSearch payload and target `icc_images_ext` URL.
- Backend logs report direct hybrid search totals and execution time.

## Risks / Open Questions

- Direct vector generation depends on local CLIP/torch and the PCA weights being available.
- Live OpenSearch query execution was run locally with a stubbed 256-dim vector to avoid downloading CLIP model assets.
- Full end-to-end embedding generation still depends on CLIP model weights being available locally or approved for download.
- Video search still uses the existing video service path.

## Status

- Implemented and locally verified with compile, unit, frontend build, mapping/pipeline inspection, and diff checks.
- Follow-up: direct query-vector generation was later changed from the initial local CLIP/PCA approach to OpenAI `text-embedding-3-small` with `dimensions=256`; see `docs/plans/openai-query-embeddings.md`.
