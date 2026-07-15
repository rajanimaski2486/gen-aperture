# kNN Nearest Neighbor Cutoff

## Goal
Reduce irrelevant vector-only candidates by making the OpenSearch kNN side of the hybrid query fetch only neighbors above a configured similarity threshold.

## Non-goals
- Do not change the lexical query fields or boosts.
- Do not change reranker scoring behavior.
- Do not change the `icc_images_ext` index mapping or OpenSearch pipeline.
- Do not add another retrieval service dependency.

## Constraints
- Keep querying the same Aiven OpenSearch domain and `icc_images_ext` index.
- Apply the cutoff inside the kNN clause so distant vector neighbors are removed before hybrid blending.
- Keep the setting tunable because the best threshold depends on the vector space and OpenSearch scoring mode.

## Acceptance criteria
- The generated hybrid query uses OpenSearch radial kNN search when `OPENSEARCH_KNN_MIN_SCORE` is above zero.
- The radial query does not send `k` at the same time as `min_score`.
- Setting `OPENSEARCH_KNN_MIN_SCORE=0` preserves the previous top-`k` behavior.
- Focused backend tests cover both radial and fallback query shapes.

## Approach
- Add `opensearch_knn_min_score` to backend settings with a default chosen from local vector-score samples.
- Move kNN clause creation into a helper so the mutually exclusive `min_score` and `k` behavior is explicit.
- Update existing direct photo search tests to assert the new radial query shape.

## Files / areas affected
- `backend/app/config.py`
- `backend/app/services/photo_search.py`
- `backend/tests/test_photo_search_direct.py`
- `docs/plans/knn-nearest-neighbor-cutoff.md`

## Verification plan
- Run focused direct photo search tests.
- Run full backend unit tests.
- Compile changed backend modules.
- Run diff whitespace validation.

## Test plan
- Before: direct hybrid query always requested top-`k` vector candidates, regardless of distance.
- Happy path: default query sends `min_score` in the kNN clause and omits `k`.
- Compatibility path: zero threshold sends the previous `k` clause.
- Proof: focused and full backend test suites pass.

## Monitoring plan
- Inspect returned result counts and relevance in local UI searches after deployment.
- Tune `OPENSEARCH_KNN_MIN_SCORE` if the cutoff is too loose or too strict for production traffic.

## Risks / open questions
- The ideal score cutoff is empirical. The default is intentionally conservative and can be adjusted from env.
- OpenSearch radial kNN requires OpenSearch 2.14+ with a supported kNN engine.

## Status
- Implemented and locally verified.
