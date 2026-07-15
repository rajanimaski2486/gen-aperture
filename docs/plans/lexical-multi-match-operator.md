# Lexical Multi Match Operator Plan

## Goal

Update the direct photo search lexical `multi_match` query so short queries require all lexical terms, while longer queries use `minimum_should_match` for a softer match.

## Non-goals

- Do not change vector/kNN query construction.
- Do not change lexical fields, boosts, filters, exclusions, or result mapping.
- Do not add dependencies or alter package manager state.

## Constraints

- Keep the diff small and focused.
- Preserve lexical fallback behavior when embedding generation fails.
- Ignore boolean connector words when deciding whether a query is short or long.
- Leave unrelated existing workspace changes untouched.

## Acceptance criteria

- Lexical queries with 1-3 meaningful terms set `operator` to `and` on `multi_match`.
- Lexical queries with 4 or more meaningful terms use `minimum_should_match` on `multi_match`.
- Boolean connector words such as `AND`, `OR`, and `NOT` do not count as lexical terms.
- Hybrid and lexical-only fallback queries share the same policy.

## Approach

- Add a helper in `PhotoSearchService` to extract/count meaningful terms from the lexical query text.
- Add a helper to apply the matching policy to the existing `multi_match` body.
- Reuse the helper from `_build_lexical_bool` so hybrid and fallback query builders stay consistent.
- Add targeted unit tests around the generated `multi_match`.

## Files / areas affected

- `backend/app/services/photo_search.py`
- `backend/tests/test_photo_search_direct.py`

## Verification plan

- Exercise `build_direct_hybrid_query` for short and long lexical queries and inspect the resulting `multi_match`.
- Exercise `execute_direct_hybrid_search` with an embedding failure to verify lexical-only fallback uses the same policy.
- Run the focused direct photo search unittest module.

## Test plan

- Before/proof: confirm current code hard-codes `operator: or` in direct lexical `multi_match`.
- Happy path: assert 1-3 meaningful terms produce `operator: and` without `minimum_should_match`.
- Sad/edge path: assert 4+ meaningful terms produce `minimum_should_match` without a hard `and` operator.
- After/proof: assert boolean connector strings such as `red AND roses` count as two meaningful terms.

## Monitoring plan

- No runtime monitoring changes. Review generated OpenSearch debug payloads after deployment to confirm expected lexical shape.

## Risks / open questions

- The exact `minimum_should_match` value is inferred from existing query-string tuning in the repo as `75%`.

## Status

- Implemented and verified with focused unit tests.
