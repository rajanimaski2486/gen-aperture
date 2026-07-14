# Gen-Aperture Architecture Review

## Current Decision Summary

Gen-Aperture now runs with server-side NVIDIA NIM credentials and direct OpenSearch image search.

Approved current decisions:

- Use FastAPI plus React/Vite for the app shell.
- Use LangGraph for the multi-agent search workflow.
- Use `NVIDIA_API_KEY` from `backend/.env`; users no longer provide OpenAI keys in the UI.
- Use the same configured OpenSearch domain for image search and conversation storage.
- Search images directly in `icc_images_ext`; do not call Search Service to obtain the image OpenSearch payload.
- Build an OpenSearch `hybrid` query in the app with lexical `multi_match` plus kNN over `dense_vector`.
- Use the `reveal-hybrid` search pipeline by default.
- Allow OpenSearch writes only to `gen-aperture-conversations`.
- Reject conversation writes above 5000 records or 5 GB index store size.

## Image Search

### Index

`icc_images_ext`

### Fields Used

- `image_id`
- `title`
- `description`
- `tags`
- `thumbnail_url`
- `medium_url`
- `pexels_url`
- `photographer`
- `width`
- `height`
- `dense_vector`

### Query Strategy

`backend/app/services/photo_search.py` generates a direct OpenSearch query:

- kNN clause over `dense_vector`
- lexical `multi_match` over `title`, `description`, `tags`, and `photographer`
- optional lexical `must_not` clauses for exclusions
- `search_pipeline=reveal-hybrid`

Older `web-index-v9` filters such as category, orientation, generated status, recency, and license count are not sent to `icc_images_ext` because those fields are not mapped there.

## Conversation Storage

### Index

`gen-aperture-conversations`

### Guardrails

- The conversation client is the only writable OpenSearch client.
- The write guardrail allows mutations only under `gen-aperture-conversations`.
- Index creation is allowed only for `gen-aperture-conversations`.
- Writes are rejected when the index has 5000 records.
- Writes are rejected when index store size reaches 5 GB.
- Photo search clients stay read-only.

## User Experience

- The frontend starts directly without an API-key modal.
- Model choice is a server-side NVIDIA model selection.
- Conversation history is loaded from OpenSearch.
- The workflow panel shows generated OpenSearch payloads for direct image search.
- Video search workflow steps may still show video Search Service metadata because that path remains separate.

## Runtime Requirements

- `backend/.env` contains `NVIDIA_API_KEY`.
- `backend/.env` contains OpenSearch endpoint and credentials.
- `OPENSEARCH_PHOTO_INDEX` is `icc_images_ext`.
- `OPENSEARCH_HYBRID_SEARCH_PIPELINE` is `reveal-hybrid`.
- The PCA model for text embeddings is available, defaulting to `ipca_10m.npz`.
- CLIP weights are available locally or the operator has approved downloading them.

## Historical Decisions Superseded

The following earlier decisions are no longer current:

- User-provided OpenAI API keys.
- Backend in-memory API-key sessions.
- `web-index-v9` as the primary image index.
- Search Service as the source of image OpenSearch payloads.
- A separate writable conversation OpenSearch endpoint by default.

See [design.md](design.md) for the current end-to-end architecture.
