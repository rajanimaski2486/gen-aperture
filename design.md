# Gen-Aperture Current Architecture

This document describes the current runtime architecture for Gen-Aperture after the NVIDIA LLM and direct OpenSearch hybrid-search changes.

## Executive Summary

Gen-Aperture is a conversational stock-image search application. Users search with natural language and can upload PDF, DOCX, or TXT briefs. The backend uses LangGraph agents, NVIDIA NIM chat-completion models, and OpenSearch for both read-only image search and guarded conversation persistence.

The image search path no longer depends on Search Service for a base OpenSearch payload. The app generates the OpenSearch query itself and sends it directly to the `icc_images_ext` index on the configured Aiven OpenSearch domain.

## Technology Stack

| Component | Technology |
| --- | --- |
| Frontend | React 18 + Vite |
| Backend | FastAPI, Python 3.11+ |
| Agent framework | LangGraph |
| LLM provider | NVIDIA NIM via OpenAI-compatible chat completions |
| Image search | OpenSearch `icc_images_ext` |
| Conversation store | OpenSearch `gen-aperture-conversations` |
| Document extraction | PDF/DOCX/TXT extraction in backend services |

## Runtime Model

Development runs two local processes:

```text
Browser
  -> Vite dev server on localhost:5173
  -> FastAPI backend on localhost:8000
  -> Aiven OpenSearch domain from OPENSEARCH_ENDPOINT
  -> NVIDIA NIM API from NVIDIA_BASE_URL
```

Production can serve the compiled React frontend from the FastAPI app when static assets are present.

## Agent Flow

```text
User message / optional brief
  -> FastAPI /api/chat
  -> Load or create conversation
  -> Extract uploaded document text and images, if present
  -> Squad Router
  -> Project Manager, for uploaded briefs
  -> Search Specialist
  -> Direct OpenSearch hybrid image search against icc_images_ext
  -> Optional Reflection Reranker
  -> Synthesizer response
  -> Guarded conversation write
```

Video and mixed video requests still use the existing video search service path. Image-only and document-assisted image search use the direct OpenSearch path.

## Direct OpenSearch Image Search

`backend/app/services/photo_search.py` owns direct image search.

The generated query uses:

- `OPENSEARCH_PHOTO_INDEX=icc_images_ext`
- `OPENSEARCH_HYBRID_SEARCH_PIPELINE=reveal-hybrid`
- `OPENSEARCH_VECTOR_FIELD=dense_vector`
- `OPENSEARCH_KNN_K=200`

The app builds a query shaped like:

```json
{
  "size": 50,
  "_source": [
    "image_id",
    "title",
    "description",
    "tags",
    "thumbnail_url",
    "medium_url",
    "pexels_url",
    "photographer",
    "width",
    "height"
  ],
  "query": {
    "hybrid": {
      "queries": [
        {
          "knn": {
            "dense_vector": {
              "vector": [0.0],
              "k": 200
            }
          }
        },
        {
          "bool": {
            "must": [
              {
                "multi_match": {
                  "query": "outdoor nature photos",
                  "fields": [
                    "title^4",
                    "description^3",
                    "tags^2",
                    "photographer"
                  ],
                  "type": "best_fields",
                  "operator": "or",
                  "fuzziness": "AUTO"
                }
              }
            ]
          }
        }
      ]
    }
  }
}
```

At runtime the vector is a real 256-dimension embedding. The backend creates it by embedding the semantic query with the configured CLIP text model and projecting it with the PCA model. By default, the PCA file is `ipca_10m.npz` at the repo root.

Unsupported filters from the older `web-index-v9` schema are intentionally ignored for `icc_images_ext` because that index does not expose category, orientation, date, generated, or license-count fields.

## Result Mapping

`icc_images_ext` hits are mapped into the existing frontend result shape:

| `icc_images_ext` field | API result field |
| --- | --- |
| `image_id` | `hadron_id`, `ext_id` when numeric |
| `description` or `title` | `description` |
| `medium_url`, `pexels_url`, `thumbnail_url` | `image_url` fallback chain |
| `thumbnail_url`, `medium_url` | `thumbnail_url` fallback chain |
| `tags` | `keywords` |
| `photographer` | `photographer` |
| `width`, `height` | `width`, `height` |

## Conversation Storage

Conversations are stored in `gen-aperture-conversations` on the same OpenSearch domain by default.

Guardrails enforce:

- Writes only to `gen-aperture-conversations`.
- Index creation only for `gen-aperture-conversations`.
- Writes rejected once the index has 5000 records.
- Writes rejected once index store size reaches 5 GB.
- Search/photo clients remain read-only.

The conversation document stores:

- `conversation_id`
- `created_at`
- `last_message_at`
- `last_user_query`
- `title`
- `message_count`
- optional uploaded `file_name` and extracted `file_content`
- nested `messages`

The app attempts to create a 7-day ISM retention policy when the OpenSearch plugin is available.

## LLM Configuration

LLM calls use the server-side NVIDIA key:

```text
NVIDIA_API_KEY=...
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
AGENT_MODEL=meta/llama-3.3-70b-instruct
IMAGE_ANALYSIS_MODEL=meta/llama-3.2-11b-vision-instruct
SEARCHBYBRIEF_MODEL=meta/llama-3.3-70b-instruct
RERANK_MODEL=meta/llama-3.3-70b-instruct
```

The request schema still accepts `openai_api_key` for backward compatibility, but it is deprecated and not used for LLM calls.

## API Surface

### `POST /api/chat`

Multipart form endpoint:

- `message`: required user message
- `conversation_id`: optional existing conversation
- `file`: optional PDF/DOCX/TXT upload
- `workflow_mode`: defaults to `agent_squad`
- `model`: optional NVIDIA model override

The response includes the assistant response, image results, workflow trace, optional reranker details, and conversation metadata.

### `GET /api/conversations/recent`

Returns the five most recent conversations for the sidebar.

### `GET /api/conversations/{conversation_id}`

Returns a full conversation document.

### `DELETE /api/conversations/{conversation_id}`

Deletes a conversation from the guarded conversation index.

### `GET /health`

Checks OpenSearch connectivity through the conversation store client.

## Local Verification

The safest local smoke test is:

```bash
cd backend
source venv/bin/activate
uvicorn app.main:app --reload
curl http://localhost:8000/health

cd ../frontend
npm run dev
```

End-to-end chat requires:

- `NVIDIA_API_KEY` in `backend/.env`
- OpenSearch endpoint and credentials in `backend/.env`
- `ipca_10m.npz` available, or `OPENSEARCH_TEXT_EMBEDDING_PCA_MODEL_PATH` set
- CLIP model weights available under `SEARCHBYBRIEF_RETRIEVER_CLIP_DOWNLOAD_ROOT`, or permission to download them

## Historical Notes

Older docs in this repo may mention user-provided OpenAI keys, `web-index-v9`, or Search Service as the source of image OpenSearch payloads. Those were superseded by the current NVIDIA and direct `icc_images_ext` design above.
