# Gen-Aperture Current Implementation Status

This file originally tracked the Phase 1 foundation. The app has since moved beyond that milestone. The current implementation includes NVIDIA-backed agents, direct OpenSearch image search, document-assisted search, reflection reranking, and guarded conversation writes.

## Current Backend

- FastAPI application in `backend/app/main.py`
- Chat endpoint in `backend/app/routers/chat.py`
- Conversation endpoints in `backend/app/routers/conversations.py`
- LangGraph orchestration in `backend/app/services/agent_squad.py`
- Direct OpenSearch image search in `backend/app/services/photo_search.py`
- Conversation persistence in `backend/app/services/conversation_store.py`
- OpenSearch read/write guardrails in `backend/app/services/opensearch_guardrails.py`
- Reflection reranker in `backend/app/services/reranker.py`
- Document extraction and image analysis for uploaded briefs

## Current Frontend

- React/Vite chat UI in `frontend/src/App.jsx`
- Sidebar conversation history
- File upload for PDF/DOCX/TXT briefs
- NVIDIA model selector
- Image result grid
- Agent workflow trace with OpenSearch payload viewer
- Reflection reranking decision panel

## Current Search Path

Image search now runs directly against `icc_images_ext`:

1. AgentSquad produces semantic and lexical search text.
2. `PhotoSearchService` embeds the semantic query.
3. The embedding is projected to the 256-dimension `dense_vector` space.
4. The backend sends an OpenSearch `hybrid` query to `icc_images_ext`.
5. The query combines kNN and lexical `multi_match` clauses.
6. The `reveal-hybrid` pipeline fuses scores.
7. Results are mapped into the existing frontend API shape.

Search Service is no longer used for the image OpenSearch payload. Video search remains on the existing video service path.

## Current Security and Guardrails

- LLM calls use server-side `NVIDIA_API_KEY`; the frontend does not collect OpenAI keys.
- The photo OpenSearch client is read-only.
- Conversation writes are allowed only to `gen-aperture-conversations`.
- Conversation writes are rejected above 5000 records or 5 GB store size.
- Uploaded files are limited to 6 MB and accepted only for supported document types.

## Run Locally

Backend:

```bash
cd backend
source venv/bin/activate
uvicorn app.main:app --reload
```

Frontend:

```bash
cd frontend
npm run dev
```

Health check:

```bash
curl http://localhost:8000/health
```

End-to-end chat requires `NVIDIA_API_KEY`, OpenSearch credentials, `ipca_10m.npz`, and available CLIP model weights.

## Current Validation Commands

```bash
PYTHONPATH=backend backend/venv/bin/python -m py_compile backend/app/config.py backend/app/services/photo_search.py backend/app/services/agent_squad.py backend/app/routers/chat.py
PYTHONPATH=backend backend/venv/bin/python -m unittest discover -s backend/tests
npm --prefix frontend run build
git diff --check
```

See [README.md](README.md), [QUICKSTART.md](QUICKSTART.md), and [design.md](design.md) for the current user and architecture docs.
