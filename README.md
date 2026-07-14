# Gen-Aperture: Agentic Stock Photo Conversational Search

AI-powered conversational interface for searching stock photos using natural language queries and document analysis.

## Features

- 🤖 **Multi-Agent AI** — LangGraph-orchestrated Squad Router, Project Manager, Search Specialist, Synthesizer, and Reflection Reranker agents
- 💬 **Natural language search** with multi-turn conversation context
- 📄 **Document upload** (PDF/DOCX/TXT) for brief-aware searching — auto-extracts visual requirements, mood, categories, and exclusions
- 🎯 **Reflection Reranking** — post-retrieval LLM reasoning pass that reorders, deduplicates, and filters results by true relevance (triggered by keywords like "best", "top ranked", "rerank", "reviewed")
- 🔐 **Server-side NVIDIA API key** via `NVIDIA_API_KEY`
- 📊 **Conversation history** with 7-day retention

## Architecture

![alt text](image.png)

- **Frontend**: React 18 + Vite
- **Backend**: FastAPI (Python 3.11+)
- **Agents**: LangGraph with NVIDIA NIM OpenAI-compatible chat completions
- **Search and storage**: One Aiven OpenSearch domain, using `icc_images_ext` for read-only image search and `gen-aperture-conversations` for guarded conversation writes

### OpenSearch Search Flow

Image and document-assisted image search no longer asks Search Service for a base payload. The backend generates the OpenSearch body directly in `PhotoSearchService`:

1. Build a local text embedding for the semantic query.
2. Project the embedding to the 256-dimension vector used by `icc_images_ext`.
3. Query `icc_images_ext` with an OpenSearch `hybrid` query:
   - kNN over `dense_vector`
   - lexical `multi_match` over `title`, `description`, `tags`, and `photographer`
4. Run the query through the `reveal-hybrid` search pipeline.
5. Map `icc_images_ext` fields (`image_id`, URLs, `tags`, `photographer`, dimensions) into the existing frontend result shape.

Explicit video and mixed video searches still use the video search service path.

## Agent Pipeline

```
User message
  ↓
Squad Router          — detects intent (relevance vs. popular) and routes
  ↓
Project Manager       — (brief uploads only) extracts visual requirements,
                         queries, categories, exclusions from PDF/DOCX/TXT
  ↓
Search Specialist     — builds + executes a direct OpenSearch hybrid
                         lexical + kNN query against icc_images_ext
  ↓
Reflection Reranker   — (trigger-phrase only) 2-pass LLM reflection:
                         Pass 1: scores all 20 candidates on relevance,
                                 criteria match, specificity, completeness
                         Pass 2: critiques ranking, identifies duplicates,
                                 flags borderline; Pass 3 (Python) builds
                                 final ordered list of ≥10 results
  ↓
Synthesizer           — combines brief analysis + results into response
```

### Reflection Reranker — trigger phrases

| User says… | Triggers reranking? |
|---|---|
| "best results", "best matching photos" | ✅ |
| "top ranked", "top-ranked" | ✅ |
| "rerank" | ✅ |
| "reflect and respond" | ✅ |
| "reviewed picks" | ✅ |
| "find a sunset photo" | ❌ |

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- Access to internal OpenSearch cluster
- NVIDIA API key configured in `backend/.env`

### Development Setup

1. **Clone and setup environment:**
```bash
cd gen-aperture
cp backend/.env.example backend/.env  # edit as needed
```

2. **Start backend:**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Backend runs on http://localhost:8000

3. **Start frontend (new terminal):**
```bash
cd frontend
npm install
npm run dev
```

Frontend runs on http://localhost:5173

4. **Open browser** → http://localhost:5173 → start chatting

### Using Docker Compose

```bash
docker-compose up
```

Access at http://localhost:5173

## Project Structure

```
gen-aperture/
├── backend/
│   ├── app/
│   │   ├── main.py                    # FastAPI application entry point
│   │   ├── config.py                  # Settings & reranker thresholds
│   │   ├── routers/
│   │   │   ├── chat.py                # POST /api/chat endpoint
│   │   │   └── conversations.py       # Conversation history endpoints
│   │   ├── services/
│   │   │   ├── agent_squad.py         # LangGraph multi-agent orchestrator
│   │   │   ├── reranker.py            # Reflection reranker (3-pass pipeline)
│   │   │   ├── photo_search.py        # Direct OpenSearch hybrid image query execution
│   │   │   ├── search_service_mcp.py  # Legacy/optional Search Service helper
│   │   │   ├── query_refinement.py    # Filter extraction (orientation, recency…)
│   │   │   ├── category_filter.py     # Category GID mapping
│   │   │   ├── file_extractor.py      # PDF/DOCX/TXT text extraction
│   │   │   ├── conversation_store.py  # OpenSearch conversation persistence
│   │   │   └── opensearch_guardrails.py # Read-only safety enforcement
│   │   └── models/
│   │       └── schemas.py             # Pydantic request/response schemas
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx                    # Chat UI, workflow panel, rerank log
│   │   ├── services/api.js            # Axios API client
│   │   └── index.css                  # Styles including rerank panel
│   └── package.json
├── briefs/                            # Sample creative briefs for testing
├── design.md                          # Technical specification
├── .github/copilot-instructions.md    # AI agent guidance
└── docker-compose.yml
```

## Workflow

![alt text](image-1.png)

## API Endpoints

### POST /api/chat
Send message and get AI response with photo results.
```bash
curl -X POST http://localhost:8000/api/chat \
  -F "message=Find the best outdoor sunset photos" \
  -F "file=@brief.pdf"
```

Response includes standard fields plus reranker output when triggered:
```json
{
  "conversation_id": "...",
  "response": "Here are the top results…",
  "results": [...],
  "search_mode": "relevance",
  "workflow_steps": [...],
  "rerank_applied": true,
  "rerank_decisions": [
    {"final_rank": 1, "hadron_id": "h001", "rerank_score": 0.97,
     "keep": true, "is_borderline": false,
     "reason": "Directly depicts golden ocean sunset.", "confidence": 0.97}
  ],
  "rerank_explanation": null
}
```

### GET /api/conversations/recent
List last 5 conversations

### GET /api/conversations/{id}
Get full conversation with messages

### GET /health
Health check

## Configuration

Backend environment variables (`backend/.env`):
```
OPENSEARCH_ENDPOINT=http://localhost:9200
OPENSEARCH_USERNAME=...
OPENSEARCH_PASSWORD=...
OPENSEARCH_PHOTO_INDEX=icc_images_ext
OPENSEARCH_HYBRID_SEARCH_PIPELINE=reveal-hybrid
OPENSEARCH_VECTOR_FIELD=dense_vector
OPENSEARCH_KNN_K=200
OPENSEARCH_TEXT_EMBEDDING_PCA_MODEL_PATH=./ipca_10m.npz
OPENSEARCH_CONVERSATION_ENDPOINT=http://localhost:9200
OPENSEARCH_CONVERSATION_INDEX=gen-aperture-conversations
OPENSEARCH_CONVERSATION_MAX_RECORDS=5000
OPENSEARCH_CONVERSATION_MAX_STORE_BYTES=5368709120
OPENSEARCH_READONLY=true
SESSION_TIMEOUT_MINUTES=30
ENVIRONMENT=development

# NVIDIA LLM (required key; defaults shown)
NVIDIA_API_KEY=...
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
AGENT_MODEL=meta/llama-3.3-70b-instruct
IMAGE_ANALYSIS_MODEL=meta/llama-3.2-11b-vision-instruct
SEARCHBYBRIEF_MODEL=meta/llama-3.3-70b-instruct
SEARCHBYBRIEF_RETRIEVER_CLIP_MODEL=ViT-B/32
SEARCHBYBRIEF_RETRIEVER_CLIP_DOWNLOAD_ROOT=/tmp/clip

# Reflection reranker thresholds (all optional — defaults shown)
RERANK_MIN_RESULTS_TARGET=10
RERANK_RELEVANCE_THRESHOLD=5.0
RERANK_BORDERLINE_THRESHOLD=3.5
RERANK_DUPLICATE_SIMILARITY_THRESHOLD=0.5
# Reflection reranker model (optional)
RERANK_MODEL=meta/llama-3.3-70b-instruct
```

## Security

- ⚠️ `NVIDIA_API_KEY` stays server-side in `backend/.env`
- OpenSearch cluster is read-only for the photo index
- Conversation writes are allowed only to `gen-aperture-conversations`
- Conversation writes are rejected at 5000 records or 5 GB index store size
- 6MB file upload limit
- 7-day conversation retention
- The OpenSearch workflow panel redacts credentials from displayed endpoint URLs


## Documentation

- [design.md](design.md) — Complete technical specification
- [.github/copilot-instructions.md](.github/copilot-instructions.md) — AI coding guidelines
- [REVIEW.md](REVIEW.md) — Design review summary
- [PHASE1-COMPLETE.md](PHASE1-COMPLETE.md) — Phase 1 completion notes

https://rajanim.github.io/agent-search-patterns/
