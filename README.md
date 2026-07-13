# Gen-Aperture: Agentic Stock Photo Conversational Search

AI-powered conversational interface for searching stock photos using natural language queries and document analysis.

## Features

- 🤖 **Multi-Agent AI** — LangGraph-orchestrated Squad Router, Project Manager, Search Specialist, Synthesizer, and Reflection Reranker agents
- 💬 **Natural language search** with multi-turn conversation context
- 📄 **Document upload** (PDF/DOCX/TXT) for brief-aware searching — auto-extracts visual requirements, mood, categories, and exclusions
- 🎯 **Reflection Reranking** — post-retrieval LLM reasoning pass that reorders, deduplicates, and filters results by true relevance (triggered by keywords like "best", "top ranked", "rerank", "reviewed")
- 🔐 **User-provided API keys** (30-min session, never stored on server)
- 📊 **Conversation history** with 7-day retention

## Architecture

![alt text](image.png)

- **Frontend**: React 18 + Vite
- **Backend**: FastAPI (Python 3.11+)
- **Agents**: LangGraph with OpenAI GPT-4o-mini
- **Storage**: OpenSearch (conversations + photo index `web-index-v9`)

## Agent Pipeline

```
User message
  ↓
Squad Router          — detects intent (relevance vs. popular) and routes
  ↓
Project Manager       — (brief uploads only) extracts visual requirements,
                         queries, categories, exclusions from PDF/DOCX/TXT
  ↓
Search Specialist     — calls Search Service MCP, builds + executes
                         OpenSearch hybrid (neural + lexical) query
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
- OpenAI API key (users provide their own)

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

4. **Open browser** → http://localhost:5173 → enter your OpenAI API key → start chatting

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
│   │   │   ├── photo_search.py        # OpenSearch photo query execution
│   │   │   ├── search_service_mcp.py  # Search Service MCP tool integration
│   │   │   ├── query_refinement.py    # Filter extraction (orientation, recency…)
│   │   │   ├── category_filter.py     # Category GID mapping
│   │   │   ├── file_extractor.py      # PDF/DOCX/TXT text extraction
│   │   │   ├── session_manager.py     # API key session handling
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
  -F "openai_api_key=sk-..." \
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
OPENSEARCH_PHOTO_INDEX=web-index-v9
OPENSEARCH_CONVERSATION_INDEX=gen-aperture-conversations
OPENSEARCH_READONLY=true
SESSION_TIMEOUT_MINUTES=30
ENVIRONMENT=development

# Agent LLM (optional — defaults shown)
AGENT_MODEL=qwen-plus
AGENT_MODEL_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
AGENT_FALLBACK_MODEL=gpt-4o-mini

# Reflection reranker thresholds (all optional — defaults shown)
RERANK_MIN_RESULTS_TARGET=10
RERANK_RELEVANCE_THRESHOLD=5.0
RERANK_BORDERLINE_THRESHOLD=3.5
RERANK_DUPLICATE_SIMILARITY_THRESHOLD=0.5
# Reflection reranker model (optional)
RERANK_MODEL=Qwen3-VL-Reranker-8B
```

## Security

- ⚠️ **API keys never stored on server** — users provide their own per session
- Session timeout: 30 minutes of inactivity, key auto-deleted
- OpenSearch cluster is read-only for the photo index
- 1MB file upload limit
- 7-day conversation retention

## Deployment

Uses Backstage FastAPI template:
```
https://backstage.shuttercorp.net/create/templates/default/add-gha-app-fastapi
```

## Documentation

- [design.md](design.md) — Complete technical specification
- [.github/copilot-instructions.md](.github/copilot-instructions.md) — AI coding guidelines
- [REVIEW.md](REVIEW.md) — Design review summary
- [PHASE1-COMPLETE.md](PHASE1-COMPLETE.md) — Phase 1 completion notes

## Support

Internal ShutterCorp project  
Team: Search Platform  
OpenSearch: `localhost:9200`

---

**Status**: Production-ready — multi-agent search with reflection reranking ✅
