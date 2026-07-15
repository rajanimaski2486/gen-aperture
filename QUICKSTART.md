# 🚀 Quick Start Guide

## What's Been Built

✅ **Backend (FastAPI)**
- Multi-agent LangGraph orchestration (Squad Router, Project Manager, Search Specialist, Synthesizer)
- Reflection Reranker — post-retrieval 3-pass LLM scoring, critique, and filtering
- Direct hybrid lexical + kNN OpenSearch queries against `icc_images_ext`
- Server-side NVIDIA NIM configuration via `NVIDIA_API_KEY`
- Agent LLM calls capped by `AGENT_LLM_TIMEOUT_SECONDS` and reranker calls capped by `RERANK_TIMEOUT_SECONDS`
- OpenSearch conversation store with 7-day retention
- PDF/DOCX/TXT file extraction for brief analysis
- Category mapping, query refinement, and exclusion filtering
- OpenSearch guardrails (read-only image search plus constrained conversation writes)

✅ **Frontend (React + Vite)**
- Chat interface with multi-turn conversation context
- Sidebar with last 5 conversation history entries
- File upload (PDF/DOCX/TXT, 6MB limit)
- Server-selected NVIDIA model control
- 5-column image result grid with description, license count, score
- 🤖 Agent Workflow panel — expandable step-by-step trace with OpenSearch payload viewer
- 🎯 Reflection Reranking Log panel — collapsible decision table showing rank, score, keep/discard verdict, reason, and confidence for every candidate
- Staged reranking progress indicator while a trigger-phrase request is pending
- Error handling & toast notifications

✅ **Infrastructure**
- Docker setup
- Docker Compose for local development

## Start the Application

### Option 1: Native Development (Recommended)

**Terminal 1 – Backend:**
```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```
Backend runs at: http://localhost:8000

**Terminal 2 – Frontend:**
```bash
cd frontend
npm install
npm run dev
```
Frontend runs at: http://localhost:5173

### Option 2: Docker Compose

```bash
docker-compose up
```
Access at: http://localhost:5173

### Option 3: Setup Script

```bash
./setup.sh
```

## Test the Application

1. Open http://localhost:5173
2. **Basic search:** type `Find outdoor nature photos` — results appear in a 5-column grid
3. **Reflection reranking:** type `Show me the best ocean sunset photos` — the loading bubble shows staged reflection-reranking progress with elapsed time, results display with a `🎯 Reranked` badge, and a collapsible Reflection Reranking Log appears below the workflow panel
4. **Brief upload:** attach a PDF/DOCX brief, type a short query — the Project Manager extracts requirements before searching
5. **Exclusion phrases:** try `mountain photos without people` — text exclusions are applied against `title`, `description`, and `tags`; older orientation/recency filters are ignored by the current `icc_images_ext` index
6. Click `🤖 Agent Workflow` to inspect each agent's reasoning, input/output, and OpenSearch payloads

Direct image search requires the configured CLIP text model and the PCA model used to project embeddings for `icc_images_ext`. By default the app looks for `ipca_10m.npz` at the repo root and CLIP weights under `SEARCHBYBRIEF_RETRIEVER_CLIP_DOWNLOAD_ROOT` (`/tmp/clip` by default).

### Reflection Reranking trigger phrases

```
best           → "show me the best travel photos"
top ranked     → "top ranked nature images"
top-ranked     → "top-ranked sunset shots"
rerank         → "rerank my results"
reviewed       → "reviewed picks only"
reflect and respond  → "reflect and respond with the most relevant images"
```

## Verify Backend Health

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{"status": "healthy", "opensearch": "connected", "environment": "development"}
```

## Test API Directly

**Standard search:**
```bash
curl -X POST http://localhost:8000/api/chat \
  -F "message=Find outdoor adventure photos"
```

**Trigger reranking:**
```bash
curl -X POST http://localhost:8000/api/chat \
  -F "message=Show me the best mountain landscape photos"
```

**Get recent conversations:**
```bash
curl http://localhost:8000/api/conversations/recent
```

## Troubleshooting

**Backend won't start:**
- Check Python version: `python3 --version` (need 3.11+)
- Check if port 8000 is available: `lsof -i :8000`
- Check OpenSearch connectivity via health endpoint

**Frontend won't start:**
- Check Node version: `node --version` (need 18+)
- Check if port 5173 is available: `lsof -i :5173`
- If dependencies are missing, install from the checked-in lockfile with `npm install`

**OpenSearch connection fails:**
- Verify you're on the internal network
- Test the configured endpoint and credentials with the backend health endpoint first: `curl http://localhost:8000/health`

**Reranking not triggering:**
- Ensure your query contains a trigger phrase (see table above)
- Check backend logs for `Reranker (text-only)` or `Reranker (brief)` log lines
- Verify `NVIDIA_API_KEY` is configured — reranker uses the same server-side key as the agents

## Development Tips

- Backend hot reload: `uvicorn ... --reload` picks up `.py` file changes automatically
- Frontend hot reload: Vite updates the browser instantly on `.jsx`/`.css` changes
- Reranker thresholds are all configurable in `backend/.env` (see README for variable names)
- Reranker model is configurable via `RERANK_MODEL` (default: `meta/llama-3.2-3b-instruct`) and capped by `RERANK_TIMEOUT_SECONDS` (default: `120`)
- View OpenSearch payloads live in the Agent Workflow panel in the UI

## File Structure

```
gen-aperture/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py                  ← NVIDIA, OpenSearch, reranker settings
│   │   ├── routers/
│   │   │   ├── chat.py
│   │   │   └── conversations.py
│   │   ├── services/
│   │   │   ├── agent_squad.py         ← LangGraph multi-agent pipeline
│   │   │   ├── reranker.py            ← Reflection reranker service
│   │   │   ├── photo_search.py        ← direct `icc_images_ext` hybrid query builder
│   │   │   ├── search_service_mcp.py  ← legacy/optional text relevance helper
│   │   │   ├── query_refinement.py
│   │   │   ├── category_filter.py
│   │   │   ├── file_extractor.py
│   │   │   ├── conversation_store.py
│   │   │   └── opensearch_guardrails.py
│   │   └── models/schemas.py          ← includes RerankerDecision schema
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx                    ← RerankLogPanel + AgentWorkflowPanel
│   │   ├── services/api.js
│   │   └── index.css
│   └── package.json
├── briefs/                            ← sample creative briefs
├── design.md
├── README.md
└── QUICKSTART.md                      ← this file
```

---

**Status**: Phase 1 Complete - Ready to Run! ✅
