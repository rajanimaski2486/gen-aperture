# 🚀 Quick Start Guide

## What's Been Built

✅ **Backend (FastAPI)**
- Multi-agent LangGraph orchestration (Squad Router, Project Manager, Search Specialist, Synthesizer)
- Reflection Reranker — post-retrieval 3-pass LLM scoring, critique, and filtering
- Hybrid neural + lexical OpenSearch queries via Search Service MCP
- Session manager (API key handling, 30-min timeout)
- OpenSearch conversation store with 7-day retention
- PDF/DOCX/TXT file extraction for brief analysis
- Category mapping, query refinement, and exclusion filtering
- OpenSearch guardrails (read-only enforcement on production clusters)

✅ **Frontend (React + Vite)**
- Chat interface with multi-turn conversation context
- Sidebar with last 5 conversation history entries
- File upload (PDF/DOCX/TXT, 1MB limit)
- API key modal with session storage
- 5-column image result grid with description, license count, score
- 🤖 Agent Workflow panel — expandable step-by-step trace with OpenSearch payload viewer
- 🎯 Reflection Reranking Log panel — collapsible decision table showing rank, score, keep/discard verdict, reason, and confidence for every candidate
- Reranking loading indicator when trigger phrase is detected
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
2. Enter your OpenAI API key when the modal appears
3. **Basic search:** type `Find outdoor nature photos` — results appear in a 5-column grid
4. **Reflection reranking:** type `Show me the best ocean sunset photos` — the loading bubble shows `🔄 Applying reflection reranking…`, results display with a `🎯 Reranked` badge, and a collapsible Reflection Reranking Log appears below the workflow panel
5. **Brief upload:** attach a PDF/DOCX brief, type a short query — the Project Manager extracts requirements before searching
6. **Filter phrases:** try `horizontal images of mountains from the last year` — orientation and recency filters are applied automatically
7. Click `🤖 Agent Workflow` to inspect each agent's reasoning, input/output, and OpenSearch payloads

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
  -F "message=Find outdoor adventure photos" \
  -F "openai_api_key=sk-..."
```

**Trigger reranking:**
```bash
curl -X POST http://localhost:8000/api/chat \
  -F "message=Show me the best mountain landscape photos" \
  -F "openai_api_key=sk-..."
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
- Clear node_modules: `rm -rf node_modules && npm install`

**OpenSearch connection fails:**
- Verify you're on the internal ShutterCorp network
- Test directly: `curl http://nelson-v1-prod.sstk-search-prod.ct.shuttercloud.org/_cluster/health`

**Reranking not triggering:**
- Ensure your query contains a trigger phrase (see table above)
- Check backend logs for `Reranker (text-only)` or `Reranker (brief)` log lines
- Verify `OPENAI_API_KEY` is valid — reranker uses the same key as the agents

## Development Tips

- Backend hot reload: `uvicorn ... --reload` picks up `.py` file changes automatically
- Frontend hot reload: Vite updates the browser instantly on `.jsx`/`.css` changes
- Reranker thresholds are all configurable in `backend/.env` (see README for variable names)
- View OpenSearch payloads live in the Agent Workflow panel in the UI

## File Structure

```
gen-aperture/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py                  ← includes reranker thresholds
│   │   ├── routers/
│   │   │   ├── chat.py
│   │   │   └── conversations.py
│   │   ├── services/
│   │   │   ├── agent_squad.py         ← LangGraph multi-agent pipeline
│   │   │   ├── reranker.py            ← Reflection reranker service
│   │   │   ├── photo_search.py
│   │   │   ├── search_service_mcp.py
│   │   │   ├── query_refinement.py
│   │   │   ├── category_filter.py
│   │   │   ├── file_extractor.py
│   │   │   ├── session_manager.py
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
