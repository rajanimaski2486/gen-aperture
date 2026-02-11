# 🚀 Quick Start Guide - Phase 1

## What's Been Built

✅ **Backend (FastAPI)**
- Session manager (API key handling, 30-min timeout)
- OpenSearch conversation store
- Chat endpoint (echo mode for Phase 1)
- Conversation management endpoints
- Health check

✅ **Frontend (React + Vite)**
- Chat interface
- Sidebar with conversation history
- File upload with preview
- API key modal
- Error handling & loading states

✅ **Infrastructure**
- Docker setup
- Docker Compose for local development
- Development scripts

## Start the Application

### Option 1: Native Development (Recommended)

**Terminal 1 - Backend:**
```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Backend will be at: http://localhost:8000

**Terminal 2 - Frontend:**
```bash
cd frontend
npm install
npm run dev
```

Frontend will be at: http://localhost:5173

### Option 2: Docker Compose

```bash
docker-compose up
```

Access at: http://localhost:5173

### Option 3: Setup Script

```bash
./setup.sh
```

Then follow the instructions printed.

## Test the Application

1. Open http://localhost:5173
2. You'll see API key modal - enter any test key (e.g., `sk-test123`)
3. Type a message: "Find outdoor photos"
4. You'll get an echo response (Phase 1 doesn't have agents yet)
5. Try uploading a file (PDF/DOCX/TXT < 1MB)
6. Click "New Chat" to start fresh conversation
7. Check sidebar for conversation history

## Verify Backend Health

```bash
curl http://localhost:8000/health
```

Should return:
```json
{
  "status": "healthy",
  "opensearch": "connected",
  "environment": "development"
}
```

## Test API Directly

**Create conversation:**
```bash
curl -X POST http://localhost:8000/api/chat \
  -F "message=Test message" \
  -F "openai_api_key=sk-test123"
```

**Get recent conversations:**
```bash
curl http://localhost:8000/api/conversations/recent
```

## What Works (Phase 1)

- ✅ Chat UI with message history
- ✅ Conversation persistence in OpenSearch
- ✅ Session management (30-min timeout)
- ✅ File upload validation & preview
- ✅ Sidebar with last 5 conversations
- ✅ New chat functionality
- ✅ API key modal with session storage
- ✅ Error handling & toast notifications
- ✅ Health check endpoint

## What's Next (Phase 2)

- [ ] File extraction (PDF/DOCX/TXT)
- [ ] LangGraph multi-agent setup
- [ ] Project Manager agent
- [ ] Search Specialist agent
- [ ] OpenSearch photo search tool
- [ ] Result formatting with image URLs

## Troubleshooting

**Backend won't start:**
- Check Python version: `python3 --version` (need 3.11+)
- Check if port 8000 is available: `lsof -i :8000`
- Check OpenSearch connectivity: Run health endpoint

**Frontend won't start:**
- Check Node version: `node --version` (need 18+)
- Check if port 5173 is available: `lsof -i :5173`
- Clear node_modules: `rm -rf node_modules && npm install`

**OpenSearch connection fails:**
- Verify you're on internal network
- Test directly: `curl http://mmr-test-v1-prod.sstk-search-prod.ct.shuttercloud.org/_cluster/health`

**Conversation index not created:**
- Check backend logs for errors
- Manually create: See DESIGN.md for index mapping

## Development Tips

**Backend hot reload:**
- Changes to .py files auto-reload
- No need to restart server

**Frontend hot reload:**
- Changes to .jsx/.css files auto-reload
- Browser updates instantly

**View backend logs:**
- Check terminal where uvicorn is running
- Logs show INFO, DEBUG, ERROR messages

**View frontend console:**
- Open browser DevTools (F12)
- Check Console tab for errors

**Debug API calls:**
- Use browser Network tab
- See request/response details

## File Structure

```
gen-aperture/
├── backend/
│   ├── app/
│   │   ├── main.py              ← FastAPI app
│   │   ├── config.py            ← Settings
│   │   ├── routers/             ← API endpoints
│   │   ├── services/            ← Business logic
│   │   └── models/              ← Data schemas
│   ├── requirements.txt
│   └── .env                     ← Configuration
├── frontend/
│   ├── src/
│   │   ├── App.jsx              ← Main component
│   │   ├── services/api.js      ← API client
│   │   └── index.css            ← Styles
│   ├── package.json
│   └── vite.config.js
├── DESIGN.md                     ← Full spec
├── README.md                     ← Overview
└── QUICKSTART.md                 ← This file
```

## Next Steps

Once Phase 1 is working:
1. Move to Phase 2 (agents)
2. Implement file extraction
3. Add LangGraph multi-agent system
4. Connect to OpenSearch photo index
5. Format and display image results

---

**Status**: Phase 1 Complete - Ready to Run! ✅
