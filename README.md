# Gen-Aperture: Agentic Stock Photo Conversational Search

AI-powered conversational interface for searching stock photos using natural language queries and document analysis.

## Features

- 🤖 **Multi-Agent AI** powered by LangGraph
- 💬 **Natural language search** with conversation context
- 📄 **Document upload** (PDF/DOCX/TXT) for context-aware searching
- 🔐 **User-provided API keys** (30-min session, never stored)
- 📊 **Conversation history** with 7-day retention

## Architecture

- **Frontend**: React 18 + Vite
- **Backend**: FastAPI (Python 3.11+)
- **Agents**: LangGraph with OpenAI GPT-4o-mini
- **Storage**: OpenSearch 3.3 (conversations + photo index)

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
cp backend/.env.example backend/.env
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

4. **Open browser:**
- Go to http://localhost:5173
- Enter your OpenAI API key when prompted
- Start chatting!

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
│   │   ├── main.py              # FastAPI application
│   │   ├── config.py            # Settings
│   │   ├── routers/             # API endpoints
│   │   │   ├── chat.py          # Chat endpoint
│   │   │   └── conversations.py # Conversation management
│   │   ├── services/            # Business logic
│   │   │   ├── session_manager.py
│   │   │   └── conversation_store.py
│   │   └── models/              # Data models
│   │       └── schemas.py
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx              # Main React component
│   │   ├── services/api.js      # API client
│   │   └── index.css            # Styles
│   └── package.json
├── DESIGN.md                     # Technical specification
├── .github/
│   └── copilot-instructions.md  # AI agent guidance
└── docker-compose.yml
```

## API Endpoints

### POST /api/chat
Send message and get AI response
```bash
curl -X POST http://localhost:8000/api/chat \
  -F "message=Find outdoor photos" \
  -F "openai_api_key=sk-..." \
  -F "file=@brief.pdf"
```

### GET /api/conversations/recent
List last 5 conversations

### GET /api/conversations/{id}
Get full conversation details

### GET /health
Health check endpoint

## Configuration

Backend environment variables (`.env`):
```
OPENSEARCH_ENDPOINT=http://mmr-test-v1-prod.sstk-search-prod.ct.shuttercloud.org
OPENSEARCH_PHOTO_INDEX=web-index-v9
OPENSEARCH_CONVERSATION_INDEX=gen-aperture-conversations
OPENSEARCH_READONLY=true
SESSION_TIMEOUT_MINUTES=30
ENVIRONMENT=development
```

## Development Phases

**Phase 1 (Current): Foundation** ✅
- Backend skeleton
- Basic React UI
- OpenSearch connection
- Session management

**Phase 2 (Next): Agents**
- LangGraph multi-agent setup
- File extraction
- OpenSearch search tool
- Photo result formatting

**Phase 3: Polish**
- Error handling
- UI improvements
- Deployment setup

## Security

- ⚠️ **API keys never stored on server** - users provide their own
- Session timeout: 30 minutes of inactivity
- Internal-only deployment (ShutterCorp network)
- 1MB file upload limit
- 7-day conversation retention

## Deployment

Uses Backstage FastAPI template:
```
https://backstage.shuttercorp.net/create/templates/default/add-gha-app-fastapi
```

## Documentation

- [DESIGN.md](DESIGN.md) - Complete technical specification
- [.github/copilot-instructions.md](.github/copilot-instructions.md) - AI coding guidelines
- [REVIEW.md](REVIEW.md) - Design review summary

## Support

Internal ShutterCorp project  
Team: Search Platform  
OpenSearch: `mmr-test-v1-prod.sstk-search-prod.ct.shuttercloud.org`

---

**Status**: Phase 1 Complete - Foundation Ready ✅
