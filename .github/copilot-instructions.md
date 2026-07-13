# Gen-Aperture: Agentic Stock Photo Conversational Search

## Architecture Overview

This is a **hybrid Node.js/Python system** with clear separation:
- **Frontend**: React + Vite (Node.js) - handles UI, file uploads, chat interface
- **Backend**: FastAPI (Python) - orchestrates AI agents, handles file extraction, proxies to OpenSearch
- **AI Layer**: AWS AgentCore (Strands & Squads) using GPT-4o-mini for reasoning

### Core Data Flow
```
User (React UI) → FastAPI endpoint → Agent Squad Router → 
  → [Project Manager Strand] → [Search Specialist Strand] → 
  → OpenSearch MCP Tool → AWS OpenSearch Domain → Response back up chain
```

## Key Architectural Patterns

### Agent Squad Structure
- **Squad Router**: GPT-4o-mini-powered orchestrator that routes based on input type
  - File present → Project Manager Strand
  - Text only → Search Specialist Strand (direct)
  
- **Project Manager Strand**: Analyzes uploaded briefs (PDF/DOCX/TXT), extracts visual requirements, themes, moods, constraints. Does NOT search - only requirements extraction.

- **Search Specialist Strand**: Executes technical searches via OpenSearch MCP Tool, interprets JSON results, formats responses for users.

### Model Context Protocol (MCP) Integration
Direct OpenSearch client integration (no MCP server needed for MVP). Future: Can add MCP if other tools need integration.

## Development Workflows

### Backend (Python)
- FastAPI handles `/chat` endpoint with multipart/form-data for file uploads
- Use `pypdf` or `textract` for text extraction from uploaded documents
- Agent Squad logic lives in backend, not frontend

### Frontend (React)
- Vite tooling for dev server and builds
- Multipart POST requests to backend `/chat` endpoint
- Local state management for current session history (sidebar)
- File upload: drag-and-drop or file picker for PDF/DOCX/TXT

## Critical Conventions

### API Key Security ⚠️
- **NEVER store OpenAI API keys in code, environment variables, or databases**
- Users provide their own API keys via UI modal
- Keys stored in backend session memory only (30-min timeout)
- Auto-deleted on inactivity - must be re-entered

### Out of Scope (Do Not Implement)
- User authentication or persistent storage (no database for user accounts)
- Modifications to OpenSearch photo index structure or data ingestion
- Payment processing or image licensing workflows
- Storing API keys anywhere persistent

### Agent Responsibilities
- **Project Manager**: Requirements analysis ONLY - never searches OpenSearch directly
- **Search Specialist**: Database interaction ONLY - receives refined queries, returns results
- Clear handoff: PM Strand → Search Specialist Strand → MCP Tool → OpenSearch

## External Dependencies

- **OpenAI API**: GPT-4o-mini for all agent reasoning (**user-provided keys**, 30-min sessions)
- **AWS OpenSearch Domain**: Pre-existing, read-only access via direct HTTP
  - Endpoint: `http://localhost:9200`
  - Photo index: `web-index-v9`
  - Conversation index: `gen-aperture-conversations` (auto-created)
- **LangGraph**: Multi-agent orchestration framework (not AWS AgentCore)

## Key Files (When Implemented)
- Frontend: React components for chat UI, file upload, session history sidebar
- Backend: FastAPI app with `/chat` endpoint, file extraction utilities
- Agent definitions: Squad router logic, Project Manager Strand, Search Specialist Strand
- MCP configuration: OpenSearch MCP server setup and tool definitions
