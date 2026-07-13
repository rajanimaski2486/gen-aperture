# Gen-Aperture Design Review - APPROVED ✅

## Summary of Decisions (2026-02-05)

### Architecture Finalized
- ✅ **Multi-agent with LangGraph** (Project Manager + Search Specialist)
- ✅ **Single Python deployment** (FastAPI serves both API and React static files)
- ✅ **User-provided OpenAI API keys** (30-min session memory, never persisted)
- ✅ **Direct OpenSearch integration** (no MCP for MVP)

### Data Specifications

#### Photo Index (web-index-v9)
- **Fields used:**
  - `hadron_id` - Unique identifier (display)
  - `ext_id` - Image ID (for URL construction)
  - `description_en` - Description (display)
  - `date_added` - Upload date (display)
  - `total_paid_license_count_all_time` - Impressions (display)
  - `keywords_en` - Search field
  - `media_type`, `is_photo` - Filters

- **Image URL pattern:**
  ```
  http://localhost:9200/assets/image-250nw-{ext_id}.jpg
  ```

#### Conversation Storage
- **Strategy:** Store `file_content` at conversation level (not per-message)
- **Index:** `gen-aperture-conversations` (auto-created on startup)
- **Retention:** 7 days with ISM policy
- **Structure:**
  ```json
  {
    "conversation_id": "uuid",
    "created_at": "timestamp",
    "last_message_at": "timestamp",
    "last_user_query": "text",
    "message_count": 5,
    "file_name": "brief.pdf",
    "file_content": "extracted text (stored once)",
    "messages": [
      {
        "message_number": 1,
        "timestamp": "...",
        "user_message": "...",
        "agent_response": "...",
        "search_results_count": 15,
        "processing_time_ms": 3200
      }
    ]
  }
  ```

### User Experience

#### Session Management
- **New Chat button** - Creates fresh conversation
- **Sidebar** - Shows last 5 conversations
- **Preview** - Each conversation shows last user query
- **Limit** - 5 conversations visible (older hidden but stored for 7 days)

#### File Upload (Option C)
1. User selects file (drag-drop or picker)
2. Frontend validates: <1MB, PDF/DOCX/TXT
3. Shows preview: "brief.pdf (842 KB) - Ready to upload"
4. User types message and hits Send
5. Backend extracts text on submission
6. Stored at conversation level for all subsequent turns

#### API Key Flow
1. User opens app → Modal: "Enter your OpenAI API key"
2. Key sent with first message
3. Backend stores in session memory (30-min timeout)
4. Timeout → Key deleted, modal reappears
5. **Never stored in database, env vars, or logs**

### Error Handling

| Error | HTTP Code | User Message |
|-------|-----------|--------------|
| OpenAI API down | 503 | "AI service temporarily unavailable" |
| Invalid API key | 401 | "Please enter a valid OpenAI API key" |
| OpenSearch down | 503 | "Photo search temporarily unavailable" |
| File too large | 413 | "File exceeds 1MB limit" |
| File read error | 400 | "Could not read file, try another" |
| No results | 200 | Agent: "No matches, try refining" |
| Session expired | 401 | "Session expired, please re-enter key" |

**Graceful degradation:** If context load fails, proceed with current message only (log warning).

### Search Strategy

#### Query Construction
```python
{
  "query": {
    "bool": {
      "must": [
        {
          "multi_match": {
            "query": "user's natural language query",
            "fields": ["description_en^2", "keywords_en^1.5"],
            "type": "best_fields"
          }
        }
      ],
      "filter": [
        {"term": {"media_type": "photo"}},
        {"term": {"is_photo": true}}
      ]
    }
  },
  "sort": [
    {"engagement_score": {"order": "desc"}},
    "_score"
  ],
  "size": 15
}
```

### Agent Architecture

#### Flow
```
User Message
  ↓
Squad Router (conditional logic)
  ↓
  ├─→ [File present] → Project Manager Agent
  │                      ↓
  │                    Extract requirements
  │                      ↓
  └─→ [Text only] ───→ Search Specialist Agent
                         ↓
                    OpenSearch Tool
                         ↓
                    Format Results
                         ↓
                    User Response
```

#### Project Manager Agent
- **Role:** Requirements analyst
- **Input:** Uploaded document text
- **Output:** Structured search requirements
- **Tools:** None (analysis only)

#### Search Specialist Agent
- **Role:** Database expert
- **Input:** Requirements OR direct query + conversation history
- **Output:** Formatted search results
- **Tools:** OpenSearch query tool

### Development Timeline

**Week 1: Foundation**
- FastAPI skeleton + static serving
- React UI (chat interface, file upload, sidebar)
- OpenSearch connection + index creation
- Basic echo endpoint

**Week 2: Agent Integration**
- LangGraph multi-agent setup
- OpenSearch search tool
- File extraction (pypdf, python-docx)
- Conversation storage/retrieval
- API key session management

**Week 3: Polish & Deploy**
- Multi-turn context handling
- Error handling + user feedback
- UI polish (loading states, error toasts)
- Backstage deployment
- 7-day ISM policy setup

**Week 4: Monitoring (if needed)**
- Usage analytics
- Performance tuning
- Agent prompt optimization

### Technical Specs

**Performance:**
- 10 RPS sustained
- 2-4s latency (text only)
- 5-8s latency (with file)
- 30-min session timeout

**Capacity:**
- Single FastAPI instance (4 cores)
- Handles 80-100 concurrent requests
- ~$25/day LLM costs (GPT-4o-mini)
- 21GB conversation data (7-day retention)

### Security

- ✅ Internal-only deployment (Internal network)
- ✅ No persistent API key storage
- ✅ 30-min session timeout with auto-cleanup
- ✅ File size limits (1MB)
- ✅ MIME type validation
- ✅ No PII collection

### Deployment

**Platform:** Backstage FastAPI template  
**Repository:** TBD - to be created  
**Endpoint:** http://localhost:8000 (example)  

**Environment:**
- `OPENSEARCH_ENDPOINT` - Hardcoded or env var
- `SESSION_TIMEOUT_MINUTES` - Default: 30
- **NO** `OPENAI_API_KEY` - Users provide their own

**Health Check:**
```
GET /health
{
  "status": "healthy",
  "opensearch": "connected"
}
```

---

## Next Steps

1. ✅ **Design approved** - All questions answered
2. 🚀 **Ready to scaffold** - Generate full project structure
3. 📝 **Implementation begins** - Week 1 foundation

**Waiting for your confirmation to start building!** 🎉

---

## Files Updated
- ✅ `DESIGN.md` - v2.0 with all specifications
- ✅ `.github/copilot-instructions.md` - Updated for API key management
- ✅ `REVIEW.md` - This summary document

**Status:** ✅ APPROVED - Ready for Implementation
