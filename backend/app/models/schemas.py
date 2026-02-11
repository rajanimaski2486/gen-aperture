"""Pydantic schemas for API requests/responses"""
from pydantic import BaseModel, Field
from typing import Optional, List, Any, Dict
from datetime import datetime


class ChatRequest(BaseModel):
    """Chat request from frontend"""
    message: str = Field(..., min_length=1, max_length=2000)
    conversation_id: Optional[str] = None
    openai_api_key: Optional[str] = Field(None, description="Required for new conversations or expired sessions")


class PhotoResult(BaseModel):
    """Photo search result"""
    hadron_id: Optional[str] = None
    ext_id: Optional[int] = None
    description: str = ""
    image_url: str = ""
    thumbnail_url: str = ""
    date_added: Optional[str] = None
    license_count: int = 0
    categories: List[Any] = []
    keywords: List[str] = []
    score: float = 0.0


class AgentWorkflowStep(BaseModel):
    """A single step in the agent workflow trace"""
    agent: str
    action: str
    reasoning: str = ""
    prompt: Optional[str] = None
    input: Optional[Dict[str, Any]] = None
    output: Optional[Dict[str, Any]] = None
    decision: Optional[str] = None
    opensearch_payload: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    """Chat response to frontend"""
    conversation_id: str
    response: str
    results: List[PhotoResult] = []
    api_key_valid: bool = True
    processing_time_ms: int = 0
    workflow_steps: List[AgentWorkflowStep] = []
    search_mode: str = "relevance"  # 'relevance' or 'popular'


class ConversationMessage(BaseModel):
    """Single message in a conversation"""
    message_number: int
    timestamp: datetime
    user_message: str
    agent_response: str
    search_results_count: int
    processing_time_ms: int


class ConversationDetail(BaseModel):
    """Full conversation details"""
    conversation_id: str
    created_at: datetime
    last_message_at: datetime
    last_user_query: str
    message_count: int
    file_name: Optional[str] = None
    messages: List[ConversationMessage]


class ConversationPreview(BaseModel):
    """Conversation preview for sidebar"""
    conversation_id: str
    last_user_query: str
    last_message_at: datetime
    message_count: int


class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: Optional[str] = None
