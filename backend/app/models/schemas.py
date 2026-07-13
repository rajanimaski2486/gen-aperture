"""Pydantic schemas for API requests/responses"""
from pydantic import BaseModel, Field
from typing import Optional, List, Any, Dict
from datetime import datetime


class ChatRequest(BaseModel):
    """Chat request from frontend"""
    message: str = Field(..., min_length=1, max_length=2000)
    conversation_id: Optional[str] = None
    openai_api_key: Optional[str] = Field(None, description="Deprecated; server uses NVIDIA_API_KEY")


class PhotoResult(BaseModel):
    """Photo or video search result"""
    hadron_id: Optional[str] = None
    ext_id: Optional[int] = None
    description: str = ""
    image_url: str = ""
    thumbnail_url: str = ""
    video_url: str = ""          # MP4 preview URL (populated for media_type=video only)
    media_type: str = "image"    # "image" or "video"
    date_added: Optional[str] = None
    license_count: int = 0
    categories: List[Any] = []
    keywords: List[str] = []
    score: float = 0.0
    is_generated: bool = False


class AgentWorkflowStep(BaseModel):
    """A single step in the agent workflow trace"""
    agent: str
    action: str
    reasoning: str = ""
    model: Optional[str] = None  # Model used by this agent
    prompt: Optional[str] = None
    input: Optional[Dict[str, Any]] = None
    output: Optional[Dict[str, Any]] = None
    decision: Optional[str] = None
    opensearch_payload: Optional[Dict[str, Any]] = None
    opensearch_url: Optional[str] = None
    search_service_endpoint: Optional[str] = None
    search_service_response: Optional[Dict[str, Any]] = None


class RerankerDecision(BaseModel):
    """Per-candidate decision from the reflection reranker"""
    final_rank: Optional[int] = None           # 1-based rank in final kept set; None if discarded
    hadron_id: Optional[str] = None
    ext_id: Optional[Any] = None
    rerank_score: float = 0.0                  # Normalised 0-1 score
    keep: bool = True                          # Whether this result is included in the final set
    is_borderline: bool = False                # True when promoted only to reach min_results_target
    reason: str = ""                           # Short explanation
    matched_criteria: List[str] = []           # Which search criteria this satisfies
    confidence: float = 0.0                    # 0-1 confidence in the keep/discard decision


class ChatResponse(BaseModel):
    """Chat response to frontend"""
    conversation_id: str
    response: str
    results: List[PhotoResult] = []
    filter_metadata: Optional[Dict[str, Any]] = None
    pdf_search_detail: Optional[Dict[str, Any]] = None
    api_key_valid: bool = True
    processing_time_ms: int = 0
    workflow_steps: List[AgentWorkflowStep] = []
    search_mode: str = "relevance"  # 'relevance' or 'popular'
    # Reranker output fields (populated only when reflection reranking was triggered)
    rerank_applied: bool = False
    rerank_decisions: Optional[List[RerankerDecision]] = None
    rerank_explanation: Optional[str] = None


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
    title: Optional[str] = None
    message_count: int
    file_name: Optional[str] = None
    messages: List[ConversationMessage]


class ConversationPreview(BaseModel):
    """Conversation preview for sidebar"""
    conversation_id: str
    last_user_query: str
    title: Optional[str] = None
    last_message_at: datetime
    message_count: int


class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: Optional[str] = None
