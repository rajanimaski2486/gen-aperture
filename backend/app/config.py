"""Application configuration"""
from pydantic import model_validator
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # OpenSearch — photo search (read-only production cluster)
    opensearch_endpoint: str = "http://localhost:9200"
    opensearch_photo_index: str = "icc_images_ext"
    opensearch_username: Optional[str] = None
    opensearch_password: Optional[str] = None

    # OpenSearch — conversation storage. Defaults to opensearch_endpoint so the
    # same domain can be used with narrow write guardrails.
    opensearch_conversation_endpoint: Optional[str] = None
    opensearch_conversation_index: str = "gen-aperture-conversations"
    opensearch_conversation_max_records: int = 5000
    opensearch_conversation_max_store_bytes: int = 5 * 1024 * 1024 * 1024
    opensearch_hybrid_search_pipeline: str = "reveal-hybrid"
    opensearch_vector_field: str = "dense_vector"
    opensearch_knn_k: int = 200
    opensearch_text_embedding_pca_model_path: Optional[str] = None

    # OpenSearch guardrails
    # If set, forces read-only mode on/off regardless of endpoint host.
    # If unset, read-only is enabled automatically for any host in
    # opensearch_readonly_hosts.
    opensearch_readonly: Optional[bool] = None
    opensearch_readonly_hosts: list[str] = [
        "localhost",
    ]
    
    # Session
    session_timeout_minutes: int = 30
    
    # Environment
    environment: str = "development"

    # NVIDIA NIM LLMs. NVIDIA exposes OpenAI-compatible chat completions, so the
    # existing OpenAI SDK/LangChain client can be pointed at the NVIDIA base URL.
    nvidia_api_key: Optional[str] = None
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    agent_model: str = "meta/llama-3.3-70b-instruct"
    agent_model_base_url: Optional[str] = None
    agent_fallback_model: Optional[str] = None
    image_analysis_model: str = "meta/llama-3.2-11b-vision-instruct"

    # Reflection Reranker
    # Minimum number of results the reranker should try to return
    rerank_min_results_target: int = 10
    # Normalised score (0-10) below which a result is considered a poor match
    rerank_relevance_threshold: float = 3.0
    # Scores in [borderline_threshold, relevance_threshold) may be promoted to hit min_results_target
    rerank_borderline_threshold: float = 2.0
    # Jaccard similarity above which two results are treated as near-duplicates
    rerank_duplicate_similarity_threshold: float = 0.5

    # SearchByBrief Stage 1 retriever
    # Modes:
    # - "embedding": CLIP + PCA + creativeImageSearchByEmbedding
    # - "text_relevance": Search Service MCP relevance query (text-only)
    # - "text-intent": Search Intent API GraphQL recommendations endpoint
    searchbybrief_retriever_mode: str = "text-intent"
    searchbybrief_retriever_endpoint: str = (
        "http://localhost:8081/graphql"
    )
    searchbybrief_retriever_collection_type: str = "APPROVED_V1"
    searchbybrief_retriever_top_k_per_lane: int = 500
    searchbybrief_retriever_use_pca: bool = True
    # Optional explicit override; when unset retriever falls back to repo ipca_10m.pkl
    searchbybrief_retriever_pca_model_path: Optional[str] = None
    searchbybrief_retriever_clip_model: str = "ViT-B/32"
    searchbybrief_retriever_clip_device: Optional[str] = None
    searchbybrief_retriever_clip_download_root: str = "/tmp/clip"
    searchbybrief_retriever_normalize_embeddings: bool = True
    searchbybrief_retriever_truncate_text: bool = False
    searchbybrief_retriever_timeout_seconds: int = 60
    searchbybrief_search_intent_endpoint: str = (
        "http://localhost:8082/graphql"
    )
    searchbybrief_search_intent_client_name: str = (
        "gen-aperture/search-results-page/retriever"
    )
    searchbybrief_search_intent_client_version: str = "1.0.0"
    # SearchByBrief Stage 0 planner
    # "v1" = full schema, "v2" = compact lanes-first output
    searchbybrief_planner_version: str = "v2"
    searchbybrief_planner_max_tokens_v1: int = 2500
    searchbybrief_planner_max_tokens_v2: int = 900
    # SearchByBrief Stage 3 curator
    # Number of parallel visual-scoring calls.
    searchbybrief_curator_concurrency: int = 6
    # Token caps for Stage 3 vision calls (lower values reduce latency).
    searchbybrief_curator_visual_max_tokens: int = 420
    searchbybrief_curator_set_audit_max_tokens: int = 560
    # Optional per-call sleep (seconds) between Stage 3 LLM calls.
    searchbybrief_curator_sleep_between_calls: float = 0.0
    # Max candidates that receive expensive visual scoring.
    searchbybrief_curator_max_visual_scoring_candidates: int = 30
    # Per-lane thumbnail count used in set-level audit.
    searchbybrief_curator_audit_top_per_lane: int = 6
    # Soft penalty applied per prior pick from the same lane during shortlist
    # interleaving. Higher => more diversity, lower => stronger score dominance.
    searchbybrief_curator_diversity_penalty: float = 0.4

    # Model used by the reflection reranker LLM passes
    rerank_model: str = "meta/llama-3.3-70b-instruct"

    # SearchByBrief planner/curator model.
    searchbybrief_model: str = "meta/llama-3.3-70b-instruct"

    # Legacy Bifrost names retained for older env files.
    bifrost_api_key: str | None = None
    bifrost_base_url: str = "https://integrate.api.nvidia.com/v1"
    bifrost_model: str = "meta/llama-3.3-70b-instruct"

    # File upload
    max_file_size_bytes: int = 6 * 1024 * 1024  # 6MB
    allowed_file_types: list[str] = ["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "text/plain"]
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"

    @model_validator(mode="after")
    def apply_dynamic_defaults(self) -> "Settings":
        if not self.opensearch_conversation_endpoint:
            self.opensearch_conversation_endpoint = self.opensearch_endpoint
        return self

    @property
    def llm_base_url(self) -> str:
        return self.agent_model_base_url or self.nvidia_base_url

    def require_nvidia_api_key(self) -> str:
        if not self.nvidia_api_key:
            raise RuntimeError("NVIDIA_API_KEY is not configured")
        return self.nvidia_api_key


# Global settings instance
settings = Settings()
