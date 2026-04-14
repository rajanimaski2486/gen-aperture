"""Application configuration"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # OpenSearch — photo search (read-only production cluster)
    opensearch_endpoint: str = "http://nelson-v1-prod.sstk-search-prod.ct.shuttercloud.org"
    opensearch_photo_index: str = "web-index-v9"

    # OpenSearch — conversation storage (separate writable cluster)
    opensearch_conversation_endpoint: str = "http://mmr-test-v1-prod.sstk-search-prod.ct.shuttercloud.org"
    opensearch_conversation_index: str = "gen-aperture-conversations"

    # OpenSearch guardrails
    # If set, forces read-only mode on/off regardless of endpoint host.
    # If unset, read-only is enabled automatically for any host in
    # opensearch_readonly_hosts.
    opensearch_readonly: Optional[bool] = None
    opensearch_readonly_hosts: list[str] = [
        "nelson-v1-prod.sstk-search-prod.ct.shuttercloud.org",
    ]
    
    # Session
    session_timeout_minutes: int = 30
    
    # Environment
    environment: str = "development"

    # Reflection Reranker
    # Minimum number of results the reranker should try to return
    rerank_min_results_target: int = 10
    # Normalised score (0-10) below which a result is considered a poor match
    rerank_relevance_threshold: float = 5.0
    # Scores in [borderline_threshold, relevance_threshold) may be promoted to hit min_results_target
    rerank_borderline_threshold: float = 3.5
    # Jaccard similarity above which two results are treated as near-duplicates
    rerank_duplicate_similarity_threshold: float = 0.5
    
    # Bifrost AI gateway (internal OpenAI-compatible proxy)
    bifrost_api_key: str
    bifrost_base_url: str = "https://bifrost.shuttercorp.net/openai"
    bifrost_model: str = "gpt-4.1"

    # File upload
    max_file_size_bytes: int = 1 * 1024 * 1024  # 1MB
    allowed_file_types: list[str] = ["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "text/plain"]
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
