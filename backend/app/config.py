"""Application configuration"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # OpenSearch
    opensearch_endpoint: str = "http://nelson-v1-prod.sstk-search-prod.ct.shuttercloud.org"
    opensearch_photo_index: str = "web-index-v9"
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
    
    # File upload
    max_file_size_bytes: int = 1 * 1024 * 1024  # 1MB
    allowed_file_types: list[str] = ["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "text/plain"]
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
