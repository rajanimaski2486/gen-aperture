"""Session manager for API key storage with 30-minute timeout"""
from datetime import datetime, timedelta
from typing import Optional, Dict
import logging

from app.config import settings

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages user sessions with OpenAI API keys in memory"""
    
    def __init__(self):
        self._sessions: Dict[str, dict] = {}
    
    def create_session(self, conversation_id: str, api_key: str) -> None:
        """Create or update a session with API key"""
        now = datetime.utcnow()
        expires_at = now + timedelta(minutes=settings.session_timeout_minutes)
        
        self._sessions[conversation_id] = {
            "api_key": api_key,
            "created_at": now,
            "last_activity": now,
            "expires_at": expires_at
        }
        
        logger.info(f"Session created for conversation {conversation_id[:8]}...")
    
    def get_api_key(self, conversation_id: str) -> Optional[str]:
        """Get API key for conversation, returns None if expired or not found"""
        session = self._sessions.get(conversation_id)
        
        if not session:
            return None
        
        now = datetime.utcnow()
        
        # Check if expired
        if now > session["expires_at"]:
            logger.info(f"Session expired for conversation {conversation_id[:8]}...")
            self.delete_session(conversation_id)
            return None
        
        # Update last activity and extend expiry
        session["last_activity"] = now
        session["expires_at"] = now + timedelta(minutes=settings.session_timeout_minutes)
        
        return session["api_key"]
    
    def delete_session(self, conversation_id: str) -> None:
        """Delete a session and remove API key from memory"""
        if conversation_id in self._sessions:
            del self._sessions[conversation_id]
            logger.info(f"Session deleted for conversation {conversation_id[:8]}...")
    
    def cleanup_expired_sessions(self) -> int:
        """Remove all expired sessions, returns count of deleted sessions"""
        now = datetime.utcnow()
        expired = [
            conv_id for conv_id, session in self._sessions.items()
            if now > session["expires_at"]
        ]
        
        for conv_id in expired:
            self.delete_session(conv_id)
        
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired sessions")
        
        return len(expired)


# Global session manager instance
session_manager = SessionManager()
