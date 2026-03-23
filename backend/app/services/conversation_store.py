"""OpenSearch conversation storage service"""
from opensearchpy import exceptions
from datetime import datetime
from typing import Optional, List, Dict
import uuid
import logging

from app.config import settings
from app.services.opensearch_guardrails import create_opensearch_client, is_readonly_endpoint

logger = logging.getLogger(__name__)


class ConversationStore:
    """Manages conversation storage in OpenSearch"""
    
    def __init__(self):
        self.readonly = is_readonly_endpoint(
            endpoint=settings.opensearch_endpoint,
            forced_readonly=settings.opensearch_readonly,
            readonly_hosts=settings.opensearch_readonly_hosts,
        )

        # Always create a client so health checks and read endpoints can work.
        # In read-only mode, the transport is patched to block any write requests.
        self.client = create_opensearch_client(
            endpoint=settings.opensearch_endpoint,
            readonly=self.readonly,
            timeout_seconds=10.0,
        )
        self.index = settings.opensearch_conversation_index

        # In read-only mode, we keep new conversations/messages in memory only.
        self._memory_conversations: Dict[str, Dict] = {}
        # Tombstone set: IDs deleted by the user during this session (used to
        # hide OpenSearch-sourced convos that can't be physically deleted in
        # read-only mode).
        self._deleted_ids: set = set()
    
    async def check_connection(self) -> bool:
        """Check if OpenSearch is accessible"""
        try:
            return self.client.ping()
        except Exception as e:
            logger.error(f"OpenSearch connection failed: {e}")
            return False
    
    async def ensure_index_exists(self) -> None:
        """Create conversation index if it doesn't exist"""
        if self.readonly:
            logger.warning(
                "OpenSearch guardrails: read-only mode enabled; skipping conversation index creation"
            )
            return

        try:
            if self.client.indices.exists(index=self.index):
                logger.info(f"Index {self.index} already exists")
                return
            
            # Create index with mapping
            mapping = {
                "mappings": {
                    "properties": {
                        "conversation_id": {"type": "keyword"},
                        "created_at": {"type": "date"},
                        "last_message_at": {"type": "date"},
                        "last_user_query": {"type": "text"},
                        "title": {"type": "keyword"},
                        "message_count": {"type": "integer"},
                        "file_name": {"type": "keyword"},
                        "file_content": {"type": "text", "index": False},
                        "messages": {
                            "type": "nested",
                            "properties": {
                                "message_number": {"type": "integer"},
                                "timestamp": {"type": "date"},
                                "user_message": {"type": "text"},
                                "agent_response": {"type": "text"},
                                "search_results_count": {"type": "integer"},
                                "processing_time_ms": {"type": "integer"}
                            }
                        }
                    }
                },
                "settings": {
                    "index": {
                        "number_of_shards": 3,
                        "number_of_replicas": 1
                    }
                }
            }
            
            self.client.indices.create(index=self.index, body=mapping)
            logger.info(f"Created index {self.index}")
            
            # Create 7-day retention policy
            await self._create_retention_policy()
            
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            raise
    
    async def _create_retention_policy(self) -> None:
        """Create ISM policy for 7-day retention"""
        policy_name = "7day_retention_policy"
        
        try:
            policy = {
                "policy": {
                    "description": "Delete conversations after 7 days",
                    "default_state": "active",
                    "states": [
                        {
                            "name": "active",
                            "actions": [],
                            "transitions": [
                                {
                                    "state_name": "delete",
                                    "conditions": {
                                        "min_index_age": "7d"
                                    }
                                }
                            ]
                        },
                        {
                            "name": "delete",
                            "actions": [{"delete": {}}]
                        }
                    ]
                }
            }
            
            # Note: This requires ISM plugin, may not work on all OpenSearch versions
            # Gracefully handle if not available
            try:
                self.client.transport.perform_request(
                    'PUT',
                    f'/_plugins/_ism/policies/{policy_name}',
                    body=policy
                )
                logger.info(f"Created ISM policy {policy_name}")
            except Exception as e:
                logger.warning(f"Could not create ISM policy (plugin may not be available): {e}")
                
        except Exception as e:
            logger.warning(f"ISM policy creation failed: {e}")
    
    async def create_conversation(
        self,
        file_name: Optional[str] = None,
        file_content: Optional[str] = None
    ) -> str:
        """Create new conversation, returns conversation_id"""
        conversation_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        
        doc = {
            "conversation_id": conversation_id,
            "created_at": now,
            "last_message_at": now,
            "last_user_query": "",
            "title": None,
            "message_count": 0,
            "file_name": file_name,
            "file_content": file_content,
            "messages": []
        }

        if self.readonly:
            self._memory_conversations[conversation_id] = doc
            logger.info(
                f"Created in-memory conversation (read-only OpenSearch): {conversation_id[:8]}..."
            )
            return conversation_id

        try:
            self.client.index(index=self.index, id=conversation_id, body=doc)
            logger.info(f"Created conversation {conversation_id[:8]}...")
            return conversation_id
        except Exception as e:
            logger.error(f"Failed to create conversation: {e}")
            raise
    
    async def set_title(self, conversation_id: str, title: str) -> None:
        """Set the permanent title for a conversation (generated from first prompt)"""
        if self.readonly:
            conv = self._memory_conversations.get(conversation_id)
            if conv:
                conv["title"] = title
            return

        try:
            self.client.update(
                index=self.index,
                id=conversation_id,
                body={"doc": {"title": title}},
            )
            logger.debug(f"Set title for conversation {conversation_id[:8]}...")
        except Exception as e:
            logger.warning(f"Failed to set conversation title: {e}")

    async def update_file_content(
        self,
        conversation_id: str,
        file_name: Optional[str],
        file_content: Optional[str],
    ) -> None:
        """Update the file content stored at the conversation level (e.g. when a new
        file is uploaded mid-conversation, overwriting the previous one)."""
        if self.readonly:
            conv = self._memory_conversations.get(conversation_id)
            if conv:
                conv["file_name"] = file_name
                conv["file_content"] = file_content
            return

        try:
            self.client.update(
                index=self.index,
                id=conversation_id,
                body={"doc": {"file_name": file_name, "file_content": file_content}},
            )
            logger.info(f"Updated file content for conversation {conversation_id[:8]}...")
        except Exception as e:
            logger.error(f"Failed to update file content: {e}")
            raise

    async def add_message(
        self,
        conversation_id: str,
        user_message: str,
        agent_response: str,
        search_results_count: int = 0,
        processing_time_ms: int = 0,
        file_name: Optional[str] = None
    ) -> None:
        """Append message to conversation"""
        if self.readonly:
            conv = self._memory_conversations.get(conversation_id)
            if not conv:
                logger.warning(
                    "OpenSearch guardrails: read-only mode enabled; "
                    f"skipping persistence for conversation {conversation_id[:8]}..."
                )
                return

            # Update file info if provided and not already set
            if file_name and not conv.get("file_name"):
                conv["file_name"] = file_name

            now = datetime.utcnow().isoformat()
            new_message = {
                "message_number": conv.get("message_count", 0) + 1,
                "timestamp": now,
                "user_message": user_message,
                "agent_response": agent_response,
                "search_results_count": search_results_count,
                "processing_time_ms": processing_time_ms,
            }
            conv.setdefault("messages", []).append(new_message)
            conv["message_count"] = conv.get("message_count", 0) + 1
            conv["last_message_at"] = now
            conv["last_user_query"] = user_message
            return

        try:
            # Get existing conversation
            doc = self.client.get(index=self.index, id=conversation_id)
            conversation = doc['_source']
            
            # Update file info if provided and not already set
            if file_name and not conversation.get('file_name'):
                conversation['file_name'] = file_name
            
            # Create new message
            now = datetime.utcnow().isoformat()
            new_message = {
                "message_number": conversation['message_count'] + 1,
                "timestamp": now,
                "user_message": user_message,
                "agent_response": agent_response,
                "search_results_count": search_results_count,
                "processing_time_ms": processing_time_ms
            }
            
            # Append and update
            conversation['messages'].append(new_message)
            conversation['message_count'] += 1
            conversation['last_message_at'] = now
            conversation['last_user_query'] = user_message
            
            # Update document
            self.client.update(
                index=self.index,
                id=conversation_id,
                body={'doc': conversation}
            )
            
            logger.debug(f"Added message to conversation {conversation_id[:8]}...")
            
        except exceptions.NotFoundError:
            logger.error(f"Conversation {conversation_id} not found")
            raise ValueError(f"Conversation {conversation_id} not found")
        except Exception as e:
            logger.error(f"Failed to add message: {e}")
            raise
    
    async def get_conversation(self, conversation_id: str) -> Optional[Dict]:
        """Retrieve full conversation"""
        mem = self._memory_conversations.get(conversation_id)
        if mem:
            return mem

        try:
            doc = self.client.get(index=self.index, id=conversation_id)
            return doc['_source']
        except exceptions.NotFoundError:
            logger.warning(f"Conversation {conversation_id} not found")
            return None
        except Exception as e:
            logger.error(f"Failed to get conversation: {e}")
            return None
    
    async def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation by ID. Returns True if deleted, False if not found."""
        # Remove from in-memory store
        if conversation_id in self._memory_conversations:
            del self._memory_conversations[conversation_id]
            self._deleted_ids.add(conversation_id)
            return True

        if self.readonly:
            # Can't physically delete from OpenSearch in read-only mode.
            # Record a tombstone so it disappears from the sidebar listing.
            self._deleted_ids.add(conversation_id)
            logger.info(f"Tombstoned conversation {conversation_id[:8]}... (read-only mode)")
            return True

        try:
            self.client.delete(index=self.index, id=conversation_id)
            self._deleted_ids.add(conversation_id)
            logger.info(f"Deleted conversation {conversation_id[:8]}...")
            return True
        except exceptions.NotFoundError:
            logger.warning(f"Conversation {conversation_id} not found for deletion")
            return False
        except Exception as e:
            logger.error(f"Failed to delete conversation: {e}")
            raise

    async def list_recent_conversations(self, limit: int = 5) -> List[Dict]:
        """Get recent conversations for sidebar"""
        try:
            # Pull from memory first (read-only mode creates conversations in-memory).
            mem_previews: List[Dict] = []
            for conv in self._memory_conversations.values():
                mem_previews.append(
                    {
                        "conversation_id": conv.get("conversation_id"),
                        "last_user_query": conv.get("last_user_query", ""),
                        "title": conv.get("title"),
                        "last_message_at": conv.get("last_message_at"),
                        "message_count": conv.get("message_count", 0),
                    }
                )

            query = {
                "query": {"match_all": {}},
                "sort": [{"last_message_at": {"order": "desc"}}],
                "size": limit,
                "_source": ["conversation_id", "last_user_query", "title", "last_message_at", "message_count"]
            }
            
            result = self.client.search(index=self.index, body=query)
            os_previews = [
                {
                    "conversation_id": h["_source"].get("conversation_id"),
                    "last_user_query": h["_source"].get("last_user_query", ""),
                    "title": h["_source"].get("title"),
                    "last_message_at": h["_source"].get("last_message_at"),
                    "message_count": h["_source"].get("message_count", 0),
                }
                for h in result.get("hits", {}).get("hits", [])
            ]

            combined = mem_previews + os_previews
            combined.sort(key=lambda d: d.get("last_message_at") or "", reverse=True)
            # Filter tombstoned conversations
            combined = [c for c in combined if c.get("conversation_id") not in self._deleted_ids]
            return combined[:limit]
            
        except Exception as e:
            logger.error(f"Failed to list conversations: {e}")
            # Fall back to memory only
            mem_only = [
                {
                    "conversation_id": conv.get("conversation_id"),
                    "last_user_query": conv.get("last_user_query", ""),
                    "title": conv.get("title"),
                    "last_message_at": conv.get("last_message_at"),
                    "message_count": conv.get("message_count", 0),
                }
                for conv in self._memory_conversations.values()
            ]
            mem_only.sort(key=lambda d: d.get("last_message_at") or "", reverse=True)
            # Filter tombstoned conversations
            mem_only = [c for c in mem_only if c.get("conversation_id") not in self._deleted_ids]
            return mem_only[:limit]


# Module-level singleton so all routers share the same in-memory state
_conversation_store_instance: Optional[ConversationStore] = None


def get_conversation_store() -> ConversationStore:
    """Return the application-wide ConversationStore singleton."""
    global _conversation_store_instance
    if _conversation_store_instance is None:
        _conversation_store_instance = ConversationStore()
    return _conversation_store_instance
