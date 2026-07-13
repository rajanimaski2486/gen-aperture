"""OpenSearch conversation storage service"""
from opensearchpy import exceptions
from datetime import datetime
from typing import Optional, List, Dict
import uuid
import logging
import json

from app.config import settings
from app.services.opensearch_guardrails import create_opensearch_client

logger = logging.getLogger(__name__)

CONVERSATION_INDEX = "gen-aperture-conversations"


class ConversationWriteLimitExceeded(RuntimeError):
    """Raised when conversation index write limits would be exceeded."""


class ConversationStore:
    """Manages conversation storage in OpenSearch"""
    
    def __init__(self):
        self.index = settings.opensearch_conversation_index
        if self.index != CONVERSATION_INDEX:
            raise ValueError(
                f"Conversation writes are allowed only to {CONVERSATION_INDEX!r}; "
                f"configured index was {self.index!r}"
            )

        self.client = create_opensearch_client(
            endpoint=settings.opensearch_conversation_endpoint,
            readonly=False,
            timeout_seconds=10.0,
            username=settings.opensearch_username,
            password=settings.opensearch_password,
            allowed_write_index=self.index,
        )
        self.readonly = False

        # In-memory fallback used only when the conversation cluster is unreachable.
        self._memory_conversations: Dict[str, Dict] = {}
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

    def _json_size_bytes(self, value: Dict) -> int:
        return len(json.dumps(value, default=str, ensure_ascii=False).encode("utf-8"))

    def _index_usage(self) -> tuple[int, int]:
        if not self.client.indices.exists(index=self.index):
            raise ConversationWriteLimitExceeded(
                f"Conversation index {self.index!r} does not exist; refusing implicit index creation"
            )

        count_response = self.client.count(index=self.index)
        count = int(count_response.get("count", 0))

        stats = self.client.indices.stats(index=self.index, metric="store")
        index_stats = stats.get("indices", {}).get(self.index, {})
        store = index_stats.get("total", {}).get("store", {})
        size_bytes = store.get("size_in_bytes")
        if size_bytes is None:
            size_bytes = (
                stats.get("_all", {})
                .get("total", {})
                .get("store", {})
                .get("size_in_bytes", 0)
            )
        return count, int(size_bytes or 0)

    def _assert_write_allowed(self, *, new_record: bool = False, projected_bytes: int = 0) -> None:
        try:
            count, size_bytes = self._index_usage()
        except ConversationWriteLimitExceeded:
            raise
        except Exception as exc:
            raise ConversationWriteLimitExceeded(
                f"Could not verify conversation index limits: {exc}"
            ) from exc

        max_records = settings.opensearch_conversation_max_records
        max_store_bytes = settings.opensearch_conversation_max_store_bytes

        if new_record and count >= max_records:
            raise ConversationWriteLimitExceeded(
                f"Conversation index {self.index!r} has {count} records; "
                f"maximum is {max_records}"
            )
        if not new_record and count > max_records:
            raise ConversationWriteLimitExceeded(
                f"Conversation index {self.index!r} has {count} records; "
                f"maximum is {max_records}"
            )

        if size_bytes >= max_store_bytes:
            raise ConversationWriteLimitExceeded(
                f"Conversation index {self.index!r} is {size_bytes} bytes; "
                f"maximum is {max_store_bytes} bytes"
            )

        if projected_bytes and size_bytes + projected_bytes > max_store_bytes:
            raise ConversationWriteLimitExceeded(
                f"Conversation write would exceed {max_store_bytes} bytes "
                f"for index {self.index!r}"
            )
    
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

        if self.readonly:  # pragma: no cover — conversation cluster is always writable
            self._memory_conversations[conversation_id] = doc
            return conversation_id

        try:
            await self.ensure_index_exists()
            self._assert_write_allowed(
                new_record=True,
                projected_bytes=self._json_size_bytes(doc),
            )
            self.client.index(index=self.index, id=conversation_id, body=doc)
            logger.info(f"Created conversation {conversation_id[:8]}...")
            return conversation_id
        except Exception as e:
            logger.error(f"Failed to create conversation: {e}")
            raise
    
    async def set_title(self, conversation_id: str, title: str) -> None:
        """Set the permanent title for a conversation (generated from first prompt)"""
        try:
            self._assert_write_allowed(projected_bytes=self._json_size_bytes({"title": title}))
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
        try:
            self._assert_write_allowed(
                projected_bytes=self._json_size_bytes(
                    {"file_name": file_name, "file_content": file_content}
                )
            )
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
            self._assert_write_allowed(projected_bytes=self._json_size_bytes(conversation))
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
        try:
            self.client.delete(index=self.index, id=conversation_id)
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
            query = {
                "query": {"match_all": {}},
                "sort": [{"last_message_at": {"order": "desc"}}],
                "size": limit,
                "_source": ["conversation_id", "last_user_query", "title", "last_message_at", "message_count"]
            }
            result = self.client.search(index=self.index, body=query)
            return [
                {
                    "conversation_id": h["_source"].get("conversation_id"),
                    "last_user_query": h["_source"].get("last_user_query", ""),
                    "title": h["_source"].get("title"),
                    "last_message_at": h["_source"].get("last_message_at"),
                    "message_count": h["_source"].get("message_count", 0),
                }
                for h in result.get("hits", {}).get("hits", [])
            ]
            
        except Exception as e:
            logger.error(f"Failed to list conversations: {e}")
            return []


# Module-level singleton so all routers share the same in-memory state
_conversation_store_instance: Optional[ConversationStore] = None


def get_conversation_store() -> ConversationStore:
    """Return the application-wide ConversationStore singleton."""
    global _conversation_store_instance
    if _conversation_store_instance is None:
        _conversation_store_instance = ConversationStore()
    return _conversation_store_instance
