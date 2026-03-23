"""Conversations management router"""
from fastapi import APIRouter, HTTPException
from typing import List
import logging

from app.models.schemas import ConversationDetail, ConversationPreview
from app.services.conversation_store import get_conversation_store

logger = logging.getLogger(__name__)
router = APIRouter()

conversation_store = get_conversation_store()


@router.get("/conversations/recent", response_model=List[ConversationPreview])
async def get_recent_conversations():
    """Get last 5 conversations for sidebar"""
    try:
        conversations = await conversation_store.list_recent_conversations(limit=5)
        return conversations
    except Exception as e:
        logger.error(f"Failed to get recent conversations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(conversation_id: str):
    """Get full conversation details"""
    try:
        conversation = await conversation_store.get_conversation(conversation_id)
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        return conversation
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a specific conversation"""
    try:
        deleted = await conversation_store.delete_conversation(conversation_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return {"deleted": True, "conversation_id": conversation_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))
