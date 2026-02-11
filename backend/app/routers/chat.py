"""Chat endpoint router"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Optional
import time
import logging

from app.models.schemas import ChatResponse, PhotoResult, AgentWorkflowStep, ErrorResponse
from app.services.session_manager import session_manager
from app.services.conversation_store import ConversationStore
from app.services.file_extractor import file_extractor
from app.services.agent_squad import AgentSquad

logger = logging.getLogger(__name__)
router = APIRouter()

conversation_store = ConversationStore()


@router.post("/chat", response_model=ChatResponse)
async def chat(
    message: str = Form(...),
    conversation_id: Optional[str] = Form(None),
    openai_api_key: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    """
    Main chat endpoint
    
    - **message**: User's query
    - **conversation_id**: UUID of existing conversation or None for new
    - **openai_api_key**: Required for new conversation or if session expired
    - **file**: Optional PDF/DOCX/TXT file (max 1MB)
    """
    start_time = time.time()
    
    try:
        # Handle new conversation
        if not conversation_id:
            if not openai_api_key:
                raise HTTPException(
                    status_code=401,
                    detail="OpenAI API key required for new conversation"
                )
            
            # Create new conversation
            conversation_id = await conversation_store.create_conversation()
            session_manager.create_session(conversation_id, openai_api_key)
            api_key = openai_api_key  # Use the key that was just provided
            logger.info(f"New conversation created: {conversation_id[:8]}...")
        else:
            # Check if API key is valid for existing conversation
            api_key = session_manager.get_api_key(conversation_id)
            if not api_key:
                # Session expired or not found
                if openai_api_key:
                    # User provided new key
                    session_manager.create_session(conversation_id, openai_api_key)
                    api_key = openai_api_key
                else:
                    raise HTTPException(
                        status_code=401,
                        detail="Session expired. Please provide OpenAI API key."
                    )
        
        # File extraction (if uploaded)
        file_content = None
        file_type = None
        file_name = None
        
        if file:
            logger.info(f"Processing uploaded file: {file.filename}")
            file_bytes = await file.read()
            extraction_result = file_extractor.extract_text(file_bytes, file.filename)
            
            if extraction_result.get('error'):
                raise HTTPException(
                    status_code=400,
                    detail=f"File extraction failed: {extraction_result['error']}"
                )
            
            file_content = extraction_result.get('text')
            file_type = extraction_result.get('file_type')
            file_name = file.filename
            logger.info(f"Extracted {len(file_content)} characters from {file.filename}")
        
        # Fetch conversation history for follow-up context
        conversation_history = []
        if conversation_id:
            existing_conv = await conversation_store.get_conversation(conversation_id)
            if existing_conv and existing_conv.get('messages'):
                for msg in existing_conv['messages']:
                    conversation_history.append({
                        'role': 'user',
                        'content': msg.get('user_message', '')
                    })
                    conversation_history.append({
                        'role': 'assistant',
                        'content': msg.get('agent_response', '')
                    })
                logger.info(f"Loaded {len(existing_conv['messages'])} prior exchanges for context")
        
        # Run agent squad
        logger.info(f"Running agent squad for conversation {conversation_id[:8]}...")
        agent_squad = AgentSquad(openai_api_key=api_key)
        
        agent_result = agent_squad.run(
            user_query=message,
            file_content=file_content,
            file_type=file_type,
            conversation_history=conversation_history
        )
        
        # Format results
        results = []
        for photo in agent_result.get('search_results', [])[:10]:
            results.append(PhotoResult(
                hadron_id=photo.get('hadron_id'),
                ext_id=photo.get('ext_id'),
                description=photo.get('description', ''),
                image_url=photo.get('image_url', ''),
                thumbnail_url=photo.get('thumbnail_url', ''),
                date_added=photo.get('date_added'),
                license_count=photo.get('license_count', 0),
                categories=photo.get('categories', []),
                keywords=photo.get('keywords', []),
                score=photo.get('score', 0.0)
            ))
        
        # Format workflow steps
        workflow_steps = []
        for step in agent_result.get('workflow_steps', []):
            workflow_steps.append(AgentWorkflowStep(
                agent=step.get('agent', ''),
                action=step.get('action', ''),
                reasoning=step.get('reasoning', ''),
                prompt=step.get('prompt'),
                input=step.get('input'),
                output=step.get('output'),
                decision=step.get('decision'),
                opensearch_payload=step.get('opensearch_payload')
            ))
        
        response_text = agent_result.get('response', 'No response generated')
        
        # Check for authentication error
        if agent_result.get('error') == 'authentication_error':
            raise HTTPException(
                status_code=401,
                detail="Invalid OpenAI API key. Please check your API key and try again."
            )
        
        # Store message in conversation
        processing_time_ms = int((time.time() - start_time) * 1000)
        await conversation_store.add_message(
            conversation_id=conversation_id,
            user_message=message,
            agent_response=response_text,
            search_results_count=len(results),
            processing_time_ms=processing_time_ms,
            file_name=file_name if file else None
        )
        
        return ChatResponse(
            conversation_id=conversation_id,
            response=response_text,
            results=results,
            api_key_valid=True,
            processing_time_ms=processing_time_ms,
            workflow_steps=workflow_steps,
            search_mode=agent_result.get('search_mode', 'relevance')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
