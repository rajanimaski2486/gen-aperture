"""Chat endpoint router"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Optional
import time
import logging

from app.models.schemas import ChatResponse, PhotoResult, AgentWorkflowStep, ErrorResponse, RerankerDecision
from app.services.session_manager import session_manager
from app.services.conversation_store import get_conversation_store
from app.services.file_extractor import file_extractor
from app.services.agent_squad import AgentSquad

logger = logging.getLogger(__name__)
router = APIRouter()

conversation_store = get_conversation_store()


def _generate_conversation_title(prompt: str, max_len: int = 60) -> str:
    """Create a short, readable title from the first user prompt."""
    # Strip leading/trailing whitespace and collapse internal whitespace
    cleaned = " ".join(prompt.strip().split())
    if len(cleaned) <= max_len:
        return cleaned
    # Truncate at a word boundary
    truncated = cleaned[:max_len].rsplit(" ", 1)[0].rstrip(",.;:!?") + "…"
    return truncated


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
        # ── Step 1: Extract file content FIRST so it can be stored with the conversation ──
        file_content = None
        file_images = None
        file_type = None
        file_name = None

        if file:
            logger.info(f"Processing uploaded file: {file.filename}")
            file_bytes = await file.read()
            extraction_result = file_extractor.extract_text_and_images(file_bytes, file.filename)

            if extraction_result.get('error'):
                raise HTTPException(
                    status_code=400,
                    detail=f"File extraction failed: {extraction_result['error']}"
                )

            file_content = extraction_result.get('text')
            file_images = extraction_result.get('images')
            file_type = extraction_result.get('file_type')
            file_name = file.filename
            logger.info(f"Extracted {len(file_content)} characters and {len(file_images)} images from {file.filename}")

        # ── Step 2: Create new conversation OR validate existing session ──
        is_new_conversation = False
        if not conversation_id:
            if not openai_api_key:
                raise HTTPException(
                    status_code=401,
                    detail="OpenAI API key required for new conversation"
                )

            # Create new conversation, storing file content at the document level
            conversation_id = await conversation_store.create_conversation(
                file_name=file_name,
                file_content=file_content,
            )
            session_manager.create_session(conversation_id, openai_api_key)
            api_key = openai_api_key
            is_new_conversation = True
            logger.info(f"New conversation created: {conversation_id[:8]}...")
        else:
            # Check if API key is valid for existing conversation
            api_key = session_manager.get_api_key(conversation_id)
            if not api_key:
                # Session expired or not found
                if openai_api_key:
                    session_manager.create_session(conversation_id, openai_api_key)
                    api_key = openai_api_key
                else:
                    raise HTTPException(
                        status_code=401,
                        detail="Session expired. Please provide OpenAI API key."
                    )

        # ── Step 3: Load conversation history AND stored file context ──
        conversation_history = []
        existing_conv = await conversation_store.get_conversation(conversation_id)

        if existing_conv:
            # Load message history for multi-turn context
            if existing_conv.get('messages'):
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

            # If no new file was uploaded, re-use the file content/images from the stored conversation
            if not file_content and existing_conv.get('file_content'):
                file_content = existing_conv['file_content']
                file_name = existing_conv.get('file_name')
                file_type = 'stored'
                # Optionally: handle file_images if you store them in conversation_store
                logger.info(f"Re-using stored file context from conversation ({file_name})")
            elif file_content and file is not None and not is_new_conversation:
                # A new file was uploaded mid-conversation — persist the updated content
                await conversation_store.update_file_content(
                    conversation_id=conversation_id,
                    file_name=file_name,
                    file_content=file_content,
                )
                logger.info(f"Updated stored file content for conversation {conversation_id[:8]}...")
        
        # Run agent squad
        logger.info(f"Running agent squad for conversation {conversation_id[:8]}...")
        agent_squad = AgentSquad(openai_api_key=api_key)
        
        agent_result = agent_squad.run(
            user_query=message,
            file_content=file_content,
            file_images=file_images,
            file_type=file_type,
            conversation_history=conversation_history
        )
        
        # Format results (unfiltered)
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
                score=photo.get('score', 0.0),
                is_generated=photo.get('is_generated', False)
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
                opensearch_payload=step.get('opensearch_payload'),
                opensearch_url=step.get('opensearch_url')
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

        # Generate and persist a human-readable title from the very first message
        if is_new_conversation:
            title = _generate_conversation_title(message)
            await conversation_store.set_title(conversation_id, title)
            logger.info(f"Saved title for conversation {conversation_id[:8]}: {title!r}")
        
        return ChatResponse(
            conversation_id=conversation_id,
            response=response_text,
            results=results,
            filter_metadata=agent_result.get('filter_metadata'),
            api_key_valid=True,
            processing_time_ms=processing_time_ms,
            workflow_steps=workflow_steps,
            search_mode=agent_result.get('search_mode', 'relevance'),
            rerank_applied=agent_result.get('rerank_applied') or False,
            rerank_decisions=[
                RerankerDecision(**d)
                for d in (agent_result.get('rerank_decisions') or [])
            ] or None,
            rerank_explanation=agent_result.get('rerank_explanation'),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
