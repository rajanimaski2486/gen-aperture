"""Chat endpoint router"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Optional
import time
import logging

from app.models.schemas import ChatResponse, PhotoResult, AgentWorkflowStep, ErrorResponse, RerankerDecision
from app.services.session_manager import session_manager
from app.services.conversation_store import get_conversation_store
from app.services.file_extractor import file_extractor
from app.services.image_analyzer import analyze_images
from app.services.agent_squad import AgentSquad
from app.services.searchbybrief.main import app as searchbybrief_workflow

logger = logging.getLogger(__name__)
router = APIRouter()
MAX_RESULTS_RETURNED = 30

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


def _safe_int(value):
    try:
        return int(str(value))
    except Exception:
        return None


def _run_searchbybrief_workflow(
    user_message: str,
    file_bytes: Optional[bytes],
    file_name: Optional[str],
) -> dict:
    """
    Run SearchByBrief LangGraph workflow and adapt output to chat endpoint shape.
    """
    state = {
        "user_request": user_message,
        "uploaded_file_bytes": file_bytes,
        "uploaded_file_name": file_name,
        "file_type": None,
        "file_images": [],
        "image_analysis": None,
        "extraction_error": None,
        "attachment_text": None,
        "search_params": None,
        "candidate_pool": [],
        "refined_pool": [],
        "stage3_candidates": [],
        "stage3_shortlist": [],
        "stage3_lane_audits": [],
        "stage3_repair_requests": [],
        "final_collection": [],
        "feedback": "",
        "iterations": 0,
    }

    final_state = searchbybrief_workflow.invoke(state)
    search_params = final_state.get("search_params")
    if hasattr(search_params, "model_dump"):
        search_params = search_params.model_dump()

    lanes = (search_params or {}).get("search_lanes", []) if isinstance(search_params, dict) else []
    lane_names = [lane.get("lane_name") for lane in lanes if isinstance(lane, dict) and lane.get("lane_name")]

    final_collection = final_state.get("final_collection") or final_state.get("stage3_shortlist") or []
    search_results = []
    for item in final_collection:
        asset_id = str(item.get("asset_id", ""))
        thumbnail_url = item.get("thumbnail_url") or ""
        score = item.get("stage3_score")
        if not isinstance(score, (int, float)):
            score = float(item.get("stage2_score") or 0.0)
        origin_lane = item.get("origin_lane_name") or "lane"
        description = (
            f"{origin_lane} · stage3={score:.3f}"
            if isinstance(score, float)
            else f"{origin_lane} asset"
        )
        search_results.append(
            {
                "hadron_id": asset_id,
                "ext_id": _safe_int(asset_id),
                "description": description,
                "image_url": thumbnail_url,
                "thumbnail_url": thumbnail_url,
                "date_added": None,
                "license_count": 0,
                "categories": [],
                "keywords": [],
                "score": score if isinstance(score, float) else 0.0,
                "is_generated": False,
            }
        )

    workflow_steps = [
        {
            "agent": "SearchByBrief Planner",
            "action": "Generate search lanes",
            "reasoning": f"Built {len(lane_names)} lane(s) from the brief and attachment context.",
            "output": {"lane_names": lane_names},
        },
        {
            "agent": "SearchByBrief Retriever",
            "action": "Retrieve lane candidates",
            "reasoning": "Retrieved candidates per lane using configured retriever mode.",
            "output": {"candidate_pool_count": len(final_state.get("candidate_pool") or [])},
        },
        {
            "agent": "SearchByBrief Curator",
            "action": "Visual scoring, filtering, dedup, and final selection",
            "reasoning": "Scored lane candidates and produced final collection.",
            "output": {
                "feedback": final_state.get("feedback"),
                "final_collection_count": len(final_collection),
                "shortlist_count": len(final_state.get("stage3_shortlist") or []),
            },
        },
    ]

    response = (
        f"SearchByBrief completed with {len(search_results)} curated result(s) "
        f"across {len(lane_names)} lane(s)."
    )
    return {
        "response": response,
        "search_results": search_results,
        "workflow_steps": workflow_steps,
        "search_mode": "relevance",
        "rerank_applied": False,
        "rerank_decisions": [],
        "rerank_explanation": None,
        "filter_metadata": None,
    }


@router.post("/chat", response_model=ChatResponse)
async def chat(
    message: str = Form(...),
    conversation_id: Optional[str] = Form(None),
    openai_api_key: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    workflow_mode: Optional[str] = Form("agent_squad"),
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
        image_analysis = None
        file_type = None
        file_name = None
        file_bytes: Optional[bytes] = None

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
            print(f"[DEBUG] Extracted {len(file_content)} chars and {len(file_images)} images from {file.filename}")  # Debug

            # Analyze extracted images for color palettes and mood
            if file_images:
                image_analysis = analyze_images(file_images)
                logger.info(f"Image analysis: {image_analysis.get('summary', '')}")
                print(f"[DEBUG] Image analysis mood_tags: {image_analysis.get('mood_tags', [])}")  # Debug
            else:
                print(f"[DEBUG] No images found in PDF — file_images is empty")  # Debug

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
        
        selected_mode = (workflow_mode or "agent_squad").strip().lower()
        if selected_mode in {"searchbybrief", "search_by_brief", "brief"}:
            logger.info(f"Running SearchByBrief workflow for conversation {conversation_id[:8]}...")
            agent_result = _run_searchbybrief_workflow(
                user_message=message,
                file_bytes=file_bytes,
                file_name=file_name,
            )
        else:
            # Run default AgentSquad workflow
            logger.info(f"Running agent squad for conversation {conversation_id[:8]}...")
            agent_squad = AgentSquad(openai_api_key=api_key)
            agent_result = agent_squad.run(
                user_query=message,
                file_content=file_content,
                file_images=file_images,
                image_analysis=image_analysis,
                file_type=file_type,
                conversation_history=conversation_history
            )
        
        # Format results (unfiltered)
        results = []
        for photo in agent_result.get('search_results', [])[:MAX_RESULTS_RETURNED]:
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
