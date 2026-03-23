"""Main FastAPI application"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import logging
from pathlib import Path

from app.config import settings
from app.routers import chat, conversations
from app.services.conversation_store import get_conversation_store

# Configure logging
logging.basicConfig(
    level=logging.INFO if settings.environment == "production" else logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Gen-Aperture API",
    description="Agentic Stock Photo Conversational Search",
    version="1.0.0"
)

# CORS middleware (for development)
if settings.environment == "development":
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Include routers
app.include_router(chat.router, prefix="/api", tags=["chat"])
app.include_router(conversations.router, prefix="/api", tags=["conversations"])


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting Gen-Aperture application...")
    
    # Initialize conversation store and create index if needed
    conversation_store = get_conversation_store()
    await conversation_store.ensure_index_exists()
    
    logger.info("Application started successfully")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    conversation_store = get_conversation_store()
    opensearch_healthy = await conversation_store.check_connection()
    
    return {
        "status": "healthy" if opensearch_healthy else "degraded",
        "opensearch": "connected" if opensearch_healthy else "disconnected",
        "environment": settings.environment
    }


# Serve static files (React frontend) in production
static_path = Path(__file__).parent / "static"
if static_path.exists() and settings.environment == "production":
    app.mount("/assets", StaticFiles(directory=str(static_path / "assets")), name="assets")
    
    @app.get("/{full_path:path}")
    async def serve_react(full_path: str):
        """Serve React app for all non-API routes"""
        if full_path.startswith("api/"):
            return {"error": "Not found"}
        
        index_path = static_path / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path))
        return {"error": "Frontend not built"}
