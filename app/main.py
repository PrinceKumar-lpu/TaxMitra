"""
FastAPI application for TaxMitra AI Tax Assistant.
Includes:
- /health endpoint
- /chat structured conversational endpoint with retry & memory
- /calculate direct deterministic tax breakdown
- /compare direct Old vs New regime comparison
- In-memory IP rate limiter
- Static files for Chat Web UI
"""

import os
import time
import logging
from collections import defaultdict
from typing import Dict, Tuple
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from dotenv import load_dotenv

load_dotenv()

from app.schemas import (
    ChatRequest,
    ChatResponse,
    CalculateTaxRequest,
    CompareRegimesRequest,
    DeductionDetails,
    FeedbackRequest,
    FeedbackResponse
)
from app.services.tools import calculate_tax, compare_regimes
from app.services.llm import llm_service
from app.services.rag import get_collection, rag_service
from app.services.observability import metrics_tracker, log_structured_event
from app.services.feedback import record_feedback, get_all_feedback, FeedbackItem

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("taxmitra")


# In-memory Rate Limiter
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
ip_request_counts: Dict[str, list] = defaultdict(list)


async def rate_limiter(request: Request):
    """Simple sliding-window rate limiter per client IP."""
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    
    # Filter out requests older than 60 seconds
    timestamps = [t for t in ip_request_counts[client_ip] if now - t < 60]
    if len(timestamps) >= RATE_LIMIT_PER_MINUTE:
        logger.warning(f"Rate limit exceeded for IP: {client_ip}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit of {RATE_LIMIT_PER_MINUTE} requests/minute exceeded. Please wait a moment."
        )
    
    timestamps.append(now)
    ip_request_counts[client_ip] = timestamps


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing TaxMitra application...")
    # Pre-check ChromaDB
    try:
        coll = get_collection()
        logger.info(f"Connected to ChromaDB collection. Document count: {coll.count()}")
    except Exception as e:
        logger.warning(f"ChromaDB initial check notice: {e}")
    yield
    logger.info("Shutting down TaxMitra...")


app = FastAPI(
    title="TaxMitra API",
    description="Deterministic AI Tax Assistant for Indian Salaried Taxpayers (FY 2024-25 / AY 2025-26)",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint checking ChromaDB and API responsiveness."""
    chroma_status = "unhealthy"
    chunks_count = 0
    try:
        coll = get_collection()
        chunks_count = coll.count()
        chroma_status = "healthy"
    except Exception as e:
        logger.error(f"Chroma health check failed: {e}")

    gemini_configured = bool(os.getenv("GEMINI_API_KEY") and os.getenv("GEMINI_API_KEY") != "your_gemini_api_key_here")

    return {
        "status": "online",
        "service": "TaxMitra",
        "version": "1.0.0",
        "financial_year": "2024-2025",
        "assessment_year": "2025-2026",
        "chromadb": {
            "status": chroma_status,
            "indexed_chunks": chunks_count
        },
        "gemini_api_configured": gemini_configured
    }


@app.get("/metrics", tags=["Observability"])
async def get_metrics():
    """Returns real-time service observability metrics."""
    return metrics_tracker.get_summary()


@app.post("/feedback", response_model=FeedbackResponse, tags=["Feedback Loop"])
async def submit_feedback(payload: FeedbackRequest):
    """Stores user thumbs-up/down ratings and reviews in SQLite."""
    try:
        row_id = record_feedback(
            FeedbackItem(
                session_id=payload.session_id,
                query=payload.query,
                bot_answer=payload.bot_answer,
                rating=payload.rating,
                comment=payload.comment,
                retrieval_mode=payload.retrieval_mode
            )
        )
        metrics_tracker.record_feedback(payload.rating)
        log_structured_event("user_feedback", {
            "feedback_id": row_id,
            "session_id": payload.session_id,
            "rating": payload.rating
        })
        return FeedbackResponse(
            id=row_id,
            status="success",
            message="Thank you! Your feedback helps TaxMitra continuously improve."
        )
    except Exception as e:
        logger.error(f"Error saving feedback: {e}")
        raise HTTPException(status_code=500, detail="Failed to save feedback.")


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(rate_limiter)], tags=["AI Assistant"])
async def chat_endpoint(payload: ChatRequest):
    """
    Main conversational endpoint:
    - Multi-layer guardrail check (injection defense, domain bounds)
    - Hybrid RAG context retrieval (dense + BM25 + RRF + reranking)
    - Deterministic tool execution for tax calculations
    - Conversation & profile memory tracking
    """
    try:
        response = llm_service.chat(
            user_message=payload.message,
            session_id=payload.session_id,
            retrieval_mode=payload.retrieval_mode
        )
        return response
    except Exception as e:
        logger.error(f"Error in /chat endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing your request: {str(e)}"
        )


@app.post("/calculate", dependencies=[Depends(rate_limiter)], tags=["Tax Engine"])
async def calculate_endpoint(payload: CalculateTaxRequest):
    """Direct deterministic calculation endpoint."""
    try:
        result = calculate_tax(
            gross_income=payload.gross_income,
            regime=payload.regime,
            deductions=payload.deductions.model_dump() if payload.deductions else None
        )
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/compare", dependencies=[Depends(rate_limiter)], tags=["Tax Engine"])
async def compare_endpoint(payload: CompareRegimesRequest):
    """Direct deterministic Old vs New regime comparison endpoint."""
    try:
        result = compare_regimes(
            gross_income=payload.gross_income,
            deductions=payload.deductions.model_dump() if payload.deductions else None
        )
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Mount static directory for Chat UI
STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_index():
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))
