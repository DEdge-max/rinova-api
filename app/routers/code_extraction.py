import logging
import time
import asyncio
from datetime import datetime, date
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Query, Depends, Request, Response
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel
from ..services.openai_service import OpenAIService
from ..dependencies.database import get_repository
from ..dependencies.services import get_openai_service
from ..models.pydantic_models import (
    ExtractionRequest,
    ExtractionResponse,
    ExtractionData,
    Metadata,
    NotesListingParams,
    NotesFilterParams,
    NotesListingResponse,
    DashboardStatistics,
    SortOrder,
    NoteType,
    ExtractionStatus,
    BatchExtractionRequest
)
from ..repositories.medical_notes import MedicalNotesRepository

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Rate limiter setup
limiter = Limiter(key_func=get_remote_address)

router = APIRouter(
    prefix="/api/v1",
    tags=["Code Extraction"]
)

# Global error handler
@router.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global error handler caught: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": str(exc)}
    )

# Add rate limit exception handler
@router.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={
            "success": False,
            "error": "Rate limit exceeded",
            "timestamp": datetime.utcnow().isoformat()
        }
    )
@router.post("/extract", response_model=ExtractionResponse, status_code=201)
@limiter.limit("10/minute")
async def extract_codes(
    request: ExtractionRequest,
    openai_service: OpenAIService = Depends(get_openai_service),
    repo: MedicalNotesRepository = Depends(get_repository)
) -> ExtractionResponse:
    """ Extract medical codes from clinical text and store them in MongoDB. """
    logger.info(f"Processing extraction request with {len(request.medical_text)} characters")
    start_time = time.time()
    
    try:
        note_id = await repo.create_note(request.medical_text)
        extracted_data = await openai_service.extract_medical_codes(request.medical_text)
        processing_time = int((time.time() - start_time) * 1000)
        
        metadata = Metadata(
            model_version="1.0",
            processing_time_ms=processing_time,
            timestamp=datetime.utcnow().isoformat(),
            note_length=len(request.medical_text)
        )
        
        data = ExtractionData(
            note_type=extracted_data.get("note_type", "brief"),
            icd10_codes=extracted_data["icd10_codes"],
            cpt_codes=extracted_data["cpt_codes"],
            documentation_gaps=extracted_data.get("documentation_gaps", []),
            metadata=metadata
        )
        
        await repo.update_extraction(note_id, data.dict())
        return ExtractionResponse(success=True, data=data, error=None)
        
    except Exception as e:
        logger.error(f"Error in extraction: {str(e)}", exc_info=True)
        return ExtractionResponse(success=False, data=None, error=str(e))

@router.post("/extract/batch", response_model=List[ExtractionResponse], status_code=201)
@limiter.limit("5/minute")
async def batch_extract_codes(
    requests: BatchExtractionRequest,
    openai_service: OpenAIService = Depends(get_openai_service),
    repo: MedicalNotesRepository = Depends(get_repository)
):
    """ Process multiple medical texts in one request asynchronously. """
    logger.info(f"Processing batch extraction with {len(requests.texts)} texts")
    tasks = [extract_codes(ExtractionRequest(medical_text=text), openai_service, repo) for text in requests.texts]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results

@router.get("/notes", response_model=NotesListingResponse)
@limiter.limit("20/minute")
async def get_notes_listing(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    repo: MedicalNotesRepository = Depends(get_repository)
):
    """ Get paginated listing of medical notes. """
    try:
        params = NotesListingParams(page=page, page_size=page_size)
        notes, summary = await repo.get_notes_listing(params)
        return NotesListingResponse(success=True, summary=summary, notes=notes, error=None)
        
    except Exception as e:
        logger.error(f"Error getting notes listing: {str(e)}", exc_info=True)
        return NotesListingResponse(success=False, summary=None, notes=[], error=str(e))

@router.get("/notes/dashboard", response_model=Dict[str, Any])
@limiter.limit("5/minute")
async def get_dashboard_statistics(
    days: int = Query(30, ge=1, le=365)
):
    """ Fetch dashboard statistics. """
    try:
        stats = await repo.get_dashboard_statistics(days)
        return {"success": True, "data": stats, "error": None}
    except Exception as e:
        return {"success": False, "data": None, "error": str(e)}

@router.get("/search", response_model=List[Dict])
@limiter.limit("15/minute")
async def search_notes(
    query: str = Query(..., min_length=1),
    limit: int = Query(10, ge=1, le=50)
):
    """ Search notes using text search. """
    try:
        return await repo.search_notes(query, limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/notes/type/{note_type}", response_model=List[Dict])
@limiter.limit("10/minute")
async def get_notes_by_type(
    note_type: NoteType,
    limit: int = Query(10, ge=1, le=50)
):
    """ Get notes of a specific type. """
    try:
        return await repo.get_notes_by_type(note_type.value, limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """ API health check endpoint """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0"
    }
