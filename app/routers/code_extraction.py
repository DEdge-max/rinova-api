import logging
import time
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Query, Depends, Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from math import ceil

from ..services.openai_service import OpenAIService
from ..repositories.medical_notes import MedicalNotesRepository
from ..models.pydantic_models import (
    ExtractionRequest,
    ExtractionResponse,
    ExtractionData,
    Metadata,
    NotesListingParams,
    NotesListingResponse,
    NotesSummary,
    NotesFilterParams,
    NoteType,
    SortOrder,
    ExtractionStatus,
    BatchExtractionRequest
)

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Rate limiter setup
limiter = Limiter(key_func=get_remote_address)

# Dependency functions
def get_openai_service():
    return OpenAIService()

def get_repository():
    return MedicalNotesRepository()

router = APIRouter(
    prefix="/api/v1",
    tags=["Code Extraction"]
)

@router.post("/extract", response_model=ExtractionResponse, status_code=201)
@limiter.limit("10/minute")
async def extract_codes(
    request: Request,
    extraction_request: ExtractionRequest,
    openai_service: OpenAIService = Depends(get_openai_service),
    repo: MedicalNotesRepository = Depends(get_repository)
) -> ExtractionResponse:
    """ Extract medical codes from clinical text and store them in MongoDB. """
    logger.info(f"Processing extraction request with {len(extraction_request.medical_text)} characters")
    start_time = time.time()
    
    try:
        note_id = await repo.create_note(extraction_request.medical_text)
        extracted_data = await openai_service.extract_medical_codes(extraction_request.medical_text)
        processing_time = int((time.time() - start_time) * 1000)
        
        metadata = Metadata(
            model_version="1.0",
            processing_time_ms=processing_time,
            timestamp=datetime.utcnow().isoformat(),
            note_length=len(extraction_request.medical_text)
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
    request: Request,
    batch_request: BatchExtractionRequest,
    openai_service: OpenAIService = Depends(get_openai_service),
    repo: MedicalNotesRepository = Depends(get_repository)
):
    """ Process multiple medical texts in one request asynchronously. """
    logger.info(f"Processing batch extraction with {len(batch_request.medical_texts)} texts")
    tasks = [
        extract_codes(
            request,
            ExtractionRequest(medical_text=text),
            openai_service,
            repo
        ) for text in batch_request.medical_texts
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results

@router.get("/notes", response_model=NotesListingResponse)
@limiter.limit("20/minute")
async def list_notes(
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    sort_by: str = "created_at",
    sort_order: SortOrder = SortOrder.DESC,
    note_type: Optional[NoteType] = None,
    status: Optional[ExtractionStatus] = None,
    search_text: Optional[str] = None,
    repo: MedicalNotesRepository = Depends(get_repository)
) -> NotesListingResponse:
    """Get paginated list of medical notes with filtering options"""
    try:
        filters = NotesFilterParams(
            note_type=note_type,
            status=status,
            search_text=search_text
        )
        
        total_notes = await repo.get_notes_count(filters)
        total_pages = ceil(total_notes / page_size)
        
        notes = await repo.get_paginated_notes(
            page=page,
            page_size=page_size,
            sort_by=sort_by,
            sort_order=sort_order,
            filters=filters
        )
        
        summary = NotesSummary(
            total_notes=total_notes,
            total_pages=total_pages,
            current_page=page,
            notes_per_page=page_size,
            has_next=page < total_pages,
            has_previous=page > 1
        )
        
        return NotesListingResponse(
            success=True,
            summary=summary,
            notes=notes,
            error=None
        )
    except Exception as e:
        logger.error(f"Error listing notes: {str(e)}")
        return NotesListingResponse(
            success=False,
            summary=NotesSummary(
                total_notes=0,
                total_pages=0,
                current_page=page,
                notes_per_page=page_size,
                has_next=False,
                has_previous=False
            ),
            notes=[],
            error=str(e)
        )

@router.get("/health")
async def health_check():
    """ API health check endpoint """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0"
    }
