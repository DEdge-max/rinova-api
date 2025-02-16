from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, List, Optional
from ..services.openai_service import OpenAIService
from ..models.pydantic_models import (
    ExtractionRequest,
    ExtractionResponse,
    ExtractionData,
    ICD10Code,
    CPTCode,
    Metadata,
    Evidence,
    DocumentationGap,
    NotesListingParams,
    NotesFilterParams,
    NotesListingResponse,
    DashboardStatistics,
    SortOrder,
    NoteType,
    ExtractionStatus
)
from ..repositories.medical_notes import MedicalNotesRepository
import time
from datetime import datetime, date
from fastapi.responses import JSONResponse

router = APIRouter(
    tags=["Code Extraction"]
)

openai_service = OpenAIService()
medical_notes_repo = MedicalNotesRepository()

@router.post(
    "/extract",
    response_model=ExtractionResponse,
    response_model_exclude_unset=True,
    summary="Extract medical codes from text",
    description="Analyzes medical text to extract ICD-10 diagnostic codes and CPT procedure codes with confidence scores, evidence, and documentation gaps"
)
async def extract_codes(
    request: ExtractionRequest
) -> ExtractionResponse:
    """
    Extract medical codes from the provided clinical text and store in MongoDB.
    Includes code evidence, documentation gaps, and note type classification.
    """
    start_time = time.time()
    
    try:
        # Store the incoming text first
        note_id = await medical_notes_repo.create_note(request.medical_text)
        
        # Extract codes using OpenAI
        extracted_data = await openai_service.extract_medical_codes(request.medical_text)
        
        # Calculate processing time
        processing_time = int((time.time() - start_time) * 1000)
        
        # Create metadata
        metadata = Metadata(
            model_version="1.0",
            processing_time_ms=processing_time,
            timestamp=datetime.utcnow().isoformat(),
            note_length=len(request.medical_text)
        )
        
        # Create extraction data with new fields
        data = ExtractionData(
            note_type=extracted_data.get("note_type", "brief"),
            icd10_codes=extracted_data["icd10_codes"],
            cpt_codes=extracted_data["cpt_codes"],
            documentation_gaps=extracted_data.get("documentation_gaps", []),
            metadata=metadata
        )
        
        # Store the extraction results
        await medical_notes_repo.update_extraction(note_id, data.dict())
        
        return ExtractionResponse(
            success=True,
            data=data,
            error=None
        )
        
    except Exception as e:
        # Log the error here if you have logging configured
        return ExtractionResponse(
            success=False,
            data=None,
            error=str(e)
        )

@router.get(
    "/notes",
    response_model=NotesListingResponse,
    summary="Get paginated notes listing",
    description="Retrieve medical notes with pagination, sorting, and filtering options"
)
async def get_notes_listing(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    sort_by: str = Query("created_at", description="Field to sort by"),
    sort_order: SortOrder = Query(SortOrder.DESCENDING, description="Sort order"),
    note_type: Optional[NoteType] = Query(None, description="Filter by note type"),
    status: Optional[ExtractionStatus] = Query(None, description="Filter by status"),
    patient_id: Optional[str] = Query(None, description="Filter by patient ID"),
    start_date: Optional[date] = Query(None, description="Filter by start date"),
    end_date: Optional[date] = Query(None, description="Filter by end date"),
    search_query: Optional[str] = Query(None, description="Text search query")
):
    """Get paginated listing of medical notes with filtering options"""
    try:
        # Convert dates to datetime if provided
        start_datetime = datetime.combine(start_date, datetime.min.time()) if start_date else None
        end_datetime = datetime.combine(end_date, datetime.max.time()) if end_date else None
        
        # Create filter parameters
        filters = NotesFilterParams(
            note_type=note_type,
            status=status,
            patient_id=patient_id,
            start_date=start_datetime,
            end_date=end_datetime,
            search_query=search_query
        )
        
        # Create listing parameters
        params = NotesListingParams(
            page=page,
            page_size=page_size,
            sort_by=sort_by,
            sort_order=sort_order,
            filters=filters
        )
        
        # Get notes and summary
        notes, summary = await medical_notes_repo.get_notes_listing(params)
        
        return NotesListingResponse(
            success=True,
            summary=summary,
            notes=notes,
            error=None
        )
        
    except Exception as e:
        return NotesListingResponse(
            success=False,
            summary=None,
            notes=[],
            error=str(e)
        )

@router.get(
    "/notes/dashboard",
    response_model=Dict[str, Any],
    summary="Get dashboard statistics",
    description="Retrieve comprehensive dashboard statistics and analytics"
)
async def get_dashboard_statistics(
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze")
):
    """Get dashboard statistics and analytics"""
    try:
        stats = await medical_notes_repo.get_dashboard_statistics(days)
        return {
            "success": True,
            "data": stats,
            "error": None
        }
    except Exception as e:
        return {
            "success": False,
            "data": None,
            "error": str(e)
        }

@router.get(
    "/recent",
    response_model=List[Dict],
    summary="Get recent extractions",
    description="Retrieve recent medical note extractions"
)
async def get_recent_extractions(limit: int = Query(10, ge=1, le=50)):
    """Get recent extractions for display/example purposes"""
    try:
        return await medical_notes_repo.get_recent_notes(limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get(
    "/extraction/{note_id}",
    response_model=Dict,
    summary="Get specific extraction",
    description="Retrieve a specific medical note extraction by ID"
)
async def get_extraction(note_id: str):
    """Get a specific extraction by ID"""
    try:
        note = await medical_notes_repo.get_note(note_id)
        if not note:
            raise HTTPException(status_code=404, detail="Note not found")
        return note
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get(
    "/search",
    response_model=List[Dict],
    summary="Search notes",
    description="Search medical notes using text query"
)
async def search_notes(
    query: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(10, ge=1, le=50, description="Maximum number of results")
):
    """Search notes using text search"""
    try:
        return await medical_notes_repo.search_notes(query, limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get(
    "/notes/type/{note_type}",
    response_model=List[Dict],
    summary="Get notes by type",
    description="Retrieve medical notes of a specific type"
)
async def get_notes_by_type(
    note_type: NoteType,
    limit: int = Query(10, ge=1, le=50)
):
    """Get notes of a specific type"""
    try:
        return await medical_notes_repo.get_notes_by_type(note_type.value, limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
