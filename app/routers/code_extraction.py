from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
from ..services.openai_service import OpenAIService
from ..models.pydantic_models import (
    ExtractionRequest,
    ExtractionResponse,
    ExtractionData,
    ICD10Code,
    CPTCode,
    Metadata,
    Evidence,
    DocumentationGap
)
from ..repositories.medical_notes import MedicalNotesRepository
import time
from datetime import datetime

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
    description="Analyzes medical text to extract ICD-10 diagnostic codes and CPT procedure codes with confidence scores, evidence, and documentation gaps",
    responses={
        200: {
            "model": ExtractionResponse,
            "description": "Successful code extraction",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "data": {
                            "note_type": "brief",
                            "icd10_codes": [
                                {
                                    "code": "E11.9",
                                    "description": "Type 2 diabetes mellitus without complications",
                                    "confidence": 0.95,
                                    "primary": True,
                                    "evidence": {
                                        "direct_quotes": ["Patient has type 2 diabetes without complications"],
                                        "reasoning": "Clear documentation of diagnosis",
                                        "guidelines_applied": ["ICD-10 guideline I.A.19"]
                                    }
                                }
                            ],
                            "cpt_codes": [
                                {
                                    "code": "99213",
                                    "description": "Office/outpatient visit for evaluation and management",
                                    "confidence": 0.92,
                                    "evidence": {
                                        "direct_quotes": ["Office visit level 3 for evaluation"],
                                        "reasoning": "Documentation supports level 3 visit",
                                        "guidelines_applied": ["E/M Guidelines 2021"]
                                    },
                                    "alternative_codes": []
                                }
                            ],
                            "documentation_gaps": [
                                {
                                    "severity": "minor",
                                    "description": "Missing specific duration of symptoms",
                                    "impact": {
                                        "affected_codes": ["99213"],
                                        "current_limitation": "Could affect E/M level",
                                        "potential_improvement": "Could support higher level E/M code"
                                    },
                                    "recommendation": {
                                        "what_to_add": "Document symptom duration",
                                        "example": "Symptoms present for 2 weeks",
                                        "rationale": "Helps establish medical necessity"
                                    }
                                }
                            ],
                            "metadata": {
                                "model_version": "1.0",
                                "processing_time_ms": 245,
                                "timestamp": "2025-02-12T10:00:00Z",
                                "note_length": 150
                            }
                        },
                        "error": None
                    }
                }
            }
        },
        422: {"model": ExtractionResponse, "description": "Validation Error"},
        500: {"model": ExtractionResponse, "description": "Internal Server Error"}
    }
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
    "/recent",
    response_model=List[Dict],
    summary="Get recent extractions",
    description="Retrieve recent medical note extractions"
)
async def get_recent_extractions(limit: int = 10):
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
