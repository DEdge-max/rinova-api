from fastapi import APIRouter, HTTPException, status, Depends
from typing import List
from bson import ObjectId
from bson.errors import InvalidId
from datetime import datetime

from app.models.pydantic_models import (
    MedicalNote,
    NoteCreate,
    NoteUpdate,
    NoteResponse,
    NotesListResponse,
    ExtractionResponse,
    CodeExtractionResult,
    QuickExtractionRequest,
    QuickExtractionResponse
)
from app.services.openai_service import openai_service
from app.repositories.medical_notes import MedicalNotesRepository

router = APIRouter(prefix="/api/v1", tags=["Code Extraction"])

# Lazy initialization of repository
notes_repository = MedicalNotesRepository()

async def get_repository():
    """Ensure repository is initialized before use."""
    await notes_repository.initialize()
    return notes_repository

def validate_object_id(id: str) -> ObjectId:
    """Validate and convert string ID to ObjectId."""
    try:
        return ObjectId(id)
    except InvalidId:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid note ID format: {id}"
        )

@router.get("/notes", response_model=NotesListResponse)
async def get_all_notes(repository: MedicalNotesRepository = Depends(get_repository)):
    """Get all medical notes with their extracted codes."""
    try:
        notes = await repository.get_all_notes()
        return NotesListResponse(
            message="Notes retrieved successfully",
            notes=notes
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve notes: {str(e)}"
        )

@router.get("/notes/{note_id}", response_model=NoteResponse)
async def get_note(
    note_id: str,
    repository: MedicalNotesRepository = Depends(get_repository)
):
    """Get a specific medical note by ID."""
    try:
        object_id = validate_object_id(note_id)
        note = await repository.get_note_by_id(str(object_id))
        if not note:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Note with ID {note_id} not found"
            )

        # Ensure extraction_result always exists
        if note.extraction_result is None:
            note.extraction_result = {
                "icd10_codes": [],
                "cpt_codes": [],
                "alternative_cpts": [],
                "modifiers": [],
                "hcpcs_codes": []
            }

        return NoteResponse(
            message="Note retrieved successfully",
            note=note
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve note: {str(e)}"
        )

@router.post("/notes", response_model=NoteResponse, status_code=status.HTTP_201_CREATED)
async def create_note(
    note: NoteCreate,
    repository: MedicalNotesRepository = Depends(get_repository)
):
    """Create a new medical note."""
    try:
        new_note_id = await repository.create_note(note)
        new_note = await repository.get_note_by_id(new_note_id)
        return NoteResponse(
            message="Note created successfully",
            note=new_note
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create note: {str(e)}"
        )

@router.put("/notes/{note_id}", response_model=NoteResponse)
async def update_note(
    note_id: str,
    note_update: NoteUpdate,
    repository: MedicalNotesRepository = Depends(get_repository)
):
    """Update an existing medical note."""
    try:
        object_id = validate_object_id(note_id)
        updated = await repository.update_note(str(object_id), note_update)
        if not updated:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Note with ID {note_id} not found"
            )
        updated_note = await repository.get_note_by_id(str(object_id))
        return NoteResponse(
            message="Note updated successfully",
            note=updated_note
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update note: {str(e)}"
        )

@router.post("/extract", response_model=ExtractionResponse)
async def extract_codes(
    note_id: str,
    repository: MedicalNotesRepository = Depends(get_repository)
):
    """Extract medical codes from a note's text using OpenAI."""
    try:
        object_id = validate_object_id(note_id)

        # Get the note
        note = await repository.get_note_by_id(str(object_id))
        if not note:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Note with ID {note_id} not found"
            )

        # Extract codes using OpenAI
        extraction_result = await openai_service.extract_codes(note.note_text)

        # Update note with extracted codes
        note_update = NoteUpdate(extraction_result=extraction_result)
        await repository.update_note(str(object_id), note_update)

        return ExtractionResponse(
            message="Codes extracted successfully",
            extraction_result=extraction_result
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to extract codes: {str(e)}"
        )

@router.post("/quick-extract", response_model=QuickExtractionResponse)
async def quick_extract(
    request: QuickExtractionRequest,
    repository: MedicalNotesRepository = Depends(get_repository)
):
    """Extract medical codes from note text and save to database with default values if needed."""
    try:
        # Create a new note with provided or default values
        note_create = NoteCreate(
            doctor_name=request.doctor_name,
            patient_name=request.patient_name,
            note_text=request.note_text,
            date=request.date or datetime.now()
        )

        # Save note to database
        new_note_id = await repository.create_note(note_create)

        # Extract codes using OpenAI
        extraction_result = await openai_service.extract_codes(request.note_text)

        # Update note with extracted codes
        note_update = NoteUpdate(extraction_result=extraction_result)
        await repository.update_note(new_note_id, note_update)

        return QuickExtractionResponse(
            message="Codes extracted successfully",
            note_id=new_note_id,
            extraction_result=extraction_result
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to extract codes: {str(e)}"
        )
