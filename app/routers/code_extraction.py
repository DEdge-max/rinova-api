from fastapi import APIRouter, HTTPException, status
from typing import List
from bson import ObjectId

from app.models.pydantic_models import (
    MedicalNote,
    NoteCreate,
    NoteUpdate,
    NoteResponse,
    NotesListResponse,
    ExtractionResponse,
    CodeExtractionResult
)
from app.services.openai_service import openai_service
from app.repositories.medical_notes import MedicalNotesRepository

router = APIRouter(prefix="/api/v1", tags=["medical-notes"])

# Initialize repository
notes_repository = MedicalNotesRepository()

@router.get("/notes", response_model=NotesListResponse)
async def get_all_notes():
    """Get all medical notes with their extracted codes."""
    try:
        notes = await notes_repository.get_all_notes()
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
async def get_note(note_id: str):
    """Get a specific medical note by ID."""
    try:
        note = await notes_repository.get_note_by_id(ObjectId(note_id))
        if not note:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Note with ID {note_id} not found"
            )
        return NoteResponse(
            message="Note retrieved successfully",
            note=note
        )
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve note: {str(e)}"
        )

@router.post("/notes", response_model=NoteResponse, status_code=status.HTTP_201_CREATED)
async def create_note(note: NoteCreate):
    """Create a new medical note."""
    try:
        # Create note without extracted codes initially
        new_note = await notes_repository.create_note(note)
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
async def update_note(note_id: str, note_update: NoteUpdate):
    """Update an existing medical note."""
    try:
        updated_note = await notes_repository.update_note(ObjectId(note_id), note_update)
        if not updated_note:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Note with ID {note_id} not found"
            )
        return NoteResponse(
            message="Note updated successfully",
            note=updated_note
        )
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update note: {str(e)}"
        )

@router.post("/extract", response_model=ExtractionResponse)
async def extract_codes(note_id: str):
    """Extract medical codes from a note's text using OpenAI."""
    try:
        # Get the note
        note = await notes_repository.get_note_by_id(ObjectId(note_id))
        if not note:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Note with ID {note_id} not found"
            )

        # Extract codes using OpenAI
        extraction_result = await openai_service.extract_codes(note.note_text)

        # Update note with extracted codes
        note_update = NoteUpdate(extraction_result=extraction_result)
        updated_note = await notes_repository.update_note(ObjectId(note_id), note_update)

        return ExtractionResponse(
            message="Codes extracted successfully",
            extraction_result=extraction_result
        )

    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to extract codes: {str(e)}"
        )
