from fastapi import APIRouter, HTTPException, status, Depends
from typing import List
from bson import ObjectId
from bson.errors import InvalidId

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
