from typing import List, Optional
from bson import ObjectId
from datetime import datetime

from app.models.pydantic_models import (
    MedicalNote,
    NoteCreate,
    NoteUpdate,
    CodeExtractionResult
)
from app.database.mongodb import get_database

class MedicalNotesRepository:
    def __init__(self):
        self.db = get_database()
        self.collection = self.db.medical_notes

    async def get_all_notes(self) -> List[MedicalNote]:
        """
        Retrieve all medical notes from the database.
        """
        try:
            cursor = self.collection.find()
            notes = []
            async for document in cursor:
                notes.append(MedicalNote(**document))
            return notes
        except Exception as e:
            print(f"Error retrieving all notes: {e}")
            raise

    async def get_note_by_id(self, note_id: ObjectId) -> Optional[MedicalNote]:
        """
        Retrieve a specific medical note by ID.
        """
        try:
            document = await self.collection.find_one({"_id": note_id})
            if document:
                return MedicalNote(**document)
            return None
        except Exception as e:
            print(f"Error retrieving note {note_id}: {e}")
            raise

    async def create_note(self, note: NoteCreate) -> MedicalNote:
        """
        Create a new medical note.
        """
        try:
            # Prepare the document
            note_dict = note.dict()
            if not note_dict.get('date'):
                note_dict['date'] = datetime.now()
            
            # Initialize empty extraction result
            note_dict['extraction_result'] = None

            # Insert into database
            result = await self.collection.insert_one(note_dict)
            
            # Retrieve the created note
            created_note = await self.get_note_by_id(result.inserted_id)
            if not created_note:
                raise ValueError("Failed to retrieve created note")
            
            return created_note
        except Exception as e:
            print(f"Error creating note: {e}")
            raise

    async def update_note(self, note_id: ObjectId, note_update: NoteUpdate) -> Optional[MedicalNote]:
        """
        Update an existing medical note.
        """
        try:
            # Get the existing note
            existing_note = await self.get_note_by_id(note_id)
            if not existing_note:
                return None

            # Prepare update data (only non-None values)
            update_data = {
                k: v for k, v in note_update.dict(exclude_unset=True).items()
                if v is not None
            }

            if not update_data:
                return existing_note

            # Update the document
            await self.collection.update_one(
                {"_id": note_id},
                {"$set": update_data}
            )

            # Retrieve and return the updated note
            updated_note = await self.get_note_by_id(note_id)
            if not updated_note:
                raise ValueError("Failed to retrieve updated note")

            return updated_note
        except Exception as e:
            print(f"Error updating note {note_id}: {e}")
            raise

    async def update_extraction_result(
        self,
        note_id: ObjectId,
        extraction_result: CodeExtractionResult
    ) -> Optional[MedicalNote]:
        """
        Update the extraction result for a specific note.
        """
        try:
            # Update the document with new extraction result
            await self.collection.update_one(
                {"_id": note_id},
                {"$set": {"extraction_result": extraction_result.dict()}}
            )

            # Retrieve and return the updated note
            updated_note = await self.get_note_by_id(note_id)
            if not updated_note:
                raise ValueError("Failed to retrieve updated note")

            return updated_note
        except Exception as e:
            print(f"Error updating extraction result for note {note_id}: {e}")
            raise

    async def delete_note(self, note_id: ObjectId) -> bool:
        """
        Delete a medical note by ID.
        """
        try:
            result = await self.collection.delete_one({"_id": note_id})
            return result.deleted_count > 0
        except Exception as e:
            print(f"Error deleting note {note_id}: {e}")
            raise
