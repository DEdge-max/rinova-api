from typing import List, Optional
from bson import ObjectId
from datetime import datetime

from app.models.pydantic_models import (
    MedicalNote,
    NoteCreate,
    NoteUpdate,
    CodeExtractionResult
)
from app.database.mongodb import db
import logging

logger = logging.getLogger(__name__)

class MedicalNotesRepository:
    def __init__(self):
        self.db = None
        self.collection = None

    async def initialize(self):
        """Initialize database connection and ensure collection exists"""
        if self.db is None:
            try:
                self.db = db.get_db()
                # Ensure collection exists
                collections = await self.db.list_collection_names()
                if 'medical_notes' not in collections:
                    await self.db.create_collection('medical_notes')
                self.collection = self.db.medical_notes
                logger.info("✅ Medical notes collection initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize medical notes collection: {str(e)}")
                raise

    async def get_all_notes(self) -> List[MedicalNote]:
        """Retrieve all medical notes from the database."""
        await self.initialize()
        try:
            cursor = self.collection.find()
            notes = []
            async for document in cursor:
                notes.append(MedicalNote(**document))
            return notes
        except Exception as e:
            logger.error(f"Error retrieving all notes: {e}")
            raise

    async def get_note_by_id(self, note_id: ObjectId) -> Optional[MedicalNote]:
        """Retrieve a specific medical note by ID."""
        await self.initialize()
        try:
            document = await self.collection.find_one({"_id": note_id})
            if document:
                return MedicalNote(**document)
            return None
        except Exception as e:
            logger.error(f"Error retrieving note {note_id}: {e}")
            raise

    async def create_note(self, note: NoteCreate) -> MedicalNote:
        """Create a new medical note."""
        await self.initialize()
        try:
            note_dict = note.dict()
            if not note_dict.get('date'):
                note_dict['date'] = datetime.now()
            
            note_dict['extraction_result'] = None
            
            result = await self.collection.insert_one(note_dict)
            
            created_note = await self.get_note_by_id(result.inserted_id)
            if not created_note:
                raise ValueError("Failed to retrieve created note")
            
            return created_note
        except Exception as e:
            logger.error(f"Error creating note: {e}")
            raise

    async def update_note(self, note_id: ObjectId, note_update: NoteUpdate) -> Optional[MedicalNote]:
        """Update an existing medical note."""
        await self.initialize()
        try:
            existing_note = await self.get_note_by_id(note_id)
            if not existing_note:
                return None

            update_data = {
                k: v for k, v in note_update.dict(exclude_unset=True).items()
                if v is not None
            }

            if not update_data:
                return existing_note

            await self.collection.update_one(
                {"_id": note_id},
                {"$set": update_data}
            )

            updated_note = await self.get_note_by_id(note_id)
            if not updated_note:
                raise ValueError("Failed to retrieve updated note")

            return updated_note
        except Exception as e:
            logger.error(f"Error updating note {note_id}: {e}")
            raise

    async def update_extraction_result(
        self,
        note_id: ObjectId,
        extraction_result: CodeExtractionResult
    ) -> Optional[MedicalNote]:
        """Update the extraction result for a specific note."""
        await self.initialize()
        try:
            await self.collection.update_one(
                {"_id": note_id},
                {"$set": {"extraction_result": extraction_result.dict()}}
            )

            updated_note = await self.get_note_by_id(note_id)
            if not updated_note:
                raise ValueError("Failed to retrieve updated note")

            return updated_note
        except Exception as e:
            logger.error(f"Error updating extraction result for note {note_id}: {e}")
            raise
