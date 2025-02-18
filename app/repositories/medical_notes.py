import logging
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

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Adjust to DEBUG for more detailed logs


class MedicalNotesRepository:
    def __init__(self, database):
        self.database = database
        self.collection = self.database["medical_notes"]

    async def initialize(self):
        """Ensure the collection is initialized."""
        collections = await self.database.list_collection_names()
        if "medical_notes" not in collections:
            await self.database.create_collection("medical_notes")
        logger.info("✅ Medical notes collection initialized")

    async def get_all_notes(self) -> List[MedicalNote]:
        """Retrieve all medical notes from the database."""
        try:
            cursor = self.collection.find()
            notes = []
            async for document in cursor:
                logger.debug(f"Raw document from DB: {document}")  # Debug log
                try:
                    note = MedicalNote(**document)
                    notes.append(note)
                except Exception as e:
                    logger.error(f"Error parsing document {document.get('_id')}: {e}")
            return notes
        except Exception as e:
            logger.error(f"Error retrieving all notes: {e}")
            raise

    async def get_note_by_id(self, note_id: str) -> Optional[MedicalNote]:
        """Retrieve a medical note by its ID."""
        try:
            document = await self.collection.find_one({"_id": ObjectId(note_id)})
            if document:
                return MedicalNote(**document)
            return None
        except Exception as e:
            logger.error(f"Error retrieving note with ID {note_id}: {e}")
            return None

    async def create_note(self, note_data: NoteCreate) -> str:
        """Insert a new medical note into the database."""
        try:
            note_dict = note_data.dict()
            note_dict["created_at"] = datetime.utcnow()
            note_dict["updated_at"] = datetime.utcnow()
            result = await self.collection.insert_one(note_dict)
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error creating note: {e}")
            raise

    async def update_note(self, note_id: str, update_data: NoteUpdate) -> bool:
        """Update an existing medical note."""
        try:
            update_dict = {k: v for k, v in update_data.dict(exclude_unset=True).items()}
            update_dict["updated_at"] = datetime.utcnow()
            result = await self.collection.update_one(
                {"_id": ObjectId(note_id)},
                {"$set": update_dict}
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Error updating note {note_id}: {e}")
            return False

    async def delete_note(self, note_id: str) -> bool:
        """Delete a medical note by its ID."""
        try:
            result = await self.collection.delete_one({"_id": ObjectId(note_id)})
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"Error deleting note {note_id}: {e}")
            return False

    async def extract_codes_for_note(self, note_id: str, extraction_result: CodeExtractionResult) -> bool:
        """Attach extracted ICD codes to a medical note."""
        try:
            result = await self.collection.update_one(
                {"_id": ObjectId(note_id)},
                {"$set": {"icd_codes": extraction_result.codes, "updated_at": datetime.utcnow()}}
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Error attaching codes to note {note_id}: {e}")
            return False
