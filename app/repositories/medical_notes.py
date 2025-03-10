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
logger.setLevel(logging.INFO)


class MedicalNotesRepository:
    def __init__(self):
        """Initialize repository with lazy database connection."""
        self.database = None
        self.collection = None

    async def initialize(self):
        """Ensure the database connection is established."""
        if self.database is None:  # ✅ FIXED: Explicit check
            self.database = db.get_db()
            if self.database is None:
                raise RuntimeError("Database connection failed: db.get_db() returned None")

            self.collection = self.database["medical_notes"]
            collections = await self.database.list_collection_names()
            if "medical_notes" not in collections:
                await self.database.create_collection("medical_notes")
            logger.info("✅ Medical notes collection initialized")

    async def get_all_notes(self) -> List[MedicalNote]:
        """Retrieve all medical notes from the database."""
        await self.initialize()
        try:
            # First repair any documents with missing fields
            await self.repair_missing_fields()
            
            cursor = self.collection.find()
            notes = []
            async for document in cursor:
                logger.debug(f"Raw document from DB: {document}")
                try:
                    # Ensure extraction_result always exists
                    document.setdefault("extraction_result", {
                        "icd10_codes": [],
                        "cpt_codes": [],
                        "alternative_cpts": [],
                        "modifiers": [],
                        "hcpcs_codes": []
                    })
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
        await self.initialize()
        try:
            document = await self.collection.find_one({"_id": ObjectId(note_id)})
            if document:
                # Ensure extraction_result always exists
                document.setdefault("extraction_result", {
                    "icd10_codes": [],
                    "cpt_codes": [],
                    "alternative_cpts": [],
                    "modifiers": [],
                    "hcpcs_codes": []
                })
                return MedicalNote(**document)
            return None
        except Exception as e:
            logger.error(f"Error retrieving note with ID {note_id}: {e}")
            return None

    async def create_note(self, note_data: NoteCreate) -> str:
        """Insert a new medical note into the database."""
        await self.initialize()
        try:
            note_dict = note_data.dict()
            note_dict["created_at"] = datetime.utcnow()
            note_dict["updated_at"] = datetime.utcnow()
            note_dict["extraction_result"] = {  # ✅ Ensure extraction_result exists
                "icd10_codes": [],
                "cpt_codes": [],
                "alternative_cpts": [],
                "modifiers": [],
                "hcpcs_codes": []
            }
            result = await self.collection.insert_one(note_dict)
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error creating note: {e}")
            raise

    async def update_note(self, note_id: str, update_data: NoteUpdate) -> bool:
        """Update an existing medical note."""
        await self.initialize()
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
        await self.initialize()
        try:
            result = await self.collection.delete_one({"_id": ObjectId(note_id)})
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"Error deleting note {note_id}: {e}")
            return False

    async def extract_codes_for_note(self, note_id: str, extraction_result: CodeExtractionResult) -> bool:
        """Attach extracted ICD codes to a medical note."""
        await self.initialize()
        try:
            result = await self.collection.update_one(
                {"_id": ObjectId(note_id)},
                {"$set": {"extraction_result": extraction_result.dict(), "updated_at": datetime.utcnow()}}
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Error attaching codes to note {note_id}: {e}")
            return False

    async def repair_missing_fields(self):
        """Repair documents with missing required fields."""
        await self.initialize()
        try:
            default_update = {
                "$set": {
                    "doctor_name": "Unknown",
                    "patient_name": "Unknown",
                    "note_text": "",
                    "date": datetime.utcnow(),
                    "extraction_result": {
                        "icd10_codes": [],
                        "cpt_codes": [],
                        "alternative_cpts": [],
                        "modifiers": [],
                        "hcpcs_codes": []
                    }
                }
            }
            
            result = await self.collection.update_many(
                {
                    "$or": [
                        {"doctor_name": {"$exists": False}},
                        {"patient_name": {"$exists": False}},
                        {"note_text": {"$exists": False}},
                        {"date": {"$exists": False}},
                        {"extraction_result": {"$exists": False}}
                    ]
                },
                default_update,
                upsert=False
            )
            logger.info(f"Repaired {result.modified_count} documents with missing fields")
        except Exception as e:
            logger.error(f"Error repairing documents: {e}")
