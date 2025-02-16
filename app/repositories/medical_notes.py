from datetime import datetime
from typing import Optional, List, Dict
from bson import ObjectId
from ..database.mongodb import db
import logging
from pymongo import IndexModel, ASCENDING, DESCENDING, TEXT

logger = logging.getLogger(__name__)

class MedicalNotesRepository:
    def __init__(self):
        """Initialize repository and ensure indexes"""
        self.ensure_indexes()

    async def ensure_indexes(self):
        """Create necessary indexes for optimized queries"""
        try:
            indexes = [
                IndexModel([("created_at", DESCENDING)], background=True),
                IndexModel([("status", ASCENDING)], background=True),
                IndexModel([("text", TEXT)], background=True),
                IndexModel([("extraction.note_type", ASCENDING)], background=True),
                IndexModel([
                    ("extraction.icd10_codes.code", ASCENDING),
                    ("extraction.cpt_codes.code", ASCENDING)
                ], background=True)
            ]
            await db.medical_notes.create_indexes(indexes)
            logger.info("Database indexes created/updated successfully")
        except Exception as e:
            logger.error(f"Failed to create indexes: {str(e)}")
            # Don't raise the error as it's not critical for operation

    async def create_note(self, text: str, source: str = "API") -> str:
        """Create a new medical note"""
        try:
            note = {
                "text": text,
                "source": source,
                "created_at": datetime.utcnow(),
                "length": len(text),
                "status": "PENDING",  # Changed from PROCESSING to match the ExtractionStatus enum
                "extraction_attempts": 0,
                "last_extraction_attempt": None
            }
            logger.info(f"Attempting to save note: {note}")
            result = await db.medical_notes.insert_one(note)
            logger.info(f"Note saved with ID: {result.inserted_id}")
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Failed to create note: {str(e)}")
            raise

    async def update_extraction(self, note_id: str, extraction_data: Dict) -> bool:
        """Update note with extraction results"""
        try:
            logger.info(f"Updating note {note_id} with extraction data")
            update_data = {
                "extraction": extraction_data,
                "status": "COMPLETED",
                "updated_at": datetime.utcnow(),
                "last_extraction_attempt": datetime.utcnow(),
                "$inc": {"extraction_attempts": 1}
            }
            
            # Remove $inc from the top level for the update
            inc_data = update_data.pop("$inc")
            
            result = await db.medical_notes.update_one(
                {"_id": ObjectId(note_id)},
                {
                    "$set": update_data,
                    "$inc": inc_data
                }
            )
            success = result.modified_count > 0
            logger.info(f"Update {'successful' if success else 'failed'} for note {note_id}")
            return success
        except Exception as e:
            logger.error(f"Failed to update note {note_id}: {str(e)}")
            raise

    async def get_note(self, note_id: str) -> Optional[Dict]:
        """Get a single note by ID"""
        try:
            note = await db.medical_notes.find_one({"_id": ObjectId(note_id)})
            if note:
                note["_id"] = str(note["_id"])
                logger.info(f"Retrieved note {note_id}")
            else:
                logger.warning(f"Note {note_id} not found")
            return note
        except Exception as e:
            logger.error(f"Failed to get note {note_id}: {str(e)}")
            raise

    async def get_recent_notes(self, limit: int = 10) -> List[Dict]:
        """Get recent notes with their extractions"""
        try:
            cursor = db.medical_notes.find({}).sort("created_at", -1).limit(limit)
            notes = []
            async for note in cursor:
                note["_id"] = str(note["_id"])
                notes.append(note)
            logger.info(f"Retrieved {len(notes)} recent notes")
            return notes
        except Exception as e:
            logger.error(f"Failed to get recent notes: {str(e)}")
            raise

    async def update_note_status(self, note_id: str, status: str) -> bool:
        """Update the status of a note"""
        try:
            result = await db.medical_notes.update_one(
                {"_id": ObjectId(note_id)},
                {
                    "$set": {
                        "status": status,
                        "updated_at": datetime.utcnow(),
                        "last_extraction_attempt": datetime.utcnow()
                    },
                    "$inc": {"extraction_attempts": 1}
                }
            )
            success = result.modified_count > 0
            logger.info(f"Status update to {status} {'successful' if success else 'failed'} for note {note_id}")
            return success
        except Exception as e:
            logger.error(f"Failed to update status for note {note_id}: {str(e)}")
            raise

    async def get_notes_by_type(self, note_type: str, limit: int = 10) -> List[Dict]:
        """Get notes of a specific type"""
        try:
            cursor = db.medical_notes.find(
                {"extraction.note_type": note_type}
            ).sort("created_at", -1).limit(limit)
            
            notes = []
            async for note in cursor:
                note["_id"] = str(note["_id"])
                notes.append(note)
            
            logger.info(f"Retrieved {len(notes)} notes of type {note_type}")
            return notes
        except Exception as e:
            logger.error(f"Failed to get notes by type {note_type}: {str(e)}")
            raise
