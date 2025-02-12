from datetime import datetime
from typing import Optional, List, Dict
from bson import ObjectId
from ..database.mongodb import db
import logging

logger = logging.getLogger(__name__)

class MedicalNotesRepository:
    async def create_note(self, text: str, source: str = "API") -> str:
        """Create a new medical note"""
        try:
            note = {
                "text": text,
                "source": source,
                "created_at": datetime.utcnow(),
                "length": len(text),
                "status": "PROCESSING"
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
            result = await db.medical_notes.update_one(
                {"_id": ObjectId(note_id)},
                {
                    "$set": {
                        "extraction": extraction_data,
                        "status": "COMPLETED",
                        "updated_at": datetime.utcnow()
                    }
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