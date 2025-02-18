import logging
from typing import List
from bson import ObjectId
from app.models.pydantic_models import MedicalNote

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Adjust to DEBUG for more detailed logs

class MedicalNotesRepository:
    def __init__(self, database):
        self.database = database
        self.collection = self.database["medical_notes"]

    async def initialize(self):
        """Ensure the collection is initialized (if needed for async connections)."""
        pass  # Add any necessary database initialization logic here

    async def get_all_notes(self) -> List[MedicalNote]:
        """Retrieve all medical notes from the database."""
        await self.initialize()
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
