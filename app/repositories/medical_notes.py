from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from bson import ObjectId
from ..database.mongodb import db
import logging
from pymongo import ASCENDING, DESCENDING
from ..models.pydantic_models import (
    DashboardStatistics,
    StatusBreakdown,
    TypeBreakdown,
    CodeFrequency,
    TimeSeriesPoint,
    ExtractionStatus,
    NoteType,
    NotesFilterParams,
    NotesListingParams,
    NotesSummary,
    NotesListingResponse,
    MedicalNote,
    SortOrder
)

logger = logging.getLogger(__name__)

class MedicalNotesRepository:
    """Repository for managing medical notes in MongoDB."""

    def _process_date_field(self, date_value: Optional[Any]) -> Optional[datetime]:
        """Convert MongoDB date fields to datetime format."""
        if isinstance(date_value, datetime):
            return date_value
        if isinstance(date_value, dict) and "$date" in date_value:
            return datetime.fromtimestamp(int(date_value["$date"]["$numberLong"]) / 1000)
        return None

    def _safe_divide(self, numerator: float, denominator: float, default: float = 0.0) -> float:
        """Safely perform division, avoiding divide-by-zero errors."""
        try:
            return numerator / (denominator if denominator != 0 else 1)
        except Exception as e:
            logger.error(f"Division error: {e} (numerator={numerator}, denominator={denominator})")
            return default

    def _basic_sanitize(self, data: Dict) -> Dict:
        """Perform basic sanitization of input data."""
        sanitized = {}
        try:
            for key, value in data.items():
                # Required field validation
                if key in ["patient_id", "note_text"] and not value:
                    raise ValueError(f"Required field {key} cannot be empty")
                    
                if value is None:
                    continue
                
                # Sanitize strings
                if isinstance(value, str):
                    value = ''.join(char for char in value if ord(char) >= 32)
                    value = value[:1000000]  # Prevent excessive string lengths
                
                # Sanitize nested dictionaries
                elif isinstance(value, dict):
                    value = self._basic_sanitize(value)
                
                # Sanitize lists
                elif isinstance(value, list):
                    value = value[:10000]  # Prevent excessive list lengths
                    value = [
                        self._basic_sanitize(item) if isinstance(item, dict)
                        else item
                        for item in value
                    ]
                
                sanitized[key] = value
            
            return sanitized
        except Exception as e:
            logger.error(f"Error sanitizing data: {str(e)}")
            raise

    async def create_note(self, note_data: Dict) -> Optional[str]:
        """Create a new medical note in the database."""
        try:
            # Basic sanitization of input
            sanitized_data = self._basic_sanitize(note_data)
            
            # Add metadata
            sanitized_data.update({
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "status": ExtractionStatus.PROCESSING.value,  # Changed from PENDING
                "extraction_attempts": 0
            })
            
            result = await db.medical_notes.insert_one(sanitized_data)
            return str(result.inserted_id) if result.inserted_id else None
            
        except Exception as e:
            logger.error(f"Error creating note: {str(e)}")
            raise

    async def get_note(self, note_id: str) -> Optional[MedicalNote]:
        """Retrieve a single note by ID with full metadata."""
        try:
            if not ObjectId.is_valid(note_id):
                logger.error(f"Invalid note ID format: {note_id}")
                return None

            note = await db.medical_notes.find_one({"_id": ObjectId(note_id)})
            
            if not note:
                logger.info(f"Note not found: {note_id}")
                return None

            # Convert ObjectId to string with safe fallback
            note["id"] = str(note.pop("_id", ""))
            
            # Process date fields
            for date_field in ["created_at", "updated_at", "last_extraction_attempt"]:
                note[date_field] = self._process_date_field(note.get(date_field))

            # Handle confidence values in extraction data
            if "extraction" in note:
                for code_type in ["icd10_codes", "cpt_codes"]:
                    for code in note["extraction"].get(code_type, []):
                        if isinstance(code.get("confidence"), dict):
                            code["confidence"] = float(code["confidence"].get("$numberDouble", 0))

            return MedicalNote(**note)

        except Exception as e:
            logger.error(f"Error retrieving note {note_id}: {str(e)}")
            raise

    async def update_extraction(self, note_id: str, extraction_data: Dict) -> bool:
        """Update note with extraction results."""
        try:
            if not ObjectId.is_valid(note_id):
                logger.error(f"Invalid note ID format: {note_id}")
                return False

            # Prepare the $set update
            set_data = {
                "extraction": extraction_data,
                "status": ExtractionStatus.COMPLETED.value,
                "updated_at": datetime.utcnow()
            }
            
            if extraction_data.get("documentation_gaps"):
                set_data["has_documentation_gaps"] = True

            # Separate $set and $inc operations
            update_query = {
                "$set": set_data,
                "$inc": {"extraction_attempts": 1}
            }

            result = await db.medical_notes.update_one(
                {"_id": ObjectId(note_id)},
                update_query
            )
            
            success = result.modified_count > 0
            logger.info(f"Extraction update {'successful' if success else 'failed'} for note {note_id}")
            return success

        except Exception as e:
            logger.error(f"Failed to update extraction for note {note_id}: {str(e)}")
            raise

    async def get_recent_notes(self, limit: int = 10) -> List[MedicalNote]:
        """Retrieve most recent notes."""
        try:
            cursor = db.medical_notes.find({})\
                .sort("created_at", DESCENDING)\
                .limit(limit)
            
            notes = []
            async for note in cursor:
                try:
                    # Convert ObjectId to string with safe fallback
                    note["id"] = str(note.pop("_id", ""))
                    
                    # Process date fields
                    for date_field in ["created_at", "updated_at", "last_extraction_attempt"]:
                        note[date_field] = self._process_date_field(note.get(date_field))

                    # Process number fields
                    for number_field in ["length", "extraction_attempts"]:
                        if number_field in note and isinstance(note[number_field], dict) and "$numberInt" in note[number_field]:
                            note[number_field] = int(note[number_field]["$numberInt"])

                    # Process extraction metadata
                    if "extraction" in note and "metadata" in note["extraction"]:
                        metadata = note["extraction"]["metadata"]
                        if isinstance(metadata.get("processing_time_ms"), dict):
                            metadata["processing_time_ms"] = int(
                                metadata["processing_time_ms"].get("$numberInt", 0)
                            )
                        if isinstance(metadata.get("note_length"), dict):
                            metadata["note_length"] = int(
                                metadata["note_length"].get("$numberInt", 0)
                            )

                    notes.append(MedicalNote(**note))
                except Exception as e:
                    logger.error(f"Error processing note in get_recent_notes: {str(e)}")
                    continue

            return notes

        except Exception as e:
            logger.error(f"Error retrieving recent notes: {str(e)}")
            raise

    def _build_filter_query(self, filters: NotesFilterParams) -> Dict:
        """Build MongoDB query from filter parameters."""
        query = {}
        try:
            if filters:
                if filters.note_type:
                    if not isinstance(filters.note_type, NoteType):
                        raise ValueError("Invalid note_type")
                    query["extraction.note_type"] = filters.note_type.value
                    
                if filters.status:
                    if not isinstance(filters.status, ExtractionStatus):
                        raise ValueError("Invalid status")
                    query["status"] = filters.status.value
                    
                if filters.search_text:
                    # Sanitize search text
                    clean_text = ''.join(char for char in filters.search_text 
                                       if char.isalnum() or char.isspace())
                    if clean_text:
                        query["$text"] = {"$search": clean_text}
                        
                if filters.start_date:
                    query["created_at"] = {"$gte": filters.start_date}
                if filters.end_date:
                    query.setdefault("created_at", {})["$lte"] = filters.end_date
                if filters.has_documentation_gaps:
                    query["extraction.documentation_gaps"] = {"$exists": True, "$ne": []}
                    
            return query
        except Exception as e:
            logger.error(f"Error building query: {str(e)}")
            raise

    async def get_notes_count(self, filters: NotesFilterParams) -> int:
        """Get total count of notes matching filters."""
        try:
            query = self._build_filter_query(filters)
            return await db.medical_notes.count_documents(query)
        except Exception as e:
            logger.error(f"Error getting notes count: {str(e)}")
            raise

    async def get_paginated_notes(
        self,
        page: int,
        page_size: int,
        sort_by: str,
        sort_order: SortOrder,
        filters: NotesFilterParams
    ) -> List[MedicalNote]:
        """Get paginated and filtered notes."""
        try:
            query = self._build_filter_query(filters)
            skip = (page - 1) * page_size
            
            sort_direction = DESCENDING if sort_order == SortOrder.DESC else ASCENDING
            cursor = db.medical_notes.find(query)\
                .sort(sort_by, sort_direction)\
                .skip(skip)\
                .limit(page_size)
            
            notes = []
            async for note in cursor:
                try:
                    # Convert ObjectId to string with safe fallback
                    note["id"] = str(note.pop("_id", ""))

                    # Process date fields
                    for date_field in ["created_at", "updated_at", "last_extraction_attempt"]:
                        note[date_field] = self._process_date_field(note.get(date_field))

                    # Process number fields
                    for number_field in ["length", "extraction_attempts"]:
                        if number_field in note and isinstance(note[number_field], dict) and "$numberInt" in note[number_field]:
                            note[number_field] = int(note[number_field]["$numberInt"])

                    # Handle nested number formats in extraction
                    if "extraction" in note and "metadata" in note["extraction"]:
                        metadata = note["extraction"]["metadata"]
                        if isinstance(metadata.get("processing_time_ms"), dict):
                            metadata["processing_time_ms"] = int(
                                metadata["processing_time_ms"].get("$numberInt", 0)
                            )
                        if isinstance(metadata.get("note_length"), dict):
                            metadata["note_length"] = int(
                                metadata["note_length"].get("$numberInt", 0)
                            )

                    # Handle confidence values in codes
                    if "extraction" in note:
                        for code_list in ["icd10_codes", "cpt_codes"]:
                            for code in note["extraction"].get(code_list, []):
                                if isinstance(code.get("confidence"), dict):
                                    code["confidence"] = float(code["confidence"].get("$numberDouble", 0))

                    notes.append(MedicalNote(**note))
                except Exception as e:
                    logger.error(f"Error processing note: {str(e)}")
                    logger.error(f"Problematic note: {note}")
                    continue

            return notes

        except Exception as e:
            logger.error(f"Error getting paginated notes: {str(e)}")
            raise

    async def get_dashboard_statistics(self, days: int = 30) -> DashboardStatistics:
        """Get comprehensive dashboard statistics."""
        try:
            start_date = datetime.utcnow() - timedelta(days=days)

            total_notes = await db.medical_notes.count_documents({})
            total_processed = await db.medical_notes.count_documents(
                {"status": ExtractionStatus.COMPLETED.value}
            )

            total_notes = max(total_notes, 1)  # Prevent zero-division

            # Status breakdown
            status_breakdown = [
                StatusBreakdown(
                    status=status,
                    count=await db.medical_notes.count_documents({"status": status.value}),
                    percentage=self._safe_divide(
                        await db.medical_notes.count_documents({"status": status.value}) * 100,
                        total_notes
                    )
                ) for status in ExtractionStatus
            ]

            # Type breakdown
            type_breakdown = [
                TypeBreakdown(
                    type=note_type,
                    count=await db.medical_notes.count_documents({"extraction.note_type": note_type.value}),
                    percentage=self._safe_divide(
                        await db.medical_notes.count_documents({"extraction.note_type": note_type.value}) * 100,
                        total_notes
                    )
                ) for note_type in NoteType
            ]

            # Processing statistics
            processing_pipeline = [
                {"$match": {"status": ExtractionStatus.COMPLETED.value}},
                {"$group": {"_id": None, "avg_time": {"$avg": "$extraction.metadata.processing_time_ms"}}}
            ]
            processing_stats = await db.medical_notes.aggregate(processing_pipeline).to_list(1)
            avg_processing_time = round(processing_stats[0]["avg_time"], 2) if processing_stats else 0

            # Fetch processed notes for quality calculations
            processed_notes = await db.medical_notes.find(
                {"status": ExtractionStatus.COMPLETED.value}
            ).to_list(None)

            documentation_quality = {
                "completeness": await self._calculate_documentation_completeness(processed_notes),
                "accuracy": await self._calculate_documentation_accuracy(processed_notes),
                "timeliness": await self._calculate_documentation_timeliness(processed_notes)
            }

            return DashboardStatistics(
                total_notes=total_notes,
                total_processed=total_processed,
                processing_success_rate=self._safe_divide(total_processed * 100, total_notes),
                avg_processing_time_ms=avg_processing_time,
                status_breakdown=status_breakdown or [],
                type_breakdown=type_breakdown or [],
                documentation_quality=documentation_quality
            )
        except Exception as e:
            logger.error(f"Failed to get dashboard statistics: {str(e)}")
            raise

    async def _calculate_documentation_completeness(self, notes: List[Dict]) -> float:
        """Calculate documentation completeness score."""
        try:
            if not notes:
                return 0.0

            total_score = sum(
                0.4 * (not note.get("extraction", {}).get("documentation_gaps", [])) +
                0.3 * self._safe_divide(
                    sum(code.get("confidence", 0) for code in 
                        note.get("extraction", {}).get("icd10_codes", []) + 
                        note.get("extraction", {}).get("cpt_codes", [])),
                    len(note.get("extraction", {}).get("icd10_codes", []) + 
                        note.get("extraction", {}).get("cpt_codes", []))
                ) +
                0.3 * any(code.get("evidence", {}).get("direct_quotes", []) 
                         for code in note.get("extraction", {}).get("icd10_codes", []) + 
                         note.get("extraction", {}).get("cpt_codes", []))
                for note in notes
            )
            return self._safe_divide(total_score * 100, len(notes))
        except Exception as e:
            logger.error(f"Error calculating documentation completeness: {str(e)}")
            return 0.0

    async def _calculate_documentation_accuracy(self, notes: List[Dict]) -> float:
        """Calculate accuracy of extracted codes."""
        try:
            if not notes:
                return 0.0

            total_score = sum(
                0.7 * self._safe_divide(
                    sum(code.get("confidence", 0) for code in 
                        note.get("extraction", {}).get("icd10_codes", []) + 
                        note.get("extraction", {}).get("cpt_codes", [])),
                    len(note.get("extraction", {}).get("icd10_codes", []) + 
                        note.get("extraction", {}).get("cpt_codes", []))
                ) +
                0.3 * all(len(code.get("evidence", {}).get("direct_quotes", [])) > 0 
                         for code in note.get("extraction", {}).get("icd10_codes", []) + 
                         note.get("extraction", {}).get("cpt_codes", []))
                for note in notes
            )
            return self._safe_divide(total_score * 100, len(notes))
        except Exception as e:
            logger.error(f"Error calculating accuracy: {str(e)}")
            return 0.0

    async def _calculate_documentation_timeliness(self, notes: List[Dict]) -> float:
        """Calculate timeliness of documentation processing."""
        try:
            if not notes:
                return 0.0

            total_score = sum(
                (1.0 if (updated_at - created_at).total_seconds() < 5 else 
                 0.8 if (updated_at - created_at).total_seconds() < 10 else 
                 0.6 if (updated_at - created_at).total_seconds() < 30 else 0.4) -
                max(0, (note.get('extraction_attempts', 1) - 1) * 0.1)
                for note in notes 
                if (created_at := self._process_date_field(note.get('created_at'))) and 
                   (updated_at := self._process_date_field(note.get('updated_at')))
            )
            return self._safe_divide(total_score * 100, len(notes))
        except Exception as e:
            logger.error(f"Error calculating timeliness: {str(e)}")
            return 0.0
