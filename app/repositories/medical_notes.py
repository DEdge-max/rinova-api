from datetime import datetime, timedelta
from typing import List, Dict, Optional
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

    async def get_notes_count(self, filters: NotesFilterParams) -> int:
        """Get total count of notes matching filters"""
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
        """Get paginated and filtered notes"""
        try:
            query = self._build_filter_query(filters)
            skip = (page - 1) * page_size
            logger.info(f"Executing query: {query}")
            
            # Determine sort direction
            sort_direction = DESCENDING if sort_order == SortOrder.DESC else ASCENDING
            cursor = db.medical_notes.find(query)\
                .sort(sort_by, sort_direction)\
                .skip(skip)\
                .limit(page_size)
            
            notes = []
            async for note in cursor:
                try:
                    # Convert ObjectId to string
                    if "_id" in note:
                        note["id"] = str(note["_id"])
                        del note["_id"]

                    # Handle MongoDB date format
                    for date_field in ["created_at", "updated_at", "last_extraction_attempt"]:
                        if date_field in note and isinstance(note[date_field], dict) and "$date" in note[date_field]:
                            note[date_field] = datetime.fromtimestamp(
                                int(note[date_field]["$date"]["$numberLong"]) / 1000
                            )

                    # Handle MongoDB number format
                    for number_field in ["length", "extraction_attempts"]:
                        if number_field in note and isinstance(note[number_field], dict) and "$numberInt" in note[number_field]:
                            note[number_field] = int(note[number_field]["$numberInt"])

                    # Handle nested number formats in extraction
                    if "extraction" in note and "metadata" in note["extraction"]:
                        if isinstance(note["extraction"]["metadata"].get("processing_time_ms"), dict):
                            note["extraction"]["metadata"]["processing_time_ms"] = int(
                                note["extraction"]["metadata"]["processing_time_ms"].get("$numberInt", 0)
                            )
                        if isinstance(note["extraction"]["metadata"].get("note_length"), dict):
                            note["extraction"]["metadata"]["note_length"] = int(
                                note["extraction"]["metadata"]["note_length"].get("$numberInt", 0)
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

    def _build_filter_query(self, filters: NotesFilterParams) -> Dict:
        """Build MongoDB query from filter parameters"""
        query = {}  # Start with empty query to get all documents if no filters
        try:
            if filters:
                if filters.note_type:
                    query["extraction.note_type"] = filters.note_type.value
                if filters.status:
                    query["status"] = filters.status.value
                if filters.search_text:
                    query["$text"] = {"$search": filters.search_text}
                if filters.start_date:
                    query["created_at"] = {"$gte": {
                        "$date": {"$numberLong": str(int(filters.start_date.timestamp() * 1000))}
                    }}
                if filters.end_date:
                    query.setdefault("created_at", {})["$lte"] = {
                        "$date": {"$numberLong": str(int(filters.end_date.timestamp() * 1000))}
                    }
                if filters.has_documentation_gaps:
                    query["extraction.documentation_gaps"] = {"$exists": True, "$ne": []}
            logger.info(f"Built MongoDB query: {query}")
            return query
        except Exception as e:
            logger.error(f"Error building query: {str(e)}")
            return {}  # Return empty query in case of errors

    async def get_dashboard_statistics(self, days: int = 30) -> DashboardStatistics:
        """Get comprehensive dashboard statistics"""
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
                    percentage=round(
                        (await db.medical_notes.count_documents({"status": status.value})) / total_notes * 100, 2
                    )
                ) for status in ExtractionStatus
            ]

            # Type breakdown
            type_breakdown = [
                TypeBreakdown(
                    type=note_type,
                    count=await db.medical_notes.count_documents({"extraction.note_type": note_type.value}),
                    percentage=round(
                        (await db.medical_notes.count_documents({"extraction.note_type": note_type.value})) / total_notes * 100, 2
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
                processing_success_rate=round(total_processed / total_notes * 100, 2),
                avg_processing_time_ms=avg_processing_time,
                status_breakdown=status_breakdown or [],
                type_breakdown=type_breakdown or [],
                documentation_quality=documentation_quality
            )
        except Exception as e:
            logger.error(f"Failed to get dashboard statistics: {str(e)}")
            raise

    async def _calculate_documentation_completeness(self, notes: List[Dict]) -> float:
        """Calculate documentation completeness"""
        try:
            total_score = sum(
                0.4 * (not note.get("extraction", {}).get("documentation_gaps", [])) +
                0.3 * sum(code.get("confidence", 0) for code in note.get("extraction", {}).get("icd10_codes", []) + note.get("extraction", {}).get("cpt_codes", [])) /
                max(len(note.get("extraction", {}).get("icd10_codes", []) + note.get("extraction", {}).get("cpt_codes", [])), 1) +
                0.3 * any(code.get("evidence", {}).get("direct_quotes", []) for code in note.get("extraction", {}).get("icd10_codes", []) + note.get("extraction", {}).get("cpt_codes", []))
                for note in notes
            )
            return (total_score / max(len(notes), 1)) * 100
        except Exception as e:
            logger.error(f"Error calculating documentation completeness: {str(e)}")
            return 0.0

    async def _calculate_documentation_accuracy(self, notes: List[Dict]) -> float:
        """Calculate accuracy of extracted codes"""
        try:
            total_score = sum(
                0.7 * sum(code.get("confidence", 0) for code in note.get("extraction", {}).get("icd10_codes", []) + note.get("extraction", {}).get("cpt_codes", [])) /
                max(len(note.get("extraction", {}).get("icd10_codes", []) + note.get("extraction", {}).get("cpt_codes", [])), 1) +
                0.3 * all(len(code.get("evidence", {}).get("direct_quotes", [])) > 0 for code in note.get("extraction", {}).get("icd10_codes", []) + note.get("extraction", {}).get("cpt_codes", []))
                for note in notes
            )
            return (total_score / max(len(notes), 1)) * 100
        except Exception as e:
            logger.error(f"Error calculating accuracy: {str(e)}")
            return 0.0

    async def _calculate_documentation_timeliness(self, notes: List[Dict]) -> float:
        """Calculate timeliness of documentation processing"""
        try:
            total_score = sum(
                (1.0 if (updated_at - created_at).total_seconds() < 5 else 
                 0.8 if (updated_at - created_at).total_seconds() < 10 else 
                 0.6 if (updated_at - created_at).total_seconds() < 30 else 0.4) -
                max(0, (note.get('extraction_attempts', 1) - 1) * 0.1)
                for note in notes if (created_at := note.get('created_at')) and (updated_at := note.get('updated_at'))
            )
            return (total_score / max(len(notes), 1)) * 100
        except Exception as e:
            logger.error(f"Error calculating timeliness: {str(e)}")
            return 0.0
