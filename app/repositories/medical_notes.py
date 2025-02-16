from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
from bson import ObjectId
from ..database.mongodb import db
import logging
from pymongo import IndexModel, ASCENDING, DESCENDING, TEXT
from ..models.pydantic_models import (
    NotesFilterParams,
    NotesListingParams,
    NotesSummary,
    DashboardStatistics,
    StatusBreakdown,
    TypeBreakdown,
    CodeFrequency,
    TimeSeriesPoint,
    ExtractionStatus,
    NoteType
)

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
                IndexModel([("patient_id", ASCENDING)], background=True),
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

    async def create_note(self, text: str, patient_id: Optional[str] = None, source: str = "API") -> str:
        """Create a new medical note"""
        try:
            note = {
                "text": text,
                "source": source,
                "patient_id": patient_id,
                "created_at": datetime.utcnow(),
                "length": len(text),
                "status": ExtractionStatus.PENDING.value,
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
                "status": ExtractionStatus.COMPLETED.value,
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

    async def build_filter_query(self, filters: Optional[NotesFilterParams]) -> Dict:
        """Build MongoDB query from filter parameters"""
        query = {}
        if not filters:
            return query

        if filters.note_type:
            query["extraction.note_type"] = filters.note_type.value
        if filters.status:
            query["status"] = filters.status.value
        if filters.patient_id:
            query["patient_id"] = filters.patient_id
        if filters.start_date:
            query["created_at"] = {"$gte": filters.start_date}
        if filters.end_date:
            if "created_at" in query:
                query["created_at"]["$lte"] = filters.end_date
            else:
                query["created_at"] = {"$lte": filters.end_date}
        if filters.search_query:
            query["$text"] = {"$search": filters.search_query}
        
        return query

    async def get_notes_listing(self, params: NotesListingParams) -> Tuple[List[Dict], NotesSummary]:
        """Get paginated notes listing with filters"""
        try:
            query = await self.build_filter_query(params.filters)
            
            # Calculate pagination values
            skip = (params.page - 1) * params.page_size
            
            # Get total count for pagination
            total_notes = await db.medical_notes.count_documents(query)
            total_pages = (total_notes + params.page_size - 1) // params.page_size
            
            # Get sorted and paginated notes
            sort_direction = ASCENDING if params.sort_order.value == "asc" else DESCENDING
            cursor = db.medical_notes.find(query) \
                .sort(params.sort_by, sort_direction) \
                .skip(skip) \
                .limit(params.page_size)
            
            notes = []
            async for note in cursor:
                note["_id"] = str(note["_id"])
                notes.append(note)
            
            # Create summary
            summary = NotesSummary(
                total_notes=total_notes,
                total_pages=total_pages,
                current_page=params.page,
                has_next=params.page < total_pages,
                has_previous=params.page > 1
            )
            
            logger.info(f"Retrieved {len(notes)} notes for page {params.page}")
            return notes, summary
        except Exception as e:
            logger.error(f"Failed to get notes listing: {str(e)}")
            raise

    async def get_dashboard_statistics(self, days: int = 30) -> DashboardStatistics:
        """Get comprehensive dashboard statistics"""
        try:
            start_date = datetime.utcnow() - timedelta(days=days)
            
            # Get basic counts
            total_notes = await db.medical_notes.count_documents({})
            total_processed = await db.medical_notes.count_documents({
                "status": ExtractionStatus.COMPLETED.value
            })
            
            # Calculate status breakdown
            status_breakdown = []
            for status in ExtractionStatus:
                count = await db.medical_notes.count_documents({"status": status.value})
                percentage = (count / total_notes * 100) if total_notes > 0 else 0
                status_breakdown.append(StatusBreakdown(
                    status=status,
                    count=count,
                    percentage=round(percentage, 2)
                ))
            
            # Calculate type breakdown
            type_breakdown = []
            for note_type in NoteType:
                count = await db.medical_notes.count_documents({
                    "extraction.note_type": note_type.value
                })
                percentage = (count / total_notes * 100) if total_notes > 0 else 0
                type_breakdown.append(TypeBreakdown(
                    type=note_type,
                    count=count,
                    percentage=round(percentage, 2)
                ))
            
            # Get processing statistics
            processing_pipeline = [
                {"$match": {"status": ExtractionStatus.COMPLETED.value}},
                {"$group": {
                    "_id": None,
                    "avg_time": {"$avg": "$extraction.metadata.processing_time_ms"}
                }}
            ]
            processing_stats = await db.medical_notes.aggregate(processing_pipeline).to_list(1)
            avg_processing_time = processing_stats[0]["avg_time"] if processing_stats else 0
            
            # Get common codes
            icd10_pipeline = [
                {"$unwind": "$extraction.icd10_codes"},
                {"$group": {
                    "_id": {
                        "code": "$extraction.icd10_codes.code",
                        "description": "$extraction.icd10_codes.description"
                    },
                    "count": {"$sum": 1}
                }},
                {"$sort": {"count": -1}},
                {"$limit": 10}
            ]
            
            cpt_pipeline = [
                {"$unwind": "$extraction.cpt_codes"},
                {"$group": {
                    "_id": {
                        "code": "$extraction.cpt_codes.code",
                        "description": "$extraction.cpt_codes.description"
                    },
                    "count": {"$sum": 1}
                }},
                {"$sort": {"count": -1}},
                {"$limit": 10}
            ]
            
            icd10_results = await db.medical_notes.aggregate(icd10_pipeline).to_list(10)
            cpt_results = await db.medical_notes.aggregate(cpt_pipeline).to_list(10)
            
            # Calculate daily extraction counts
            daily_pipeline = [
                {"$match": {"created_at": {"$gte": start_date}}},
                {"$group": {
                    "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$created_at"}},
                    "count": {"$sum": 1}
                }},
                {"$sort": {"_id": 1}}
            ]
            daily_counts = await db.medical_notes.aggregate(daily_pipeline).to_list(None)
            
            # Format code frequencies
            common_icd10_codes = [
                CodeFrequency(
                    code=r["_id"]["code"],
                    description=r["_id"]["description"],
                    count=r["count"],
                    percentage=round(r["count"] / total_processed * 100, 2)
                ) for r in icd10_results
            ]
            
            common_cpt_codes = [
                CodeFrequency(
                    code=r["_id"]["code"],
                    description=r["_id"]["description"],
                    count=r["count"],
                    percentage=round(r["count"] / total_processed * 100, 2)
                ) for r in cpt_results
            ]
            
            # Format time series data
            daily_extraction_counts = [
                TimeSeriesPoint(
                    date=datetime.strptime(d["_id"], "%Y-%m-%d"),
                    count=d["count"]
                ) for d in daily_counts
            ]
            
            # Calculate documentation quality metrics
            documentation_quality = {
                "completeness": await self._calculate_documentation_completeness(),
                "accuracy": await self._calculate_documentation_accuracy(),
                "timeliness": await self._calculate_documentation_timeliness()
            }
            
            return DashboardStatistics(
                total_notes=total_notes,
                total_processed=total_processed,
                processing_success_rate=round(total_processed / total_notes * 100, 2) if total_notes > 0 else 0,
                avg_processing_time_ms=round(avg_processing_time, 2),
                status_breakdown=status_breakdown,
                type_breakdown=type_breakdown,
                common_icd10_codes=common_icd10_codes,
                common_cpt_codes=common_cpt_codes,
                daily_extraction_counts=daily_extraction_counts,
                documentation_quality=documentation_quality
            )
        except Exception as e:
            logger.error(f"Failed to get dashboard statistics: {str(e)}")
            raise

    async def _calculate_documentation_completeness(self) -> float:
        """Calculate documentation completeness score"""
        try:
            pipeline = [
                {"$match": {"status": ExtractionStatus.COMPLETED.value}},
                {"$project": {
                    "has_icd10": {"$cond": [{"$gt": [{"$size": "$extraction.icd10_codes"}, 0]}, 1, 0]},
                    "has_cpt": {"$cond": [{"$gt": [{"$size": "$extraction.cpt_codes"}, 0]}, 1, 0]},
                    "has_gaps": {"$cond": [{"$gt": [{"$size": "$extraction.documentation_gaps"}, 0]}, 0, 1]}
                }},
                {"$group": {
                    "_id": None,
                    "avg_completeness": {
                        "$avg": {"$divide": [{"$add": ["$has_icd10", "$has_cpt", "$has_gaps"]}, 3]}
                    }
                }}
            ]
            result = await db.medical_notes.aggregate(pipeline).to_list(1)
            return round(result[0]["avg_completeness"] * 100, 2) if result else 0
        except Exception as e:
            logger.error(f"Failed to calculate documentation completeness: {str(e)}")
            return 0

    async def _calculate_documentation_accuracy(self) -> float:
        """Calculate documentation accuracy score based on confidence scores"""
        try:
            pipeline = [
                {"$match": {"status": ExtractionStatus.COMPLETED.value}},
                {"$project": {
                    "icd10_confidence": {"$avg": "$extraction.icd10_codes.confidence"},
                    "cpt_confidence": {"$avg": "$extraction.cpt_codes.confidence"}
                }},
                {"$group": {
                    "_id": None,
                    "avg_accuracy": {
                        "$avg": {"$multiply": [
                            {"$avg": ["$icd10_confidence", "$cpt_confidence"]},
                            100
                        ]}
                    }
                }}
            ]
            result = await db.medical_notes.aggregate(pipeline).to_list(1)
            return round(result[0]["avg_accuracy"], 2) if result else 0
        except Exception as e:
            logger.error(f"Failed to calculate documentation accuracy: {str(e)}")
            return 0

    async def _calculate_documentation_timeliness(self) -> float:
        """Calculate documentation timeliness score"""
        try:
            threshold = datetime.utcnow() - timedelta(hours=24)
            total = await db.medical_notes.count_documents({})
            if total == 0:
                return 0
                
            processed_within_threshold = await db.medical_notes.count_documents({
                "status": ExtractionStatus.COMPLETED.value,
                "last_extraction_attempt": {"$gte": threshold}
            })
            
            return round((processed_within_threshold / total) * 100, 2)
        except Exception as e:
            logger.error(f"Failed to calculate documentation timeliness: {str(e)}")
            return 0

    # Keep existing methods unchanged
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

    async def search_notes(self, query: str, limit: int = 10) -> List[Dict]:
        """Search notes using text search"""
        try:
            cursor = db.medical_notes.find(
                {"$text": {"$search": query}}
            ).sort("created_at", -1).limit(limit)
            
            notes = []
            async for note in cursor:
                note["_id"] = str(note["_id"])
                notes.append(note)
            
            logger.info(f"Found {len(notes)} notes matching query: {query}")
            return notes
        except Exception as e:
            logger.error(f"Failed to search notes: {str(e)}")
            raise
