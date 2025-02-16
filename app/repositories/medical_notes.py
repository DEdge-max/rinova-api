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
            
            # Calculate total occurrences of codes
            total_icd10_occurrences = sum(r["count"] for r in icd10_results)
            total_cpt_occurrences = sum(r["count"] for r in cpt_results)

            # Format code frequencies with capped percentages
            common_icd10_codes = [
                CodeFrequency(
                    code=r["_id"]["code"],
                    description=r["_id"]["description"],
                    count=r["count"],
                    percentage=min(100, round(r["count"] / max(total_icd10_occurrences, 1) * 100, 2))
                ) for r in icd10_results
            ]

            common_cpt_codes = [
                CodeFrequency(
                    code=r["_id"]["code"],
                    description=r["_id"]["description"],
                    count=r["count"],
                    percentage=min(100, round(r["count"] / max(total_cpt_occurrences, 1) * 100, 2))
                ) for r in cpt_results
            ]
            
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
