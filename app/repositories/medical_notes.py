from datetime import datetime, timedelta
from typing import List, Dict
from bson import ObjectId
from ..database.mongodb import db
import logging
from pymongo import IndexModel, ASCENDING, DESCENDING, TEXT
from ..models.pydantic_models import (
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
        """Initialize repository"""
        pass

    async def get_dashboard_statistics(self, days: int = 30) -> DashboardStatistics:
        """Get comprehensive dashboard statistics"""
        try:
            start_date = datetime.utcnow() - timedelta(days=days)

            total_notes = await db.medical_notes.count_documents({})
            total_processed = await db.medical_notes.count_documents(
                {"status": ExtractionStatus.COMPLETED.value}
            )

            # Ensure zero-division protection
            total_notes = max(total_notes, 1)

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

            # Common ICD-10 codes
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
            icd10_results = await db.medical_notes.aggregate(icd10_pipeline).to_list(10)
            common_icd10_codes = [
                CodeFrequency(
                    code=r["_id"]["code"],
                    description=r["_id"]["description"],
                    count=r["count"],
                    percentage=round(r["count"] / max(total_processed, 1) * 100, 2)
                ) for r in icd10_results
            ]

            # Common CPT codes
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
            cpt_results = await db.medical_notes.aggregate(cpt_pipeline).to_list(10)
            common_cpt_codes = [
                CodeFrequency(
                    code=r["_id"]["code"],
                    description=r["_id"]["description"],
                    count=r["count"],
                    percentage=round(r["count"] / max(total_processed, 1) * 100, 2)
                ) for r in cpt_results
            ]

            # Daily extraction counts
            daily_pipeline = [
                {"$match": {"created_at": {"$gte": start_date}}},
                {"$group": {"_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$created_at"}}, "count": {"$sum": 1}}},
                {"$sort": {"_id": 1}}
            ]
            daily_counts = await db.medical_notes.aggregate(daily_pipeline).to_list(None)
            daily_extraction_counts = [
                TimeSeriesPoint(
                    date=datetime.strptime(d["_id"], "%Y-%m-%d"),
                    count=d["count"]
                ) for d in daily_counts
            ]

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
                common_icd10_codes=common_icd10_codes or [],
                common_cpt_codes=common_cpt_codes or [],
                daily_extraction_counts=daily_extraction_counts or [],
                documentation_quality=documentation_quality or {"completeness": 0, "accuracy": 0, "timeliness": 0}
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
