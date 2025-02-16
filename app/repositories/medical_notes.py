from datetime import datetime, timedelta
from typing import List, Dict
from bson import ObjectId
from ..database.mongodb import db
import logging
from pymongo import IndexModel, ASCENDING, DESCENDING, TEXT
from ..models.pydantic_models import (
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
        """Initialize repository"""
        pass

    async def get_dashboard_statistics(self, days: int = 30) -> DashboardStatistics:
        """Get comprehensive dashboard statistics"""
        try:
            start_date = datetime.utcnow() - timedelta(days=days)
            
            total_notes = await db.medical_notes.count_documents({})
            total_processed = await db.medical_notes.count_documents({
                "status": ExtractionStatus.COMPLETED.value
            })
            
            status_breakdown = [
                StatusBreakdown(
                    status=status,
                    count=await db.medical_notes.count_documents({"status": status.value}),
                    percentage=round((await db.medical_notes.count_documents({"status": status.value}) / max(total_notes, 1)) * 100, 2)
                ) for status in ExtractionStatus
            ]
            
            type_breakdown = [
                TypeBreakdown(
                    type=note_type,
                    count=await db.medical_notes.count_documents({"extraction.note_type": note_type.value}),
                    percentage=round((await db.medical_notes.count_documents({"extraction.note_type": note_type.value}) / max(total_notes, 1)) * 100, 2)
                ) for note_type in NoteType
            ]
            
            processed_notes = await db.medical_notes.find({
                "status": ExtractionStatus.COMPLETED.value
            }).to_list(None)
            
            documentation_quality = {
                "completeness": await self._calculate_documentation_completeness(processed_notes),
                "accuracy": await self._calculate_documentation_accuracy(processed_notes),
                "timeliness": await self._calculate_documentation_timeliness(processed_notes)
            }
            
            return DashboardStatistics(
                total_notes=total_notes,
                total_processed=total_processed,
                processing_success_rate=round(total_processed / max(total_notes, 1) * 100, 2),
                status_breakdown=status_breakdown,
                type_breakdown=type_breakdown,
                documentation_quality=documentation_quality
            )
        except Exception as e:
            logger.error(f"Failed to get dashboard statistics: {str(e)}")
            raise

    async def _calculate_documentation_completeness(self, notes: List[Dict]) -> float:
        try:
            total_score = sum(
                0.4 * (not note.get('extraction', {}).get('documentation_gaps', [])) +
                0.3 * sum(code.get('confidence', 0) for code in note.get('extraction', {}).get('icd10_codes', []) + note.get('extraction', {}).get('cpt_codes', [])) / max(len(note.get('extraction', {}).get('icd10_codes', []) + note.get('extraction', {}).get('cpt_codes', [])), 1) +
                0.3 * any(code.get('evidence', {}).get('direct_quotes', []) for code in note.get('extraction', {}).get('icd10_codes', []) + note.get('extraction', {}).get('cpt_codes', []))
                for note in notes
            )
            return (total_score / max(len(notes), 1)) * 100
        except Exception as e:
            logger.error(f"Error calculating documentation completeness: {str(e)}")
            return 0.0

    async def _calculate_documentation_accuracy(self, notes: List[Dict]) -> float:
        try:
            total_score = sum(
                0.7 * (sum(code.get('confidence', 0) for code in note.get('extraction', {}).get('icd10_codes', []) + note.get('extraction', {}).get('cpt_codes', [])) / max(len(note.get('extraction', {}).get('icd10_codes', []) + note.get('extraction', {}).get('cpt_codes', [])), 1)) +
                0.3 * all(len(code.get('evidence', {}).get('direct_quotes', [])) > 0 for code in note.get('extraction', {}).get('icd10_codes', []) + note.get('extraction', {}).get('cpt_codes', []))
                for note in notes
            )
            return (total_score / max(len(notes), 1)) * 100
        except Exception as e:
            logger.error(f"Error calculating accuracy: {str(e)}")
            return 0.0

    async def _calculate_documentation_timeliness(self, notes: List[Dict]) -> float:
        try:
            total_score = sum(
                (1.0 if (updated_at - created_at).total_seconds() < 5 else 0.8 if (updated_at - created_at).total_seconds() < 10 else 0.6 if (updated_at - created_at).total_seconds() < 30 else 0.4) -
                max(0, (note.get('extraction_attempts', 1) - 1) * 0.1)
                for note in notes if (created_at := note.get('created_at')) and (updated_at := note.get('updated_at'))
            )
            return (total_score / max(len(notes), 1)) * 100
        except Exception as e:
            logger.error(f"Error calculating timeliness: {str(e)}")
            return 0.0
