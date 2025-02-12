from fastapi import APIRouter, Query, HTTPException
from typing import List, Dict, Any
from datetime import datetime, timedelta
from ..database.mongodb import get_database

router = APIRouter(prefix="/api/v1/analytics", tags=["analytics"])

@router.get("/extraction-stats")
async def get_extraction_statistics(
    days: int = Query(30, description="Number of days to analyze")
):
    db = get_database()
    start_date = datetime.utcnow() - timedelta(days=days)
    
    try:
        # Get total counts
        total_extractions = await db.extraction_results.count_documents({
            "created_at": {"$gte": start_date}
        })
        
        # Get success rate
        success_count = await db.extraction_results.count_documents({
            "created_at": {"$gte": start_date},
            "status": "completed"
        })
        
        # Calculate average processing time
        pipeline = [
            {
                "$match": {
                    "created_at": {"$gte": start_date},
                    "processing_time": {"$exists": True}
                }
            },
            {
                "$group": {
                    "_id": None,
                    "avg_processing_time": {"$avg": "$processing_time"}
                }
            }
        ]
        avg_time_result = await db.extraction_results.aggregate(pipeline).to_list(1)
        avg_processing_time = avg_time_result[0]["avg_processing_time"] if avg_time_result else 0
        
        return {
            "total_extractions": total_extractions,
            "success_rate": (success_count / total_extractions * 100) if total_extractions > 0 else 0,
            "avg_processing_time": avg_processing_time,
            "period_days": days
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/common-codes")
async def get_common_codes(
    limit: int = Query(10, ge=1, le=50),
    days: int = Query(30, description="Number of days to analyze")
):
    db = get_database()
    start_date = datetime.utcnow() - timedelta(days=days)
    
    try:
        pipeline = [
            {
                "$match": {
                    "created_at": {"$gte": start_date},
                    "extracted_codes": {"$exists": True}
                }
            },
            {"$unwind": "$extracted_codes"},
            {
                "$group": {
                    "_id": "$extracted_codes.code",
                    "count": {"$sum": 1},
                    "description": {"$first": "$extracted_codes.description"}
                }
            },
            {"$sort": {"count": -1}},
            {"$limit": limit}
        ]
        
        results = await db.extraction_results.aggregate(pipeline).to_list(limit)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))