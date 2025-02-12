from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from ..database.mongodb import get_database
from ..models.pydantic_models import ExtractionStatus, MedicalNote
from ..core.config import Settings

router = APIRouter(prefix="/admin", tags=["Admin"])

@router.get("/system/stats")
async def get_system_stats():
    """
    Get system-wide statistics and performance metrics.
    """
    db = get_database()
    
    try:
        current_time = datetime.utcnow()
        last_24h = current_time - timedelta(hours=24)
        last_7d = current_time - timedelta(days=7)
        
        # Gather system statistics
        stats = {
            "total_notes": await db.medical_notes.count_documents({}),
            "total_extractions": await db.extraction_results.count_documents({}),
            "last_24h": {
                "notes_added": await db.medical_notes.count_documents(
                    {"created_at": {"$gte": last_24h}}
                ),
                "extractions_performed": await db.extraction_results.count_documents(
                    {"created_at": {"$gte": last_24h}}
                )
            },
            "last_7d": {
                "notes_added": await db.medical_notes.count_documents(
                    {"created_at": {"$gte": last_7d}}
                ),
                "extractions_performed": await db.extraction_results.count_documents(
                    {"created_at": {"$gte": last_7d}}
                )
            },
            "status_counts": {}
        }
        
        # Get counts by status
        pipeline = [
            {"$group": {"_id": "$status", "count": {"$sum": 1}}}
        ]
        status_counts = await db.medical_notes.aggregate(pipeline).to_list(None)
        stats["status_counts"] = {doc["_id"]: doc["count"] for doc in status_counts}
        
        return {"success": True, "data": stats}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching system stats: {str(e)}")

@router.get("/performance/metrics")
async def get_performance_metrics(
    timeframe: str = Query("24h", description="Timeframe for metrics (24h, 7d, 30d)")
):
    """
    Get detailed performance metrics for the system.
    """
    db = get_database()
    
    try:
        current_time = datetime.utcnow()
        if timeframe == "24h":
            start_time = current_time - timedelta(hours=24)
        elif timeframe == "7d":
            start_time = current_time - timedelta(days=7)
        elif timeframe == "30d":
            start_time = current_time - timedelta(days=30)
        else:
            raise HTTPException(status_code=400, detail="Invalid timeframe")
            
        pipeline = [
            {
                "$match": {
                    "created_at": {"$gte": start_time}
                }
            },
            {
                "$group": {
                    "_id": None,
                    "avg_processing_time": {"$avg": "$metadata.processing_time_ms"},
                    "max_processing_time": {"$max": "$metadata.processing_time_ms"},
                    "total_extractions": {"$sum": 1},
                    "success_count": {
                        "$sum": {"$cond": [{"$eq": ["$status", "completed"]}, 1, 0]}
                    }
                }
            }
        ]
        
        metrics = await db.extraction_results.aggregate(pipeline).to_list(1)
        if not metrics:
            return {"success": True, "data": {"message": "No data for the specified timeframe"}}
            
        metrics = metrics[0]
        metrics["success_rate"] = (metrics["success_count"] / metrics["total_extractions"]) * 100
        
        return {"success": True, "data": metrics}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching performance metrics: {str(e)}")

@router.post("/maintenance/cleanup")
async def cleanup_old_records(
    days: int = Query(30, ge=7, description="Number of days to keep records")
):
    """
    Clean up old records from the database.
    """
    db = get_database()
    
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Delete old records but keep successful extractions
        delete_result = await db.medical_notes.delete_many({
            "created_at": {"$lt": cutoff_date},
            "status": {"$in": ["failed", "pending"]}
        })
        
        return {
            "success": True,
            "data": {
                "deleted_count": delete_result.deleted_count,
                "cutoff_date": cutoff_date.isoformat()
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during cleanup: {str(e)}")

@router.post("/maintenance/reprocess")
async def reprocess_failed_extractions(
    max_items: int = Query(100, ge=1, le=1000, description="Maximum number of items to reprocess")
):
    """
    Reprocess failed extractions.
    """
    db = get_database()
    
    try:
        # Find failed extractions
        failed_notes = await db.medical_notes.find(
            {"status": "failed"}
        ).limit(max_items).to_list(None)
        
        # Update status to pending for reprocessing
        note_ids = [note["_id"] for note in failed_notes]
        
        if note_ids:
            await db.medical_notes.update_many(
                {"_id": {"$in": note_ids}},
                {"$set": {"status": "pending", "updated_at": datetime.utcnow()}}
            )
        
        return {
            "success": True,
            "data": {
                "reprocessing_count": len(note_ids),
                "note_ids": note_ids
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during reprocessing: {str(e)}")

@router.get("/queue/status")
async def get_queue_status():
    """
    Get the current status of the extraction queue.
    """
    db = get_database()
    
    try:
        pipeline = [
            {
                "$group": {
                    "_id": "$status",
                    "count": {"$sum": 1},
                    "avg_wait_time": {
                        "$avg": {
                            "$subtract": [datetime.utcnow(), "$created_at"]
                        }
                    }
                }
            }
        ]
        
        queue_stats = await db.medical_notes.aggregate(pipeline).to_list(None)
        
        # Format the results
        status_stats = {}
        for stat in queue_stats:
            status_stats[stat["_id"]] = {
                "count": stat["count"],
                "avg_wait_time_minutes": round(stat["avg_wait_time"] / 60000, 2)
                if stat["avg_wait_time"] else 0
            }
            
        return {"success": True, "data": status_stats}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching queue status: {str(e)}")

@router.post("/maintenance/optimize")
async def optimize_database():
    """
    Perform database optimization tasks.
    """
    db = get_database()
    
    try:
        # Create/update indexes
        await db.medical_notes.create_index([("created_at", -1)])
        await db.medical_notes.create_index([("status", 1)])
        await db.medical_notes.create_index([("patient_id", 1)])
        await db.medical_notes.create_index([("content", "text")])
        
        # Run database stats
        db_stats = await db.command("dbStats")
        
        return {
            "success": True,
            "data": {
                "message": "Database optimization completed",
                "stats": {
                    "collections": db_stats["collections"],
                    "indexes": db_stats["indexes"],
                    "size_mb": round(db_stats["dataSize"] / (1024 * 1024), 2)
                }
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during optimization: {str(e)}")