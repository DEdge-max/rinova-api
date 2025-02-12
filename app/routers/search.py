from fastapi import APIRouter, Query, HTTPException
from typing import List, Optional
from datetime import datetime
from ..database.mongodb import db  # Using your existing db import
from pymongo import DESCENDING

router = APIRouter(prefix="/api/v1/search", tags=["search"])

@router.get("/notes")
async def search_medical_notes(
    query: str = Query(None, description="Text search query"),
    start_date: datetime = Query(None, description="Start date for filtering"),
    end_date: datetime = Query(None, description="End date for filtering"),
    status: str = Query(None, description="Extraction status"),
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100)
):
    try:
        # Build query filter
        filter_query = {}
        if query:
            filter_query["$text"] = {"$search": query}
        if start_date and end_date:
            filter_query["created_at"] = {
                "$gte": start_date,
                "$lte": end_date
            }
        if status:
            filter_query["status"] = status
            
        # Get total count for pagination
        total_count = await db.client[db.db_name]["medical_notes"].count_documents(filter_query)
        
        # Get paginated results
        cursor = db.client[db.db_name]["medical_notes"].find(
            filter_query
        ).sort("created_at", DESCENDING).skip(skip).limit(limit)
        
        notes = await cursor.to_list(length=limit)
        
        return {
            "success": True,
            "data": {
                "total": total_count,
                "notes": notes,
                "page": {
                    "current": skip // limit + 1,
                    "size": limit,
                    "total_pages": (total_count + limit - 1) // limit
                }
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/extractions")
async def search_extractions(
    query: str = Query(None, description="Text search query"),
    start_date: datetime = Query(None, description="Start date for filtering"),
    end_date: datetime = Query(None, description="End date for filtering"),
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100)
):
    try:
        # Build query filter
        filter_query = {}
        if query:
            filter_query["$text"] = {"$search": query}
        if start_date and end_date:
            filter_query["created_at"] = {
                "$gte": start_date,
                "$lte": end_date
            }
            
        # Get total count
        total_count = await db.client[db.db_name]["extraction_results"].count_documents(filter_query)
        
        # Get paginated results
        cursor = db.client[db.db_name]["extraction_results"].find(
            filter_query
        ).sort("created_at", DESCENDING).skip(skip).limit(limit)
        
        extractions = await cursor.to_list(length=limit)
        
        return {
            "success": True,
            "data": {
                "total": total_count,
                "extractions": extractions,
                "page": {
                    "current": skip // limit + 1,
                    "size": limit,
                    "total_pages": (total_count + limit - 1) // limit
                }
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))