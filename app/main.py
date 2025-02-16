from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from datetime import datetime, timedelta
import uvicorn
import openai
import logging
import os
from typing import Dict, Any

from app.routers import code_extraction
from app.database.mongodb import db

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app instance
app = FastAPI(
    title="Rinova API",
    description="Medical code extraction API using OpenAI with enhanced analytics & system monitoring",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Rate Limiting Setup
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)  # Add this line to properly handle rate limiting

# Add API Routers (removed prefix as it's already in the router)
app.include_router(
    code_extraction.router,
    tags=["Code Extraction"]
)

# CORS Configuration
origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["Content-Type", "Authorization", "Accept", "X-Requested-With"],
    expose_headers=["Content-Length"],
    max_age=3600,
)

# System Health Cache
system_status_cache = {
    "last_check": None,
    "status": None
}

async def check_system_health() -> Dict[str, Any]:
    """Comprehensive system health check with MongoDB & OpenAI monitoring"""
    current_time = datetime.utcnow()

    if system_status_cache["last_check"] and current_time - system_status_cache["last_check"] < timedelta(minutes=5):
        return system_status_cache["status"]

    status = {
        "status": "online",
        "timestamp": current_time.isoformat(),
        "api_version": app.version,
        "services": {}
    }

    # MongoDB Health Check
    try:
        await db.client.admin.command('ping')
        db_stats = await db.db.command("dbStats")
        collections = await db.db.list_collection_names()

        try:
            recent_stats = await db.medical_notes.aggregate([
                {"$match": {"created_at": {"$gte": current_time - timedelta(hours=24)}}},
                {"$group": {
                    "_id": "$status",
                    "count": {"$sum": 1},
                    "avg_processing_time": {"$avg": "$extraction.metadata.processing_time_ms"}
                }}
            ]).to_list(None)
        except Exception:
            recent_stats = []

        status["services"]["mongodb"] = {
            "status": "healthy",
            "collections": collections,
            "size_mb": round(db_stats["dataSize"] / (1024 * 1024), 2),
            "recent_extractions": {
                "total": sum(stat["count"] for stat in recent_stats),
                "by_status": {stat["_id"]: stat["count"] for stat in recent_stats},
                "avg_processing_time": round(
                    sum(stat.get("avg_processing_time", 0) for stat in recent_stats) /
                    len(recent_stats) if recent_stats else 0, 2
                )
            }
        }
    except Exception as e:
        logger.error(f"MongoDB health check failed: {str(e)}")
        status["services"]["mongodb"] = {"status": "unhealthy", "error": str(e)}

    # OpenAI API Check
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        status["services"]["openai"] = {"status": "unhealthy", "error": "Missing API key"}
    else:
        try:
            client = openai.OpenAI(api_key=openai_api_key)
            models = openai.Model.list()
            status["services"]["openai"] = {"status": "healthy", "models_available": len(models.data)}
        except Exception as e:
            logger.error(f"OpenAI API health check failed: {str(e)}")
            status["services"]["openai"] = {"status": "unhealthy", "error": str(e)}

    system_status_cache["last_check"] = current_time
    system_status_cache["status"] = status

    return status

# Database Connection Handling
@app.on_event("startup")
async def startup_db_client():
    """Initialize database connection and perform startup checks"""
    try:
        await db.connect_to_mongodb()
        await db.client.admin.command('ping')
        logger.info("Connected to MongoDB!")
        await check_system_health()
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_db_client():
    """Clean up connections on shutdown"""
    await db.close_mongodb_connection()
    logger.info("Disconnected from MongoDB.")

# Root Endpoint
@app.get("/", tags=["Health"])
async def root():
    return {
        "status": "success",
        "message": "Welcome to Rinova API",
        "timestamp": datetime.utcnow().isoformat(),
        "version": app.version,
        "docs": "/docs"
    }

# Enhanced Health Check with Rate Limiting
@app.get("/health", tags=["Health"])
@limiter.limit("5/minute")
async def health_check(request: Request):  # Added request parameter
    return await check_system_health()
# Exception Handlers
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    """Handle rate limit exceptions"""
    return JSONResponse(
        status_code=429,
        content={
            "success": False,
            "error": {
                "code": 429,
                "message": "Too many requests",
                "detail": str(exc),
                "timestamp": datetime.utcnow().isoformat()
            },
            "data": None
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "code": exc.status_code,
                "message": exc.detail,
                "timestamp": datetime.utcnow().isoformat()
            },
            "data": None
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle all other exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": {
                "code": 500,
                "message": "Internal server error",
                "detail": str(exc),
                "timestamp": datetime.utcnow().isoformat()
            },
            "data": None
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=True,
        workers=int(os.getenv("WORKERS", "1"))
    )
