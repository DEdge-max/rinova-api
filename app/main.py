from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
import uvicorn
import openai
from app.routers import code_extraction
from app.database.mongodb import db
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app instance
app = FastAPI(
    title="Rinova API",
    description="Medical code extraction API for Rinova using OpenAI for ICD-10 and CPT code identification",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add router
app.include_router(
    code_extraction.router,
    prefix="/api/v1",
    tags=["Code Extraction"]
)

# CORS configuration - Updated for development and production
origins = [
    "http://localhost:3000",      # React default dev port
    "http://localhost:5173",      # Vite default dev port
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    # Add your frontend production URL when you have it
    # "https://your-frontend-domain.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "Accept"],
    max_age=3600,  # Cache preflight requests for 1 hour
)

@app.on_event("startup")
async def startup_db_client():
    try:
        await db.connect_to_mongodb()
        # Test the connection
        await db.client.admin.command('ping')
        logger.info("Connected to MongoDB!")
        logger.info(f"Using database: {db.db.name}")
        # List all collections
        collections = await db.db.list_collection_names()
        logger.info(f"Available collections: {collections}")
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_db_client():
    await db.close_mongodb_connection()
    logger.info("Disconnected from MongoDB.")

@app.get("/", tags=["Health"])
async def root():
    return {
        "status": "success",
        "message": "Welcome to Rinova API",
        "timestamp": datetime.now().isoformat(),
        "version": app.version,
        "docs_url": "/docs"
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint for MongoDB and OpenAI API"""
    
    # MongoDB Check
    try:
        await db.client.admin.command('ping')
        mongodb_status = "healthy"
    except Exception as e:
        logger.error(f"MongoDB health check failed: {str(e)}")
        mongodb_status = f"unhealthy: {str(e)}"

    # OpenAI API Check
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        openai_status = "unhealthy: Missing API key"
    else:
        try:
            openai.api_key = openai_api_key
            openai.Model.list()  # Ping OpenAI API
            openai_status = "healthy"
        except Exception as e:
            logger.error(f"OpenAI API health check failed: {str(e)}")
            openai_status = f"unhealthy: {str(e)}"

    return {
        "status": "online",
        "timestamp": datetime.now().isoformat(),
        "api_version": app.version,
        "services": {
            "mongodb": mongodb_status,
            "openai": openai_status
        }
    }

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "data": None,
            "error": {
                "code": exc.status_code,
                "message": exc.detail,
                "timestamp": datetime.now().isoformat()
            }
        }
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)