from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

# Existing Extraction Models
class ExtractionRequest(BaseModel):
    """Request model for code extraction."""
    medical_text: str = Field(
        ..., 
        min_length=1,
        description="Medical text to extract codes from",
        example="Patient has type 2 diabetes without complications and hypertension. Office visit level 3 for evaluation."
    )

class ICD10Code(BaseModel):
    """Represents an ICD-10 medical diagnostic code."""
    code: str = Field(
        ..., 
        description="The ICD-10 code value",
        example="E11.9"
    )
    description: str = Field(
        ..., 
        description="Official description of the code",
        example="Type 2 diabetes mellitus without complications"
    )
    confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Confidence score between 0 and 1",
        example=0.95
    )
    primary: bool = Field(
        ..., 
        description="Whether this is the primary diagnosis",
        example=True
    )

class CPTCode(BaseModel):
    """Represents a CPT procedure code."""
    code: str = Field(
        ..., 
        description="The CPT code value",
        example="99213"
    )
    description: str = Field(
        ..., 
        description="Official description of the code",
        example="Office/outpatient visit for evaluation and management"
    )
    confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Confidence score between 0 and 1",
        example=0.92
    )
    category: str = Field(
        ..., 
        description="Category of the procedure",
        example="Evaluation and Management"
    )

class Metadata(BaseModel):
    """Metadata about the extraction process."""
    model_version: str = Field(
        ..., 
        description="Version of the extraction model",
        example="1.0"
    )
    processing_time_ms: int = Field(
        ..., 
        description="Processing time in milliseconds",
        example=245
    )
    timestamp: str = Field(
        ..., 
        description="Timestamp of the extraction",
        example="2025-02-12T10:00:00Z"
    )
    note_length: int = Field(
        ..., 
        description="Length of the input text",
        example=150
    )

class ExtractionData(BaseModel):
    """Container for extracted medical codes."""
    icd10_codes: List[ICD10Code] = Field(
        ..., 
        description="List of extracted ICD-10 codes"
    )
    cpt_codes: List[CPTCode] = Field(
        ..., 
        description="List of extracted CPT codes"
    )
    metadata: Metadata = Field(
        ..., 
        description="Extraction process metadata"
    )

class ExtractionResponse(BaseModel):
    """API response model."""
    success: bool = Field(
        ..., 
        description="Whether the extraction was successful",
        example=True
    )
    data: Optional[ExtractionData] = Field(
        None, 
        description="Extracted codes and metadata"
    )
    error: Optional[str] = Field(
        None, 
        description="Error message if any",
        example=None
    )

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "data": {
                    "icd10_codes": [
                        {
                            "code": "E11.9",
                            "description": "Type 2 diabetes mellitus without complications",
                            "confidence": 0.95,
                            "primary": True
                        },
                        {
                            "code": "I10",
                            "description": "Essential (primary) hypertension",
                            "confidence": 0.88,
                            "primary": False
                        }
                    ],
                    "cpt_codes": [
                        {
                            "code": "99213",
                            "description": "Office/outpatient visit for evaluation and management",
                            "confidence": 0.92,
                            "category": "Evaluation and Management"
                        }
                    ],
                    "metadata": {
                        "model_version": "1.0",
                        "processing_time_ms": 245,
                        "timestamp": "2025-02-12T10:00:00Z",
                        "note_length": 150
                    }
                },
                "error": None
            }
        }

# New Models for Search and Analytics

class ExtractionStatus(str, Enum):
    """Enumeration of possible extraction statuses."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class MedicalNote(BaseModel):
    """Model for storing medical notes."""
    id: str = Field(..., description="Unique identifier for the note")
    content: str = Field(..., description="The medical note text")
    patient_id: Optional[str] = Field(None, description="Patient identifier")
    created_at: datetime = Field(..., description="Note creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    status: ExtractionStatus = Field(..., description="Current extraction status")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "note_123",
                "content": "Patient presents with...",
                "patient_id": "P12345",
                "created_at": "2025-02-12T10:00:00Z",
                "updated_at": "2025-02-12T10:05:00Z",
                "status": "completed",
                "metadata": {"source": "EMR", "department": "Cardiology"}
            }
        }

class SearchRequest(BaseModel):
    """Model for search requests."""
    query: Optional[str] = Field(None, description="Text search query")
    start_date: Optional[datetime] = Field(None, description="Start date for filtering")
    end_date: Optional[datetime] = Field(None, description="End date for filtering")
    status: Optional[ExtractionStatus] = Field(None, description="Filter by status")
    skip: int = Field(0, ge=0, description="Number of records to skip")
    limit: int = Field(20, ge=1, le=100, description="Maximum number of records to return")

class SearchResponse(BaseModel):
    """Model for search responses."""
    total: int = Field(..., description="Total number of matching records")
    results: List[MedicalNote] = Field(..., description="Search results")
    has_more: bool = Field(..., description="Whether more results are available")

class AnalyticsTimeframe(BaseModel):
    """Model for analytics timeframe specification."""
    start_date: datetime = Field(..., description="Start of analysis period")
    end_date: datetime = Field(..., description="End of analysis period")

class ExtractionStatistics(BaseModel):
    """Model for extraction statistics."""
    total_extractions: int = Field(..., description="Total number of extractions")
    success_rate: float = Field(..., ge=0.0, le=100.0, description="Success rate percentage")
    avg_processing_time_ms: float = Field(..., description="Average processing time in milliseconds")
    extraction_counts: Dict[ExtractionStatus, int] = Field(..., description="Counts by status")
    daily_counts: Dict[str, int] = Field(..., description="Daily extraction counts")

class CommonCode(BaseModel):
    """Model for common code statistics."""
    code: str = Field(..., description="The medical code")
    count: int = Field(..., description="Number of occurrences")
    description: str = Field(..., description="Code description")
    percentage: float = Field(..., description="Percentage of total extractions")

class CodeAnalytics(BaseModel):
    """Model for code analytics response."""
    total_codes: int = Field(..., description="Total number of codes analyzed")
    common_icd10_codes: List[CommonCode] = Field(..., description="Most common ICD-10 codes")
    common_cpt_codes: List[CommonCode] = Field(..., description="Most common CPT codes")
    timeframe: AnalyticsTimeframe = Field(..., description="Analysis timeframe")

class PaginationParams(BaseModel):
    """Common pagination parameters."""
    skip: int = Field(0, ge=0, description="Number of records to skip")
    limit: int = Field(20, ge=1, le=100, description="Maximum number of records to return")

    @validator('limit')
    def validate_limit(cls, v):
        if v > 100:
            raise ValueError("Maximum limit is 100 records")
        return v