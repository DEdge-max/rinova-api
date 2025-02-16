from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class Evidence(BaseModel):
    """Model for evidence supporting code selection."""
    direct_quotes: List[str] = Field(
        default_factory=list,
        description="Relevant text from note supporting this code"
    )
    reasoning: str = Field(
        default="",
        description="Detailed explanation of why this code was selected"
    )
    guidelines_applied: List[str] = Field(
        default_factory=list,
        description="Specific coding guidelines used"
    )

class AlternativeCode(BaseModel):
    """Model for alternative CPT codes."""
    code: str = Field(
        ...,
        description="The alternative CPT code",
        example="99214"
    )
    description: str = Field(
        ...,
        description="Description of the alternative code",
        example="Office/outpatient visit, established patient, moderate complexity"
    )
    required_documentation: str = Field(
        ...,
        description="Documentation needed to support this code",
        example="Must document moderate complexity medical decision making"
    )
    why_considered: str = Field(
        ...,
        description="Explanation of why this is a potential alternative",
        example="Visit complexity suggests potential for higher level code"
    )

class DocumentationGapImpact(BaseModel):
    """Model for documentation gap impact details."""
    affected_codes: List[str] = Field(
        default_factory=list,
        description="Codes impacted by this gap"
    )
    current_limitation: str = Field(
        default="",
        description="How this affects current code selection"
    )
    potential_improvement: str = Field(
        default="",
        description="What better codes could be used with proper documentation"
    )

class DocumentationGapRecommendation(BaseModel):
    """Model for documentation gap recommendations."""
    what_to_add: str = Field(
        default="",
        description="Specific guidance on documentation needed"
    )
    example: str = Field(
        default="",
        description="Example of proper documentation"
    )
    rationale: str = Field(
        default="",
        description="Why this documentation is important"
    )

class DocumentationGap(BaseModel):
    """Model for documentation gaps."""
    severity: str = Field(
        ...,
        description="Severity level of the gap",
        example="moderate"
    )
    description: str = Field(
        ...,
        description="Specific missing or ambiguous element"
    )
    impact: DocumentationGapImpact = Field(
        default_factory=DocumentationGapImpact,
        description="Impact of the documentation gap"
    )
    recommendation: DocumentationGapRecommendation = Field(
        default_factory=DocumentationGapRecommendation,
        description="Recommendations for improvement"
    )

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
    evidence: Optional[Evidence] = Field(
        None,
        description="Evidence supporting code selection"
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
    evidence: Optional[Evidence] = Field(
        None,
        description="Evidence supporting code selection"
    )
    alternative_codes: List[AlternativeCode] = Field(
        default_factory=list,
        description="Potential alternative codes"
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
    note_type: str = Field(
        default="brief",
        description="Type of medical note"
    )
    icd10_codes: List[ICD10Code] = Field(
        ..., 
        description="List of extracted ICD-10 codes"
    )
    cpt_codes: List[CPTCode] = Field(
        ..., 
        description="List of extracted CPT codes"
    )
    documentation_gaps: List[DocumentationGap] = Field(
        default_factory=list,
        description="List of identified documentation gaps"
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
                    "note_type": "brief",
                    "icd10_codes": [
                        {
                            "code": "E11.9",
                            "description": "Type 2 diabetes mellitus without complications",
                            "confidence": 0.95,
                            "primary": True,
                            "evidence": {
                                "direct_quotes": ["Patient has type 2 diabetes without complications"],
                                "reasoning": "Clear documentation of diagnosis",
                                "guidelines_applied": ["ICD-10 guideline I.A.19"]
                            }
                        }
                    ],
                    "cpt_codes": [
                        {
                            "code": "99213",
                            "description": "Office/outpatient visit for evaluation and management",
                            "confidence": 0.92,
                            "evidence": {
                                "direct_quotes": ["Office visit level 3 for evaluation"],
                                "reasoning": "Documentation supports level 3 visit",
                                "guidelines_applied": ["E/M Guidelines 2021"]
                            },
                            "alternative_codes": []
                        }
                    ],
                    "documentation_gaps": [],
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

class ExtractionStatus(str, Enum):
    """Enumeration of possible extraction statuses."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class NoteType(str, Enum):
    """Enumeration of note types."""
    BRIEF = "brief"
    COMPREHENSIVE = "comprehensive"
    EMERGENCY = "emergency"
    OPERATIVE = "operative"
    PROGRESS = "progress"
    CONSULTATION = "consultation"

class SortOrder(str, Enum):
    """Enumeration for sort orders."""
    ASC = "asc"
    DESC = "desc"

class MedicalNote(BaseModel):
    """Model for storing medical notes."""
    id: str = Field(..., description="Unique identifier for the note")
    text: str = Field(..., description="The medical note text")
    source: str = Field(..., description="Source of the note")
    patient_id: Optional[str] = Field(None, description="Patient identifier")
    created_at: datetime = Field(..., description="Note creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    length: int = Field(..., description="Length of the note")
    status: str = Field(..., description="Current extraction status")
    extraction_attempts: int = Field(..., description="Number of extraction attempts")
    last_extraction_attempt: datetime = Field(..., description="Timestamp of last extraction attempt")
    extraction: Dict[str, Any] = Field(..., description="Extraction results")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "note_123",
                "text": "Patient presents with...",
                "source": "API",
                "patient_id": "P12345",
                "created_at": "2025-02-12T10:00:00Z",
                "updated_at": "2025-02-12T10:05:00Z",
                "length": 659,
                "status": "completed",
                "extraction_attempts": 1,
                "last_extraction_attempt": "2025-02-12T10:05:00Z",
                "extraction": {
                    "note_type": "emergency",
                    "icd10_codes": [],
                    "cpt_codes": [],
                    "documentation_gaps": []
                }
            }
        }
        allow_population_by_field_name = True
        populate_by_name = True

    @validator('id', pre=True)
    def convert_object_id(cls, v):
        if hasattr(v, "$oid"):
            return v["$oid"]
        return str(v)

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

class StatusBreakdown(BaseModel):
    """Model for status counts breakdown."""
    pending: int = Field(
        default=0,
        description="Number of notes in pending status"
    )
    in_progress: int = Field(
        default=0,
        description="Number of notes in progress"
    )
    completed: int = Field(
        default=0,
        description="Number of completed notes"
    )
    failed: int = Field(
        default=0,
        description="Number of failed notes"
    )
    total: int = Field(
        default=0,
        description="Total number of notes"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "pending": 50,
                "in_progress": 25,
                "completed": 400,
                "failed": 10,
                "total": 485
            }
        }

class NotesFilterParams(BaseModel):
    """Parameters for filtering notes."""
    note_type: Optional[NoteType] = Field(None, description="Filter by note type")
    status: Optional[ExtractionStatus] = Field(None, description="Filter by extraction status")
    start_date: Optional[datetime] = Field(None, description="Filter by start date")
    end_date: Optional[datetime] = Field(None, description="Filter by end date")
    search_text: Optional[str] = Field(None, description="Search in note content")
    has_documentation_gaps: Optional[bool] = Field(None, description="Filter notes with documentation gaps")
    
    class Config:
        json_schema_extra = {
            "example": {
                "note_type": "brief",
                "status": "completed",
                "start_date": "2025-01-01T00:00:00Z",
                "end_date": "2025-12-31T23:59:59Z",
                "search_text": "diabetes",
                "has_documentation_gaps": True
            }
        }

class NotesListingParams(BaseModel):
    """Parameters for notes listing endpoint."""
    page: int = Field(1, ge=1, description="Page number")
    page_size: int = Field(20, ge=1, le=100, description="Items per page")
    sort_by: str = Field("created_at", description="Field to sort by")
    sort_order: SortOrder = Field(SortOrder.DESC, description="Sort order")
    filters: Optional[NotesFilterParams] = None

    class Config:
        json_schema_extra = {
            "example": {
                "page": 1,
                "page_size": 20,
                "sort_by": "created_at",
                "sort_order": "desc",
                "filters": {
                    "note_type": "brief",
                    "status": "completed"
                }
            }
        }

class NotesSummary(BaseModel):
    """Summary statistics for notes listing."""
    total_notes: int = Field(..., description="Total number of notes")
    total_pages: int = Field(..., description="Total number of pages")
    current_page: int = Field(..., description="Current page number")
    notes_per_page: int = Field(..., description="Number of notes per page")
    has_next: bool = Field(..., description="Whether there are more pages")
    has_previous: bool = Field(..., description="Whether there are previous pages")

class NotesListingResponse(BaseModel):
    """Response model for notes listing."""
    success: bool = Field(..., description="Whether the request was successful")
    summary: NotesSummary = Field(..., description="Summary statistics for the listing")
    notes: List[MedicalNote] = Field(..., description="List of medical notes")
    error: Optional[str] = Field(None, description="Error message if any")

class CodeFrequency(BaseModel):
    """Model for code frequency statistics."""
    code: str = Field(..., description="The medical code")
    description: str = Field(..., description="Description of the code")
    count: int = Field(..., description="Number of occurrences")
    percentage: float = Field(..., ge=0.0, le=100.0, description="Percentage of total codes")

class TimeSeriesPoint(BaseModel):
    """Model for time series data points."""
    date: datetime = Field(..., description="Date of the data point")
    count: int = Field(..., description="Count/value for this date")

class TypeBreakdown(BaseModel):
    """Model for note type counts breakdown."""
    type: NoteType = Field(..., description="The note type")
    count: int = Field(..., description="Number of notes of this type")
    percentage: float = Field(..., ge=0.0, le=100.0, description="Percentage of total notes")

class DocumentationQuality(BaseModel):
    """Model for documentation quality metrics."""
    completeness: float = Field(..., ge=0.0, le=100.0, description="Documentation completeness score")
    accuracy: float = Field(..., ge=0.0, le=100.0, description="Documentation accuracy score")
    timeliness: float = Field(..., ge=0.0, le=100.0, description="Documentation timeliness score")

class DashboardStatistics(BaseModel):
    """Model for dashboard statistics."""
    total_notes: int = Field(..., description="Total number of notes in the system")
    total_processed: int = Field(..., description="Number of notes that have been processed")
    processing_success_rate: float = Field(..., ge=0.0, le=100.0, description="Success rate of note processing as a percentage")
    avg_processing_time_ms: float = Field(..., description="Average processing time in milliseconds")
    status_breakdown: List[StatusBreakdown] = Field(..., description="Detailed breakdown of notes by status")
    type_breakdown: List[TypeBreakdown] = Field(..., description="Breakdown of notes by type")
    common_icd10_codes: List[CodeFrequency] = Field(..., description="Most common ICD-10 codes")
    common_cpt_codes: List[CodeFrequency] = Field(..., description="Most common CPT codes")
    daily_extraction_counts: List[TimeSeriesPoint] = Field(..., description="Daily extraction counts")
    documentation_quality: DocumentationQuality = Field(..., description="Documentation quality metrics")

class BatchExtractionRequest(BaseModel):
    """Request model for batch code extraction."""
    medical_texts: List[str] = Field(
        ..., 
        min_items=1,
        description="List of medical texts to extract codes from",
        example=["Patient has type 2 diabetes without complications.", 
                "Patient presents with hypertension."]
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "medical_texts": [
                    "Patient has type 2 diabetes without complications and hypertension.",
                    "Follow-up visit for blood pressure monitoring. BP 140/90."
                ]
            }
        }

class BatchExtractionResponse(BaseModel):
    """API response model for batch extraction."""
    success: bool = Field(
        ..., 
        description="Whether the batch extraction was successful",
        example=True
    )
    results: List[ExtractionResponse] = Field(
        ...,
        description="List of extraction results, one per input text"
    )
    error: Optional[str] = Field(
        None, 
        description="Error message if any",
        example=None
    )
