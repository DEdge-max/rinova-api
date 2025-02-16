from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


# --- ENUMS ---
class SortOrder(str, Enum):
    """Enumeration for sort order options."""
    ASCENDING = "asc"
    DESCENDING = "desc"


class NoteType(str, Enum):
    """Enumeration for medical note types."""
    BRIEF = "brief"
    CONSULTATION = "consultation"
    PROGRESS = "progress"
    DISCHARGE = "discharge"
    PROCEDURE = "procedure"
    OTHER = "other"


class ExtractionStatus(str, Enum):
    """Enumeration of possible extraction statuses."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


# --- EVIDENCE, ALTERNATIVES, DOCUMENTATION GAPS ---
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
    code: str = Field(..., description="The alternative CPT code")
    description: str = Field(..., description="Description of the alternative code")
    required_documentation: str = Field(..., description="Documentation needed to support this code")
    why_considered: str = Field(..., description="Explanation of why this is a potential alternative")


class DocumentationGapImpact(BaseModel):
    """Model for documentation gap impact details."""
    affected_codes: List[str] = Field(default_factory=list, description="Codes impacted by this gap")
    current_limitation: str = Field(default="", description="How this affects current code selection")
    potential_improvement: str = Field(default="", description="What better codes could be used with proper documentation")


class DocumentationGapRecommendation(BaseModel):
    """Model for documentation gap recommendations."""
    what_to_add: str = Field(default="", description="Specific guidance on documentation needed")
    example: str = Field(default="", description="Example of proper documentation")
    rationale: str = Field(default="", description="Why this documentation is important")


class DocumentationGap(BaseModel):
    """Model for documentation gaps."""
    severity: str = Field(..., description="Severity level of the gap")
    description: str = Field(..., description="Specific missing or ambiguous element")
    impact: DocumentationGapImpact = Field(default_factory=DocumentationGapImpact, description="Impact of the documentation gap")
    recommendation: DocumentationGapRecommendation = Field(default_factory=DocumentationGapRecommendation, description="Recommendations for improvement")


# --- ICD-10 & CPT CODES ---
class ICD10Code(BaseModel):
    """Represents an ICD-10 medical diagnostic code."""
    code: str = Field(..., description="The ICD-10 code value")
    description: str = Field(..., description="Official description of the code")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    primary: bool = Field(..., description="Whether this is the primary diagnosis")
    evidence: Optional[Evidence] = Field(None, description="Evidence supporting code selection")


class CPTCode(BaseModel):
    """Represents a CPT procedure code."""
    code: str = Field(..., description="The CPT code value")
    description: str = Field(..., description="Official description of the code")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    category: str = Field(..., description="Category of the procedure")
    evidence: Optional[Evidence] = Field(None, description="Evidence supporting code selection")
    alternative_codes: List[AlternativeCode] = Field(default_factory=list, description="Potential alternative codes")


# --- METADATA & EXTRACTION RESPONSE ---
class Metadata(BaseModel):
    """Metadata about the extraction process."""
    model_version: str = Field(..., description="Version of the extraction model")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    timestamp: str = Field(..., description="Timestamp of the extraction")
    note_length: int = Field(..., description="Length of the input text")


class ExtractionData(BaseModel):
    """Container for extracted medical codes."""
    note_type: NoteType = Field(..., description="Type of medical note")
    icd10_codes: List[ICD10Code] = Field(..., description="List of extracted ICD-10 codes")
    cpt_codes: List[CPTCode] = Field(..., description="List of extracted CPT codes")
    documentation_gaps: List[DocumentationGap] = Field(default_factory=list, description="List of identified documentation gaps")
    metadata: Metadata = Field(..., description="Extraction process metadata")


class ExtractionResponse(BaseModel):
    """API response model."""
    success: bool = Field(..., description="Whether the extraction was successful")
    data: Optional[ExtractionData] = Field(None, description="Extracted codes and metadata")
    error: Optional[str] = Field(None, description="Error message if any")


# --- SEARCH, PAGINATION, & ANALYTICS ---
class MedicalNote(BaseModel):
    """Model for storing medical notes."""
    id: str = Field(..., description="Unique identifier for the note")
    content: str = Field(..., description="The medical note text")
    patient_id: Optional[str] = Field(None, description="Patient identifier")
    created_at: datetime = Field(..., description="Note creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    status: ExtractionStatus = Field(..., description="Current extraction status")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


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


class PaginationParams(BaseModel):
    """Common pagination parameters."""
    skip: int = Field(0, ge=0, description="Number of records to skip")
    limit: int = Field(20, ge=1, le=100, description="Maximum number of records to return")

    @validator('limit')
    def validate_limit(cls, v):
        if v > 100:
            raise ValueError("Maximum limit is 100 records")
        return v
