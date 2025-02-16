from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime
from enum import Enum

# New Enums
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

# Existing Models
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
        description="The alternative CPT code"
    )
    description: str = Field(
        ...,
        description="Description of the alternative code"
    )
    required_documentation: str = Field(
        ...,
        description="Documentation needed to support this code"
    )
    why_considered: str = Field(
        ...,
        description="Explanation of why this is a potential alternative"
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
        description="Severity level of the gap"
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

# New Models for Notes Listing and Filtering
class NotesFilterParams(BaseModel):
    """Parameters for filtering medical notes."""
    note_type: Optional[NoteType] = Field(None, description="Filter by note type")
    status: Optional[ExtractionStatus] = Field(None, description="Filter by extraction status")
    start_date: Optional[datetime] = Field(None, description="Filter by start date")
    end_date: Optional[datetime] = Field(None, description="Filter by end date")
    search_query: Optional[str] = Field(None, description="Text search query")
    patient_id: Optional[str] = Field(None, description="Filter by patient ID")

class NotesListingParams(BaseModel):
    """Parameters for notes listing with pagination and sorting."""
    page: int = Field(1, ge=1, description="Page number")
    page_size: int = Field(20, ge=1, le=100, description="Items per page")
    sort_by: str = Field("created_at", description="Field to sort by")
    sort_order: SortOrder = Field(SortOrder.DESCENDING, description="Sort order")
    filters: Optional[NotesFilterParams] = Field(None, description="Filter parameters")

class NotesSummary(BaseModel):
    """Summary information for notes listing."""
    total_notes: int = Field(..., description="Total number of notes")
    total_pages: int = Field(..., description="Total number of pages")
    current_page: int = Field(..., description="Current page number")
    has_next: bool = Field(..., description="Whether there is a next page")
    has_previous: bool = Field(..., description="Whether there is a previous page")

class NotesListingResponse(BaseModel):
    """Response model for notes listing endpoint."""
    success: bool = Field(..., description="Operation success status")
    summary: NotesSummary = Field(..., description="Page information")
    notes: List[MedicalNote] = Field(..., description="List of medical notes")
    error: Optional[str] = Field(None, description="Error message if any")

# Dashboard Statistics Models
class CodeFrequency(BaseModel):
    """Frequency statistics for a specific code."""
    code: str = Field(..., description="The code value")
    description: str = Field(..., description="Code description")
    count: int = Field(..., description="Number of occurrences")
    percentage: float = Field(..., ge=0.0, le=100.0, description="Percentage of total")

class StatusBreakdown(BaseModel):
    """Breakdown of notes by status."""
    status: ExtractionStatus = Field(..., description="Extraction status")
    count: int = Field(..., description="Number of notes")
    percentage: float = Field(..., ge=0.0, le=100.0, description="Percentage of total")

class TypeBreakdown(BaseModel):
    """Breakdown of notes by type."""
    type: NoteType = Field(..., description="Note type")
    count: int = Field(..., description="Number of notes")
    percentage: float = Field(..., ge=0.0, le=100.0, description="Percentage of total")

class TimeSeriesPoint(BaseModel):
    """Data point for time series analysis."""
    date: datetime = Field(..., description="Date of measurement")
    count: int = Field(..., description="Number of occurrences")

class DashboardStatistics(BaseModel):
    """Comprehensive dashboard statistics."""
    total_notes: int = Field(..., description="Total number of notes")
    total_processed: int = Field(..., description="Total processed notes")
    processing_success_rate: float = Field(
        ..., 
        ge=0.0, 
        le=100.0, 
        description="Success rate percentage"
    )
    avg_processing_time_ms: float = Field(
        ..., 
        description="Average processing time in milliseconds"
    )
    status_breakdown: List[StatusBreakdown] = Field(
        ..., 
        description="Notes by status"
    )
    type_breakdown: List[TypeBreakdown] = Field(
        ..., 
        description="Notes by type"
    )
    common_icd10_codes: List[CodeFrequency] = Field(
        ..., 
        description="Most common ICD-10 codes"
    )
    common_cpt_codes: List[CodeFrequency] = Field(
        ..., 
        description="Most common CPT codes"
    )
    daily_extraction_counts: List[TimeSeriesPoint] = Field(
        ..., 
        description="Daily extraction counts"
    )
    documentation_quality: Dict[str, float] = Field(
        ..., 
        description="Documentation quality metrics"
    )

# Keep all existing models unchanged
[... rest of the existing models remain the same ...]
