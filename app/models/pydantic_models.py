from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
from datetime import datetime
from bson import ObjectId

class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v, handler):
        if isinstance(v, ObjectId):
            return v
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        return {"type": "string"}

class BaseCode(BaseModel):
    code: str = Field(..., description="The medical code")
    description: str = Field(..., description="Description of the code")
    confidence_score: float = Field(..., ge=0, le=100, description="Confidence score (0-100%)")
    suggestions: List[str] = Field(default=[], description="Suggestions for improving documentation")

class ICD10Code(BaseCode):
    pass

class CPTCode(BaseCode):
    pass

class HCPCSCode(BaseCode):
    pass

class Modifier(BaseModel):
    modifier: str = Field(..., description="The modifier code")
    description: str = Field(..., description="Description of the modifier")
    confidence_score: float = Field(..., ge=0, le=100)
    suggestions: List[str] = Field(default=[])

class AlternativeCPT(BaseCode):
    justification: str = Field(..., description="Justification for why this code could apply")
    missing_documentation: List[str] = Field(..., description="Required documentation to support this code")

class CodeExtractionResult(BaseModel):
    icd10_codes: List[ICD10Code] = Field(default=[], description="Extracted ICD-10 codes")
    cpt_codes: List[CPTCode] = Field(default=[], description="Extracted CPT codes")
    alternative_cpts: List[AlternativeCPT] = Field(default=[], description="Potential alternative CPT codes")
    modifiers: List[Modifier] = Field(default=[], description="Applicable modifiers")
    hcpcs_codes: List[HCPCSCode] = Field(default=[], description="Extracted HCPCS codes")

class UpdatedCodes(BaseModel):
    icd10_codes: Optional[List[ICD10Code]] = Field(default=None, description="Final sorted ICD-10 codes")
    cpt_codes: Optional[List[CPTCode]] = Field(default=None, description="Final sorted CPT codes")
    modifiers: Optional[List[Modifier]] = Field(default=None, description="Final sorted modifiers")
    hcpcs_codes: Optional[List[HCPCSCode]] = Field(default=None, description="Final sorted HCPCS codes")
    alternative_cpts: Optional[List[AlternativeCPT]] = Field(default=None, description="Remaining alternative CPTs")

class MedicalNote(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_encoders={ObjectId: str}
    )
    
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    doctor_name: Optional[str] = Field(default=None)
    patient_name: Optional[str] = Field(default=None)
    date: Optional[datetime] = Field(default_factory=datetime.now)
    note_text: Optional[str] = Field(default=None)
    extraction_result: Optional[CodeExtractionResult] = None

class NoteCreate(BaseModel):
    doctor_name: str
    patient_name: str
    note_text: str
    date: Optional[datetime] = None

class NoteUpdate(BaseModel):
    doctor_name: Optional[str] = None
    patient_name: Optional[str] = None
    note_text: Optional[str] = None
    date: Optional[datetime] = None
    extraction_result: Optional[CodeExtractionResult] = None

class NoteResponse(BaseModel):
    message: str
    note: MedicalNote

class NotesListResponse(BaseModel):
    message: str
    notes: List[MedicalNote]

class ExtractionResponse(BaseModel):
    message: str
    extraction_result: CodeExtractionResult

# New models for quick-extract endpoint
class QuickExtractionRequest(BaseModel):
    note_text: str
    
class QuickExtractionResponse(BaseModel):
    message: str
    note_id: str
    extraction_result: CodeExtractionResult
