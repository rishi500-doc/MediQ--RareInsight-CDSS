from pydantic import BaseModel, Field
from typing import List, Optional

class PatientData(BaseModel):
    clinical_notes: Optional[str] = Field(None, description="Free-text clinical notes")
    symptoms: List[str] = Field(default_factory=list, description="List of patient symptoms")
    age: Optional[int] = Field(None, ge=0, le=120, description="Patient age in years")
    gender: Optional[str] = Field(None, description="Patient gender (male/female/other)")
    family_history: bool = Field(False, description="Presence of heritable genetic history")
    genomic_info: Optional[str] = Field(None, description="Specific genetic variants")

class DiagnosticResult(BaseModel):
    disease: str
    description: str
    source: str
    score: float = 0.0
    confidence_score: Optional[float] = Field(None, description="AI confidence probability")

class ClinicalReport(BaseModel):
    alignment: str
    reasoning: str
    recommendations: str
