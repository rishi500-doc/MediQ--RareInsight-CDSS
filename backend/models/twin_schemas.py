"""
Pydantic schemas for the Patient Digital Twin module.
These models define the API contract for all twin endpoints.
"""
from __future__ import annotations
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
from datetime import date, datetime
from enum import Enum


# ─── Enumerations ─────────────────────────────────────────────────────────────

class DiagnosticStatus(str, Enum):
    undiagnosed = "undiagnosed"
    suspected   = "suspected"
    confirmed   = "confirmed"
    excluded    = "excluded"

class HPOTermStatus(str, Enum):
    active   = "active"
    resolved = "resolved"
    excluded = "excluded"

class EventType(str, Enum):
    symptom_onset    = "symptom_onset"
    symptom_resolved = "symptom_resolved"
    lab_result       = "lab_result"
    imaging          = "imaging"
    diagnosis        = "diagnosis"
    treatment_start  = "treatment_start"
    treatment_end    = "treatment_end"
    hospitalization  = "hospitalization"
    genetic_result   = "genetic_result"
    family_history   = "family_history"
    encounter        = "encounter"
    agent_note       = "agent_note"
    rag_analysis     = "rag_analysis"

class Severity(str, Enum):
    info     = "info"
    low      = "low"
    medium   = "medium"
    high     = "high"
    critical = "critical"

class TreatmentResponse(str, Enum):
    excellent = "excellent"
    good      = "good"
    partial   = "partial"
    none      = "none"
    adverse   = "adverse"
    unknown   = "unknown"
    ongoing   = "ongoing"

class VariantClassification(str, Enum):
    pathogenic        = "Pathogenic"
    likely_pathogenic = "Likely_Pathogenic"
    vus               = "VUS"
    likely_benign     = "Likely_Benign"
    benign            = "Benign"
    unknown           = "Unknown"


# ─── Sub-models ───────────────────────────────────────────────────────────────

class Demographics(BaseModel):
    age_at_creation : Optional[int]   = Field(None, ge=0, le=120, description="Patient age in years")
    date_of_birth   : Optional[date]  = None
    gender          : Optional[str]   = None
    ethnicity       : Optional[str]   = None
    country         : Optional[str]   = None
    consanguinity   : bool            = False


class HPOTermInput(BaseModel):
    hpo_term_id : str  = Field(..., pattern=r"^HP:\d{7}$", description="HPO term ID e.g. HP:0002099")
    hpo_label   : str  = Field(..., description="Human-readable HPO label")
    onset_date  : Optional[date]        = None
    status      : HPOTermStatus         = HPOTermStatus.active
    severity    : Optional[float]       = Field(None, ge=0.0, le=1.0)
    source      : str                   = "clinician"


class GeneticVariantInput(BaseModel):
    gene_symbol    : str = Field(..., description="HGNC gene symbol e.g. CFTR")
    hgvs_notation  : Optional[str] = None
    zygosity       : Optional[str] = None
    classification : Optional[VariantClassification] = VariantClassification.unknown
    clinvar_id     : Optional[str] = None
    omim_id        : Optional[str] = None
    panel_tested   : Optional[str] = None
    inheritance    : Optional[str] = None


class FamilyHistoryInput(BaseModel):
    relationship  : Optional[str] = None
    condition     : Optional[str] = None
    hpo_term_id   : Optional[str] = None
    consanguinity : bool          = False
    notes         : Optional[str] = None


class LabResultInput(BaseModel):
    test_date       : date
    test_name       : str
    loinc_code      : Optional[str]   = None
    value_numeric   : Optional[float] = None
    value_text      : Optional[str]   = None
    unit            : Optional[str]   = None
    reference_low   : Optional[float] = None
    reference_high  : Optional[float] = None
    interpretation  : Optional[str]   = None
    hpo_associations: List[str]       = Field(default_factory=list)
    encounter_id    : Optional[str]   = None


class ImagingInput(BaseModel):
    imaging_date     : date
    modality         : Optional[str] = None
    body_region      : Optional[str] = None
    findings         : Optional[str] = None
    impression       : Optional[str] = None
    hpo_associations : List[str]     = Field(default_factory=list)
    encounter_id     : Optional[str] = None


class TreatmentInput(BaseModel):
    drug_name    : str
    drug_brand   : Optional[str]             = None
    rxnorm_code  : Optional[str]             = None
    indication   : Optional[str]             = None
    start_date   : date
    end_date     : Optional[date]            = None
    dose         : Optional[str]             = None
    route        : Optional[str]             = None
    response     : TreatmentResponse         = TreatmentResponse.unknown
    notes        : Optional[str]             = None
    encounter_id : Optional[str]             = None


class EncounterInput(BaseModel):
    encounter_date  : date
    encounter_type  : str          = "outpatient"
    chief_complaint : Optional[str] = None
    clinical_notes  : Optional[str] = None
    clinician_id    : Optional[str] = None
    facility        : Optional[str] = None


# ─── Request Schemas ──────────────────────────────────────────────────────────

class TwinCreateRequest(BaseModel):
    """Request body for POST /twin/create"""
    patient_id      : str         = Field(..., description="Your internal patient identifier")
    demographics    : Demographics = Field(default_factory=Demographics)
    hpo_terms       : List[HPOTermInput]        = Field(default_factory=list)
    symptoms        : List[str]                 = Field(default_factory=list,
                        description="Free-text symptoms (auto-mapped to HPO if possible)")
    family_history  : List[FamilyHistoryInput]  = Field(default_factory=list)
    genetic_variants: List[GeneticVariantInput] = Field(default_factory=list)
    lab_results     : List[LabResultInput]      = Field(default_factory=list)
    imaging         : List[ImagingInput]        = Field(default_factory=list)
    treatments      : List[TreatmentInput]      = Field(default_factory=list)
    clinical_notes  : Optional[str]             = None
    diagnostic_status: DiagnosticStatus         = DiagnosticStatus.undiagnosed
    confirmed_diagnoses: List[str]              = Field(default_factory=list)


class TwinUpdateRequest(BaseModel):
    """Request body for PUT /twin/{twin_id} — all fields optional"""
    demographics    : Optional[Demographics]            = None
    hpo_terms       : Optional[List[HPOTermInput]]      = None
    symptoms        : Optional[List[str]]               = None
    family_history  : Optional[List[FamilyHistoryInput]]= None
    genetic_variants: Optional[List[GeneticVariantInput]]= None
    lab_results     : Optional[List[LabResultInput]]    = None
    imaging         : Optional[List[ImagingInput]]      = None
    treatments      : Optional[List[TreatmentInput]]    = None
    clinical_notes  : Optional[str]                     = None
    diagnostic_status: Optional[DiagnosticStatus]       = None
    confirmed_diagnoses: Optional[List[str]]            = None


class TimelineEventCreate(BaseModel):
    """Request body for POST /twin/{twin_id}/events"""
    event_date  : date
    event_type  : EventType
    title       : str
    description : Optional[str]  = None
    severity    : Severity        = Severity.info
    hpo_terms   : List[str]       = Field(default_factory=list)
    source_id   : Optional[str]   = None
    source_table: Optional[str]   = None
    metadata    : Dict[str, Any]  = Field(default_factory=dict)


# ─── Response Schemas ─────────────────────────────────────────────────────────

class TwinSummaryResponse(BaseModel):
    """Lightweight twin summary (list views)"""
    twin_id             : str
    patient_id          : str
    created_at          : datetime
    last_updated        : datetime
    version             : int
    diagnostic_status   : str
    confirmed_diagnoses : List[str]
    data_completeness   : float
    confidence_score    : float
    demographics        : Dict[str, Any]
    active_hpo_count    : int = 0


class TwinDetailResponse(TwinSummaryResponse):
    """Full twin detail (individual twin view)"""
    hpo_terms       : List[Dict[str, Any]] = Field(default_factory=list)
    lab_results     : List[Dict[str, Any]] = Field(default_factory=list)
    treatments      : List[Dict[str, Any]] = Field(default_factory=list)
    genetic_variants: List[Dict[str, Any]] = Field(default_factory=list)
    family_history  : List[Dict[str, Any]] = Field(default_factory=list)
    imaging         : List[Dict[str, Any]] = Field(default_factory=list)
    clinical_notes  : Optional[str]        = None


class TimelineEventResponse(BaseModel):
    event_id        : str
    twin_id         : str
    event_date      : date
    event_timestamp : datetime
    event_type      : str
    title           : str
    description     : Optional[str]
    severity        : str
    hpo_terms       : List[str]
    metadata        : Dict[str, Any]


class TimelineResponse(BaseModel):
    twin_id : str
    events  : List[TimelineEventResponse]
    total   : int


class SimilarPatientResponse(BaseModel):
    twin_id          : str
    similarity_score : float
    jaccard_score    : Optional[float] = None
    metadata         : Dict[str, Any]
    text_summary     : str


class ProgressionPrediction(BaseModel):
    horizon_days    : int
    predicted_hpo   : List[Dict[str, Any]]
    risk_level      : str
    confidence      : float
    recommendations : List[str]
    model_version   : str
    trigger_terms   : List[str]


class TwinHealthResponse(BaseModel):
    db_available     : bool
    chromadb_available: bool
    twin_count       : int
