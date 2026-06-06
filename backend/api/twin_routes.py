"""
Digital Twin API Routes — /api/v1/twin/

All endpoints for creating, reading, updating, deleting, and analysing
patient digital twins. Mounted in main.py under /api/v1.
"""
from __future__ import annotations
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from backend.models.twin_schemas import (
    TwinCreateRequest,
    TwinUpdateRequest,
    TimelineEventCreate,
)
from backend.twin.builder   import TwinBuilder
from backend.twin.timeline  import TimelineEngine
from backend.utils.common   import get_logger

logger      = get_logger("CDSS.API.Twin")
twin_router = APIRouter(prefix="/twin", tags=["Digital Twin"])

# Singletons — instantiated once per worker process
_builder  = TwinBuilder()
_timeline = TimelineEngine()


# ─── Health ───────────────────────────────────────────────────────────────────

@twin_router.get("/health", summary="Twin system health check")
async def twin_health():
    """Check PostgreSQL and ChromaDB availability for the twin module."""
    from backend.db.database import is_db_available
    from backend.twin.similarity import get_collection_stats

    db_ok    = await is_db_available()
    chroma   = {}
    chroma_ok = False

    try:
        chroma    = get_collection_stats()
        chroma_ok = True
    except Exception as e:
        chroma = {"error": str(e)}

    twin_count = 0
    if db_ok:
        try:
            twins = await _builder.list_twins(limit=1, offset=0)
            from backend.db.database import get_db_pool
            pool = await get_db_pool()
            async with pool.acquire() as conn:
                twin_count = await conn.fetchval("SELECT COUNT(*) FROM patient_twins")
        except Exception:
            pass

    return {
        "db_available"      : db_ok,
        "chromadb_available": chroma_ok,
        "twin_count"        : twin_count,
        "chromadb_stats"    : chroma,
    }


# ─── CRUD ─────────────────────────────────────────────────────────────────────

@twin_router.post("/create", status_code=201, summary="Create a new patient twin")
async def create_twin(request: TwinCreateRequest):
    """
    Create a Patient Digital Twin.

    - Stores demographics, HPO terms, labs, genetics, treatments in PostgreSQL
    - Generates a PubMedBERT embedding and upserts to ChromaDB
    - Records an initial timeline event

    **patient_id** is your internal identifier (MRN, UUID, etc.)
    """
    try:
        result = await _builder.create_twin(request)
        logger.info(f"Twin created: {result['twin_id']}")
        return result
    except Exception as e:
        logger.error(f"Twin creation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@twin_router.get("/{twin_id}", summary="Get full twin detail")
async def get_twin(twin_id: str):
    """
    Retrieve the complete patient twin including all HPO terms, labs,
    genetic variants, imaging, treatments, and family history.
    """
    twin = await _builder.get_twin(twin_id)
    if not twin:
        raise HTTPException(status_code=404, detail=f"Twin '{twin_id}' not found.")
    return twin


@twin_router.get("/", summary="List all twins (paginated)")
async def list_twins(
    limit : int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0,  ge=0),
):
    """Return a paginated list of patient twin summaries."""
    twins = await _builder.list_twins(limit=limit, offset=offset)
    return {"twins": twins, "count": len(twins), "offset": offset, "limit": limit}


@twin_router.put("/{twin_id}", summary="Update an existing twin")
async def update_twin(twin_id: str, request: TwinUpdateRequest):
    """
    Partially update a patient twin.
    - New HPO terms, labs, and treatments are **appended** (not replaced)
    - Demographics and status fields are **overwritten** if provided
    - Automatically refreshes the ChromaDB embedding
    """
    twin = await _builder.get_twin(twin_id)
    if not twin:
        raise HTTPException(status_code=404, detail=f"Twin '{twin_id}' not found.")
    try:
        return await _builder.update_twin(twin_id, request)
    except Exception as e:
        logger.error(f"Twin update failed [{twin_id}]: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@twin_router.delete("/{twin_id}", summary="Delete a twin permanently")
async def delete_twin(twin_id: str):
    """
    Permanently delete a patient twin and all associated records.
    Also removes the ChromaDB embedding.
    ⚠️ This action is irreversible.
    """
    twin = await _builder.get_twin(twin_id)
    if not twin:
        raise HTTPException(status_code=404, detail=f"Twin '{twin_id}' not found.")
    return await _builder.delete_twin(twin_id)


# ─── Timeline ─────────────────────────────────────────────────────────────────

@twin_router.post("/{twin_id}/events", status_code=201, summary="Add a timeline event")
async def add_timeline_event(twin_id: str, event: TimelineEventCreate):
    """
    Record a new clinical event on the patient's longitudinal timeline.

    Event types: symptom_onset | symptom_resolved | lab_result | imaging |
    diagnosis | treatment_start | treatment_end | hospitalization |
    genetic_result | family_history | encounter | agent_note | rag_analysis
    """
    twin = await _builder.get_twin(twin_id)
    if not twin:
        raise HTTPException(status_code=404, detail=f"Twin '{twin_id}' not found.")
    try:
        return await _timeline.add_event(twin_id, event)
    except Exception as e:
        logger.error(f"Timeline event failed [{twin_id}]: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@twin_router.get("/{twin_id}/timeline", summary="Get patient timeline")
async def get_timeline(
    twin_id   : str,
    days_back : int          = Query(default=730, ge=1,   le=3650,
                                     description="Days back to query"),
    event_type: Optional[str]= Query(default=None,
                                     description="Filter by event type"),
    severity  : Optional[str]= Query(default=None,
                                     description="Filter by severity"),
):
    """
    Retrieve the chronological event timeline for a patient twin.
    Supports filtering by event type and severity.
    """
    twin = await _builder.get_twin(twin_id)
    if not twin:
        raise HTTPException(status_code=404, detail=f"Twin '{twin_id}' not found.")
    return await _timeline.get_timeline(twin_id, days_back, event_type, severity)


@twin_router.get("/{twin_id}/timeline/trends", summary="Lab value trend over time")
async def get_lab_trend(twin_id: str, test_name: str = Query(...)):
    """
    Return time-series values for a specific lab test.
    Includes trend direction (increasing / decreasing / stable) for chart rendering.
    """
    return await _timeline.get_lab_trend(twin_id, test_name)


@twin_router.get("/{twin_id}/timeline/milestones", summary="Diagnostic milestones and delay")
async def get_milestones(twin_id: str):
    """
    Return key diagnostic milestones and compute the patient's diagnostic delay
    (days from first symptom to confirmed diagnosis).
    """
    twin = await _builder.get_twin(twin_id)
    if not twin:
        raise HTTPException(status_code=404, detail=f"Twin '{twin_id}' not found.")
    return await _timeline.get_milestone_summary(twin_id)


@twin_router.get("/{twin_id}/timeline/alerts", summary="Recent high/critical events")
async def get_alerts(twin_id: str, limit: int = Query(default=5, ge=1, le=20)):
    """Return the most recent high and critical severity events for this twin."""
    return await _timeline.get_recent_alerts(twin_id, limit)


# ─── Similar Patients ─────────────────────────────────────────────────────────

@twin_router.get("/{twin_id}/similar", summary="Find phenotypically similar patients")
async def get_similar_patients(
    twin_id     : str,
    n           : int   = Query(default=5, ge=1, le=20,
                                description="Number of similar patients to return"),
    min_score   : float = Query(default=0.60, ge=0.0, le=1.0,
                                description="Minimum cosine similarity threshold"),
):
    """
    Find the top-N most phenotypically similar patients using PubMedBERT
    embeddings and cosine similarity in ChromaDB.

    Returns similarity_score (embedding cosine) and jaccard_score (HPO term overlap).
    """
    twin = await _builder.get_twin(twin_id)
    if not twin:
        raise HTTPException(status_code=404, detail=f"Twin '{twin_id}' not found.")

    from backend.twin.similarity import find_similar_patients
    try:
        matches = find_similar_patients(twin_id, n_results=n, min_similarity=min_score)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Add explanation of similarity (overlapping terms)
    query_hpo_dict = {
        h.get("hpo_term_id", ""): h.get("hpo_label", h.get("hpo_term_id", ""))
        for h in twin.get("hpo_terms", []) if h.get("status", "active") == "active"
    }
    query_diag_set = set(twin.get("confirmed_diagnoses", []))
    
    for match in matches:
        match_id = match["twin_id"]
        match_twin = await _builder.get_twin(match_id)
        if match_twin:
            m_hpo_dict = {
                h.get("hpo_term_id", ""): h.get("hpo_label", h.get("hpo_term_id", ""))
                for h in match_twin.get("hpo_terms", []) if h.get("status", "active") == "active"
            }
            m_diag_set = set(match_twin.get("confirmed_diagnoses", []))
            
            overlap_hpo_ids = set(query_hpo_dict.keys()) & set(m_hpo_dict.keys())
            overlap_hpo_labels = [m_hpo_dict[hpo_id] for hpo_id in overlap_hpo_ids if hpo_id]
            overlap_diagnoses = list(query_diag_set & m_diag_set)
            
            match["overlapping_hpo_terms"] = overlap_hpo_labels
            match["overlapping_diagnoses"] = overlap_diagnoses
        else:
            match["overlapping_hpo_terms"] = []
            match["overlapping_diagnoses"] = []

    # Optionally store results to similarity index
    if matches:
        try:
            from backend.db.database import get_db_pool
            pool = await get_db_pool()
            async with pool.acquire() as conn:
                await conn.executemany("""
                    INSERT INTO twin_similarity_index
                        (twin_id_a, twin_id_b, similarity_score, jaccard_score)
                    VALUES ($1::uuid, $2::uuid, $3, $4)
                    ON CONFLICT (twin_id_a, twin_id_b)
                    DO UPDATE SET
                        similarity_score = EXCLUDED.similarity_score,
                        jaccard_score    = EXCLUDED.jaccard_score,
                        computed_at      = NOW()
                """, [
                    (twin_id, m["twin_id"], m["similarity_score"],
                     m.get("jaccard_score", 0.0))
                    for m in matches
                ])
        except Exception:
            pass  # Non-fatal

    return {
        "query_twin_id"  : twin_id,
        "similar_patients": matches,
        "count"          : len(matches),
        "min_score"      : min_score,
    }


# ─── Disease Progression ──────────────────────────────────────────────────────

@twin_router.post("/{twin_id}/predict/progression",
                  summary="Predict disease progression")
async def predict_progression(
    twin_id        : str,
    horizon_months : int = Query(default=6, ge=1, le=24,
                                 description="Prediction horizon in months"),
):
    """
    Predict which symptoms or complications this patient is likely to develop
    next, based on their current HPO term profile and clinical rules.

    Phase 1: Rule-based prediction engine.
    Phase 3: XGBoost / Transformer model (trained on rare disease cohorts).
    """
    twin = await _builder.get_twin(twin_id)
    if not twin:
        raise HTTPException(status_code=404, detail=f"Twin '{twin_id}' not found.")

    from backend.twin.progression import (
        predict_progression_rule_based,
        store_prediction,
    )

    hpo_terms  = twin.get("hpo_terms", [])
    prediction = predict_progression_rule_based(hpo_terms, horizon_months)

    # Persist prediction for audit trail
    try:
        pred_id = await store_prediction(twin_id, prediction)
        prediction["prediction_id"] = pred_id
    except Exception as e:
        logger.warning(f"Prediction storage failed: {e}")

    return {"twin_id": twin_id, **prediction}


# ─── RAG + Twin Combined Analysis ─────────────────────────────────────────────

@twin_router.post("/{twin_id}/analyze", summary="Run RAG analysis using twin context")
async def analyze_twin(twin_id: str):
    """
    Run the existing RAG pipeline enriched with this patient's twin context.

    - Builds a standardised HPO-based query from the twin's phenotype
    - Retrieves matching rare disease candidates from ChromaDB
    - Finds phenotypically similar patients
    - Returns all results for the frontend to display

    This is the bridge between the Digital Twin and the existing CDSS.
    """
    twin = await _builder.get_twin(twin_id)
    if not twin:
        raise HTTPException(status_code=404, detail=f"Twin '{twin_id}' not found.")

    from backend.twin.rag_bridge import TwinRAGBridge
    bridge = TwinRAGBridge()

    try:
        result = await bridge.query_with_twin_context(twin, n_results=5)

        # Record analysis as a timeline event
        await _timeline.add_event(twin_id, TimelineEventCreate(
            event_date  = __import__("datetime").date.today(),
            event_type  = "rag_analysis",
            title       = "RAG Disease Analysis Run",
            description = f"Retrieved {len(result.get('disease_candidates', []))} candidates. "
                          f"Found {len(result.get('similar_patients', []))} similar patients.",
            severity    = "info",
            metadata    = {"rag_query": result.get("rag_query", "")[:200]},
        ))

        return result

    except Exception as e:
        logger.error(f"Twin RAG analysis failed [{twin_id}]: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
