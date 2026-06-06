"""
Twin Builder — Creates, updates, and deletes Patient Digital Twins.

Responsibilities:
  - Write twin records to PostgreSQL (demographics, HPO terms, labs, etc.)
  - Generate and upsert the patient embedding to ChromaDB
  - Compute data completeness score
  - Record all changes as timeline events
"""
from __future__ import annotations
import json
import uuid
from datetime import datetime, date
from typing import Optional

from backend.db.database import get_db_pool
from backend.utils.common import get_logger
from backend.models.twin_schemas import TwinCreateRequest, TwinUpdateRequest

logger = get_logger("CDSS.Twin.Builder")


# ─── Completeness Scoring ──────────────────────────────────────────────────────
COMPLETENESS_WEIGHTS = {
    "has_demographics"    : 0.15,
    "has_hpo_terms"       : 0.25,
    "has_lab_results"     : 0.20,
    "has_genetic_variants": 0.20,
    "has_family_history"  : 0.10,
    "has_treatments"      : 0.10,
}

def _compute_completeness(request: TwinCreateRequest) -> float:
    score = 0.0
    demo = request.demographics
    if demo.age_at_creation or demo.date_of_birth or demo.gender:
        score += COMPLETENESS_WEIGHTS["has_demographics"]
    if request.hpo_terms or request.symptoms:
        score += COMPLETENESS_WEIGHTS["has_hpo_terms"]
    if request.lab_results:
        score += COMPLETENESS_WEIGHTS["has_lab_results"]
    if request.genetic_variants:
        score += COMPLETENESS_WEIGHTS["has_genetic_variants"]
    if request.family_history:
        score += COMPLETENESS_WEIGHTS["has_family_history"]
    if request.treatments:
        score += COMPLETENESS_WEIGHTS["has_treatments"]
    return round(score, 3)


# ─── Core Builder ─────────────────────────────────────────────────────────────

class TwinBuilder:
    """
    Handles all CRUD operations for Patient Digital Twins.
    After writing to PostgreSQL, it delegates embedding to the SimilarityEngine.
    """

    async def create_twin(self, request: TwinCreateRequest) -> dict:
        """
        Create a new patient twin.
        Returns the created twin_id and summary.
        """
        pool = await get_db_pool()
        
        # ── Resolve and map any HPO terms or free-text symptoms ──
        from backend.hpo.hpo_mapper import HPOMapper
        from backend.models.twin_schemas import HPOTermInput
        mapper = HPOMapper()
        resolved_hpos = []
        
        # 1. Start with HPO terms list, cleaning quotes and mapping HP:0000000
        if request.hpo_terms:
            for h in request.hpo_terms:
                label = h.hpo_label.strip().strip('"').strip("'")
                term_id = h.hpo_term_id
                if term_id == "HP:0000000" or not term_id:
                    mapped = mapper.map_symptom(label)
                    if mapped:
                        term_id = mapped["hpo_id"]
                        label = mapped["term"]
                h.hpo_term_id = term_id
                h.hpo_label = label
                resolved_hpos.append(h)
                
        # 2. Map free-text symptoms list and merge into HPO terms list
        if request.symptoms:
            for sym in request.symptoms:
                sym_clean = sym.strip().strip('"').strip("'")
                if not sym_clean:
                    continue
                mapped = mapper.map_symptom(sym_clean)
                if mapped:
                    if not any(h.hpo_term_id == mapped["hpo_id"] for h in resolved_hpos):
                        resolved_hpos.append(HPOTermInput(
                            hpo_term_id=mapped["hpo_id"],
                            hpo_label=mapped["term"],
                            status="active",
                            source="clinician"
                        ))
                else:
                    if not any(h.hpo_label.lower() == sym_clean.lower() for h in resolved_hpos):
                        resolved_hpos.append(HPOTermInput(
                            hpo_term_id="HP:0000000",
                            hpo_label=sym_clean,
                            status="active",
                            source="clinician"
                        ))
        request.hpo_terms = resolved_hpos

        completeness = _compute_completeness(request)
        demographics = request.demographics.model_dump(mode="json")

        async with pool.acquire() as conn:
            async with conn.transaction():
                # ── 1. Insert core twin record ──────────────────────────────
                twin = await conn.fetchrow("""
                    INSERT INTO patient_twins
                        (patient_id, demographics, diagnostic_status,
                         confirmed_diagnoses, data_completeness, clinical_notes)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    RETURNING twin_id, created_at, last_updated, version
                """,
                    request.patient_id,
                    json.dumps(demographics),
                    request.diagnostic_status.value,
                    request.confirmed_diagnoses,
                    completeness,
                    request.clinical_notes,
                )
                twin_id = str(twin["twin_id"])
                logger.info(f"Created twin {twin_id} for patient {request.patient_id}")

                # ── 2. Insert HPO terms ─────────────────────────────────────
                if request.hpo_terms:
                    await conn.executemany("""
                        INSERT INTO twin_hpo_terms
                            (twin_id, hpo_term_id, hpo_label, onset_date,
                             status, severity, source)
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                        ON CONFLICT (twin_id, hpo_term_id)
                        WHERE status != 'excluded' DO NOTHING
                    """, [
                        (twin_id, h.hpo_term_id, h.hpo_label,
                         h.onset_date, h.status.value,
                         h.severity, h.source)
                        for h in request.hpo_terms
                    ])

                # ── 3. Insert Lab Results ───────────────────────────────────
                if request.lab_results:
                    await conn.executemany("""
                        INSERT INTO twin_lab_results
                            (twin_id, test_date, test_name, loinc_code,
                             value_numeric, value_text, unit,
                             reference_low, reference_high,
                             interpretation, hpo_associations)
                        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
                    """, [
                        (twin_id, l.test_date, l.test_name, l.loinc_code,
                         l.value_numeric, l.value_text, l.unit,
                         l.reference_low, l.reference_high,
                         l.interpretation, l.hpo_associations)
                        for l in request.lab_results
                    ])

                # ── 4. Insert Genetic Variants ──────────────────────────────
                if request.genetic_variants:
                    await conn.executemany("""
                        INSERT INTO twin_genetic_variants
                            (twin_id, gene_symbol, hgvs_notation, zygosity,
                             classification, clinvar_id, omim_id,
                             panel_tested, inheritance)
                        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
                    """, [
                        (twin_id, v.gene_symbol, v.hgvs_notation, v.zygosity,
                         v.classification.value if v.classification else None,
                         v.clinvar_id, v.omim_id, v.panel_tested, v.inheritance)
                        for v in request.genetic_variants
                    ])

                # ── 5. Insert Family History ────────────────────────────────
                if request.family_history:
                    await conn.executemany("""
                        INSERT INTO twin_family_history
                            (twin_id, relationship, condition,
                             hpo_term_id, consanguinity, notes)
                        VALUES ($1,$2,$3,$4,$5,$6)
                    """, [
                        (twin_id, f.relationship, f.condition,
                         f.hpo_term_id, f.consanguinity, f.notes)
                        for f in request.family_history
                    ])

                # ── 6. Insert Treatments ────────────────────────────────────
                if request.treatments:
                    await conn.executemany("""
                        INSERT INTO twin_treatments
                            (twin_id, drug_name, drug_brand, rxnorm_code,
                             indication, start_date, end_date, dose, route,
                             response, notes)
                        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
                    """, [
                        (twin_id, t.drug_name, t.drug_brand, t.rxnorm_code,
                         t.indication, t.start_date, t.end_date, t.dose, t.route,
                         t.response.value, t.notes)
                        for t in request.treatments
                    ])

                # ── 7. Insert Imaging ───────────────────────────────────────
                if request.imaging:
                    await conn.executemany("""
                        INSERT INTO twin_imaging
                            (twin_id, imaging_date, modality, body_region,
                             findings, impression, hpo_associations)
                        VALUES ($1,$2,$3,$4,$5,$6,$7)
                    """, [
                        (twin_id, img.imaging_date, img.modality, img.body_region,
                         img.findings, img.impression, img.hpo_associations)
                        for img in request.imaging
                    ])

                # ── 8. Record "twin created" timeline event ─────────────────
                hpo_ids = [h.hpo_term_id for h in request.hpo_terms]
                await conn.execute("""
                    INSERT INTO twin_timeline_events
                        (twin_id, event_date, event_type, title, description,
                         severity, hpo_terms, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                    twin_id,
                    date.today(),
                    "encounter",
                    "Patient Digital Twin Created",
                    f"Initial twin created with {len(request.hpo_terms)} HPO terms, "
                    f"{len(request.lab_results)} lab results, "
                    f"{len(request.genetic_variants)} genetic variants.",
                    "info",
                    hpo_ids,
                    json.dumps({"patient_id": request.patient_id,
                                "completeness": completeness}),
                )

        # ── 9. Generate and store ChromaDB embedding (non-blocking) ─────────
        try:
            from backend.twin.similarity import upsert_twin_embedding
            twin_data = await self.get_twin(twin_id)
            if twin_data:
                embedding = upsert_twin_embedding(twin_id, twin_data)
                # Store chroma_doc_id reference back to PG
                async with (await get_db_pool()).acquire() as conn:
                    await conn.execute(
                        "UPDATE patient_twins SET chroma_doc_id = $1 WHERE twin_id = $2::uuid",
                        twin_id, twin_id
                    )
                logger.info(f"ChromaDB embedding upserted for twin {twin_id}")
        except Exception as e:
            # Non-fatal — twin is still created in PG
            logger.warning(f"ChromaDB embedding failed (will retry): {e}")

        return {
            "twin_id"      : twin_id,
            "patient_id"   : request.patient_id,
            "created_at"   : twin["created_at"].isoformat(),
            "completeness" : completeness,
            "message"      : "Patient Digital Twin created successfully.",
        }

    async def get_twin(self, twin_id: str) -> Optional[dict]:
        """Fetch the full twin record including all sub-tables."""
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM patient_twins WHERE twin_id = $1::uuid", twin_id
            )
            if not row:
                return None

            twin = dict(row)
            twin["demographics"] = (
                json.loads(twin["demographics"])
                if isinstance(twin["demographics"], str) else twin["demographics"]
            )

            # Load sub-tables
            twin["hpo_terms"] = [dict(r) for r in await conn.fetch(
                "SELECT * FROM twin_hpo_terms WHERE twin_id=$1::uuid ORDER BY recorded_at",
                twin_id
            )]
            twin["lab_results"] = [dict(r) for r in await conn.fetch(
                "SELECT * FROM twin_lab_results WHERE twin_id=$1::uuid ORDER BY test_date DESC",
                twin_id
            )]
            twin["genetic_variants"] = [dict(r) for r in await conn.fetch(
                "SELECT * FROM twin_genetic_variants WHERE twin_id=$1::uuid",
                twin_id
            )]
            twin["family_history"] = [dict(r) for r in await conn.fetch(
                "SELECT * FROM twin_family_history WHERE twin_id=$1::uuid",
                twin_id
            )]
            twin["treatments"] = [dict(r) for r in await conn.fetch(
                "SELECT * FROM twin_treatments WHERE twin_id=$1::uuid ORDER BY start_date DESC",
                twin_id
            )]
            twin["imaging"] = [dict(r) for r in await conn.fetch(
                "SELECT * FROM twin_imaging WHERE twin_id=$1::uuid ORDER BY imaging_date DESC",
                twin_id
            )]

            # Convert UUID and datetime objects for JSON serialisation
            twin = _serialise(twin)
            return twin

    async def list_twins(self, limit: int = 50, offset: int = 0) -> list:
        """Return paginated list of twin summaries."""
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT
                    pt.*,
                    COUNT(th.id) FILTER (WHERE th.status = 'active') AS active_hpo_count
                FROM patient_twins pt
                LEFT JOIN twin_hpo_terms th ON th.twin_id = pt.twin_id
                GROUP BY pt.twin_id
                ORDER BY pt.last_updated DESC
                LIMIT $1 OFFSET $2
            """, limit, offset)

            result = []
            for row in rows:
                d = dict(row)
                d["demographics"] = (
                    json.loads(d["demographics"])
                    if isinstance(d["demographics"], str) else d["demographics"]
                )
                result.append(_serialise(d))
            return result

    async def update_twin(self, twin_id: str, request: TwinUpdateRequest) -> dict:
        """
        Partial update — only provided fields are changed.
        Appends new HPO terms, labs, treatments (does not overwrite).
        """
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                # Update core fields if provided
                if request.demographics or request.diagnostic_status or \
                        request.confirmed_diagnoses or request.clinical_notes is not None:
                    await conn.execute("""
                        UPDATE patient_twins SET
                            demographics      = COALESCE($2::jsonb, demographics),
                            diagnostic_status = COALESCE($3, diagnostic_status),
                            confirmed_diagnoses = COALESCE($4, confirmed_diagnoses),
                            clinical_notes    = COALESCE($5, clinical_notes),
                            last_updated      = NOW(),
                            version           = version + 1
                        WHERE twin_id = $1::uuid
                    """,
                        twin_id,
                        json.dumps(request.demographics.model_dump(mode="json"))
                            if request.demographics else None,
                        request.diagnostic_status.value
                            if request.diagnostic_status else None,
                        request.confirmed_diagnoses,
                        request.clinical_notes,
                    )

                # Resolve and map new HPO terms or free-text symptoms if provided
                from backend.hpo.hpo_mapper import HPOMapper
                from backend.models.twin_schemas import HPOTermInput
                mapper = HPOMapper()
                resolved_hpos = []

                if request.hpo_terms:
                    for h in request.hpo_terms:
                        label = h.hpo_label.strip().strip('"').strip("'")
                        term_id = h.hpo_term_id
                        if term_id == "HP:0000000" or not term_id:
                            mapped = mapper.map_symptom(label)
                            if mapped:
                                term_id = mapped["hpo_id"]
                                label = mapped["term"]
                        h.hpo_term_id = term_id
                        h.hpo_label = label
                        resolved_hpos.append(h)

                if request.symptoms:
                    for sym in request.symptoms:
                        sym_clean = sym.strip().strip('"').strip("'")
                        if not sym_clean:
                            continue
                        mapped = mapper.map_symptom(sym_clean)
                        if mapped:
                            if not any(h.hpo_term_id == mapped["hpo_id"] for h in resolved_hpos):
                                resolved_hpos.append(HPOTermInput(
                                    hpo_term_id=mapped["hpo_id"],
                                    hpo_label=mapped["term"],
                                    status="active",
                                    source="clinician"
                                ))
                        else:
                            if not any(h.hpo_label.lower() == sym_clean.lower() for h in resolved_hpos):
                                resolved_hpos.append(HPOTermInput(
                                    hpo_term_id="HP:0000000",
                                    hpo_label=sym_clean,
                                    status="active",
                                    source="clinician"
                                ))

                request.hpo_terms = resolved_hpos

                # Append new HPO terms
                if request.hpo_terms:
                    await conn.executemany("""
                        INSERT INTO twin_hpo_terms
                            (twin_id, hpo_term_id, hpo_label, onset_date,
                             status, severity, source)
                        VALUES ($1,$2,$3,$4,$5,$6,$7)
                        ON CONFLICT (twin_id, hpo_term_id)
                        WHERE status != 'excluded' DO NOTHING
                    """, [
                        (twin_id, h.hpo_term_id, h.hpo_label, h.onset_date,
                         h.status.value, h.severity, h.source)
                        for h in request.hpo_terms
                    ])


                # Append new lab results
                if request.lab_results:
                    await conn.executemany("""
                        INSERT INTO twin_lab_results
                            (twin_id, test_date, test_name, loinc_code,
                             value_numeric, value_text, unit, reference_low,
                             reference_high, interpretation, hpo_associations)
                        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
                    """, [
                        (twin_id, l.test_date, l.test_name, l.loinc_code,
                         l.value_numeric, l.value_text, l.unit, l.reference_low,
                         l.reference_high, l.interpretation, l.hpo_associations)
                        for l in request.lab_results
                    ])

                # Append new treatments
                if request.treatments:
                    await conn.executemany("""
                        INSERT INTO twin_treatments
                            (twin_id, drug_name, drug_brand, rxnorm_code,
                             indication, start_date, end_date, dose, route,
                             response, notes)
                        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
                    """, [
                        (twin_id, t.drug_name, t.drug_brand, t.rxnorm_code,
                         t.indication, t.start_date, t.end_date, t.dose, t.route,
                         t.response.value, t.notes)
                        for t in request.treatments
                    ])

                # Timeline event for the update
                await conn.execute("""
                    INSERT INTO twin_timeline_events
                        (twin_id, event_date, event_type, title, severity, metadata)
                    VALUES ($1, NOW()::date, 'encounter', 'Twin Updated', 'info', $2)
                """, twin_id, json.dumps({"fields_updated": [
                    k for k, v in request.model_dump(exclude_unset=True).items() if v
                ]}))

        # Refresh ChromaDB embedding
        try:
            from backend.twin.similarity import upsert_twin_embedding
            twin_data = await self.get_twin(twin_id)
            if twin_data:
                upsert_twin_embedding(twin_id, twin_data)
        except Exception as e:
            logger.warning(f"ChromaDB refresh failed after update: {e}")

        return {"twin_id": twin_id, "message": "Twin updated successfully."}

    async def delete_twin(self, twin_id: str) -> dict:
        """Permanently delete a twin and all its records (cascade)."""
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM patient_twins WHERE twin_id = $1::uuid", twin_id
            )
        # Remove from ChromaDB
        try:
            from backend.twin.similarity import TWIN_COLLECTION
            TWIN_COLLECTION.delete(ids=[twin_id])
        except Exception:
            pass
        return {"twin_id": twin_id, "message": "Twin deleted.", "db_result": result}


# ─── Serialisation Helper ──────────────────────────────────────────────────────

def _serialise(obj):
    """Recursively convert non-JSON-serialisable types (UUID, datetime, date)."""
    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_serialise(i) for i in obj]
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat()
    elif hasattr(obj, "__str__") and type(obj).__name__ in ("UUID", "Record"):
        return str(obj)
    return obj
