-- ─────────────────────────────────────────────────────────────────────────────
-- ProHealth CDSS — Digital Twin Database Schema
-- Migration: 001_create_digital_twin_schema.sql
-- Run via: backend/db/init_db.py (called automatically on app startup)
-- ─────────────────────────────────────────────────────────────────────────────

-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ─── Core Patient Twin ───────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS patient_twins (
    twin_id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id          TEXT NOT NULL UNIQUE,          -- external patient reference ID
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_updated        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    version             INTEGER NOT NULL DEFAULT 1,

    -- Demographics stored as JSONB for schema flexibility
    demographics        JSONB NOT NULL DEFAULT '{}',

    -- Vector store reference (embedding stored in ChromaDB, doc ID here)
    chroma_doc_id       TEXT,
    embedding_model     TEXT DEFAULT 'NLP4Science/pubmedbert-base-embeddings',

    -- Diagnostic status
    diagnostic_status   TEXT NOT NULL DEFAULT 'undiagnosed'
                        CHECK (diagnostic_status IN
                            ('undiagnosed', 'suspected', 'confirmed', 'excluded')),
    confirmed_diagnoses TEXT[] NOT NULL DEFAULT '{}',

    -- Data quality signals
    data_completeness   FLOAT NOT NULL DEFAULT 0.0
                        CHECK (data_completeness BETWEEN 0 AND 1),
    confidence_score    FLOAT NOT NULL DEFAULT 0.0
                        CHECK (confidence_score BETWEEN 0 AND 1),

    -- Free-text notes from clinician
    clinical_notes      TEXT,

    -- Cached RAG summary (refreshed by twin builder agent)
    rag_summary         JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_twins_patient_id ON patient_twins(patient_id);
CREATE INDEX IF NOT EXISTS idx_twins_status ON patient_twins(diagnostic_status);
CREATE INDEX IF NOT EXISTS idx_twins_updated ON patient_twins(last_updated DESC);


-- ─── HPO Phenotype Terms ──────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS twin_hpo_terms (
    id              BIGSERIAL PRIMARY KEY,
    twin_id         UUID NOT NULL REFERENCES patient_twins(twin_id) ON DELETE CASCADE,
    hpo_term_id     VARCHAR(12) NOT NULL,              -- e.g. HP:0002099
    hpo_label       TEXT NOT NULL,
    onset_date      DATE,
    status          TEXT NOT NULL DEFAULT 'active'
                    CHECK (status IN ('active', 'resolved', 'excluded')),
    severity        FLOAT CHECK (severity BETWEEN 0 AND 1),
    source          TEXT NOT NULL DEFAULT 'clinician'  -- clinician | nlp | agent
                    CHECK (source IN ('clinician', 'nlp', 'agent', 'import')),
    recorded_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_hpo_twin ON twin_hpo_terms(twin_id);
CREATE INDEX IF NOT EXISTS idx_hpo_term ON twin_hpo_terms(hpo_term_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_hpo_twin_term
    ON twin_hpo_terms(twin_id, hpo_term_id)
    WHERE status != 'excluded';


-- ─── Clinical Encounters ─────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS twin_encounters (
    encounter_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    twin_id             UUID NOT NULL REFERENCES patient_twins(twin_id) ON DELETE CASCADE,
    encounter_date      DATE NOT NULL,
    encounter_type      TEXT DEFAULT 'outpatient'
                        CHECK (encounter_type IN
                            ('outpatient', 'inpatient', 'emergency', 'telehealth', 'procedure')),
    chief_complaint     TEXT,
    clinical_notes      TEXT,
    clinician_id        TEXT,
    facility            TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_enc_twin ON twin_encounters(twin_id, encounter_date DESC);


-- ─── Lab Results ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS twin_lab_results (
    lab_id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    twin_id             UUID NOT NULL REFERENCES patient_twins(twin_id) ON DELETE CASCADE,
    encounter_id        UUID REFERENCES twin_encounters(encounter_id) ON DELETE SET NULL,
    test_date           DATE NOT NULL,
    test_name           TEXT NOT NULL,
    loinc_code          VARCHAR(20),
    value_numeric       FLOAT,
    value_text          TEXT,
    unit                TEXT,
    reference_low       FLOAT,
    reference_high      FLOAT,
    interpretation      TEXT
                        CHECK (interpretation IN
                            ('normal', 'low', 'high', 'critical_low',
                             'critical_high', 'abnormal', 'pending')),
    hpo_associations    TEXT[] NOT NULL DEFAULT '{}',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_lab_twin ON twin_lab_results(twin_id, test_date DESC);
CREATE INDEX IF NOT EXISTS idx_lab_name ON twin_lab_results(twin_id, test_name);


-- ─── Imaging Findings ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS twin_imaging (
    imaging_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    twin_id             UUID NOT NULL REFERENCES patient_twins(twin_id) ON DELETE CASCADE,
    encounter_id        UUID REFERENCES twin_encounters(encounter_id) ON DELETE SET NULL,
    imaging_date        DATE NOT NULL,
    modality            TEXT,
    body_region         TEXT,
    findings            TEXT,
    impression          TEXT,
    hpo_associations    TEXT[] NOT NULL DEFAULT '{}',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_img_twin ON twin_imaging(twin_id, imaging_date DESC);


-- ─── Genetic Variants ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS twin_genetic_variants (
    variant_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    twin_id             UUID NOT NULL REFERENCES patient_twins(twin_id) ON DELETE CASCADE,
    gene_symbol         TEXT NOT NULL,
    hgvs_notation       TEXT,
    zygosity            TEXT CHECK (zygosity IN ('homozygous', 'heterozygous', 'hemizygous', 'unknown')),
    classification      TEXT CHECK (classification IN
                            ('Pathogenic', 'Likely_Pathogenic', 'VUS',
                             'Likely_Benign', 'Benign', 'Unknown')),
    clinvar_id          TEXT,
    omim_id             TEXT,
    panel_tested        TEXT,
    inheritance         TEXT,
    recorded_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_gene_twin ON twin_genetic_variants(twin_id);
CREATE INDEX IF NOT EXISTS idx_gene_symbol ON twin_genetic_variants(gene_symbol);


-- ─── Family History ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS twin_family_history (
    id              BIGSERIAL PRIMARY KEY,
    twin_id         UUID NOT NULL REFERENCES patient_twins(twin_id) ON DELETE CASCADE,
    relationship    TEXT,
    condition       TEXT,
    hpo_term_id     TEXT,
    consanguinity   BOOLEAN NOT NULL DEFAULT FALSE,
    notes           TEXT,
    recorded_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_fam_twin ON twin_family_history(twin_id);


-- ─── Treatments ──────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS twin_treatments (
    treatment_id    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    twin_id         UUID NOT NULL REFERENCES patient_twins(twin_id) ON DELETE CASCADE,
    encounter_id    UUID REFERENCES twin_encounters(encounter_id) ON DELETE SET NULL,
    drug_name       TEXT NOT NULL,
    drug_brand      TEXT,
    rxnorm_code     TEXT,
    indication      TEXT,
    start_date      DATE NOT NULL,
    end_date        DATE,
    dose            TEXT,
    route           TEXT,
    response        TEXT DEFAULT 'unknown'
                    CHECK (response IN
                        ('excellent', 'good', 'partial', 'none',
                         'adverse', 'unknown', 'ongoing')),
    notes           TEXT,
    recorded_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_tx_twin ON twin_treatments(twin_id, start_date DESC);
CREATE INDEX IF NOT EXISTS idx_tx_drug ON twin_treatments(drug_name);


-- ─── Unified Timeline Events ─────────────────────────────────────────────────
-- Single queryable log of every clinically significant event for a patient.
-- Source tables write here automatically via the TimelineEngine.
CREATE TABLE IF NOT EXISTS twin_timeline_events (
    event_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    twin_id         UUID NOT NULL REFERENCES patient_twins(twin_id) ON DELETE CASCADE,
    event_date      DATE NOT NULL,
    event_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    event_type      TEXT NOT NULL
                    CHECK (event_type IN (
                        'symptom_onset', 'symptom_resolved', 'lab_result', 'imaging',
                        'diagnosis', 'treatment_start', 'treatment_end',
                        'hospitalization', 'genetic_result', 'family_history',
                        'encounter', 'agent_note', 'rag_analysis'
                    )),
    title           TEXT NOT NULL,
    description     TEXT,
    severity        TEXT DEFAULT 'info'
                    CHECK (severity IN ('info', 'low', 'medium', 'high', 'critical')),
    hpo_terms       TEXT[] NOT NULL DEFAULT '{}',
    source_id       UUID,              -- FK to source record (lab_id, encounter_id, etc.)
    source_table    TEXT,              -- which table the event originated from
    metadata        JSONB NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_timeline_twin_date
    ON twin_timeline_events(twin_id, event_date DESC);
CREATE INDEX IF NOT EXISTS idx_timeline_type
    ON twin_timeline_events(twin_id, event_type);


-- ─── Progression Predictions (Audit Trail) ───────────────────────────────────
CREATE TABLE IF NOT EXISTS twin_progression_predictions (
    prediction_id   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    twin_id         UUID NOT NULL REFERENCES patient_twins(twin_id) ON DELETE CASCADE,
    predicted_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    model_version   TEXT NOT NULL DEFAULT 'rule-based-v1',
    horizon_days    INTEGER NOT NULL DEFAULT 180,
    predicted_hpo   TEXT[] NOT NULL DEFAULT '{}',
    risk_level      TEXT CHECK (risk_level IN ('low', 'medium', 'high', 'critical')),
    confidence      FLOAT CHECK (confidence BETWEEN 0 AND 1),
    trigger_terms   TEXT[] NOT NULL DEFAULT '{}',
    raw_output      JSONB NOT NULL DEFAULT '{}',
    actual_outcome  JSONB DEFAULT NULL  -- filled in later for model evaluation
);

CREATE INDEX IF NOT EXISTS idx_pred_twin ON twin_progression_predictions(twin_id, predicted_at DESC);


-- ─── Similar Patient Index ───────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS twin_similarity_index (
    id              BIGSERIAL PRIMARY KEY,
    twin_id_a       UUID NOT NULL REFERENCES patient_twins(twin_id) ON DELETE CASCADE,
    twin_id_b       UUID NOT NULL REFERENCES patient_twins(twin_id) ON DELETE CASCADE,
    similarity_score FLOAT NOT NULL CHECK (similarity_score BETWEEN 0 AND 1),
    jaccard_score   FLOAT CHECK (jaccard_score BETWEEN 0 AND 1),
    computed_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (twin_id_a, twin_id_b),
    CHECK (twin_id_a <> twin_id_b)
);

CREATE INDEX IF NOT EXISTS idx_sim_a ON twin_similarity_index(twin_id_a, similarity_score DESC);
CREATE INDEX IF NOT EXISTS idx_sim_b ON twin_similarity_index(twin_id_b, similarity_score DESC);
