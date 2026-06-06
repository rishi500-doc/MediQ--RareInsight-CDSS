# ProHealth Rare Disease CDSS — Complete Documentation

## Overview

ProHealth Rare Disease CDSS is an AI-powered Clinical Decision Support System focused on rare disease diagnosis and patient digital twins. It combines Retrieval-Augmented Generation (RAG), HPO mapping, PubMedBERT embeddings, and an event-driven digital twin to support clinicians with diagnostic candidates, evidence summaries, and longitudinal patient analysis.

- Backend: FastAPI application serving a dashboard UI and JSON APIs.
- Key features: symptom-to-HPO mapping, hybrid retrieval + reranking, streaming reasoning, digital twins (CRUD + timeline), similarity search using embeddings.

## Repository Structure (high-level)

- `backend/` — Main application code and services
  - `main.py` — FastAPI app, CORS, template mounting, lifespan hooks
  - `api/` — API routers and endpoints
    - `routes.py` — RAG/analysis endpoints (`/api/v1/analyze`, `/api/v1/ingest`)
    - `twin_routes.py` — Digital Twin endpoints (`/api/v1/twin/...`)
  - `ingestion/` — Ingestion pipeline and indexer
  - `retriever/` — Retrieval engines, hybrid retriever, disease API client
  - `models/` — Pydantic schemas for request/response models
  - `nlp/` — Clinical NLP components (NER, symptom normalization, negation)
  - `twin/` — Digital twin builder, timeline, similarity, progression
  - `db/` — Database pool and migrations
  - `embeddings/` — Embedding manager
  - `vector_db/` — Local Chroma artifacts (for development)
- `data/` — Reference data (HPO dictionaries, clinical synonyms, CSVs)
- `templates/`, `static/` — Frontend templates and assets
- `requirements.txt`, `pyproject.toml` — Python dependencies
- `run_dev.bat` — Windows helper for running the dev server

## Design & Architecture

- FastAPI-based backend exposing a single-page dashboard and REST endpoints.
- Lifespan hook in `backend/main.py` runs DB migrations on startup when `DATABASE_URL` is set.
- Routers are mounted under `/api/v1`:
  - General RAG/analysis router in `backend/api/routes.py`
  - Digital Twin router in `backend/api/twin_routes.py` (prefix `/twin`)
- Hybrid retrieval: `HybridRetriever` integrates vector search (Chroma) + lexical/reranker to produce diagnostic candidates.
- Streaming reasoning: `stream_clinical_reasoning` yields server-sent events for progressive AI output.
- Digital Twin: Twins are persisted to PostgreSQL and embeddings stored in ChromaDB. The twin module provides health checks, CRUD, timelines, similarity matching, progression prediction, and RAG analysis bridging.

## Installation & Setup

Prerequisites:
- Python 3.10+ (recommended)
- PostgreSQL (optional for twin features; required to enable DB-backed twin features)
- Optional: a local ChromaDB instance or the included `vector_db/chroma.sqlite3` for dev

1. Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Environment variables

Copy `.env.example` (if present) to `.env` and set the following variables as needed:

- `DATABASE_URL` — PostgreSQL connection string (enables digital twin features and migrations)
- Any API keys or model endpoints used by the reasoning or embedding services

Notes:
- If `DATABASE_URL` is not set, `main.py` will log a warning and skip DB migrations; the dashboard remains available but twin features are disabled.

## Running the Application (development)

- Windows helper (provided):

```powershell
.\run_dev.bat
```

- Or run with Uvicorn directly:

```powershell
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/` to access the dashboard UI. The dashboard template is served from `templates/index.html` and static assets from `static/`.

## Key API Endpoints (summary)

All API endpoints are prefixed with `/api/v1` as configured in `backend/main.py`.

- `POST /api/v1/analyze`
  - Accepts `PatientData` (see `backend/models/schemas.py`) and runs the diagnostic pipeline:
    - Optional NLP parsing of `clinical_notes` via `ClinicalNLPEngine`
    - HPO mapping, query building, hybrid retrieval (via `HybridRetriever`)
    - BioMistral evidence compression and streaming reasoning
    - Drug interaction checks
  - Returns a `text/event-stream` SSE streaming response produced by `stream_clinical_reasoning`.

- `POST /api/v1/ingest`
  - Triggers the ingestion pipeline (`IngestionManager.ingest_pipeline`) to update the knowledge base and vector index.
  - Returns ingestion `status` and updated knowledge base `stats`.

Digital Twin endpoints (prefix `/api/v1/twin`):

- `GET /api/v1/twin/health` — Health check for twin subsystem (Postgres + Chroma availability).
- `POST /api/v1/twin/create` — Create a new patient twin. Accepts `TwinCreateRequest`.
- `GET /api/v1/twin/{twin_id}` — Retrieve full twin details.
- `GET /api/v1/twin/` — List twins (paginated).
- `PUT /api/v1/twin/{twin_id}` — Update twin (partial append semantics for HPO/labs/treatments).
- `DELETE /api/v1/twin/{twin_id}` — Delete twin permanently (removes Chroma embedding).
- `POST /api/v1/twin/{twin_id}/events` — Add a timeline event.
- `GET /api/v1/twin/{twin_id}/timeline` — Retrieve timeline with filters.
- `GET /api/v1/twin/{twin_id}/timeline/trends?test_name=...` — Lab trend time-series.
- `GET /api/v1/twin/{twin_id}/timeline/milestones` — Diagnostic milestones and delay calculation.
- `GET /api/v1/twin/{twin_id}/timeline/alerts` — Recent high/critical events.
- `GET /api/v1/twin/{twin_id}/similar?n=5&min_score=0.6` — Find similar patients using embeddings + HPO overlap.
- `POST /api/v1/twin/{twin_id}/predict/progression?horizon_months=6` — Predict progression (rule-based, with future model plans).
- `POST /api/v1/twin/{twin_id}/analyze` — Run the RAG pipeline enriched with twin context (bridging twin -> RAG).

Refer to `backend/api/routes.py` and `backend/api/twin_routes.py` for actionable parameter names, request/response models, and error handling.

## Data & Models

- `data/hpo/` and `backend/hpo/` hold HPO dictionaries and metadata used for mapping clinical phenotype tokens to HPO terms.
- `data/cases_clean.csv` — Example/seed cases used in testing or ingestion.
- Clinical synonym files and hallmark lists live under `data/` for normalisation.
- Models and schemas used for requests/responses are defined in `backend/models/`.

## Database & Migrations

- `backend/db/init_db.py` exposes `run_migrations()` which is invoked on startup when `DATABASE_URL` is set.
- SQL migrations present under `backend/db/migrations/` (e.g., `001_create_digital_twin_schema.sql`).
- Database connection pooling and helpers are provided in `backend/db/database.py`.

## Embeddings & Vector DB

- Embeddings are managed by `backend/embeddings/embedding_manager.py` and used with ChromaDB.
- A development Chroma artifact is present under `vector_db/` — `chroma.sqlite3` and a UUID-named folder. In production, configure a proper Chroma instance or vector DB.

## Ingestion Pipeline

- The ingestion pipeline lives in `backend/ingestion/` and includes:
  - `fetcher.py` — gathers source literature and evidence
  - `chunker.py` — splits documents into passages
  - `metadata_extractor.py` — extracts structured metadata
  - `vector_indexer.py` — encodes passages and upserts to the vector index
  - `ingestion_manager.py` — orchestrates the pipeline and exposes a programmatic `ingest_pipeline()` method

Trigger ingestion manually via `POST /api/v1/ingest`.

## NLP & Clinical Parsing

- `backend/nlp/` contains specialized clinical processing:
  - `medical_ner.py` — named entity recognition for medical mentions
  - `negation_detector.py` — removes negated findings
  - `symptom_normalizer.py` and `temporal_parser.py` — normalize and extract symptom/timing
  - `clinical_nlp.py` — high-level orchestrator used in `routes.analyze`

## Logging & Observability

- Logger instances are created via `backend/utils/common.py` and used across modules.
- The FastAPI app exposes health and static endpoints; add a metrics exporter or middleware for Prometheus if desired.

## Testing

- A `tests/` directory exists for unit and integration tests. Run tests with your test runner (e.g., `pytest`).

## Development Notes

- `main.py` warns and disables twin features if `DATABASE_URL` is not set. This allows quick dev runs without Postgres.
- Streaming endpoints use SSE (`text/event-stream`) for progressive streaming of reasoning results.
- Gene extraction: `routes.py` uses a regex to extract HGNC-like gene symbols from free text and merges these with NLP-extracted genes.

## Security & Privacy

- Patient data (PHI) handling: This project assumes deployment in a secure, HIPAA-compliant environment when processing real patient data. Ensure database encryption, access controls, logging policies, and secure model endpoints.
- Delete operations (e.g., `DELETE /api/v1/twin/{twin_id}`) are irreversible — consider adding soft-delete flags and audit trails for safety.

## Deployment Recommendations

- Use environment-specific `.env` files and secrets management (Azure Key Vault, AWS Secrets Manager, etc.).
- Run migrations at startup or from CI/CD using `backend/db/init_db.py` functions.
- Run workers for ingestion and long-running tasks separately from the FastAPI server to avoid blocking.
- Persist Chroma and Postgres to durable storage; back up embeddings and DB snapshots regularly.

## Extensibility

- Add model-based progression prediction in `backend/twin/progression` using training pipelines and model registry.
- Add authentication/authorization middleware to protect API endpoints.
- Add OpenAPI docs enhancements (tags, examples) — FastAPI generates `/docs` and `/redoc` automatically.

## Files of Interest

- `backend/main.py` — App entrypoint and router mounting
- `backend/api/routes.py` — Diagnostic RAG endpoints
- `backend/api/twin_routes.py` — Twin CRUD + analysis
- `backend/ingestion/` — Ingestion pipeline
- `backend/retriever/` — Hybrid retriever & engine
- `backend/twin/` — Twin builder, timeline, similarity, progression

## Next Steps (suggested)

- Add a populated `README.md` with quickstart commands and example requests.
- Add Postman/HTTPie collection with sample API calls for `analyze` and twin CRUD.
- Add CI test runner and linting in `pyproject.toml` or GitHub Actions workflow.

---

*Generated by an automated documentation pass. For clarification or to expand any section (API examples, request/response schemas, diagrams), tell me which area you'd like expanded.*
