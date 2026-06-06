# ProHealth Rare Disease CDSS: Architecture & API Walkthrough

The **ProHealth Rare Disease Clinical Decision Support System (CDSS)** is an advanced, AI-powered diagnostic platform. It uses a **Retrieval-Augmented Generation (RAG)** pipeline combined with **Human Phenotype Ontology (HPO)** mapping and **Real-Time Streaming AI** to help clinicians diagnose complex genetic and rare diseases.

This document breaks down how the system works end-to-end, with a specific deep dive into the API architecture.

---

## 🏗️ 1. High-Level Architecture

The system follows a modern decoupled architecture using Python (FastAPI) for the backend and Vanilla JS/HTML/CSS for the frontend dashboard.

**The Diagnostic Flow:**
1. **Frontend**: Clinician inputs patient data (age, symptoms, family history) into the dashboard.
2. **API Endpoint**: Data is sent to the FastAPI `POST /api/v1/analyze` endpoint.
3. **Query Expansion**: Symptoms are mapped to standardized HPO terms.
4. **Vector Retrieval**: A hybrid search (vector embeddings + keyword matching) queries a ChromaDB database of rare diseases.
5. **Clinical Reranking**: Results are scored based on clinical hallmarks, age appropriateness, and genetic risk factors.
6. **AI Reasoning**: DeepSeek R1 streams a highly structured clinical evaluation back to the frontend in real-time.
7. **UI Update**: The dashboard updates its gauges, risk flags, and markdown reports dynamically.

---

## 🛠️ 2. Detailed Code Walkthrough

### A. The FastAPI Entrypoint (`backend/main.py`)
This is the heart of the web server. It initializes FastAPI, configures CORS, and serves the static frontend assets.
*   **`@app.get("/")`**: Renders the `index.html` template.
*   **`@app.get("/rag.png")`**: Serves the anatomy visualization image.
*   **`app.include_router(api_router)`**: Mounts the main diagnostic logic under the `/api/v1` prefix.

### B. The Diagnostic API (`backend/api/routes.py`)
This file defines the core `POST /analyze` endpoint. It orchestrates the entire pipeline.
*   **Input Validation**: It uses Pydantic (`PatientData` from `schemas.py`) to ensure the incoming JSON has valid symptoms and (optional) age.
*   **Execution**:
    1. It evaluates simple "Red Flags" (e.g., if "seizure" or "neurological" is present, it flags high risk).
    2. It passes the symptoms to the `HybridRetriever`.
    3. It passes the retrieved results to the `ClinicalReranker`.
    4. It hands the highly-ranked results to the `stream_clinical_reasoning` function.
*   **Output**: It returns a `StreamingResponse` with `media_type="text/event-stream"`, allowing the frontend to receive data chunk-by-chunk.

### C. Retrieval Pipeline (`backend/retriever/` & `backend/reranker/`)
This is the medical "brain" of the system.
*   **`ClinicalQueryBuilder`**: Translates messy human input (e.g., "sticky stool") into medical terminology (e.g., "Meconium ileus").
*   **`HybridRetriever`**: Connects to ChromaDB. It uses `PubMedBERT` to convert the query into a math vector and finds diseases with similar vectors.
*   **`ClinicalReranker`**: The initial vector search isn't perfect. The Reranker steps in to adjust scores. It boosts a disease's score if the patient has a known "Hallmark" symptom (loaded from `data/hallmarks.json`) or if their family history strongly aligns with a genetic disorder.

### D. The AI Streaming Engine (`backend/reasoning/clinical_ai.py`)
This script handles the connection to DeepSeek R1 via OpenRouter. It is highly optimized for real-time UI updates.
*   **The Prompt**: It instructs the AI to use strict Markdown delimiters (`[SECTION_ALIGNMENT]`, `[SECTION_REASONING]`). This forces the AI to structure its medical thoughts perfectly.
*   **Server-Sent Events (SSE)**: Instead of making the clinician wait 10 seconds for the whole AI response, it `yields` data instantly.
    *   First, it yields a `metadata` JSON packet containing the exact vector scores and risk flags.
    *   Then, as DeepSeek types, it yields `content` and `reasoning` packets continuously.

### E. The Frontend Dashboard (`templates/index.html`)
The frontend is a Single Page Application (SPA).
*   **Safe Rendering**: The JS uses `try/catch` blocks and `.innerHTML` safety checks to ensure the UI doesn't crash if data is missing.
*   **Stream Parsing**: It reads the `TextDecoder` stream from the API. When it sees `type: 'metadata'`, it instantly draws the severity gauges and populates the Top Matching Diseases list.
*   **Real-time Markdown**: As `type: 'content'` chunks arrive, it splits them based on the `[SECTION_*]` delimiters and routes the text to the correct dashboard cards (Alignment, Reasoning, Recommendations).

---

## 📡 3. The API Payload Specification

Here is exactly what the frontend sends and receives from the backend.

### Request: `POST /api/v1/analyze`
```json
{
  "symptoms": ["chronic cough", "salty skin", "meconium ileus", "poor weight gain"],
  "age": 2,
  "gender": "male",
  "family_history": true
}
```

### Response: Server-Sent Events (SSE) Stream
The API responds with a continuous stream of events formatted as `data: {...}\n\n`.

**Event 1: Metadata (Instant)**
```json
data: {
  "type": "metadata",
  "rare_flag": true,
  "score": 12,
  "possible_conditions": [
    {
      "disease": "Cystic Fibrosis",
      "final_score": 0.94,
      "reasoning": ["Salty skin is a hallmark", "Positive family history"]
    }
  ]
}
```

**Events 2-N: AI Reasoning (Streaming)**
```json
data: {"type": "content", "delta": "[SECTION_ALIGNMENT]\n### 1. DISEASE "}
data: {"type": "content", "delta": "ALIGNMENT ANALYSIS\nThe patient presents..."}
```

## 🚀 4. Summary of Recent Fixes
During our recent work, we hardened this architecture significantly:
1.  **Fixed 422 Errors**: Made `age` optional in `schemas.py` to prevent API rejection if the clinician leaves it blank.
2.  **UI Hardening**: Added global error catchers in `index.html` to prevent crashes during stream parsing.
3.  **Visual Polish**: Added a "Synthesizing Clinical Data" overlay mask and fixed CSS gauge alignments for a premium, production-ready feel.
4.  **Enforced AI Formatting**: Upgraded the `clinical_ai.py` prompt to force the generation of a specific "Genetic Summary", preventing the AI from rambling conversationally.
