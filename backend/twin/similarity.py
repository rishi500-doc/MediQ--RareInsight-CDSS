"""
Similarity Engine — Finds phenotypically similar patients using ChromaDB.

Uses PubMedBERT embeddings (same model as the existing RAG pipeline)
to convert patient twin profiles into dense vectors and perform cosine
similarity search.

A separate ChromaDB collection ("patient_twins") is used so that
patient data never mingles with the rare disease knowledge base.
"""
from __future__ import annotations
import json
import os
from typing import List, Dict, Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from backend.utils.common import get_logger

logger = get_logger("CDSS.Twin.Similarity")

# ─── ChromaDB Client ──────────────────────────────────────────────────────────
# Re-use the same persistent ChromaDB path as the existing RAG pipeline
_CHROMA_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
_CHROMA_CLIENT = chromadb.PersistentClient(path=_CHROMA_PATH)

TWIN_COLLECTION = _CHROMA_CLIENT.get_or_create_collection(
    name="patient_twins",
    metadata={"hnsw:space": "cosine"},
)

# ─── Embedding Model ──────────────────────────────────────────────────────────
# Same PubMedBERT model used by the existing HybridRetriever for consistency
_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL", "NeuML/pubmedbert-base-embeddings"
)
_model: Optional[SentenceTransformer] = None


def _get_model() -> SentenceTransformer:
    """Lazy-load the embedding model (only once)."""
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {_MODEL_NAME}")
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


# ─── Profile → Text Representation ───────────────────────────────────────────

def build_twin_text_representation(twin: dict) -> str:
    """
    Convert a patient twin dict into a rich medical text string for embedding.

    The quality of this function directly determines the quality of patient
    matching. Order: phenotype (most important) → genetics → labs → demographics.
    """
    parts: List[str] = []

    # 1. Active HPO terms (most discriminative signal)
    hpo_terms = twin.get("hpo_terms", [])
    active_labels = [
        h.get("hpo_label") or h.get("label", "")
        for h in hpo_terms
        if h.get("status", "active") == "active" and h.get("hpo_label") or h.get("label")
    ]
    if active_labels:
        parts.append(f"Active phenotype: {', '.join(active_labels[:12])}")

    # 2. Resolved HPO terms (still clinically relevant)
    resolved_labels = [
        h.get("hpo_label") or h.get("label", "")
        for h in hpo_terms
        if h.get("status") == "resolved" and h.get("hpo_label") or h.get("label")
    ]
    if resolved_labels:
        parts.append(f"Resolved symptoms: {', '.join(resolved_labels[:6])}")

    # 3. Confirmed diagnoses
    diagnoses = twin.get("confirmed_diagnoses", [])
    if diagnoses:
        parts.append(f"Confirmed diagnoses: {', '.join(diagnoses)}")

    # 4. Pathogenic genetic variants
    variants = twin.get("genetic_variants", [])
    pathogenic = [
        f"{v.get('gene_symbol', '')} {v.get('hgvs_notation', '')}".strip()
        for v in variants
        if v.get("classification") in ["Pathogenic", "Likely_Pathogenic"]
    ]
    if pathogenic:
        parts.append(f"Pathogenic variants: {', '.join(pathogenic[:5])}")

    # 5. Abnormal lab results
    labs = twin.get("lab_results", [])
    abnormal_labs = list(dict.fromkeys(
        l["test_name"] for l in labs
        if l.get("interpretation") in
           ["high", "low", "critical_high", "critical_low", "abnormal"]
    ))
    if abnormal_labs:
        parts.append(f"Abnormal labs: {', '.join(abnormal_labs[:8])}")

    # 6. Demographics (lower weight — placed last)
    demo = twin.get("demographics", {})
    age_str = f"Age {demo.get('age_at_creation', '?')} years" if demo else ""
    gender  = demo.get("gender", "") if demo else ""
    ethnicity = demo.get("ethnicity", "") if demo else ""
    demo_parts = [p for p in [age_str, gender, ethnicity] if p]
    if demo_parts:
        parts.append(" ".join(demo_parts))

    # 7. Family history flag
    fam = twin.get("family_history", [])
    if any(f.get("consanguinity") for f in fam):
        parts.append("Consanguineous parents")
    if fam:
        conditions = [f.get("condition", "") for f in fam if f.get("condition")]
        if conditions:
            parts.append(f"Family history: {', '.join(conditions[:3])}")

    text = ". ".join(filter(None, parts))
    return text or "Unknown patient profile"


# ─── Upsert Embedding ─────────────────────────────────────────────────────────

def upsert_twin_embedding(twin_id: str, twin_data: dict) -> list:
    """
    Generate a PubMedBERT embedding for the twin and upsert it into ChromaDB.
    Returns the embedding vector.
    """
    model = _get_model()
    text  = build_twin_text_representation(twin_data)
    embedding = model.encode(text, normalize_embeddings=True).tolist()

    # Build metadata for ChromaDB (must be flat key-value, no nested dicts)
    hpo_terms = twin_data.get("hpo_terms", [])
    active_hpo_ids = [
        h.get("hpo_term_id") or h.get("term_id", "")
        for h in hpo_terms
        if h.get("status", "active") == "active"
    ]
    demo = twin_data.get("demographics") or {}

    TWIN_COLLECTION.upsert(
        ids=[twin_id],
        embeddings=[embedding],
        documents=[text],
        metadatas=[{
            "twin_id"            : twin_id,
            "patient_id"         : str(twin_data.get("patient_id", "")),
            "hpo_terms_json"     : json.dumps(active_hpo_ids),
            "diagnoses_json"     : json.dumps(twin_data.get("confirmed_diagnoses", [])),
            "diagnostic_status"  : str(twin_data.get("diagnostic_status", "undiagnosed")),
            "age"                : int(demo.get("age_at_creation") or 0),
            "gender"             : str(demo.get("gender") or ""),
        }],
    )
    logger.info(f"Upserted ChromaDB embedding for twin {twin_id} ({len(active_hpo_ids)} HPO terms)")
    return embedding


# ─── Similarity Search ────────────────────────────────────────────────────────

def find_similar_patients(
    query_twin_id: str,
    n_results: int = 10,
    min_similarity: float = 0.60,
) -> List[Dict]:
    """
    Find the top-N most phenotypically similar patient twins.
    Uses cosine similarity on PubMedBERT embeddings stored in ChromaDB.

    Args:
        query_twin_id:  The twin to find matches for
        n_results:      Maximum number of results to return
        min_similarity: Minimum cosine similarity threshold (0–1)

    Returns:
        List of dicts with twin_id, similarity_score, metadata, text_summary
    """
    # Retrieve the query twin's embedding from ChromaDB
    result = TWIN_COLLECTION.get(ids=[query_twin_id], include=["embeddings", "metadatas"])

    if not result["ids"]:
        logger.warning(f"Twin {query_twin_id} not found in ChromaDB — embed it first.")
        return []

    query_embedding = result["embeddings"][0]
    query_metadata  = result["metadatas"][0] if result["metadatas"] else {}

    # Query for neighbours (fetch extra to account for self-exclusion)
    fetch_n = min(n_results + 1, TWIN_COLLECTION.count())
    if fetch_n == 0:
        return []

    matches = TWIN_COLLECTION.query(
        query_embeddings=[query_embedding],
        n_results=fetch_n,
        include=["documents", "metadatas", "distances"],
    )

    similar: List[Dict] = []
    for i, doc_id in enumerate(matches["ids"][0]):
        if doc_id == query_twin_id:
            continue  # exclude self

        distance   = matches["distances"][0][i]
        similarity = 1.0 - distance  # cosine distance → similarity

        if similarity < min_similarity:
            continue

        meta = matches["metadatas"][0][i] if matches["metadatas"] else {}

        # Compute HPO Jaccard similarity as a secondary signal
        query_hpo_ids  = json.loads(query_metadata.get("hpo_terms_json", "[]"))
        match_hpo_ids  = json.loads(meta.get("hpo_terms_json", "[]"))
        jaccard = _jaccard(query_hpo_ids, match_hpo_ids)

        similar.append({
            "twin_id"         : doc_id,
            "similarity_score": round(similarity, 4),
            "jaccard_score"   : round(jaccard, 4),
            "metadata"        : meta,
            "text_summary"    : matches["documents"][0][i]
                                if matches["documents"] else "",
        })

    # Sort by embedding similarity descending
    similar.sort(key=lambda x: -x["similarity_score"])
    return similar[:n_results]


def find_similar_by_text(
    query_text: str,
    n_results: int = 5,
    min_similarity: float = 0.60,
) -> List[Dict]:
    """
    Find similar patients given a raw symptom / HPO text query.
    Useful for the main RAG analyze endpoint to find relevant twins
    even before a twin has been created for the current patient.
    """
    model     = _get_model()
    embedding = model.encode(query_text, normalize_embeddings=True).tolist()

    if TWIN_COLLECTION.count() == 0:
        return []

    matches = TWIN_COLLECTION.query(
        query_embeddings=[embedding],
        n_results=min(n_results, TWIN_COLLECTION.count()),
        include=["documents", "metadatas", "distances"],
    )

    results = []
    for i, doc_id in enumerate(matches["ids"][0]):
        distance   = matches["distances"][0][i]
        similarity = 1.0 - distance
        if similarity < min_similarity:
            continue
        results.append({
            "twin_id"         : doc_id,
            "similarity_score": round(similarity, 4),
            "metadata"        : matches["metadatas"][0][i] if matches["metadatas"] else {},
            "text_summary"    : matches["documents"][0][i] if matches["documents"] else "",
        })

    return results


def get_collection_stats() -> dict:
    """Return ChromaDB collection statistics."""
    return {
        "collection_name": "patient_twins",
        "twin_count"     : TWIN_COLLECTION.count(),
        "chroma_path"    : _CHROMA_PATH,
    }


# ─── HPO Jaccard ──────────────────────────────────────────────────────────────

def _jaccard(set_a: list, set_b: list) -> float:
    """Compute Jaccard similarity between two HPO term ID lists."""
    sa, sb = set(set_a), set(set_b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)
