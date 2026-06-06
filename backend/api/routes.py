from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from backend.models.schemas import PatientData
from backend.retriever.hybrid_retriever import HybridRetriever
from backend.reasoning.clinical_ai import stream_clinical_reasoning
from backend.nlp.clinical_nlp import ClinicalNLPEngine
from backend.ingestion.ingestion_manager import IngestionManager
from backend.reasoning.biomistral_engine import BioMistralEngine
from backend.retriever.disease_api_client import disease_api_client
from backend.utils.common import get_logger
from backend.retriever.engine import get_knowledge_base_stats
from backend.criteria import check_drug_interactions
import asyncio
import re

# Regex for standard HGNC gene symbol format: 1 uppercase letter + up to 7 alphanumeric chars
_GENE_SYMBOL_RE = re.compile(r'\b([A-Z][A-Z0-9]{1,7})\b')

nlp_engine = ClinicalNLPEngine()
ingestion_manager = IngestionManager()
biomistral_engine = BioMistralEngine()

# Initialize the production-grade hybrid retriever
hybrid_retriever = HybridRetriever()

def apply_clinical_rules(data: PatientData):
    patterns = []
    score = 0
    rare_flag = False
    
    if len(data.symptoms) >= 3:
        patterns.append("Multisystemic presentation detected")
        score += 40
        rare_flag = True
    
    if data.family_history:
        patterns.append("Genetic/Heritable history flag")
        score += 30
        rare_flag = True
        
    return {
        "rare_flag": rare_flag,
        "patterns": patterns,
        "score": score
    }

def get_recommendations(data: PatientData, patterns: dict):
    recs = [
        "Specialist referral to Clinical Genetics",
        "Whole Exome Sequencing (WES) consultation",
        "Baseline echocardiogram for syndromic screening"
    ]
    if data.family_history:
        recs.append("First-degree relative cascade screening recommended")
    if data.age is not None and data.age < 18:
        recs.append("Pediatric genetics consultation advised")
    return recs

router = APIRouter()
logger = get_logger("CDSS.API")

@router.post("/analyze")
async def analyze_patient(data: PatientData):
    """
    Production diagnostic endpoint.
    Flow: Input -> HPO Mapping -> Query Building -> Hybrid Reranked Retrieval -> Streaming AI Reasoning.
    """
    logger.info(f"New diagnostic request received: {len(data.symptoms)} symptoms detected.")
    
    try:
        # NLP Stage: Parse free-text clinical notes if provided
        nlp_summary = None
        if data.clinical_notes:
            logger.info("Processing unstructured clinical notes via NLP Engine.")
            extracted_data = await nlp_engine.process_clinical_note(data.clinical_notes)
            nlp_summary = extracted_data  # Preserve full extraction for UI transparency
            
            # Merge extracted symptoms with any manual symptoms
            if extracted_data.get("symptoms"):
                data.symptoms.extend(extracted_data["symptoms"])
            
            # Optionally add genes to genomic_info
            if extracted_data.get("genes"):
                if data.genomic_info:
                    data.genomic_info += ", " + ", ".join(extracted_data["genes"])
                else:
                    data.genomic_info = ", ".join(extracted_data["genes"])
                    
        # Ensure unique symptoms
        data.symptoms = list(set(data.symptoms))

        # ── Gene Symbol Extraction ────────────────────────────────────────────
        # PatientData.genomic_info is a free-text string (e.g. "COL5A1 / COL5A2 variants").
        # The reranker and reasoning engine expect a structured `genes` list.
        # Parse HGNC-format gene tokens (ALL-CAPS, 2-8 chars) from the string.
        extracted_genes: list = []
        if data.genomic_info:
            extracted_genes = list(dict.fromkeys(_GENE_SYMBOL_RE.findall(data.genomic_info)))
        # Also absorb any genes the NLP engine found in clinical notes
        if nlp_summary and nlp_summary.get("genes"):
            for g in nlp_summary["genes"]:
                if g not in extracted_genes:
                    extracted_genes.append(g)
        logger.info(f"Genes resolved from genomic_info + NLP: {extracted_genes}")
        # ─────────────────────────────────────────────────────────────────────

        # 1. Pipeline Stage: Clinical Rule Analysis & Pattern Recognition
        # Provides immediate rule-based insights
        patterns = apply_clinical_rules(data)
        
        # Build enriched patient dict with structured genes list for downstream scoring
        patient_dict = {**data.dict(), "genes": extracted_genes}

        # 2. Pipeline Stage: Hybrid Retrieval & Clinical Reranking
        # This now includes HPO mapping, Query Building, and Stage 2 Reranking internally
        diagnostic_candidates = await hybrid_retriever.retrieve_async(patient_dict)

        
        if not diagnostic_candidates:
            logger.warning("No diagnostic candidates found for the patient profile.")
        
        # 3. Pipeline Stage: Specialist Recommendations
        recommendations = get_recommendations(data, patterns)
        
        # 4. Pipeline Stage: BioMistral Literature Compression
        biomistral_summary = await biomistral_engine.summarize_evidence(diagnostic_candidates)
        biomistral_summary["clinical_findings"].extend(recommendations)

        # 4b. Fetch condition-specific genes from HPO for the top 3 candidates (parallel)
        #     These are passed into the metadata so the Genomic Insights card shows
        #     genes associated with the actual conditions being reasoned about.
        top_conditions = [c.get("disease", "") for c in diagnostic_candidates[:3] if c.get("disease")]
        gene_lists = await asyncio.gather(
            *[disease_api_client.get_disease_genes(d) for d in top_conditions]
        )
        # Flatten + deduplicate HPO condition genes, preserving order
        condition_genes: list = []
        for genes in gene_lists:
            for g in genes:
                if g and g not in condition_genes:
                    condition_genes.append(g)
        # Merge with any BioMistral-extracted genes
        for g in biomistral_summary.get("genes", []):
            if g and g not in condition_genes:
                condition_genes.append(g)

        # ── Patient-supplied genes take priority in the Genomic Insights card ──
        # Prepend extracted_genes so COL5A1/COL5A2 (entered by clinician) always
        # appear first, before the HPO condition-inferred genes.
        merged_genes: list = list(extracted_genes)  # start with patient-supplied
        for g in condition_genes:
            if g not in merged_genes:
                merged_genes.append(g)
        biomistral_summary["genes"] = merged_genes
        logger.info(f"Final merged genes (patient + condition): {merged_genes[:12]}")

        # ── Drug Interaction Checks ───────────────────────────────────────────
        medications = nlp_summary.get("medications", []) if nlp_summary else []
        top_diseases = [d.get("disease", "") for d in diagnostic_candidates[:3]]
        drug_alerts = check_drug_interactions(medications, top_diseases)
        if drug_alerts:
            logger.warning(f"Found {len(drug_alerts)} drug interaction alerts for patient.")

        # 5. Pipeline Stage: Streaming AI Reasoning (DeepSeek R1)
        # Emits metadata immediately followed by a continuous reasoning stream
        return StreamingResponse(
            stream_clinical_reasoning(
                patient_data=patient_dict,           # ← enriched dict with genes list
                hpo_terms=data.symptoms,
                reranked_evidence=diagnostic_candidates,
                biomistral_summary=biomistral_summary,
                patterns_data=patterns,
                recommendations=recommendations,
                nlp_summary=nlp_summary,
                drug_alerts=drug_alerts
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive"
            }
        )
        
    except ValueError as ve:
        logger.error(f"Validation Error in diagnostic pipeline: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Critical System Error in diagnostic pipeline: {e}")
        # Return a structured 500 error if the pipeline fails before streaming starts
        raise HTTPException(status_code=500, detail="Internal Clinical Engine Error")

@router.post("/ingest")
async def trigger_ingestion():
    """Trigger the manual evidence ingestion pipeline."""
    try:
        await ingestion_manager.ingest_pipeline()
        stats = get_knowledge_base_stats()
        return {"status": "success", "message": "Knowledge base updated successfully!", "stats": stats}
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
