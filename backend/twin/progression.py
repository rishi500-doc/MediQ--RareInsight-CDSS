"""
Disease Progression Predictor — Phase 1 (Rule-Based MVP).

Uses clinical HPO co-occurrence rules derived from rare disease literature.
Phase 3 will replace/extend this with an XGBoost / Transformer ML model.
"""
from __future__ import annotations
from typing import List, Dict, Any
from backend.utils.common import get_logger

logger = get_logger("CDSS.Twin.Progression")

# ─── Rule Knowledge Base ──────────────────────────────────────────────────────
# Format: frozenset of trigger HPO IDs → predicted outcomes
#
# Each rule specifies:
#   predicted_hpo:         HPO terms likely to appear next
#   diagnosis_hint:        Probable diagnosis
#   urgency:               low | medium | high | critical
#   recommended_tests:     Investigations to order
#   timeframe_months:      Expected time window for predicted symptoms
#
# Sources: Orphanet natural history data + OMIM clinical synopses
# Expand this list as you observe patterns in your patient cohort.

PROGRESSION_RULES: Dict[frozenset, Dict[str, Any]] = {

    # Cystic Fibrosis — Respiratory + FTT pattern
    frozenset(["HP:0002099", "HP:0001508"]): {
        "predicted_hpo"     : ["HP:0012379", "HP:0006528", "HP:0001959"],
        "predicted_labels"  : ["Exocrine pancreatic insufficiency",
                               "Recurrent respiratory infections", "Malnutrition"],
        "diagnosis_hint"    : "Cystic Fibrosis",
        "urgency"           : "high",
        "recommended_tests" : ["Sweat Chloride Test", "CFTR Genetic Panel",
                               "Fecal Elastase", "Spirometry (FEV1)"],
        "timeframe_months"  : 6,
        "confidence"        : 0.85,
    },

    # Phenylketonuria — Intellectual disability + Hypopigmentation
    frozenset(["HP:0001249", "HP:0001010"]): {
        "predicted_hpo"     : ["HP:0001263", "HP:0000739"],
        "predicted_labels"  : ["Global developmental delay", "Anxiety"],
        "diagnosis_hint"    : "Phenylketonuria (PKU)",
        "urgency"           : "high",
        "recommended_tests" : ["Serum Phenylalanine", "Plasma Amino Acids",
                               "PAH Genetic Panel"],
        "timeframe_months"  : 3,
        "confidence"        : 0.80,
    },

    # Mitochondrial Disease — Developmental delay + Ophthalmoplegia
    frozenset(["HP:0001263", "HP:0000486"]): {
        "predicted_hpo"     : ["HP:0000505", "HP:0003128", "HP:0003011"],
        "predicted_labels"  : ["Visual impairment", "Lactic acidosis",
                               "Abnormality of skeletal muscles"],
        "diagnosis_hint"    : "Mitochondrial Disease",
        "urgency"           : "high",
        "recommended_tests" : ["Serum Lactate", "Muscle Biopsy (ETC complex)",
                               "Mitochondrial DNA Panel", "Brain MRI"],
        "timeframe_months"  : 12,
        "confidence"        : 0.75,
    },

    # Marfan Syndrome — Aortic dilation + Tall stature
    frozenset(["HP:0004942", "HP:0001519"]): {
        "predicted_hpo"     : ["HP:0002616", "HP:0007924", "HP:0001083"],
        "predicted_labels"  : ["Aortic root aneurysm", "Subluxation of the lens",
                               "Ectopia lentis"],
        "diagnosis_hint"    : "Marfan Syndrome",
        "urgency"           : "critical",
        "recommended_tests" : ["Echocardiogram", "Slit Lamp Eye Exam",
                               "FBN1 Genetic Panel", "Aortic CT Angiography"],
        "timeframe_months"  : 6,
        "confidence"        : 0.88,
    },

    # Wilson Disease — Liver disease + Neurological symptoms
    frozenset(["HP:0001392", "HP:0001268"]): {
        "predicted_hpo"     : ["HP:0002315", "HP:0001337", "HP:0000533"],
        "predicted_labels"  : ["Headache", "Tremor", "Corneal opacity (Kayser-Fleischer)"],
        "diagnosis_hint"    : "Wilson Disease",
        "urgency"           : "high",
        "recommended_tests" : ["Serum Ceruloplasmin", "24h Urine Copper",
                               "Slit Lamp (K-F rings)", "ATP7B Genetic Panel",
                               "Liver Biopsy"],
        "timeframe_months"  : 12,
        "confidence"        : 0.82,
    },

    # Gaucher Disease — Hepatosplenomegaly + Bone pain
    frozenset(["HP:0001433", "HP:0002653"]): {
        "predicted_hpo"     : ["HP:0001903", "HP:0002240", "HP:0000939"],
        "predicted_labels"  : ["Anaemia", "Hepatomegaly", "Osteoporosis"],
        "diagnosis_hint"    : "Gaucher Disease",
        "urgency"           : "medium",
        "recommended_tests" : ["Beta-glucocerebrosidase enzyme assay",
                               "GBA1 Genetic Panel", "Bone Marrow Biopsy",
                               "Bone Density DEXA"],
        "timeframe_months"  : 12,
        "confidence"        : 0.78,
    },

    # Pompe Disease — Muscle weakness + Cardiomegaly (infantile)
    frozenset(["HP:0003325", "HP:0001640"]): {
        "predicted_hpo"     : ["HP:0002093", "HP:0001270", "HP:0003701"],
        "predicted_labels"  : ["Respiratory insufficiency", "Motor delay",
                               "Proximal muscle weakness"],
        "diagnosis_hint"    : "Pompe Disease (Glycogen Storage Disease Type II)",
        "urgency"           : "critical",
        "recommended_tests" : ["Acid Alpha-Glucosidase enzyme assay",
                               "GAA Genetic Panel", "Echocardiogram", "CK Level"],
        "timeframe_months"  : 3,
        "confidence"        : 0.87,
    },

    # Primary Ciliary Dyskinesia — Situs inversus + Recurrent infections
    frozenset(["HP:0001696", "HP:0006532"]): {
        "predicted_hpo"     : ["HP:0002110", "HP:0012182", "HP:0000405"],
        "predicted_labels"  : ["Bronchiectasis", "Male infertility",
                               "Conductive hearing impairment"],
        "diagnosis_hint"    : "Primary Ciliary Dyskinesia",
        "urgency"           : "medium",
        "recommended_tests" : ["Nasal Nitric Oxide", "Electron Microscopy of cilia",
                               "DNAI1/DNAI2 Genetic Panel",
                               "Semen Analysis (motility)"],
        "timeframe_months"  : 12,
        "confidence"        : 0.80,
    },
}

# Priority map for sorting urgency
_URGENCY_ORDER = {"critical": 3, "high": 2, "medium": 1, "low": 0}


# ─── Prediction Engine ────────────────────────────────────────────────────────

def predict_progression_rule_based(
    active_hpo_terms: List[Dict],
    horizon_months: int = 6,
) -> Dict[str, Any]:
    """
    Rule-based disease progression predictor (Phase 1 MVP).

    Checks the patient's active HPO terms against all defined rules.
    Returns a ranked list of predictions, each with predicted HPO terms,
    recommended investigations, and a risk level.

    Args:
        active_hpo_terms: List of HPO term dicts from twin_hpo_terms table
        horizon_months:   Prediction horizon in months

    Returns:
        Dict with predictions, risk_level, and recommended tests.
    """
    hpo_id_set = frozenset(
        h.get("hpo_term_id") or h.get("term_id", "")
        for h in active_hpo_terms
        if h.get("status", "active") == "active"
    )

    predictions: List[Dict] = []
    all_tests: List[str]   = []

    for trigger_combo, outcome in PROGRESSION_RULES.items():
        # Check if ALL trigger terms are present in patient's active phenotype
        if trigger_combo.issubset(hpo_id_set):
            # Only include if predicted timeframe is within horizon
            rule_months = outcome.get("timeframe_months", 12)
            if rule_months <= horizon_months:
                predictions.append({
                    "diagnosis_hint"   : outcome["diagnosis_hint"],
                    "predicted_hpo"    : outcome["predicted_hpo"],
                    "predicted_labels" : outcome.get("predicted_labels", []),
                    "confidence"       : outcome["confidence"],
                    "urgency"          : outcome["urgency"],
                    "recommended_tests": outcome["recommended_tests"],
                    "trigger_terms"    : list(trigger_combo),
                    "timeframe_months" : rule_months,
                    "method"           : "rule_based_v1",
                })
                all_tests.extend(outcome["recommended_tests"])

    # Sort by urgency then confidence
    predictions.sort(
        key=lambda x: (_URGENCY_ORDER.get(x["urgency"], 0), x["confidence"]),
        reverse=True,
    )

    # Overall risk level = highest urgency across all triggered rules
    overall_risk = "low"
    if predictions:
        overall_risk = predictions[0]["urgency"]

    # Deduplicate recommended tests preserving order
    seen_tests: set = set()
    unique_tests: List[str] = []
    for t in all_tests:
        if t not in seen_tests:
            seen_tests.add(t)
            unique_tests.append(t)

    logger.info(
        f"Progression prediction: {len(predictions)} rules triggered, "
        f"risk={overall_risk}, tests={len(unique_tests)}"
    )

    return {
        "predictions"          : predictions,
        "risk_level"           : overall_risk,
        "recommended_tests"    : unique_tests,
        "model_version"        : "rule_based_v1",
        "horizon_months"       : horizon_months,
        "active_hpo_count"     : len(hpo_id_set),
        "rules_evaluated"      : len(PROGRESSION_RULES),
        "rules_triggered"      : len(predictions),
    }


async def store_prediction(twin_id: str, prediction: dict) -> str:
    """Persist the prediction to PostgreSQL for audit trail."""
    from backend.db.database import get_db_pool
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            INSERT INTO twin_progression_predictions
                (twin_id, model_version, horizon_days, predicted_hpo,
                 risk_level, confidence, trigger_terms, raw_output)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb)
            RETURNING prediction_id
        """,
            twin_id,
            prediction.get("model_version", "rule_based_v1"),
            prediction.get("horizon_months", 6) * 30,
            [p["predicted_hpo"][0] for p in prediction.get("predictions", []) if p.get("predicted_hpo")],
            prediction.get("risk_level", "low"),
            max((p.get("confidence", 0) for p in prediction.get("predictions", [])), default=0.0),
            [t for p in prediction.get("predictions", []) for t in p.get("trigger_terms", [])],
            __import__("json").dumps(prediction),
        )
    return str(row["prediction_id"])
