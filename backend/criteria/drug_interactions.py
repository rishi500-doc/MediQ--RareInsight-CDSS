"""
Rare Disease CDSS - Drug Interaction Engine
Checks for contraindicated medications based on top predicted diseases.
"""

from typing import List, Dict, Any

INTERACTION_RULES = [
    {
        "diseases": ["marfan", "loeys-dietz", "ehlers-danlos syndrome, vascular type", "aortic aneurysm"],
        "drugs": ["lisinopril", "captopril", "enalapril", "ramipril", "ace inhibitor", "ace inhibitors"],
        "severity": "high",
        "message": "Patient is on an ACE inhibitor. For Marfan/Loeys-Dietz spectrum disorders, guidelines strongly recommend Losartan (ARB) or Beta-blockers (e.g., Atenolol) instead, to better reduce aortic root dilation.",
        "reference": "2022 ACC/AHA Guidelines for Aortic Disease"
    },
    {
        "diseases": ["ehlers-danlos", "ehlers danlos", "vascular eds"],
        "drugs": ["aspirin", "ibuprofen", "naproxen", "diclofenac", "nsaid", "nsaids", "warfarin", "heparin"],
        "severity": "medium",
        "message": "NSAIDs or blood thinners detected. Use with caution in Ehlers-Danlos due to tissue fragility, capillary fragility, and increased risk of severe bruising or spontaneous bleeding.",
        "reference": "EDS Clinical Management Guidelines"
    },
    {
        "diseases": ["ataxia-telangiectasia", "ataxia telangiectasia"],
        "drugs": ["bleomycin", "radiotherapy", "fluorouracil"],
        "severity": "high",
        "message": "Extreme radiosensitivity and sensitivity to radiomimetic drugs (like Bleomycin) in A-T. Can cause fatal tissue toxicity. Avoid these agents.",
        "reference": "A-T Clinical Guidelines"
    },
    {
        "diseases": ["cystic fibrosis"],
        "drugs": ["ibuprofen"],
        "severity": "low",
        "message": "High-dose ibuprofen is sometimes used in CF to slow lung function decline, but requires careful pharmacokinetic monitoring to avoid GI/renal toxicity.",
        "reference": "Cystic Fibrosis Foundation Guidelines"
    }
]

def check_drug_interactions(medications: List[str], top_diseases: List[str]) -> List[Dict[str, str]]:
    """
    Checks extracted medications against the top predicted diseases for known contraindications.
    """
    alerts = []
    if not medications or not top_diseases:
        return alerts
        
    meds_lower = [m.lower() for m in medications]
    diseases_lower = [d.lower() for d in top_diseases]
    
    for rule in INTERACTION_RULES:
        # Check if any rule disease matches any top disease
        disease_match = False
        for rd in rule["diseases"]:
            if any(rd in td for td in diseases_lower):
                disease_match = True
                break
                
        if disease_match:
            # Check if any rule drug is in patient medications
            for rdrug in rule["drugs"]:
                if any(rdrug in pm for pm in meds_lower):
                    alerts.append({
                        "severity": rule["severity"],
                        "message": rule["message"],
                        "reference": rule["reference"],
                        "trigger_drug": rdrug
                    })
                    break  # Only alert once per rule
                    
    return alerts
