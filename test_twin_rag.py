import json
import requests

from backend.twin.similarity import upsert_twin_embedding, find_similar_patients


import uuid
suffix = uuid.uuid4().hex[:6]

twin_001 = {
    "patient_id": f"patient_001_{suffix}",
    "hpo_terms": [
        {"hpo_term_id": "HP:0000252", "hpo_label": "Microcephaly", "status": "active"},
        {"hpo_term_id": "HP:0001250", "hpo_label": "Seizures", "status": "active"},
        {"hpo_term_id": "HP:0001516", "hpo_label": "Global developmental delay", "status": "active"}
    ],
    "genetic_variants": [
        {"gene_symbol": "SCN1A", "hgvs_notation": "c.4313A>G", "classification": "Pathogenic"}
    ],
    "lab_results": [
        {"test_name": "EEG", "test_date": "2024-01-01", "interpretation": "abnormal"},
        {"test_name": "Blood lactate", "test_date": "2024-01-02", "interpretation": "normal"}
    ],
    "demographics": {
        "age_at_creation": 4,
        "gender": "female",
        "ethnicity": "Caucasian"
    },
    "family_history": [
        {"condition": "Epilepsy", "consanguinity": False}
    ],
    "clinical_notes": "Child with early-onset seizures, developmental delay, and microcephaly.",
    "confirmed_diagnoses": [],
    "diagnostic_status": "undiagnosed"
}

twin_002 = {
    "patient_id": f"patient_002_{suffix}",
    "hpo_terms": [
        {"hpo_term_id": "HP:0000252", "hpo_label": "Microcephaly", "status": "active"},
        {"hpo_term_id": "HP:0001250", "hpo_label": "Seizures", "status": "active"},
        {"hpo_term_id": "HP:0001263", "hpo_label": "Autistic behavior", "status": "active"}
    ],
    "genetic_variants": [
        {"gene_symbol": "SCN1A", "hgvs_notation": "c.1234C>T", "classification": "Likely_Pathogenic"}
    ],
    "lab_results": [
        {"test_name": "EEG", "test_date": "2024-01-01", "interpretation": "abnormal"}
    ],
    "demographics": {
        "age_at_creation": 5,
        "gender": "female",
        "ethnicity": "Hispanic"
    },
    "family_history": [
        {"condition": "Febrile seizures", "consanguinity": False}
    ],
    "clinical_notes": "Female child with refractory seizures and microcephaly.",
    "confirmed_diagnoses": ["Dravet syndrome"],
    "diagnostic_status": "confirmed"
}

twin_003 = {
    "patient_id": f"patient_003_{suffix}",
    "hpo_terms": [
        {"hpo_term_id": "HP:0002494", "hpo_label": "Hypermobile joints", "status": "active"},
        {"hpo_term_id": "HP:0001371", "hpo_label": "Skin hyperextensibility", "status": "active"}
    ],
    "genetic_variants": [
        {"gene_symbol": "COL5A1", "hgvs_notation": "c.3017G>A", "classification": "Pathogenic"}
    ],
    "lab_results": [],
    "demographics": {
        "age_at_creation": 16,
        "gender": "male",
        "ethnicity": "Asian"
    },
    "family_history": [],
    "clinical_notes": "Teen with connective tissue features and joint hypermobility.",
    "confirmed_diagnoses": ["Ehlers-Danlos syndrome"],
    "diagnostic_status": "confirmed"
}


def main():
    import time
    print("Creating twins via API (PostgreSQL + ChromaDB)...")
    
    # Create twins
    res1 = requests.post("http://127.0.0.1:8000/api/v1/twin/create", json=twin_001)
    if res1.status_code != 201:
        print("Failed to create twin 1:", res1.text)
        return
    twin_id_1 = res1.json()["twin_id"]
    
    requests.post("http://127.0.0.1:8000/api/v1/twin/create", json=twin_002)
    requests.post("http://127.0.0.1:8000/api/v1/twin/create", json=twin_003)
    
    print("Creation complete. Waiting briefly for ChromaDB indexing...")
    time.sleep(2)  # Give ChromaDB a moment

    print(f"\nFinding similar patients for {twin_id_1} via API:")
    res_similar = requests.get(f"http://127.0.0.1:8000/api/v1/twin/{twin_id_1}/similar?n=5&min_score=0.55")
    if res_similar.status_code == 200:
        print(json.dumps(res_similar.json(), indent=2))
    else:
        print("Error finding similar:", res_similar.status_code, res_similar.text)

    print(f"\nCalling twin RAG analyze endpoint for {twin_id_1}:")
    resp = requests.post(f"http://127.0.0.1:8000/api/v1/twin/{twin_id_1}/analyze")
    print("Status:", resp.status_code)
    try:
        print("Response JSON:", json.dumps(resp.json(), indent=2))
    except Exception:
        print("Response text:", resp.text[:800])

if __name__ == "__main__":
    main()
