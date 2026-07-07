import streamlit as st
import asyncio
from backend.models.schemas import PatientData
from backend.retriever.hybrid_retriever import HybridRetriever
import re
from backend.retriever.disease_api_client import disease_api_client
from backend.reasoning.clinical_ai import stream_clinical_reasoning

# Initialize backend components (cached so they only load once)
@st.cache_resource
def get_backend_components():
    return HybridRetriever()

hybrid_retriever = get_backend_components()

# Regex for standard HGNC gene symbol format: 1 uppercase letter + up to 7 alphanumeric chars
_GENE_SYMBOL_RE = re.compile(r'\b([A-Z][A-Z0-9]{1,7})\b')

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

def get_recommendations(data: PatientData, patterns: dict, merged_genes: list, top_conditions: list):
    recs = [
        "Baseline echocardiogram for syndromic screening"
    ]
    if data.family_history:
        recs.append("First-degree relative cascade screening recommended")
    if data.age is not None and data.age < 18:
        recs.append("Pediatric genetics consultation advised")
        
    # Department mapping based on symptoms and conditions
    symptoms_text = " ".join(data.symptoms).lower()
    conditions_text = " ".join(top_conditions).lower()
    combined_text = symptoms_text + " " + conditions_text
    
    departments = ["Medical Genetics"] # Always required for rare diseases
    
    if any(k in combined_text for k in ["aortic", "heart", "cardio", "aneurysm", "valve", "vascular"]):
        departments.append("Cardiology")
    if any(k in combined_text for k in ["lens", "eye", "vision", "blind", "ectopia", "retina"]):
        departments.append("Ophthalmology")
    if any(k in combined_text for k in ["stature", "bone", "scoliosis", "arachnodactyly", "contractural", "joint"]):
        departments.append("Orthopedics")
    if any(k in combined_text for k in ["neuro", "seizure", "brain", "cognitive", "developmental"]):
        departments.append("Neurology")
    if any(k in combined_text for k in ["lipodystrophy", "metabolic", "thyroid", "growth"]):
        departments.append("Endocrinology")
        
    recs.append(f"Recommended Clinic Consults: {', '.join(departments)}")
        
    # Dynamic diagnostic tests based on retrieved data
    if merged_genes:
        recs.append(f"Targeted genetic panel sequencing for: {', '.join(merged_genes)}")
    else:
        recs.append("Whole Exome Sequencing (WES) consultation")
        
    if top_conditions:
        recs.append(f"Clinical evaluation for suspected: {top_conditions[0]}")
        
    return recs

st.set_page_config(page_title="MediQ Rare Disease RAG", layout="wide")

st.title("🩺 MediQ Rare Disease RAG")
st.markdown("Retrieval-Augmented Generation system for identifying rare disease evidence.")

with st.sidebar:
    st.header("Patient Profile")
    age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)
    gender = st.selectbox("Gender", ["male", "female", "other"])
    family_history = st.checkbox("Positive Family History")
    
    st.header("Clinical Findings")
    symptoms_input = st.text_area("Symptoms (comma-separated)", "tall stature, lens dislocation, aortic aneurysm")
    genes_input = st.text_input("Genes / Genomic Info", "FBN1")
    
    analyze_btn = st.button("Retrieve Evidence", type="primary", use_container_width=True)

async def run_pipeline(patient_data: PatientData):
    # ── Gene Symbol Extraction ────────────────────────────────────────────
    extracted_genes = []
    if patient_data.genomic_info:
        extracted_genes = list(dict.fromkeys(_GENE_SYMBOL_RE.findall(patient_data.genomic_info)))
    
    # 1. Pipeline Stage: Clinical Rule Analysis & Pattern Recognition
    patterns = apply_clinical_rules(patient_data)
    patient_dict = {**patient_data.dict(), "genes": extracted_genes}
    
    # 2. Pipeline Stage: Hybrid Retrieval
    diagnostic_candidates = await hybrid_retriever.retrieve_async(patient_dict)
    
    # 4b. Fetch condition-specific genes for the top candidates
    top_conditions = [c.get("disease", "") for c in diagnostic_candidates[:3] if c.get("disease")]
    gene_lists = await asyncio.gather(
        *[disease_api_client.get_disease_genes(d) for d in top_conditions]
    )
    
    condition_genes = []
    for genes in gene_lists:
        for g in genes:
            if g and g not in condition_genes:
                condition_genes.append(g)
                
    merged_genes = list(extracted_genes)
    for g in condition_genes:
        if g not in merged_genes:
            merged_genes.append(g)

    # 3. Pipeline Stage: Specialist & Diagnostic Recommendations
    recommendations = get_recommendations(patient_data, patterns, merged_genes, top_conditions)

    return patient_dict, diagnostic_candidates, merged_genes, patterns, recommendations

if analyze_btn:
    # Prepare payload
    symptoms = [s.strip() for s in symptoms_input.split(",") if s.strip()]
    
    patient_data = PatientData(
        age=age,
        gender=gender,
        family_history=family_history,
        symptoms=symptoms,
        genomic_info=genes_input.strip() if genes_input.strip() else None
    )
    
    st.markdown("### Analysis Results")
    
    try:
        with st.spinner("Searching knowledge base..."):
            patient_dict, diagnostic_candidates, merged_genes, patterns, recommendations = asyncio.run(
                run_pipeline(patient_data)
            )
            
            if not diagnostic_candidates:
                st.warning("No diagnostic candidates found matching these symptoms.")
            else:
                st.markdown("#### Top Possible Conditions & Accuracy")
                for candidate in diagnostic_candidates[:5]:
                    disease = candidate.get("disease", "Unknown Disease")
                    score = candidate.get("confidence_score", candidate.get("final_score", 0.0))
                    score_val = min(max(float(score), 0.0), 1.0)
                    
                    # Custom compact HTML progress bar
                    st.markdown(f"""
                    <div style="display: flex; align-items: center; margin-bottom: 8px;">
                        <div style="width: 250px; font-weight: 600; font-size: 14px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{disease}</div>
                        <div style="flex-grow: 1; max-width: 300px; background-color: rgba(255, 255, 255, 0.1); border-radius: 4px; height: 8px; margin: 0 15px;">
                            <div style="width: {int(score_val * 100)}%; background-color: #4da6ff; height: 100%; border-radius: 4px;"></div>
                        </div>
                        <div style="font-size: 13px; color: #a0aab5; width: 50px;">{int(score_val * 100)}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                st.markdown("---")
                st.markdown("### 🤖 DeepSeek Clinical Analysis")
                
                top_conditions = [c.get("disease", "") for c in diagnostic_candidates[:3] if c.get("disease")]
                
                with st.container(border=True):
                    st.caption("✨ AI-generated diagnostic reasoning based on retrieved RAG evidence")
                    st.write_stream(stream_clinical_reasoning(patient_data, merged_genes, top_conditions, recommendations))
            
    except Exception as e:
        st.error(f"Error executing pipeline: {e}")
