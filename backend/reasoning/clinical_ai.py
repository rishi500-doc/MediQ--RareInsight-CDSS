import os
from openai import OpenAI
from dotenv import load_dotenv
from backend.models.schemas import PatientData

load_dotenv()

def stream_clinical_reasoning(patient_data: PatientData, merged_genes: list, top_conditions: list, recommendations: list):
    """
    Streams step-by-step reasoning from DeepSeek-R1 based on the retrieved RAG evidence.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        yield "⚠️ **Error:** `OPENROUTER_API_KEY` is not set in the `.env` file."
        return

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    prompt = f"""
    You are an expert clinical geneticist. You have been provided with a patient profile, their clinical symptoms, and a set of highly relevant medical literature and disease candidates retrieved from a vector database (RAG). 
    Based on this retrieved evidence, think step-by-step to provide a detailed clinical analysis, differential diagnosis, and recommend specific clinical actions.

    Patient Profile:
    - Age: {patient_data.age}
    - Gender: {patient_data.gender}
    - Family History: {patient_data.family_history}
    - Symptoms: {', '.join(patient_data.symptoms)}
    
    Retrieved Top Conditions (RAG):
    {', '.join(top_conditions) if top_conditions else 'None'}
    
    Involved Mutated Genes (RAG):
    {', '.join(merged_genes) if merged_genes else 'None'}
    
    System Recommendations:
    {', '.join(recommendations) if recommendations else 'None'}
    
    Format your output using Markdown, but DO NOT wrap your response in markdown code blocks (```markdown). Output the raw markdown text directly. Write a concise but highly clinical step-by-step analysis.
    
    IMPORTANT: You MUST include two clear Markdown headings at the bottom of your reasoning:
    1. "### 🧬 Mutated Genes Involved" (List the genes provided above)
    2. "### 🏥 Patient Encounter Recommendations" (List the system recommendations provided above and any clinical additions you suggest)
    """

    try:
        response = client.chat.completions.create(
            model="deepseek/deepseek-chat",
            messages=[
                {"role": "user", "content": prompt}
            ],
            stream=True,
            max_tokens=1000
        )

        for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                yield content
    except Exception as e:
        yield f"\n\n⚠️ **Error generating reasoning:** {str(e)}"
