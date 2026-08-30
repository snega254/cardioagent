"""
CardioAgent response generator with data-driven severity.
NO RULES - uses severity scorer.
"""

import os
import streamlit as st
from google import genai
from severity_scorer import SeverityScorer

CLASS_FULL_NAMES = {
    "NORM": "Normal ECG",
    "MI": "Myocardial Infarction (possible)",
    "STTC": "ST/T-Wave Change",
    "CD": "Conduction Disturbance",
    "HYP": "Hypertrophy (possible)",
}

FRIENDLY_DESCRIPTIONS = {
    "NORM": "a normal ECG pattern",
    "MI": "a pattern associated with possible myocardial infarction",
    "STTC": "an ST/T-wave-related pattern",
    "CD": "a conduction-related pattern",
    "HYP": "a pattern associated with possible hypertrophy",
}

GEMINI_MODEL = "gemini-3.6-flash"

SAFETY_RULES = """
IMPORTANT SAFETY RULES:
- The ECGConvNet model produced a pattern prediction. You did NOT directly analyze the raw ECG waveform.
- Do NOT claim you personally detected ECG waves, ST segments, QRS morphology, or other clinical findings 
  unless those findings are explicitly provided in the input below.
- Do NOT convert model confidence into a guaranteed diagnosis.
- Do NOT invent symptoms, patient history, measurements, or clinical findings.
- Do NOT claim Grad-CAM proves a specific ECG wave or interval is abnormal.
- Use the retrieved medical evidence as supporting context. If evidence is insufficient, say so.
- Do not recommend medication or treatment as an autonomous prescription.
- Do not tell the user they definitely have a disease.
- State plainly that this is a research prototype and professional clinical interpretation is required.
- If the user reports emergency symptoms (chest pain, shortness of breath, severe dizziness, etc.), 
  clearly instruct them to seek immediate emergency care.
""".strip()


def get_api_key():
    try:
        key = st.secrets.get("GEMINI_API_KEY")
        if key:
            return key
    except Exception:
        pass
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to .streamlit/secrets.toml"
        )
    return key


@st.cache_resource
def get_client():
    return genai.Client(api_key=get_api_key())


def get_severity(predicted_class, symptoms=None, heart_rate=None, 
                 qtc=None, qrs=None, patient_info=None, measurements=None):
    """
    Data-driven severity (NO RULES).
    Uses severity scorer with multiple factors.
    """
    scorer = SeverityScorer()
    
    ecg_prediction = {'prediction': predicted_class, 'confidence': 0.7}
    
    if patient_info is None:
        patient_info = {'age': 50, 'sex': 'Unknown', 'symptoms': symptoms or ''}
    
    if measurements is None:
        measurements = {
            'heart_rate': heart_rate,
            'qtc_interval': qtc,
            'qrs_duration': qrs
        }
    
    result = scorer.calculate(ecg_prediction, patient_info, measurements)
    return result['level']


def build_explanation_prompt(predicted_class, confidence, features, xai, 
                              retrieved_passages, severity, patient_info,
                              measurements, severity_score, severity_evidence):
    full_name = CLASS_FULL_NAMES.get(predicted_class, predicted_class)
    friendly = FRIENDLY_DESCRIPTIONS.get(predicted_class, "an ECG pattern")
    
    evidence_text = ""
    if retrieved_passages:
        evidence_text = "\n".join(
            f"- {text[:500]}..." for _, text, _ in retrieved_passages[:3]
        )
    else:
        evidence_text = "No specific evidence retrieved."
    
    hr = features.get("heart_rate", "Not available")
    n_rpeaks = features.get("n_rpeaks", "Not available")
    start_sec = xai.get("region_start_sec", 0)
    end_sec = xai.get("region_end_sec", 0)
    
    measurement_text = ""
    if measurements:
        measurement_text = f"""
Extended ECG Measurements:
- Heart Rate: {measurements.get('heart_rate', 'N/A')} bpm
- RR Variability: {measurements.get('hrv', 'N/A')}
- Rhythm Regularity: {measurements.get('rhythm_regularity', 'N/A')}
- PR Interval: {measurements.get('pr_interval', 'N/A')} ms
- QRS Duration: {measurements.get('qrs_duration', 'N/A')} ms
- QT Interval: {measurements.get('qt_interval', 'N/A')} ms
- QTc Interval: {measurements.get('qtc_interval', 'N/A')} ms
"""
    
    patient_text = ""
    if patient_info:
        patient_text = f"""
Patient Information:
- Age: {patient_info.get('age', 'Not provided')}
- Sex: {patient_info.get('sex', 'Not provided')}
- Symptoms: {patient_info.get('symptoms', 'Not provided')}
- Reason for ECG: {patient_info.get('reason', 'Not provided')}
- Clinical History: {patient_info.get('history', 'Not provided')}
"""
    
    evidence_list = "\n".join(f"- {e}" for e in severity_evidence[:5])
    
    return f"""
You are CardioAgent, an AI-powered explainable conversational ECG clinical decision-support assistant.

{SAFETY_RULES}

The system has analyzed an ECG recording and produced the following results:

PREDICTION: {full_name}
PATTERN: {friendly}
MODEL CONFIDENCE: {confidence*100:.1f}%
SEVERITY ASSESSMENT: {severity}
SEVERITY SCORE: {severity_score:.2f}

Contributing factors:
{evidence_list}

ECG FEATURES:
- Heart Rate: {hr} bpm
- R-peaks detected: {n_rpeaks}
- Grad-CAM important region: {start_sec:.2f}s to {end_sec:.2f}s

{measurement_text}

{patient_text}

MEDICAL EVIDENCE RETRIEVED:
{evidence_text}

Generate a clinical decision-support explanation using this structure:

## AI ECG Interpretation
[Brief, plain-language statement of what the ECG pattern appears to be]

## Severity / Clinical Priority
[Explain the severity assessment and why it was assigned]
- Severity Level: {severity}
- Severity Score: {severity_score:.2f}

## ECG Measurements
[Summarize available measurements with clinical context]

## Important Signal Region (Grad-CAM)
[Explain what the highlighted region indicates without overclaiming]

## Medical Context
[Use retrieved evidence to explain the pattern's clinical significance]

## Clinical Considerations
[Specific items for clinician review]

## What to Discuss with Your Doctor
[Safe, general guidance]

## Limitations
[Research prototype, requires clinical validation]

Keep it concise but useful. Use clear medical language appropriate for clinicians. 
Always distinguish between AI findings and clinical diagnosis.
""".strip()


def build_chatbot_prompt(analysis_context, chat_history, user_question):
    """Build prompt for conversational chatbot."""
    history_text = ""
    if chat_history:
        turns = [f"{'User' if h['role'] == 'user' else 'CardioAgent'}: {h['text']}"
                 for h in chat_history[-5:]]
        history_text = "\n\nPrevious conversation:\n" + "\n".join(turns)
    
    return f"""
You are CardioAgent, a conversational ECG clinical decision-support assistant.

{SAFETY_RULES}

Current analysis context:
{analysis_context}

{history_text}

User's question: {user_question}

Provide a helpful, clinically-oriented response. 
- If the question asks about something not in the context, say "This wasn't measured by the system."
- Keep responses concise but informative.
- Use bullet points for clarity when appropriate.
- Always include appropriate safety disclaimers.
""".strip()


def generate_llm_response(prompt):
    client = get_client()
    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    if response is None:
        raise RuntimeError("Gemini returned an empty response.")
    text = getattr(response, "text", None)
    if not text:
        raise RuntimeError("Gemini did not return any text.")
    return text.strip()


def compose_response(predicted_class, confidence, features, xai, retrieved_passages,
                      patient_info=None, measurements=None):
    """
    Compose response using data-driven severity (NO RULES).
    """
    scorer = SeverityScorer()
    
    ecg_prediction = {'prediction': predicted_class, 'confidence': confidence}
    
    # Build patient info
    if patient_info is None:
        patient_info = {}
    
    # Build measurements
    if measurements is None:
        measurements = {}
    
    # Calculate severity
    severity_result = scorer.calculate(ecg_prediction, patient_info, measurements)
    severity = severity_result['level']
    severity_score = severity_result['score']
    severity_evidence = severity_result['evidence']
    
    prompt = build_explanation_prompt(
        predicted_class, confidence, features, xai, retrieved_passages,
        severity, patient_info, measurements, severity_score, severity_evidence
    )
    
    try:
        response = generate_llm_response(prompt)
        return response, severity
    except Exception as e:
        raise RuntimeError(f"Explanation generation failed: {e}")


def generate_chat_response(analysis_context, chat_history, user_question):
    prompt = build_chatbot_prompt(analysis_context, chat_history, user_question)
    try:
        return generate_llm_response(prompt)
    except Exception as e:
        raise RuntimeError(f"Chat generation failed: {e}")