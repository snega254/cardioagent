"""
CardioAgent response generator with severity assessment and medication considerations.
"""
import os
import streamlit as st
from google import genai

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

SEVERITY_LEVELS = {
    "NORM": "Low concern",
    "STTC": "Moderate concern",
    "CD": "Moderate concern",
    "HYP": "Moderate concern",
    "MI": "High concern",
}

SEVERITY_DESCRIPTIONS = {
    "Low concern": "Routine clinical review is appropriate.",
    "Moderate concern": "Clinical review should be prioritized.",
    "High concern": "Prompt clinical evaluation is recommended.",
    "Urgent review": "Urgent clinical evaluation is strongly recommended.",
    "Emergency review": "Emergency evaluation is recommended. Seek immediate care.",
}

GEMINI_MODEL = "gemini-3.6-flash"

SAFETY_RULES = """
IMPORTANT SAFETY RULES:
- The ECGConvNet model produced a pattern prediction. You did NOT directly analyze the raw ECG waveform.
- Do NOT claim you personally detected ECG waves, ST segments, QRS morphology, or other clinical findings 
  unless those findings are explicitly provided in the input below.
- Do NOT convert model confidence into a guaranteed diagnosis.
- Do NOT invent symptoms, patient history, measurements, or clinical findings.
- Do NOT claim Grad-CAM proves a specific ECG wave or interval is abnormal — Grad-CAM shows which
  part of the signal the model weighted most; it does not by itself identify a clinical feature.
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


def get_severity(predicted_class, symptoms=None, heart_rate=None):
    """Determine severity based on prediction and available clinical context."""
    base_severity = SEVERITY_LEVELS.get(predicted_class, "Moderate concern")
    
    # Adjust severity based on symptoms
    if symptoms:
        symptom_text = symptoms.lower() if isinstance(symptoms, str) else ""
        emergency_symptoms = ["chest pain", "shortness of breath", "severe dizziness", 
                             "fainting", "loss of consciousness", "palpitations", "sweating"]
        for es in emergency_symptoms:
            if es in symptom_text:
                if base_severity == "Low concern":
                    return "Moderate concern"
                elif base_severity == "Moderate concern":
                    return "High concern"
                elif base_severity == "High concern":
                    return "Urgent review"
    
    # Adjust based on heart rate extremes
    if heart_rate:
        if heart_rate > 150 or heart_rate < 30:
            if base_severity != "Low concern":
                return "Urgent review"
    
    return base_severity


def build_explanation_prompt(predicted_class, confidence, features, xai, 
                              retrieved_passages, severity, patient_info,
                              measurements):
    full_name = CLASS_FULL_NAMES.get(predicted_class, predicted_class)
    friendly = FRIENDLY_DESCRIPTIONS.get(predicted_class, "an ECG pattern")
    
    severity_desc = SEVERITY_DESCRIPTIONS.get(severity, "")
    
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
    
    # Include extended measurements if available
    measurement_text = ""
    if measurements:
        measurement_text = f"""
Extended ECG Measurements:
- Heart Rate: {measurements.get('heart_rate', 'N/A')} bpm
- RR Variability: {measurements.get('rr_variability', 'N/A')}
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
    
    return f"""
You are CardioAgent, an AI-powered explainable conversational ECG clinical decision-support assistant.

{SAFETY_RULES}

The system has analyzed an ECG recording and produced the following results:

PREDICTION: {full_name}
PATTERN: {friendly}
MODEL CONFIDENCE: {confidence*100:.1f}%
SEVERITY ASSESSMENT: {severity}
{severity_desc}

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

## ECG Measurements
[Summarize available measurements with clinical context]

## Important Signal Region (Grad-CAM)
[Explain what the highlighted region indicates without overclaiming]

## Medical Context
[Use retrieved evidence to explain the pattern's clinical significance]

## Clinical Considerations
[Specific items for clinician review]

## Medication Considerations (For Physician Review)
[Only if clinically indicated and supported by evidence. MUST state "Requires physician confirmation" for each item]

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
                 for h in chat_history[-5:]]  # Keep last 5 turns for context
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
    severity = get_severity(
        predicted_class,
        patient_info.get("symptoms") if patient_info else None,
        features.get("heart_rate")
    )
    
    prompt = build_explanation_prompt(
        predicted_class, confidence, features, xai, retrieved_passages,
        severity, patient_info, measurements
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


# ============================================================
# NEW: Clinical Reasoning Functions for Report Mode
# ============================================================

def build_clinical_reasoning_prompt(ecg_data, patient, guidelines, severity_level):
    """
    Build prompt for clinical reasoning from report data.
    Used by clinical_report.py for evidence-grounded reasoning.
    """
    # Build measurement summary
    measurements = []
    
    # Handle both dict and object inputs
    if hasattr(ecg_data, 'to_dict'):
        ecg_dict = ecg_data.to_dict()
    elif isinstance(ecg_data, dict):
        ecg_dict = ecg_data
    else:
        ecg_dict = {}
    
    # Build measurement list from available data
    measurement_fields = {
        'heart_rate': 'Heart Rate',
        'pr_interval': 'PR Interval',
        'qrs_duration': 'QRS Duration',
        'qt_interval': 'QT Interval',
        'qtc_interval': 'QTc Interval',
        'rhythm': 'Rhythm',
        'axis': 'Axis',
        'st_segment': 'ST Segment',
        't_wave': 'T Wave',
        'p_wave': 'P Wave',
        'bundle_branch': 'Bundle Branch',
        'machine_interpretation': 'Machine Interpretation'
    }
    
    for key, label in measurement_fields.items():
        value = ecg_dict.get(key)
        if value:
            # Format units
            if key in ['heart_rate']:
                measurements.append(f"{label}: {value} bpm")
            elif key in ['pr_interval', 'qrs_duration', 'qt_interval', 'qtc_interval']:
                measurements.append(f"{label}: {value} ms")
            else:
                measurements.append(f"{label}: {value}")
    
    # Add abnormalities
    abnormalities = ecg_dict.get('abnormalities', [])
    if abnormalities:
        measurements.append(f"Abnormalities: {', '.join(abnormalities)}")
    
    measurement_text = "\n".join(measurements) if measurements else "No measurements provided."
    
    # Build patient context
    patient_text = ""
    if patient:
        if hasattr(patient, 'age') and patient.age:
            patient_text += f"Age: {patient.age}\n"
        if hasattr(patient, 'sex') and patient.sex:
            patient_text += f"Sex: {patient.sex}\n"
        if hasattr(patient, 'symptoms') and patient.symptoms:
            patient_text += f"Symptoms: {patient.symptoms}\n"
        if hasattr(patient, 'history') and patient.history:
            patient_text += f"History: {patient.history}\n"
        if hasattr(patient, 'vitals') and patient.vitals:
            vitals_text = ", ".join(f"{k}: {v}" for k, v in patient.vitals.items())
            patient_text += f"Vitals: {vitals_text}\n"
    
    # Build guidelines
    guideline_text = ""
    if guidelines:
        for g in guidelines[:3]:
            source = g.get('source', 'Unknown')
            text = g.get('text', '')[:500]
            guideline_text += f"\n- Source: {source}\n  {text}...\n"
    else:
        guideline_text = "No specific guidelines retrieved."
    
    # Build severity text
    if isinstance(severity_level, dict):
        severity_text = severity_level.get('level', 'routine review')
        severity_evidence = "\n".join(f"- {e}" for e in severity_level.get('evidence', []))
    else:
        severity_text = str(severity_level)
        severity_evidence = "Based on available measurements."
    
    return f"""
You are CardioAgent, an explainable multimodal clinical decision-support assistant.

CRITICAL RULES:
- You must not invent information.
- Use only the information provided below.
- If information is missing, explicitly state that it is missing.
- Do not present possible diagnoses as confirmed diagnoses.
- Do not claim clinical validation.
- Distinguish observed findings from model-derived findings.

=== ECG MEASUREMENTS ===
{measurement_text}

=== PATIENT CONTEXT ===
{patient_text}

=== SEVERITY ASSESSMENT ===
Level: {severity_text}

Supporting evidence:
{severity_evidence}

=== RETRIEVED GUIDELINES ===
{guideline_text}

=== INSTRUCTIONS ===
Generate a structured clinical decision-support response with these sections:

1. **ECG Observations** - Summarize the key ECG findings from the available information.

2. **Clinical Significance** - Explain what these findings may mean, in clinical terms. Use cautious language.

3. **Risk / Triage Support** - Based on available evidence, provide a supported level such as:
   - Routine review
   - Prompt clinical review
   - Urgent evaluation may be appropriate

4. **Red Flags** - List any concerning findings supported by the available evidence.

5. **Possible Considerations** - Use "Possible considerations include..." Never present as confirmed diagnoses.

6. **Clinician Review** - What the clinician should specifically review. Mention missing information.

7. **Questions / Missing Information** - What additional information would materially improve interpretation.

8. **Explanation** - Detailed explanation of the reasoning in understandable language.

9. **Evidence / Sources** - Identify the RAG evidence used where available.

10. **Disclaimer** - This is decision-support software and does not replace professional medical evaluation.

Keep responses clear, structured, and evidence-grounded.
Do not invent findings not present in the data.
Do not make definitive diagnoses.
"""


def generate_clinical_response(ecg_data, patient, guidelines, severity_level):
    """
    Generate clinical response using the clinical reasoning prompt.
    
    Args:
        ecg_data: ECGReportData object or dict with measurements
        patient: PatientContext object or dict with patient info
        guidelines: List of retrieved guidelines
        severity_level: String or dict with severity assessment
    
    Returns:
        String containing the clinical reasoning response
    """
    prompt = build_clinical_reasoning_prompt(ecg_data, patient, guidelines, severity_level)
    try:
        return generate_llm_response(prompt)
    except Exception as e:
        raise RuntimeError(f"Clinical reasoning generation failed: {e}")