"""
Clinical Triage Assessment — Evidence-grounded reasoning only.
No hardcoded deterministic rule engine.
"""

from respond import generate_llm_response


def generate_triage_assessment(patient, symptoms, vitals, ecg_findings, rag_guidelines):
    """
    Generate triage assessment using evidence-grounded LLM reasoning.
    
    Args:
        patient: {"age", "sex", "history", "medications": [...]}
        symptoms: {"chief_complaint", "onset", "pain_severity", "associated": [...]}
        vitals: {"bp", "hr", "spo2", "rr", "systolic_bp"}
        ecg_findings: {"predicted_class", "confidence", "heart_rate", "gradcam_leads", "gradcam_window"}
        rag_guidelines: [{"source": ..., "text": ...}, ...]
    
    Returns:
        (triage_text, None) where triage_text is the LLM-generated assessment
    """
    
    # Build guidelines text
    guideline_text = ""
    if rag_guidelines:
        for g in rag_guidelines[:3]:
            source = g.get('source', 'Unknown')
            text = g.get('text', '')[:500]
            guideline_text += f"\nSource: {source}\n{text}...\n"
    else:
        guideline_text = "No specific guidelines retrieved."
    
    # Build vitals text
    vitals_text = ""
    if vitals:
        vitals_items = []
        if vitals.get('bp'):
            vitals_items.append(f"BP: {vitals.get('bp')}")
        if vitals.get('hr'):
            vitals_items.append(f"HR: {vitals.get('hr')} bpm")
        if vitals.get('spo2'):
            vitals_items.append(f"SpO2: {vitals.get('spo2')}%")
        if vitals.get('rr'):
            vitals_items.append(f"RR: {vitals.get('rr')}")
        if vitals.get('systolic_bp'):
            vitals_items.append(f"Systolic BP: {vitals.get('systolic_bp')} mmHg")
        vitals_text = ", ".join(vitals_items) if vitals_items else "Not provided"
    
    # Build symptoms text
    symptoms_text = ""
    if symptoms:
        symptoms_parts = []
        if symptoms.get('chief_complaint'):
            symptoms_parts.append(f"Chief complaint: {symptoms.get('chief_complaint')}")
        if symptoms.get('onset'):
            symptoms_parts.append(f"Onset: {symptoms.get('onset')}")
        if symptoms.get('pain_severity'):
            symptoms_parts.append(f"Pain severity: {symptoms.get('pain_severity')}/10")
        if symptoms.get('associated'):
            symptoms_parts.append(f"Associated: {', '.join(symptoms.get('associated', []))}")
        symptoms_text = ", ".join(symptoms_parts) if symptoms_parts else "Not provided"
    
    prompt = f"""
You are CardioAgent, a clinical decision-support assistant.

CRITICAL RULES:
- You must not invent information.
- Use only the information provided below.
- If information is missing, explicitly state that it is missing.
- Do not present possible diagnoses as confirmed diagnoses.
- Do not claim clinical validation.
- Distinguish observed findings from model-derived findings.
- Every medication/disposition item is a RECOMMENDATION requiring clinician confirmation.

PATIENT:
- Age: {patient.get('age', 'unknown')}
- Sex: {patient.get('sex', 'unknown')}
- History: {patient.get('history', 'Not provided')}
- Current medications: {', '.join(patient.get('medications', [])) or 'None reported'}

PRESENTATION:
{symptoms_text}

VITALS:
{vitals_text}

ECG FINDINGS:
- Predicted class: {ecg_findings.get('predicted_class', 'Not available')}
- Confidence: {ecg_findings.get('confidence', 'Not available')}
- Heart rate: {ecg_findings.get('heart_rate', 'Not available')} bpm
- Grad-CAM region: {ecg_findings.get('gradcam_window', 'Not available')}
- Grad-CAM leads: {ecg_findings.get('gradcam_leads', 'Not available')}

RETRIEVED GUIDELINES:
{guideline_text}

Generate a structured triage assessment with these sections:

1. **Observations** - What the ECG and clinical information show. Be specific.

2. **Clinical Significance** - What these findings may mean, using cautious language.

3. **Risk / Triage Support** - Provide a supported level:
   - Routine review
   - Prompt clinical review
   - Urgent evaluation may be appropriate

4. **Red Flags** - List any concerning findings supported by available evidence.

5. **Possible Considerations** - Use "Possible considerations include..." Never present as confirmed diagnoses.

6. **Clinician Review Items** - What the clinician should specifically review.

7. **Missing Information** - What additional information would help.

8. **Explanation** - Detailed reasoning in understandable language.

9. **Disclaimer** - This is decision-support software and does not replace professional medical evaluation.

Keep responses clear, structured, and evidence-grounded.
Do not invent findings not present in the data.
Do not make definitive diagnoses.
"""

    try:
        response = generate_llm_response(prompt)
        return response, None  # No flags from rule engine
    except Exception as e:
        raise RuntimeError(f"Triage assessment generation failed: {e}")


# ============================================================
# Legacy compatibility functions (preserved for existing code)
# ============================================================

def check_contraindications(**kwargs):
    """
    Legacy function for compatibility with existing code.
    Returns empty TriageFlags-like object.
    """
    # Create a simple object with the expected attributes
    class Flags:
        def __init__(self):
            self.nitrate_contraindicated = False
            self.nitrate_reasons = []
            self.beta_blocker_caution = False
            self.beta_blocker_reasons = []
            self.hypotensive = False
            self.bradycardic = False
            self.tachycardic = False
            self.hypoxic = False
            self.inferior_mi_pattern = False
    
    return Flags()