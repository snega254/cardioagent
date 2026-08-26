"""
CardioAgent response generator.

Uses Gemini to turn CardioAgent's structured pipeline output (ECGConvNet
prediction, Grad-CAM region, heart rate/R-peaks, RAG evidence) into a
plain-language explanation.

The LLM does NOT diagnose the ECG. It only explains results that were
already produced upstream by the trained model, Grad-CAM, the heart-rate
algorithm, and RAG retrieval. The prompt explicitly forbids inventing
findings, and every value passed into it comes from the actual pipeline,
never hardcoded or guessed.

IMPORTANT LIMITATION (state this plainly, don't hide it): this module
requires a real Gemini API key and network access to Google's API, which
could not be tested in the environment this was written in (no network
route to generativelanguage.googleapis.com from that sandbox). The prompt
construction and error-handling logic were verified directly; the actual
API call was not.
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

# Short, friendly phrasing for the main UI headline — rephrasing of the
# same categories above, not a new claim. Kept separate from
# CLASS_FULL_NAMES so the technical name and the friendly phrasing can
# each be edited independently.
FRIENDLY_DESCRIPTIONS = {
    "NORM": "a normal ECG pattern",
    "MI": "a pattern associated with possible myocardial infarction",
    "STTC": "an ST/T-wave-related pattern",
    "CD": "a conduction-related pattern",
    "HYP": "a pattern associated with possible hypertrophy",
}

GEMINI_MODEL = "gemini-3.6-flash"


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
            "GEMINI_API_KEY is not set. Add it to .streamlit/secrets.toml:\n"
            'GEMINI_API_KEY = "your-api-key"'
        )
    return key


@st.cache_resource
def get_client():
    return genai.Client(api_key=get_api_key())


def _format_evidence(retrieved_passages):
    if not retrieved_passages:
        return "No relevant medical evidence was retrieved from the knowledge base."
    parts = []
    for i, (source, text, score) in enumerate(retrieved_passages, start=1):
        parts.append(f"Evidence {i}\nSource: {source}\nSimilarity score: {score:.3f}\n\n{text}")
    return "\n\n".join(parts)


SAFETY_RULES = """
IMPORTANT:
- The ECGConvNet model produced the prediction. You did NOT directly analyze the raw ECG waveform.
- Do NOT claim you personally detected ECG waves, ST segments, QRS morphology, P waves, or other
  clinical findings unless those findings are explicitly provided in the input below.
- Do NOT convert model confidence into a guaranteed diagnosis.
- Do NOT invent symptoms, patient history, measurements, or clinical findings.
- Do NOT claim Grad-CAM proves a specific ECG wave or interval is abnormal — Grad-CAM shows which
  part of the signal the model weighted most; it does not by itself identify a clinical feature.
- Use the retrieved medical evidence as supporting context. If evidence is insufficient, say so.
- Do not recommend medication or treatment.
- Do not tell the user they definitely have a disease.
- State plainly that this is a research prototype and professional clinical interpretation is required.
""".strip()


def build_prompt(predicted_class, confidence, region_start_sec, region_end_sec,
                  retrieved_passages, heart_rate=None, n_rpeaks=None):
    full_name = CLASS_FULL_NAMES.get(predicted_class, predicted_class)
    evidence_text = _format_evidence(retrieved_passages)
    hr_text = f"{heart_rate:.1f} beats per minute" if heart_rate is not None else "Not available"
    rpeak_text = str(n_rpeaks) if n_rpeaks is not None else "Not available"

    return f"""
You are the explanation assistant inside CardioAgent, an AI-assisted ECG research prototype.
Your job is to explain the output of an ECG machine-learning pipeline clearly and cautiously.

{SAFETY_RULES}

The system output is:

Predicted ECG category: {full_name}
Internal model class: {predicted_class}
Calibrated model confidence: {confidence*100:.1f}%
Heart rate: {hr_text}
Number of detected R-peaks: {rpeak_text}
Grad-CAM important region: approximately {region_start_sec:.2f}s to {region_end_sec:.2f}s

Medical evidence retrieved from the CardioAgent knowledge base:
{evidence_text}

Generate a clear explanation using this structure:
1. RESULT — briefly state what category the model predicted.
2. WHAT THE MODEL FOUND — explain the prediction and confidence; make clear confidence is not a
   confirmed probability of disease.
3. ECG MEASUREMENTS — mention available heart rate / R-peak info; do not invent measurements.
4. IMPORTANT SIGNAL REGION — explain that Grad-CAM identified this time region as influential; do
   not claim it corresponds to a specific wave/abnormality unless the evidence explicitly says so.
5. MEDICAL CONTEXT — use the retrieved evidence to explain what the predicted category generally means.
6. WHAT TO DO NEXT — safe, general guidance (review with a healthcare professional, compare with
   previous ECGs if appropriate, consider symptoms/history). No medication or treatment advice.
7. LIMITATIONS — this is an AI research prototype, predictions require validation, Grad-CAM shows
   attribution not diagnosis, RAG evidence is contextual, professional interpretation is required.

Keep it concise but useful. Use simple language suitable for a patient or project evaluator.
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


def compose_response(predicted_class, confidence, region_start_sec, region_end_sec,
                      retrieved_passages, heart_rate=None, n_rpeaks=None):
    prompt = build_prompt(predicted_class, confidence, region_start_sec, region_end_sec,
                           retrieved_passages, heart_rate, n_rpeaks)
    try:
        return generate_llm_response(prompt)
    except Exception as e:
        raise RuntimeError(f"Gemini explanation generation failed: {e}") from e
