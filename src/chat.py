"""
Chatbot follow-up Q&A for CardioAgent.
"""
from respond import generate_chat_response, FRIENDLY_DESCRIPTIONS

SUGGESTED_QUESTIONS = [
    "Why did the AI classify this ECG this way?",
    "What did the AI focus on?",
    "Is this finding concerning?",
    "Explain the ECG in simple language.",
    "What measurements support this finding?",
    "What should a doctor review?",
    "What medications may be considered?",
    "What are the important contraindications?",
    "Compare this ECG with my previous ECG.",
    "Explain the severity assessment.",
    "What additional clinical information is needed?",
]


def build_chat_context(analysis, ecg_record, measurements=None):
    """Build a concise context for the chatbot."""
    pred = analysis.get("prediction")
    friendly = FRIENDLY_DESCRIPTIONS.get(pred, "an ECG pattern") if pred else "an ECG pattern"
    severity = analysis.get("severity", "Not assessed")
    features = analysis.get("features") or {}
    xai = analysis.get("xai") or {}
    
    context = f"""
ECG Analysis Summary:
- Pattern: {friendly.capitalize()}
- Severity: {severity}
- Heart Rate: {features.get('heart_rate', 'N/A')} bpm
- R-peaks: {features.get('n_rpeaks', 'N/A')}
- Grad-CAM region: {xai.get('region_start_sec', 0):.2f}s to {xai.get('region_end_sec', 0):.2f}s
"""
    
    # Add extended measurements if available
    if features.get("pr_interval"):
        context += f"- PR Interval: {features.get('pr_interval')} ms\n"
    if features.get("qrs_duration"):
        context += f"- QRS Duration: {features.get('qrs_duration')} ms\n"
    if features.get("qtc_interval"):
        context += f"- QTc Interval: {features.get('qtc_interval')} ms\n"
    if features.get("rhythm_regularity"):
        context += f"- Rhythm: {features.get('rhythm_regularity')}\n"
    
    return context


def ask_chatbot(analysis, ecg_record, chat_history, user_question, measurements=None):
    context = build_chat_context(analysis, ecg_record, measurements)
    try:
        return generate_chat_response(context, chat_history, user_question)
    except Exception as e:
        raise RuntimeError(f"CardioAgent could not generate a response: {e}") from e