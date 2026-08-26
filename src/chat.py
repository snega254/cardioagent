"""
Chatbot follow-up Q&A for CardioAgent.

Every question is answered using ONLY the structured data already
produced by the pipeline for the current analysis (prediction, heart
rate, Grad-CAM region, RAG evidence) plus the conversation so far — never
new claims about the ECG that the pipeline didn't actually produce.

IMPORTANT LIMITATION (stated plainly): requires a real Gemini API key and
network access; could not be tested against the real API from the
sandbox this was written in (no network route to Google's API). The
context-building and history-formatting logic were verified directly.
"""
from respond import CLASS_FULL_NAMES, SAFETY_RULES, generate_llm_response


def build_context_block(analysis, ecg_record):
    pred = analysis.get("prediction")
    pred_full = CLASS_FULL_NAMES.get(pred, pred) if pred else "Not available"
    conf = analysis.get("confidence")
    features = analysis.get("features") or {}
    xai = analysis.get("xai") or {}
    rag_sources = analysis.get("rag_sources") or []

    evidence_lines = "\n".join(
        f"- [{s.get('source')}] {s.get('text')}" for s in rag_sources
    ) or "None retrieved."

    return f"""
CURRENT ECG ANALYSIS CONTEXT (this is the only source of truth about this
recording — do not add findings that aren't listed here):

Predicted category: {pred_full}
Calibrated model confidence: {f"{conf*100:.1f}%" if conf is not None else "Not available"}
Heart rate: {f"{features.get('heart_rate'):.1f} bpm" if features.get("heart_rate") is not None else "Not reliably determined"}
R-peaks detected: {features.get("n_rpeaks", "Not available")}
Sampling rate: {ecg_record.get("sampling_rate", "Not available")} Hz
Number of leads: {ecg_record.get("n_leads", "Not available")}
Duration: {ecg_record.get("duration_sec", "Not available")} s
Grad-CAM important region: {f"{xai.get('region_start_sec'):.2f}s to {xai.get('region_end_sec'):.2f}s" if xai.get("region_start_sec") is not None else "Not available"}

Retrieved medical evidence:
{evidence_lines}
""".strip()


def build_chat_prompt(analysis, ecg_record, chat_history, user_question):
    context = build_context_block(analysis, ecg_record)

    history_text = ""
    if chat_history:
        turns = [f"{'User' if h['role'] == 'user' else 'CardioAgent'}: {h['text']}"
                 for h in chat_history]
        history_text = "\n\nPrevious conversation:\n" + "\n".join(turns)

    return f"""
You are CardioAgent, an AI-assisted ECG explanation assistant answering a
follow-up question about one specific ECG analysis.

{SAFETY_RULES}

{context}
{history_text}

The user's new question: {user_question}

Answer using only the context above and general medical knowledge about
what these categories/terms mean. If the question asks for something not
covered by the context (e.g. a specific waveform measurement that wasn't
extracted), say plainly that this wasn't measured by the system rather
than guessing. Match the depth of your answer to the question — keep
simple questions short, and go deeper only if asked to explain more.
""".strip()


def ask_chatbot(analysis, ecg_record, chat_history, user_question):
    prompt = build_chat_prompt(analysis, ecg_record, chat_history, user_question)
    try:
        return generate_llm_response(prompt)
    except Exception as e:
        raise RuntimeError(f"CardioAgent could not generate a response: {e}") from e


SUGGESTED_QUESTIONS = [
    "Why did you predict this?",
    "What does this finding mean?",
    "What part of the ECG influenced the model?",
    "Can you explain this in simple terms?",
    "What should I discuss with my doctor?",
    "What additional ECG information would be useful?",
]
