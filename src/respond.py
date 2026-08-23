"""
Final response composer.

Deliberately template-based rather than LLM-based: given 8GB RAM / tight
time constraints, and because a template cannot introduce claims beyond
what the model/attribution/retrieval steps already produced, this
guarantees no hallucinated medical content. State this explicitly as a
design decision in the paper, not an omission.

All non-ASCII punctuation avoided in printed output — Windows PowerShell's
default console encoding mangles characters like em-dashes.
"""

CLASS_FULL_NAMES = {
    "NORM": "Normal ECG",
    "MI": "Myocardial Infarction (possible)",
    "STTC": "ST/T-Wave Change",
    "CD": "Conduction Disturbance",
    "HYP": "Hypertrophy (possible)",
}


def compose_response(predicted_class, confidence, region_start_sec, region_end_sec,
                      retrieved_passages):
    full_name = CLASS_FULL_NAMES.get(predicted_class, predicted_class)

    evidence_lines = []
    for source, text, score in retrieved_passages:
        evidence_lines.append(f"  - [source: {source}, similarity: {score:.3f}] {text}")
    evidence_block = "\n".join(evidence_lines) if evidence_lines else "  (no relevant passage retrieved)"

    response = f"""
ECG Analysis:
The system analyzed the provided ECG recording and identified patterns most
consistent with the "{full_name}" category, based on the trained model's
output.

Prediction:
{full_name} (calibrated model confidence: {confidence*100:.1f}%)

Important ECG Region:
The model's Grad-CAM attribution indicates the most influential part of the
signal was approximately between second {region_start_sec:.2f} and
second {region_end_sec:.2f} of the recording.

Explanation:
The prediction above was most strongly driven by the highlighted time
region of the waveform. This does not by itself identify a specific
clinical feature (e.g., a specific wave or segment) - that mapping would
require further validation - but it tells you where in the signal the
model's decision came from.

Evidence (retrieved from knowledge base):
{evidence_block}

Confidence/Uncertainty:
Reported confidence has been calibrated via temperature scaling on a
held-out validation set, but has not been validated for reliability at
the individual-prediction level. Treat it as an approximate indicator,
not a precise probability.

Disclaimer:
This is an AI-assisted research prototype output, not a confirmed medical
diagnosis. It must not be used for treatment decisions. Please consult a
licensed healthcare professional for accurate diagnosis and care.
""".strip()

    return response
