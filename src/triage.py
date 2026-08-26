"""
Clinical triage assessment for CardioAgent.

HARD RULE, stated once here and enforced in the prompt itself, not left
as a convention: every medication, disposition, or treatment-adjacent
item this module produces is a RECOMMENDATION requiring a licensed
clinician's confirmation before any action is taken. This module never
frames output as an autonomous order, and the prompt explicitly forbids
the LLM from doing so regardless of how urgently a case reads.

Objective threshold/contraindication checks (BP, HR, SpO2, medication
timing) are computed deterministically in triage_rules.py, NOT left to
the LLM to notice — the LLM receives them as pre-computed facts to
narrate and ground in guidelines, not as numbers it has to reason about
from scratch under time pressure.

IMPORTANT LIMITATION (stated plainly): requires a real Gemini API key
and network access; the actual API call could not be tested from the
sandbox this was written in. Prompt construction and the deterministic
flag logic (triage_rules.py) were both verified directly.
"""
from respond import generate_llm_response
from triage_rules import check_contraindications

HUMAN_CONFIRMATION_RULE = """
NON-NEGOTIABLE RULE: You are a decision-support tool, not an autonomous
prescriber. Every medication, disposition, or treatment-adjacent item you
list is a RECOMMENDATION that requires a licensed clinician's confirmation
before anything is administered or acted on. Never phrase an item as an
order you are issuing. Always frame the checklist as "For Physician
Confirmation" — this applies even in Tier 1 / emergency cases. Urgency
changes the recommended timeline, never the requirement for human
confirmation.
""".strip()


def build_triage_prompt(patient, symptoms, vitals, ecg_findings, rag_guidelines, flags):
    contraindication_lines = []
    if flags.nitrate_contraindicated:
        contraindication_lines.append("NITRATES CONTRAINDICATED:\n  - " +
                                       "\n  - ".join(flags.nitrate_reasons))
    if flags.beta_blocker_caution:
        contraindication_lines.append("BETA-BLOCKER CAUTION:\n  - " +
                                       "\n  - ".join(flags.beta_blocker_reasons))
    contraindication_text = ("\n".join(contraindication_lines) if contraindication_lines
                              else "No threshold-based contraindications flagged by the "
                                   "deterministic safety check.")

    guideline_text = "\n\n".join(
        f"Source: {g.get('source')}\n{g.get('text')}" for g in rag_guidelines
    ) if rag_guidelines else "No specific guideline excerpts retrieved for this case."

    return f"""
You are CardioAgent, a clinical decision-support tool for emergency triage
and point-of-care cardiac assessment.

{HUMAN_CONFIRMATION_RULE}

Ground every action item strictly in the retrieved guideline excerpts
provided below. Do not invent a guideline citation. If the guidelines
don't cover something, say so rather than filling the gap.

PATIENT: Age {patient.get('age', 'unknown')}, Sex {patient.get('sex', 'unknown')}
History: {patient.get('history', 'Not provided')}
Current medications: {', '.join(patient.get('medications', [])) or 'None reported'}

PRESENTATION: {symptoms.get('chief_complaint', 'Not provided')}
Onset: {symptoms.get('onset', 'Not provided')}
Pain severity: {symptoms.get('pain_severity', 'Not provided')}
Associated symptoms: {', '.join(symptoms.get('associated', [])) or 'None reported'}

VITALS: BP {vitals.get('bp', 'Not provided')}, HR {vitals.get('hr', 'Not provided')} bpm,
SpO2 {vitals.get('spo2', 'Not provided')}%, RR {vitals.get('rr', 'Not provided')}

ECG MODEL FINDINGS:
Predicted class: {ecg_findings.get('predicted_class', 'Not available')}
Confidence: {f"{ecg_findings.get('confidence')*100:.1f}%" if ecg_findings.get('confidence') is not None else 'Not available'}
Heart rate: {ecg_findings.get('heart_rate', 'Not available')} bpm
Grad-CAM: leads {ecg_findings.get('gradcam_leads', 'Not available')}, window
{ecg_findings.get('gradcam_window', 'Not available')}

DETERMINISTICALLY-COMPUTED SAFETY FLAGS (pre-checked in code, not for you
to re-derive — narrate and ground these, don't second-guess the arithmetic):
{contraindication_text}

RETRIEVED GUIDELINE EXCERPTS:
{guideline_text}

Produce your response in exactly this structure:

### 1. TRIAGE SUMMARY
- Urgency Level: [pick one, with reasoning grounded in the findings above]
- Primary Clinical Finding
- Target Care Window

### 2. WAVEFORM AUDIT & EXPLAINABILITY
Explain the Grad-CAM attribution and interval findings in clinical terms.
State plainly what Grad-CAM does and doesn't prove.

### 3. RECOMMENDED BEDSIDE PROTOCOL (For Physician Confirmation)
Checklist format. Every item is a recommendation pending clinician sign-off.

### 4. SAFETY & CONTRAINDICATION ALERTS
Narrate the deterministic flags above in clinical context. Do not add new
contraindications not supported by the flags or the guidelines.

### 5. GUIDELINE GROUNDING & CITATIONS
Cite only the guideline excerpts actually provided above.
""".strip()


def generate_triage_assessment(patient, symptoms, vitals, ecg_findings, rag_guidelines):
    """
    patient: {"age", "sex", "history", "medications": [...]}
    symptoms: {"chief_complaint", "onset", "pain_severity", "associated": [...]}
    vitals: {"bp", "hr", "spo2", "rr", "systolic_bp"}
    ecg_findings: {"predicted_class", "confidence", "heart_rate",
                    "gradcam_leads": [...], "gradcam_window"}
    rag_guidelines: [{"source": ..., "text": ...}, ...]
    """
    flags = check_contraindications(
        systolic_bp=vitals.get("systolic_bp"),
        heart_rate=ecg_findings.get("heart_rate") or vitals.get("hr"),
        spo2=vitals.get("spo2"),
        current_medications=patient.get("medications", []),
        predicted_class=ecg_findings.get("predicted_class"),
        gradcam_leads=ecg_findings.get("gradcam_leads"),
        hours_since_pde5_inhibitor=patient.get("hours_since_pde5_inhibitor"),
    )

    prompt = build_triage_prompt(patient, symptoms, vitals, ecg_findings, rag_guidelines, flags)

    try:
        text = generate_llm_response(prompt)
        return text, flags
    except Exception as e:
        raise RuntimeError(f"Triage assessment generation failed: {e}") from e
