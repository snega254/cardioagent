"""
Deterministic clinical threshold/contraindication checks for the triage
module. These are plain arithmetic comparisons against the rules stated
in the triage system prompt — computed in code, not left to the LLM to
notice on its own. The LLM receives these as pre-computed facts, not as
numbers it has to reason about from scratch.

This module makes no diagnosis and issues no orders — it only flags
threshold conditions for a human clinician to review. Every flag is
phrased as "review before X", never as an instruction to act.
"""
from dataclasses import dataclass, field


@dataclass
class TriageFlags:
    nitrate_contraindicated: bool = False
    nitrate_reasons: list = field(default_factory=list)
    beta_blocker_caution: bool = False
    beta_blocker_reasons: list = field(default_factory=list)
    hypotensive: bool = False
    bradycardic: bool = False
    tachycardic: bool = False
    hypoxic: bool = False
    inferior_mi_pattern: bool = False


def check_contraindications(systolic_bp=None, heart_rate=None, spo2=None,
                             current_medications=None, predicted_class=None,
                             gradcam_leads=None, hours_since_pde5_inhibitor=None):
    """
    All inputs are the real, measured/reported values — this function
    never invents a value. Any input left as None is simply not checked
    (not assumed to be normal, not assumed to be abnormal).
    """
    flags = TriageFlags()
    current_medications = current_medications or []
    gradcam_leads = gradcam_leads or []

    if systolic_bp is not None and systolic_bp < 90:
        flags.hypotensive = True

    if heart_rate is not None:
        if heart_rate < 50:
            flags.bradycardic = True
        elif heart_rate > 100:
            flags.tachycardic = True

    if spo2 is not None and spo2 < 90:
        flags.hypoxic = True

    inferior_leads = {"II", "III", "aVF"}
    predicted_class_upper = (predicted_class or "").upper()
    is_mi_prediction = ("MI" == predicted_class_upper or
                         "MYOCARDIAL INFARCTION" in predicted_class_upper or
                         predicted_class_upper.startswith("MI "))
    if gradcam_leads and inferior_leads.intersection(set(gradcam_leads)):
        if is_mi_prediction:
            flags.inferior_mi_pattern = True

    # Nitrate contraindication logic
    if flags.hypotensive:
        flags.nitrate_contraindicated = True
        flags.nitrate_reasons.append(f"Systolic BP {systolic_bp} mmHg is below the 90 mmHg threshold.")
    if flags.inferior_mi_pattern:
        flags.nitrate_contraindicated = True
        flags.nitrate_reasons.append(
            "Inferior-lead MI pattern detected — right ventricular infarction "
            "must be excluded (right-sided leads V3R/V4R) before considering "
            "any preload-reducing agent."
        )
    pde5_inhibitors = {"sildenafil", "tadalafil", "vardenafil", "avanafil"}
    for med in current_medications:
        med_lower = med.lower()
        if any(p in med_lower for p in pde5_inhibitors):
            if hours_since_pde5_inhibitor is None or hours_since_pde5_inhibitor <= 48:
                flags.nitrate_contraindicated = True
                window_text = (f"taken {hours_since_pde5_inhibitor}h ago"
                                if hours_since_pde5_inhibitor is not None
                                else "timing not specified — assume within window until confirmed")
                flags.nitrate_reasons.append(
                    f"PDE-5 inhibitor ({med}) {window_text}; within the 24-48h "
                    f"nitrate contraindication window."
                )

    # Beta-blocker caution logic
    if flags.hypotensive:
        flags.beta_blocker_caution = True
        flags.beta_blocker_reasons.append(f"Systolic BP {systolic_bp} mmHg is already hypotensive.")
    if flags.bradycardic:
        flags.beta_blocker_caution = True
        flags.beta_blocker_reasons.append(f"Heart rate {heart_rate} bpm is bradycardic.")
    if flags.tachycardic:
        flags.beta_blocker_caution = True
        flags.beta_blocker_reasons.append(
            f"Heart rate {heart_rate} bpm is tachycardic — per the stated triage rule "
            f"this is flagged for review, though tachycardia alone is not always a "
            f"beta-blocker contraindication in every clinical context."
        )

    return flags
