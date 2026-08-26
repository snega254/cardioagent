"""
Multimodal Reasoning — Combines Signal, Report, and Patient Modalities.
"""

from typing import Optional, Dict, Any, List
import torch
import numpy as np
import os

from clinical_report import ECGReportData, PatientContext, clinical_report_pipeline
from preprocessing import SUPERCLASSES
from model import ECGConvNet
from gradcam import grad_cam_1d, top_attributed_region
from heart_rate import detect_heart_rate, extract_ecg_measurements
from rag import retrieve
from respond import compose_response, FRIENDLY_DESCRIPTIONS, CLASS_FULL_NAMES, generate_llm_response
from ecg_io import prepare_for_model


def multimodal_analysis(
    ecg_signal: Optional[np.ndarray] = None,
    fs: Optional[float] = None,
    ecg_report: Optional[ECGReportData] = None,
    patient: Optional[PatientContext] = None,
    model_checkpoint: str = "checkpoint.pt",
    rag_store_path: str = "vector_store.pkl"
) -> Dict[str, Any]:
    """
    Full multimodal analysis combining signal + report + patient context.
    
    Returns:
        {
            "signal_output": {...} or None,
            "report_output": {...} or None,
            "combined_reasoning": "...",
            "severity": "...",
            "guidelines": []
        }
    """
    
    result = {
        "signal_output": None,
        "report_output": None,
        "combined_reasoning": None,
        "severity": "routine review",
        "severity_evidence": [],
        "guidelines": [],
        "gradcam": None,
        "measurements": {},
        "triage": None,
        "modes_used": []
    }
    
    # === SIGNAL MODE (if raw ECG provided) ===
    if ecg_signal is not None and fs is not None:
        result["modes_used"].append("signal")
        try:
            # Load model
            if not os.path.exists(model_checkpoint):
                result["signal_output"] = {"error": f"Model checkpoint not found: {model_checkpoint}"}
            else:
                model = ECGConvNet()
                ckpt = torch.load(model_checkpoint, map_location="cpu")
                model.load_state_dict(ckpt["model_state"])
                model.log_temperature.data = ckpt["log_temperature"]
                model.eval()
                
                # Preprocess
                model_input, _ = prepare_for_model(ecg_signal, fs)
                x_tensor = torch.from_numpy(model_input).unsqueeze(0)
                
                # Predict
                with torch.no_grad():
                    logits = model(x_tensor)
                    probs = model.calibrated_probs(logits)[0]
                
                pred_idx = int(torch.argmax(probs).item())
                pred_class = SUPERCLASSES[pred_idx]
                confidence = float(probs[pred_idx].item())
                
                # Grad-CAM
                cam = grad_cam_1d(model, x_tensor, pred_idx)
                start_sec, end_sec, center_sec = top_attributed_region(cam, fs=100)
                
                # Heart rate
                hr_result = detect_heart_rate(ecg_signal[:, 0], fs)
                measurements = extract_ecg_measurements(ecg_signal[:, 0], fs)
                if hr_result.get("heart_rate"):
                    measurements["heart_rate"] = hr_result["heart_rate"]
                
                # RAG
                query = f"{pred_class} {CLASS_FULL_NAMES.get(pred_class, '')} ECG"
                passages = []
                try:
                    raw = retrieve(query, store_path=rag_store_path, top_k=3)
                    passages = [{"source": s, "text": t, "score": sc} for s, t, sc in raw]
                except:
                    pass
                
                # LLM explanation
                features = {
                    "heart_rate": measurements.get("heart_rate"),
                    "n_rpeaks": hr_result.get("n_rpeaks", 0),
                    "hr_reliable": hr_result.get("reliable", False)
                }
                xai = {
                    "region_start_sec": start_sec,
                    "region_end_sec": end_sec,
                    "center_sec": center_sec
                }
                
                try:
                    explanation, severity = compose_response(
                        pred_class, confidence, features, xai,
                        [(p["source"], p["text"], p["score"]) for p in passages],
                        patient.to_dict() if patient else None,
                        measurements
                    )
                except Exception as e:
                    explanation = f"LLM explanation unavailable: {e}"
                    severity = "prompt clinical review"
                
                result["signal_output"] = {
                    "prediction": pred_class,
                    "confidence": confidence,
                    "friendly_name": FRIENDLY_DESCRIPTIONS.get(pred_class, pred_class),
                    "gradcam_region": {"start_sec": start_sec, "end_sec": end_sec},
                    "explanation": explanation,
                    "features": features,
                    "measurements": measurements,
                    "rag_sources": passages
                }
                
                result["gradcam"] = cam
                result["measurements"] = measurements
                result["severity"] = severity
                result["guidelines"] = passages
                
        except Exception as e:
            result["signal_output"] = {"error": str(e)}
    
    # === REPORT MODE (if ECG report provided) ===
    if ecg_report is not None:
        result["modes_used"].append("report")
        try:
            if patient is None:
                patient = PatientContext()
            
            report_result = clinical_report_pipeline(ecg_report, patient, True, rag_store_path)
            result["report_output"] = report_result
            
            # Use report severity if signal not available
            if result["signal_output"] is None:
                result["severity"] = report_result.get("severity", {}).get("level", "routine review")
                result["severity_evidence"] = report_result.get("severity", {}).get("evidence", [])
            
            result["guidelines"] = report_result.get("guidelines_used", [])
            
        except Exception as e:
            result["report_output"] = {"error": str(e)}
    
    # === Combined Reasoning (if both modes available) ===
    if result["signal_output"] and result["report_output"]:
        try:
            signal_pred = result["signal_output"].get("friendly_name", "N/A")
            report_abnormalities = result["report_output"].get("ecg_data", {}).get("abnormalities", [])
            report_severity = result["report_output"].get("severity", {}).get("level", "N/A")
            
            combined_prompt = f"""
Combine these two analyses of the same patient:

SIGNAL ANALYSIS (from raw ECG):
- Prediction: {signal_pred}
- Heart Rate: {result['measurements'].get('heart_rate', 'N/A')} bpm
- Severity: {result['severity']}

REPORT ANALYSIS (from ECG measurements):
- Abnormalities: {', '.join(report_abnormalities) if report_abnormalities else 'None reported'}
- Severity: {report_severity}

PATIENT:
{patient.to_dict() if patient else 'Not provided'}

Generate a combined clinical summary that synthesizes both data sources.
Highlight any discrepancies between signal and report findings.
Provide a unified severity assessment.
Use cautious, evidence-grounded language.
"""
            combined = generate_llm_response(combined_prompt)
            result["combined_reasoning"] = combined
        except Exception as e:
            result["combined_reasoning"] = f"Combined reasoning unavailable: {e}"
    
    # === Emergency symptom warning ===
    if patient and patient.has_emergency_symptoms():
        result["emergency_warning"] = "⚠️ Emergency symptoms reported — seek immediate professional care"
    
    return result