"""
Generates a professional PDF report from real analysis data. Every value
placed in the PDF is passed in explicitly by the caller — this module
never invents or defaults a clinical value; if something wasn't computed
upstream, the caller must pass None and it renders as "Not available"
rather than being silently omitted or fabricated.
"""
import os

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (Paragraph, SimpleDocTemplate, Spacer, Table,
                                 TableStyle)

ACCENT = colors.HexColor("#2E6BA8")


def _val(x, suffix=""):
    if x is None:
        return "Not available"
    return f"{x}{suffix}"


def generate_pdf_report(output_path, patient, ecg_record, analysis, class_full_names):
    """
    patient: dict with name, age, sex, username
    ecg_record: dict with filename, source_type, sampling_rate, n_leads,
                duration_sec, upload_time
    analysis: dict with prediction, confidence, features (dict, may
              include heart_rate/n_rpeaks/hr_reliable), xai (dict with
              region_start_sec/region_end_sec), rag_sources (list of
              dicts with source/text/score), explanation
    """
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("CardioTitle", parent=styles["Title"], textColor=ACCENT)
    heading_style = ParagraphStyle("CardioHeading", parent=styles["Heading2"], textColor=ACCENT,
                                    spaceBefore=14, spaceAfter=6)
    body_style = styles["Normal"]

    doc = SimpleDocTemplate(output_path, pagesize=letter,
                             topMargin=0.7 * inch, bottomMargin=0.7 * inch)
    story = []

    story.append(Paragraph("CardioAgent ECG Analysis Report", title_style))
    story.append(Paragraph(f"Generated: {ecg_record.get('upload_time', 'N/A')}", body_style))
    story.append(Spacer(1, 12))

    # Patient info
    story.append(Paragraph("Patient Information", heading_style))
    patient_table = Table([
        ["Name", _val(patient.get("name"))],
        ["Age", _val(patient.get("age"))],
        ["Sex", _val(patient.get("sex"))],
        ["Username", _val(patient.get("username"))],
    ], colWidths=[1.8 * inch, 4 * inch])
    patient_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.grey),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(patient_table)

    # ECG recording info
    story.append(Paragraph("ECG Recording", heading_style))
    ecg_table = Table([
        ["Filename", _val(ecg_record.get("filename"))],
        ["Format", _val(ecg_record.get("source_type"))],
        ["Sampling rate", _val(ecg_record.get("sampling_rate"), " Hz")],
        ["Number of leads", _val(ecg_record.get("n_leads"))],
        ["Duration", _val(ecg_record.get("duration_sec"), " s")],
    ], colWidths=[1.8 * inch, 4 * inch])
    ecg_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.grey),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(ecg_table)

    # Features
    features = analysis.get("features") or {}
    story.append(Paragraph("Extracted ECG Features", heading_style))
    if features:
        rows = [[k.replace("_", " ").title(), _val(v)] for k, v in features.items()]
        feat_table = Table(rows, colWidths=[1.8 * inch, 4 * inch])
        feat_table.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("TEXTCOLOR", (0, 0), (0, -1), colors.grey),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(feat_table)
    else:
        story.append(Paragraph("No features available.", body_style))

    # Prediction
    story.append(Paragraph("Model Prediction", heading_style))
    pred = analysis.get("prediction")
    pred_full = class_full_names.get(pred, pred) if pred else "Not available"
    conf = analysis.get("confidence")
    conf_str = f"{conf*100:.1f}%" if conf is not None else "Not available"
    story.append(Paragraph(f"<b>Prediction:</b> {pred_full}", body_style))
    story.append(Paragraph(f"<b>Confidence:</b> {conf_str}", body_style))

    # XAI
    xai = analysis.get("xai") or {}
    story.append(Paragraph("Explainable AI (Grad-CAM)", heading_style))
    if xai.get("region_start_sec") is not None:
        story.append(Paragraph(
            f"The most influential region of the signal for this prediction "
            f"was approximately {xai['region_start_sec']:.2f}s to "
            f"{xai['region_end_sec']:.2f}s.", body_style))
    else:
        story.append(Paragraph("No XAI region available.", body_style))

    # RAG evidence
    story.append(Paragraph("Retrieved Medical Evidence", heading_style))
    rag_sources = analysis.get("rag_sources") or []
    if rag_sources:
        for src in rag_sources:
            story.append(Paragraph(
                f"<b>Source:</b> {src.get('source', 'unknown')} "
                f"(similarity: {src.get('score', 0):.3f})", body_style))
            story.append(Paragraph(src.get("text", ""), body_style))
            story.append(Spacer(1, 6))
    else:
        story.append(Paragraph("No evidence retrieved for this analysis.", body_style))

    # Explanation
    story.append(Paragraph("CardioAgent Explanation", heading_style))
    story.append(Paragraph(analysis.get("explanation") or "Not available.", body_style))

    # Disclaimer
    story.append(Spacer(1, 16))
    disclaimer_style = ParagraphStyle("Disclaimer", parent=body_style,
                                       textColor=colors.HexColor("#8a1f1f"),
                                       borderColor=colors.HexColor("#8a1f1f"),
                                       borderWidth=0.5, borderPadding=8)
    story.append(Paragraph(
        "<b>Disclaimer:</b> This is an AI-assisted research prototype output, "
        "not a confirmed medical diagnosis. It must not be used for treatment "
        "decisions. Please consult a licensed healthcare professional.",
        disclaimer_style))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    doc.build(story)
    return output_path
