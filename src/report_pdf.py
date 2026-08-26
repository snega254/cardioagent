"""
Generates a human-readable clinical PDF report.
"""
import os
import re
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ACCENT = colors.HexColor("#2E6BA8")
WARNING = colors.HexColor("#CC0000")


def _val(x, suffix=""):
    if x is None or x == "None" or x == "":
        return "Not available"
    return f"{x}{suffix}"


def generate_pdf_report(output_path, patient, ecg_record, analysis, class_full_names):
    """Generate a human-readable clinical PDF report."""
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("CardioTitle", parent=styles["Title"], 
                                  textColor=ACCENT, fontSize=18, spaceAfter=6)
    subtitle_style = ParagraphStyle("CardioSubtitle", parent=styles["Heading2"],
                                     textColor=ACCENT, spaceBefore=12, spaceAfter=6)
    heading_style = ParagraphStyle("CardioHeading", parent=styles["Heading3"],
                                    textColor=ACCENT, spaceBefore=10, spaceAfter=4)
    body_style = styles["Normal"]
    
    doc = SimpleDocTemplate(output_path, pagesize=letter,
                             topMargin=0.7*inch, bottomMargin=0.7*inch)
    story = []
    
    # ---- Header ----
    story.append(Paragraph("❤️ CardioAgent", title_style))
    story.append(Paragraph("ECG Clinical Decision-Support Report", subtitle_style))
    story.append(Spacer(1, 10))
    
    # Generated time
    story.append(Paragraph(f"Report generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", 
                           body_style))
    story.append(Spacer(1, 12))
    
    # ---- Patient Information ----
    story.append(Paragraph("Patient Information", heading_style))
    patient_data = [
        ["Name", _val(patient.get("name"))],
        ["Age", _val(patient.get("age"))],
        ["Sex", _val(patient.get("sex"))],
        ["Email", _val(patient.get("email"))],
    ]
    patient_table = Table(patient_data, colWidths=[1.5*inch, 4*inch])
    patient_table.setStyle(TableStyle([
        ("FONTSIZE", (0,0), (-1,-1), 10),
        ("TEXTCOLOR", (0,0), (0,-1), colors.grey),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(patient_table)
    story.append(Spacer(1, 8))
    
    # ---- Mode Type ----
    mode_type = analysis.get("mode_type", "research")
    if mode_type == "report":
        story.append(Paragraph(f"<b>Analysis Mode:</b> Report / Parameter Analysis", body_style))
    else:
        story.append(Paragraph(f"<b>Analysis Mode:</b> Research / Signal Analysis", body_style))
    story.append(Spacer(1, 8))
    
    # ---- ===== CLINICAL REPORT MODE SECTIONS ===== ----
    if mode_type == "report":
        # Patient Context from report mode
        patient_context = analysis.get("patient_context", {})
        if patient_context:
            story.append(Paragraph("Patient Context", heading_style))
            context_data = [
                ["Age", _val(patient_context.get("age"))],
                ["Sex", _val(patient_context.get("sex"))],
                ["Symptoms", _val(patient_context.get("symptoms"))],
                ["History", _val(patient_context.get("history"))],
            ]
            if patient_context.get("vitals"):
                for k, v in patient_context["vitals"].items():
                    context_data.append([k.capitalize(), _val(v)])
            
            context_table = Table(context_data, colWidths=[1.5*inch, 4*inch])
            context_table.setStyle(TableStyle([
                ("FONTSIZE", (0,0), (-1,-1), 10),
                ("TEXTCOLOR", (0,0), (0,-1), colors.grey),
                ("BOTTOMPADDING", (0,0), (-1,-1), 4),
            ]))
            story.append(context_table)
            story.append(Spacer(1, 8))
        
        # ECG Measurements (from report)
        features = analysis.get("features") or {}
        if features:
            story.append(Paragraph("ECG Measurements (from report)", heading_style))
            report_measurements = []
            
            measurement_map = {
                'heart_rate': ('Heart Rate', ' bpm'),
                'pr_interval': ('PR Interval', ' ms'),
                'qrs_duration': ('QRS Duration', ' ms'),
                'qt_interval': ('QT Interval', ' ms'),
                'qtc_interval': ('QTc Interval', ' ms'),
                'rhythm': ('Rhythm', ''),
                'axis': ('Axis', ''),
                'st_segment': ('ST Segment', ''),
                't_wave': ('T Wave', ''),
                'bundle_branch': ('Bundle Branch', ''),
                'machine_interpretation': ('Machine Interpretation', ''),
            }
            
            for key, (label, suffix) in measurement_map.items():
                value = features.get(key)
                if value:
                    report_measurements.append([label, _val(value, suffix)])
            
            if report_measurements:
                meas_table = Table(report_measurements, colWidths=[1.8*inch, 4*inch])
                meas_table.setStyle(TableStyle([
                    ("FONTSIZE", (0,0), (-1,-1), 10),
                    ("TEXTCOLOR", (0,0), (0,-1), colors.grey),
                    ("BOTTOMPADDING", (0,0), (-1,-1), 4),
                ]))
                story.append(meas_table)
                story.append(Spacer(1, 8))
        
        # Clinical Reasoning
        clinical_reasoning = analysis.get("clinical_reasoning", {})
        if clinical_reasoning:
            story.append(Paragraph("Clinical Decision Support", heading_style))
            
            # Severity / Triage Support
            triage_support = clinical_reasoning.get('triage_support', 'Not available')
            story.append(Paragraph(f"<b>Severity / Triage Support:</b> {triage_support}", body_style))
            
            # Observations
            observations = clinical_reasoning.get('observations')
            if observations:
                story.append(Paragraph("<b>Observations:</b>", body_style))
                # Clean and split observations
                obs_clean = re.sub(r'\*\*', '', observations)
                for para in obs_clean.split('\n'):
                    if para.strip():
                        story.append(Paragraph(para.strip(), body_style))
            
            # Red Flags
            red_flags = clinical_reasoning.get('red_flags', [])
            if red_flags:
                story.append(Paragraph("<b>Red Flags:</b>", body_style))
                for flag in red_flags:
                    story.append(Paragraph(f"• {flag}", body_style))
            
            # Missing Information
            missing_info = clinical_reasoning.get('missing_information', '')
            if missing_info:
                story.append(Paragraph("<b>Missing Information:</b>", body_style))
                story.append(Paragraph(missing_info, body_style))
            
            story.append(Spacer(1, 8))
    
    # ---- ===== RESEARCH MODE SECTIONS ===== ----
    else:
        # ECG Summary (from signal)
        story.append(Paragraph("ECG Summary", heading_style))
        ecg_data = [
            ["Recording", _val(ecg_record.get("filename"))],
            ["Format", _val(ecg_record.get("source_type"))],
            ["Sampling rate", _val(ecg_record.get("sampling_rate"), " Hz")],
            ["Leads", _val(ecg_record.get("n_leads"))],
            ["Duration", _val(ecg_record.get("duration_sec"), " s")],
        ]
        ecg_table = Table(ecg_data, colWidths=[1.5*inch, 4*inch])
        ecg_table.setStyle(TableStyle([
            ("FONTSIZE", (0,0), (-1,-1), 10),
            ("TEXTCOLOR", (0,0), (0,-1), colors.grey),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ]))
        story.append(ecg_table)
        story.append(Spacer(1, 8))
        
        # AI Interpretation (from signal)
        story.append(Paragraph("AI ECG Interpretation", heading_style))
        pred = analysis.get("prediction")
        friendly = "Normal ECG pattern" if pred == "NORM" else "ECG pattern requiring clinical review"
        story.append(Paragraph(f"<b>Finding:</b> {friendly}", body_style))
        story.append(Spacer(1, 4))
        
        # ECG Measurements (from signal)
        features = analysis.get("features") or {}
        story.append(Paragraph("ECG Measurements", heading_style))
        measurements_data = [
            ["Heart Rate", _val(features.get("heart_rate"), " bpm")],
            ["R-peaks detected", _val(features.get("n_rpeaks"))],
            ["R-peaks reliable", "Yes" if features.get("hr_reliable") else "No"],
        ]
        measurements_table = Table(measurements_data, colWidths=[2*inch, 3.5*inch])
        measurements_table.setStyle(TableStyle([
            ("FONTSIZE", (0,0), (-1,-1), 10),
            ("TEXTCOLOR", (0,0), (0,-1), colors.grey),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ]))
        story.append(measurements_table)
        story.append(Spacer(1, 8))
        
        # Grad-CAM
        xai = analysis.get("xai") or {}
        story.append(Paragraph("Model Attribution (Grad-CAM)", heading_style))
        if xai.get("region_start_sec") is not None:
            story.append(Paragraph(
                f"The model placed greatest importance on the signal from approximately "
                f"{xai['region_start_sec']:.2f}s to {xai['region_end_sec']:.2f}s.",
                body_style
            ))
            story.append(Paragraph(
                "Grad-CAM shows which part of the ECG most influenced the model's prediction. "
                "It does not independently prove a specific clinical abnormality.",
                body_style
            ))
        else:
            story.append(Paragraph("Attribution information not available.", body_style))
        story.append(Spacer(1, 8))
    
    # ---- ===== SHARED SECTIONS ===== ----
    
    # Explanation (shared)
    explanation = analysis.get("explanation")
    if explanation:
        story.append(Paragraph("Detailed Explanation", heading_style))
        # Clean up markdown
        clean_explanation = re.sub(r'##\s*', '', explanation)
        clean_explanation = re.sub(r'\*\*', '', clean_explanation)
        
        for para in clean_explanation.split('\n\n'):
            if para.strip():
                story.append(Paragraph(para.strip(), body_style))
                story.append(Spacer(1, 4))
    story.append(Spacer(1, 8))
    
    # Medical Evidence / RAG Sources
    rag_sources = analysis.get("rag_sources") or []
    if rag_sources:
        story.append(Paragraph("Medical Context / Evidence", heading_style))
        for src in rag_sources[:2]:
            source_name = src.get('source', 'unknown')
            text = src.get('text', '')
            if len(text) > 300:
                text = text[:300] + "..."
            story.append(Paragraph(f"<b>Source:</b> {source_name}", body_style))
            story.append(Paragraph(text, body_style))
            story.append(Spacer(1, 6))
        story.append(Spacer(1, 4))
    
    # Clinical Considerations (shared)
    story.append(Paragraph("Clinical Considerations", heading_style))
    
    # Get from clinical reasoning if available, otherwise use defaults
    clinical_reasoning = analysis.get("clinical_reasoning", {})
    clinical_considerations = analysis.get("clinical_considerations", [])
    
    if clinical_reasoning.get('clinician_review'):
        story.append(Paragraph(clinical_reasoning.get('clinician_review'), body_style))
    elif clinical_considerations:
        for item in clinical_considerations:
            story.append(Paragraph(f"• {item}", body_style))
    else:
        story.append(Paragraph(
            "The following items are suggested for clinical review based on the ECG findings.",
            body_style
        ))
        story.append(Paragraph(
            "1. Review the original ECG waveform and Grad-CAM highlighted region.",
            body_style
        ))
        story.append(Paragraph(
            "2. Correlate ECG findings with patient symptoms and clinical history.",
            body_style
        ))
        story.append(Paragraph(
            "3. Consider additional diagnostic tests if clinically indicated.",
            body_style
        ))
        story.append(Paragraph(
            "4. Compare with previous ECGs if available.",
            body_style
        ))
    story.append(Spacer(1, 8))
    
    # Medication Considerations
    medications = analysis.get("medications", [])
    if medications:
        story.append(Paragraph("Medication Considerations (For Physician Review)", heading_style))
        for med in medications:
            if isinstance(med, dict):
                story.append(Paragraph(f"<b>{med.get('name', 'Medication')}:</b> {med.get('reason', '')}", body_style))
                if med.get('contraindications'):
                    story.append(Paragraph(f"Contraindications: {med.get('contraindications')}", body_style))
            else:
                story.append(Paragraph(f"• {med}", body_style))
            story.append(Spacer(1, 4))
        story.append(Spacer(1, 8))
    
    # Limitations
    story.append(Paragraph("Limitations", heading_style))
    story.append(Paragraph(
        "This ECG analysis is generated by an AI research prototype and should be "
        "interpreted by a qualified healthcare professional.",
        body_style
    ))
    story.append(Spacer(1, 4))
    
    # Severity (if available)
    severity = analysis.get("severity", "Not assessed")
    if severity != "Not assessed":
        severity_color = "green" if "Low" in severity or "routine" in severity else "orange" if "Moderate" in severity or "prompt" in severity else "red"
        story.append(Paragraph(f"<b>Severity Assessment:</b> <font color='{severity_color}'>{severity}</font>", body_style))
        story.append(Spacer(1, 4))
    
    # ---- Disclaimer ----
    story.append(Spacer(1, 8))
    disclaimer_style = ParagraphStyle("Disclaimer", parent=body_style,
                                       textColor=colors.HexColor("#8a1f1f"),
                                       borderColor=colors.HexColor("#8a1f1f"),
                                       borderWidth=0.5, borderPadding=8)
    story.append(Paragraph(
        "<b>⚠️ IMPORTANT DISCLAIMER:</b> This is an AI-assisted research prototype. "
        "It is not a confirmed medical diagnosis and must not be used alone for "
        "treatment decisions. Always consult a licensed healthcare professional.",
        disclaimer_style
    ))
    
    # Build PDF
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    doc.build(story)
    return output_path