"""
Clinical Report Mode — ECG Report + Symptoms + Patient Context
No raw ECG signal required. Uses measurements and patient information.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import re
from severity_scorer import SeverityScorer


@dataclass
class PatientContext:
    """Patient information entered by user."""
    age: Optional[int] = None
    sex: Optional[str] = None
    symptoms: Optional[str] = None
    history: Optional[str] = None
    vitals: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "age": self.age,
            "sex": self.sex,
            "symptoms": self.symptoms,
            "history": self.history,
            "vitals": self.vitals or {}
        }
    
    def has_emergency_symptoms(self) -> bool:
        if not self.symptoms:
            return False
        emergency = [
            "chest pain", "chest discomfort", "shortness of breath",
            "severe dizziness", "fainting", "loss of consciousness",
            "palpitations", "sweating", "nausea with chest pain"
        ]
        symptom_lower = self.symptoms.lower()
        return any(es in symptom_lower for es in emergency)
    
    def get_urgency_indicator(self) -> str:
        if self.has_emergency_symptoms():
            return "Warning: Emergency symptoms reported — seek immediate professional care"
        return ""


@dataclass
class ECGReportData:
    """ECG measurements from report."""
    heart_rate: Optional[float] = None
    pr_interval: Optional[float] = None
    qrs_duration: Optional[float] = None
    qt_interval: Optional[float] = None
    qtc_interval: Optional[float] = None
    rhythm: Optional[str] = None
    axis: Optional[str] = None
    st_segment: Optional[str] = None
    t_wave: Optional[str] = None
    p_wave: Optional[str] = None
    q_waves: Optional[str] = None
    bundle_branch: Optional[str] = None
    machine_interpretation: Optional[str] = None
    abnormalities: Optional[List[str]] = None
    raw_report_text: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "heart_rate": self.heart_rate,
            "pr_interval": self.pr_interval,
            "qrs_duration": self.qrs_duration,
            "qt_interval": self.qt_interval,
            "qtc_interval": self.qtc_interval,
            "rhythm": self.rhythm,
            "axis": self.axis,
            "st_segment": self.st_segment,
            "t_wave": self.t_wave,
            "p_wave": self.p_wave,
            "q_waves": self.q_waves,
            "bundle_branch": self.bundle_branch,
            "machine_interpretation": self.machine_interpretation,
            "abnormalities": self.abnormalities or [],
            "raw_report_text": self.raw_report_text
        }
    
    def has_data(self) -> bool:
        return any([
            self.heart_rate, self.pr_interval, self.qrs_duration,
            self.qt_interval, self.qtc_interval, self.rhythm,
            self.st_segment, self.t_wave, self.machine_interpretation,
            self.abnormalities, self.raw_report_text
        ])


def parse_report_text(text: str) -> ECGReportData:
    """Parse raw ECG report text into structured ECGReportData."""
    data = ECGReportData(raw_report_text=text)
    
    if not text or not text.strip():
        return data
    
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    
    # Heart Rate
    hr_patterns = [
        r'(?:HR|Heart Rate|Heart rate)[:\s]+(\d+)',
        r'(\d+)\s*(?:bpm|BPM)',
        r'heart rate[:\s]+(\d+)',
        r'HR\s*=\s*(\d+)'
    ]
    for pattern in hr_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            data.heart_rate = float(match.group(1))
            break
    
    # PR Interval
    pr_patterns = [
        r'(?:PR|PR interval|P-R)[:\s]+(\d+)',
        r'PR\s*=\s*(\d+)',
        r'PR interval[:\s]+(\d+)\s*(?:ms|msec)?'
    ]
    for pattern in pr_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            data.pr_interval = float(match.group(1))
            break
    
    # QRS Duration
    qrs_patterns = [
        r'(?:QRS|QRS duration)[:\s]+(\d+)',
        r'QRS\s*=\s*(\d+)',
        r'QRS duration[:\s]+(\d+)\s*(?:ms|msec)?'
    ]
    for pattern in qrs_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            data.qrs_duration = float(match.group(1))
            break
    
    # QT Interval
    qt_patterns = [
        r'(?:QT|QT interval)[:\s]+(\d+)',
        r'QT\s*=\s*(\d+)',
        r'QT interval[:\s]+(\d+)\s*(?:ms|msec)?'
    ]
    for pattern in qt_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            data.qt_interval = float(match.group(1))
            break
    
    # QTc Interval
    qtc_patterns = [
        r'(?:QTc|QTc interval|QTcB|QTcF)[:\s]+(\d+)',
        r'QTc\s*=\s*(\d+)',
        r'QTc interval[:\s]+(\d+)\s*(?:ms|msec)?'
    ]
    for pattern in qtc_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            data.qtc_interval = float(match.group(1))
            break
    
    # Rhythm
    rhythm_patterns = [
        (r'(sinus rhythm|sinus arrhythmia|sinus bradycardia|sinus tachycardia)', 'Sinus rhythm'),
        (r'(atrial fibrillation|a-fib|afib)', 'Atrial Fibrillation'),
        (r'(atrial flutter|a-flutter|aflutter)', 'Atrial Flutter'),
        (r'(ventricular tachycardia|vtach)', 'Ventricular Tachycardia'),
        (r'(ventricular fibrillation|vfib)', 'Ventricular Fibrillation'),
        (r'(bradycardia|brady)', 'Bradycardia'),
        (r'(tachycardia|tachy)', 'Tachycardia'),
        (r'(irregularly irregular|irregular rhythm)', 'Irregular'),
        (r'(normal sinus|nsr|sinus)', 'Sinus rhythm')
    ]
    for pattern, value in rhythm_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            data.rhythm = value
            break
    
    # Axis
    axis_patterns = [
        (r'(normal axis|axis normal)', 'Normal'),
        (r'(left axis deviation|left axis|lax|LAD)', 'Left Axis Deviation'),
        (r'(right axis deviation|right axis|rax|RAD)', 'Right Axis Deviation'),
        (r'(extreme axis|extreme deviation)', 'Extreme Axis Deviation'),
        (r'axis[:\s]+(\d+°)', None)
    ]
    for pattern, value in axis_patterns:
        if value:
            if re.search(pattern, text, re.IGNORECASE):
                data.axis = value
                break
        else:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                data.axis = match.group(1)
                break
    
    # ST Segment
    st_patterns = [
        (r'(st elevation|ste)', 'Elevation'),
        (r'(st depression|std)', 'Depression'),
        (r'(normal st|st normal|no st changes)', 'Normal'),
        (r'(st changes)', 'Changes')
    ]
    for pattern, value in st_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            data.st_segment = value
            break
    
    # T Wave
    t_patterns = [
        (r'(t wave inversion|t inversion|inverted t|negative t|t-wave inversion)', 'Inversion'),
        (r'(flat t|t wave flat|flattened t)', 'Flat'),
        (r'(biphasic t|t wave biphasic|biphasic)', 'Biphasic'),
        (r'(peaked t|tall t|hyperacute t)', 'Peaked'),
        (r'(normal t|t wave normal|no t wave changes)', 'Normal'),
    ]
    for pattern, value in t_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            data.t_wave = value
            break
    
    # Bundle Branch
    if re.search(r'lbbb|left bundle branch block', text, re.IGNORECASE):
        data.bundle_branch = 'LBBB'
    elif re.search(r'rbbb|right bundle branch block', text, re.IGNORECASE):
        data.bundle_branch = 'RBBB'
    elif re.search(r'bundle branch block|bbb', text, re.IGNORECASE):
        data.bundle_branch = 'Bundle Branch Block'
    
    # Q Waves
    if re.search(r'(pathological q waves|q waves)', text, re.IGNORECASE):
        data.q_waves = 'Pathological'
    
    # P Wave
    if re.search(r'(p wave abnormality|bifid p|p pulmonale)', text, re.IGNORECASE):
        data.p_wave = 'Abnormal'
    elif re.search(r'(normal p|p wave normal)', text, re.IGNORECASE):
        data.p_wave = 'Normal'
    
    # Machine Interpretation
    interp_patterns = [
        r'(?:Interpretation|Conclusion|Impressions|Findings|Summary|Diagnosis)[:\s]+([^.]+\.[^.]+)',
        r'(?:Interpretation|Conclusion|Impressions)[:\s]+([^\n]+)'
    ]
    for pattern in interp_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            data.machine_interpretation = match.group(1).strip()
            break
    
    # Abnormalities List
    data.abnormalities = []
    abnormality_patterns = [
        ('ST elevation', r'st elevation'),
        ('ST depression', r'st depression'),
        ('T wave inversion', r't wave inversion'),
        ('T wave flattening', r'flat t|t wave flat'),
        ('Q waves', r'q waves'),
        ('LVH', r'lvh|left ventricular hypertrophy'),
        ('RVH', r'rvh|right ventricular hypertrophy'),
        ('Atrial Fibrillation', r'atrial fibrillation'),
        ('Bundle Branch Block', r'bundle branch block'),
        ('Prolonged QT', r'prolonged qt'),
        ('Short QT', r'short qt'),
        ('LVH with strain', r'lvh with strain'),
        ('ST-T changes', r'st-t changes|st t changes'),
        ('Ischemic changes', r'ischemic'),
        ('Myocardial Infarction', r'myocardial infarction|mi'),
        ('Conduction disturbance', r'conduction disturbance'),
        ('Hypertrophy', r'hypertrophy')
    ]
    
    for label, pattern in abnormality_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            if label not in data.abnormalities:
                data.abnormalities.append(label)
    
    if data.t_wave == 'Inversion' and 'T wave inversion' not in data.abnormalities:
        data.abnormalities.append('T wave inversion')
    
    return data


def determine_severity_from_report(ecg_data: ECGReportData, patient: PatientContext):
    """
    Data-driven severity from report data.
    NO RULES - uses severity scorer.
    """
    scorer = SeverityScorer()
    
    # Build prediction
    ecg_prediction = {'prediction': 'Report-based', 'confidence': 0.7}
    
    # Build patient info
    patient_info = {
        'age': patient.age if patient else 50,
        'sex': patient.sex if patient else 'Unknown',
        'symptoms': patient.symptoms if patient else ''
    }
    
    # Build measurements
    measurements = {
        'heart_rate': ecg_data.heart_rate,
        'qtc_interval': ecg_data.qtc_interval,
        'qrs_duration': ecg_data.qrs_duration,
        'pr_interval': ecg_data.pr_interval,
        'st_elevation': ecg_data.st_segment == 'Elevation',
        'st_depression': ecg_data.st_segment == 'Depression',
        't_inversion': ecg_data.t_wave == 'Inversion'
    }
    
    result = scorer.calculate(ecg_prediction, patient_info, measurements)
    return result


def clinical_report_pipeline(ecg_data: ECGReportData,
                            patient: PatientContext,
                            use_rag: bool = True,
                            rag_store_path: str = "vector_store.pkl") -> Dict[str, Any]:
    """Complete clinical report pipeline using data-driven severity."""
    
    # Step 1: Determine severity (NO RULES)
    severity_result = determine_severity_from_report(ecg_data, patient)
    
    # Step 2: Retrieve guidelines (if RAG enabled)
    retrieved = []
    if use_rag:
        try:
            query_parts = []
            if ecg_data.heart_rate:
                query_parts.append(f"heart rate {ecg_data.heart_rate}")
            if ecg_data.st_segment:
                query_parts.append(f"ST {ecg_data.st_segment}")
            if ecg_data.rhythm:
                query_parts.append(f"rhythm {ecg_data.rhythm}")
            if ecg_data.abnormalities:
                query_parts.append(" ".join(ecg_data.abnormalities))
            if patient and patient.symptoms:
                query_parts.append(patient.symptoms)
            
            query = "ECG " + " ".join(query_parts) if query_parts else "ECG interpretation guidelines"
            
            from rag import retrieve
            raw = retrieve(query, store_path=rag_store_path, top_k=3)
            retrieved = [{"source": s, "text": t, "score": sc} for s, t, sc in raw]
        except Exception as e:
            retrieved = [{"text": f"RAG retrieval unavailable: {e}"}]
    
    # Step 3: Build LLM prompt
    prompt = build_clinical_prompt(ecg_data, patient, retrieved, severity_result)
    
    # Step 4: Generate LLM response
    try:
        from respond import generate_llm_response
        response = generate_llm_response(prompt)
    except Exception as e:
        response = f"LLM reasoning unavailable: {e}\n\nSeverity assessment: {severity_result.get('level', 'routine review')}"
    
    return {
        "ecg_data": ecg_data.to_dict(),
        "patient": patient.to_dict() if patient else {},
        "severity": severity_result,
        "guidelines_used": retrieved,
        "llm_response": response,
        "full_output": {
            "observations": response,
            "triage_support": severity_result.get('level', 'routine review'),
            "severity_score": severity_result.get('score', 0.5),
            "evidence": severity_result.get('evidence', []),
            "explanation": response,
            "disclaimer": "This is decision-support software and does not replace professional medical evaluation."
        }
    }


def build_clinical_prompt(ecg_data: ECGReportData, 
                          patient: PatientContext,
                          guidelines: List[Dict],
                          severity_result: Dict[str, Any]) -> str:
    """Build LLM prompt for clinical reasoning."""
    
    measurements = []
    if ecg_data.heart_rate:
        measurements.append(f"Heart Rate: {ecg_data.heart_rate} bpm")
    if ecg_data.pr_interval:
        measurements.append(f"PR Interval: {ecg_data.pr_interval} ms")
    if ecg_data.qrs_duration:
        measurements.append(f"QRS Duration: {ecg_data.qrs_duration} ms")
    if ecg_data.qt_interval:
        measurements.append(f"QT Interval: {ecg_data.qt_interval} ms")
    if ecg_data.qtc_interval:
        measurements.append(f"QTc Interval: {ecg_data.qtc_interval} ms")
    if ecg_data.rhythm:
        measurements.append(f"Rhythm: {ecg_data.rhythm}")
    if ecg_data.axis:
        measurements.append(f"Axis: {ecg_data.axis}")
    if ecg_data.st_segment:
        measurements.append(f"ST Segment: {ecg_data.st_segment}")
    if ecg_data.t_wave:
        measurements.append(f"T Wave: {ecg_data.t_wave}")
    if ecg_data.abnormalities:
        measurements.append(f"Abnormalities: {', '.join(ecg_data.abnormalities)}")
    if ecg_data.machine_interpretation:
        measurements.append(f"Machine Interpretation: {ecg_data.machine_interpretation}")
    
    measurement_text = "\n".join(measurements) if measurements else "No measurements provided."
    
    patient_text = ""
    if patient:
        if patient.age:
            patient_text += f"Age: {patient.age}\n"
        if patient.sex:
            patient_text += f"Sex: {patient.sex}\n"
        if patient.symptoms:
            patient_text += f"Symptoms: {patient.symptoms}\n"
        if patient.history:
            patient_text += f"History: {patient.history}\n"
        if patient.vitals:
            vitals_text = ", ".join(f"{k}: {v}" for k, v in patient.vitals.items())
            patient_text += f"Vitals: {vitals_text}\n"
    
    guideline_text = ""
    if guidelines:
        for g in guidelines[:3]:
            source = g.get('source', 'Unknown')
            text = g.get('text', '')[:500]
            guideline_text += f"\n- Source: {source}\n  {text}...\n"
    else:
        guideline_text = "No specific guidelines retrieved."
    
    severity_level = severity_result.get('level', 'routine review')
    severity_score = severity_result.get('score', 0.5)
    severity_evidence = severity_result.get('evidence', [])
    evidence_text = "\n".join(f"- {e}" for e in severity_evidence[:5])
    
    return f"""You are CardioAgent, an explainable multimodal clinical decision-support assistant.

CRITICAL RULES:
- You must not invent information.
- Use only the information provided below.
- If information is missing, explicitly state that it is missing.
- Do not present possible diagnoses as confirmed diagnoses.
- Do not claim clinical validation.
- Distinguish observed findings from model-derived findings.

=== ECG MEASUREMENTS ===
{measurement_text}

=== PATIENT CONTEXT ===
{patient_text}

=== SEVERITY ASSESSMENT (Data-driven) ===
Level: {severity_level}
Score: {severity_score:.2f}

Contributing factors:
{evidence_text}

=== RETRIEVED GUIDELINES ===
{guideline_text}

=== INSTRUCTIONS ===
Generate a structured clinical decision-support response with these sections:

1. **ECG Observations** - Summarize the key ECG findings from the available information.

2. **Clinical Significance** - Explain what these findings may mean, in clinical terms. Use cautious language.

3. **Risk / Triage Support** - Based on available evidence, provide a supported level.

4. **Red Flags** - List any concerning findings supported by the available evidence.

5. **Possible Considerations** - Use "Possible considerations include..." Never present as confirmed diagnoses.

6. **Clinician Review** - What the clinician should specifically review. Mention missing information.

7. **Questions / Missing Information** - What additional information would materially improve interpretation.

8. **Explanation** - Detailed explanation of the reasoning in understandable language.

9. **Evidence / Sources** - Identify the RAG evidence used where available.

10. **Disclaimer** - This is decision-support software and does not replace professional medical evaluation.

Keep responses clear, structured, and evidence-grounded.
Do not invent findings not present in the data.
Do not make definitive diagnoses.
"""