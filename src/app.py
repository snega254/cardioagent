"""
CardioAgent — Explainable Multimodal ECG Clinical Decision-Support Assistant
"""

import os
import uuid
import torch
import streamlit as st
import auth
from db import CardioDB

# ECG imports
from ecg_io import ECGLoadError, load_wfdb_pair, validate_signal, prepare_for_model
from gradcam import grad_cam_1d, top_attributed_region, create_gradcam_visualization
from heart_rate import detect_heart_rate, extract_ecg_measurements
from model import ECGConvNet
from preprocessing import SUPERCLASSES
from rag import build_query, retrieve
from report_pdf import generate_pdf_report
from respond import (
    CLASS_FULL_NAMES, FRIENDLY_DESCRIPTIONS, compose_response, generate_chat_response
)
from clinical_report import ECGReportData, parse_report_text, clinical_report_pipeline
from patient_context import PatientContext
from report_parser import parse_pdf_file_object
from patient_management import PatientManager, get_patient_display_name, format_symptoms

CHECKPOINT = "checkpoint.pt"
STORE_PATH = "vector_store.pkl"
ECG_FILES_DIR = "ecg_files"
REPORTS_DIR = "reports"

st.set_page_config(
    page_title="CardioAgent — ECG Clinical Decision Support",
    page_icon="❤️",
    layout="wide"
)

# ===== CSS THEME =====
st.markdown("""
<style>
    /* Professional clinical theme */
    .stApp {
        font-size: 15px;
        background-color: #f5f7fa;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #1a2332;
        font-weight: 600;
    }
    
    .main-header {
        color: #1a2332;
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 4px;
    }
    
    .main-subheader {
        color: #5a6a7a;
        font-size: 15px;
        margin-bottom: 20px;
    }
    
    .card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        border: 1px solid #e8ecf0;
        margin-bottom: 16px;
    }
    
    .result-finding {
        font-size: 20px;
        font-weight: 600;
        color: #1a2332;
        padding: 12px 16px;
        background: #f0f5ff;
        border-radius: 8px;
        border-left: 4px solid #2E6BA8;
        margin: 8px 0;
    }
    
    .severity-low { color: #28a745; }
    .severity-moderate { color: #fd7e14; }
    .severity-high { color: #dc3545; }
    .severity-urgent { color: #8b0000; font-weight: bold; }
    
    .emergency-warning {
        background: #fef3f2;
        border-left: 4px solid #dc3545;
        padding: 14px 18px;
        border-radius: 8px;
        margin: 12px 0;
    }
    
    .patient-summary {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 8px 0;
        border: 1px solid #e8ecf0;
    }
    
    .analysis-option {
        padding: 24px;
        border-radius: 12px;
        border: 2px solid #e8ecf0;
        background: white;
        text-align: center;
        cursor: pointer;
        transition: all 0.2s;
        min-height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    
    .analysis-option:hover {
        border-color: #2E6BA8;
        box-shadow: 0 4px 12px rgba(46, 107, 168, 0.1);
    }
    
    .analysis-option .icon {
        font-size: 36px;
        margin-bottom: 8px;
    }
    
    .analysis-option .title {
        font-weight: 600;
        font-size: 18px;
        color: #1a2332;
    }
    
    .analysis-option .desc {
        color: #5a6a7a;
        font-size: 13px;
        margin-top: 4px;
    }
    
    .compare-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #e8ecf0;
        min-height: 200px;
    }
    
    .compare-card .label {
        color: #5a6a7a;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .compare-card .date {
        color: #1a2332;
        font-size: 15px;
        font-weight: 500;
    }
    
    .compare-card .result {
        margin-top: 12px;
        padding: 10px 14px;
        background: #f8f9fa;
        border-radius: 6px;
        font-size: 14px;
    }
    
    .change-summary {
        background: #f0f5ff;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #d0e0f0;
        margin: 12px 0;
    }
    
    .change-summary .title {
        font-weight: 600;
        font-size: 16px;
        color: #1a2332;
    }
    
    .stButton button {
        font-size: 15px;
        padding: 10px 24px;
        border-radius: 8px;
        font-weight: 500;
    }
    
    .stButton button[kind="primary"] {
        background-color: #2E6BA8;
        color: white;
    }
    
    .stButton button[kind="primary"]:hover {
        background-color: #1a4f7a;
    }
    
    .stSelectbox label, .stNumberInput label, .stTextInput label {
        font-weight: 500;
        color: #1a2332;
    }
    
    .stExpander {
        border: 1px solid #e8ecf0 !important;
        border-radius: 8px !important;
    }
    
    .stExpander summary {
        font-weight: 500;
        color: #1a2332;
    }
    
    .sidebar .sidebar-content {
        background-color: white;
    }
    
    .stChatInput textarea {
        font-size: 15px !important;
        min-height: 45px !important;
    }
    
    hr {
        margin: 20px 0;
        border-color: #e8ecf0;
    }
</style>
""", unsafe_allow_html=True)


# ===== DATABASE =====
@st.cache_resource
def get_db():
    return CardioDB()

try:
    cdb = get_db()
    db_error = None
except Exception as e:
    cdb = None
    db_error = str(e)


@st.cache_resource
def get_patient_manager():
    return PatientManager(cdb) if cdb else None

patient_manager = get_patient_manager()


@st.cache_resource
def load_model():
    model = ECGConvNet()
    if os.path.exists(CHECKPOINT):
        ckpt = torch.load(CHECKPOINT, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        model.log_temperature.data = ckpt["log_temperature"]
        model.eval()
    return model


# ===== SESSION STATE =====
for key, default in [
    ("user_id", None),
    ("email", None),
    ("name", None),
    ("page", "Dashboard"),
    ("selected_patient_id", None),
    ("selected_analysis_id", None),
    ("analysis_type", None),
    ("chat_histories", {}),
    ("current_analysis", None),
    ("current_signal", None),
    ("current_fs", None),
    ("current_lead_names", None),
    ("current_ecg_data", None),
    ("current_patient", None),
    ("analysis_complete", False),
    ("show_analysis_input", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ===== AUTHENTICATION =====
def do_login(email, password):
    user = cdb.get_user_by_email(email)
    if user is None:
        return False, "No account with that email address."
    if not auth.verify_password(password, user["password_hash"]):
        return False, "Incorrect password."
    st.session_state.user_id = user["id"]
    st.session_state.email = user["email"]
    st.session_state.name = user["name"]
    return True, ""


def do_register(email, password, name):
    ok, msg = auth.validate_email(email)
    if not ok:
        return False, msg
    ok, msg = auth.validate_password(password)
    if not ok:
        return False, msg
    ok, msg = auth.validate_name(name)
    if not ok:
        return False, msg
    if cdb.get_user_by_email(email) is not None:
        return False, "That email is already registered."
    pw_hash = auth.hash_password(password)
    cdb.create_user(email, pw_hash, name)
    return True, "Account created. Please log in."


# ===== SIDEBAR =====
def render_sidebar():
    with st.sidebar:
        st.markdown("## ❤️ CardioAgent")
        st.caption("ECG Clinical Decision Support")
        
        if db_error:
            st.error("Database not connected.")
            with st.expander("Setup"):
                st.code(db_error)
            st.stop()
        
        if st.session_state.user_id is None:
            render_login()
            return
        
        st.success(f"👤 {st.session_state.name}")
        st.divider()
        
        # Navigation
        nav = st.radio(
            "Navigation",
            ["Dashboard", "New Analysis", "History", "Compare"],
            index=0,
            label_visibility="collapsed"
        )
        st.session_state.page = nav
        
        # Patient info if selected
        if st.session_state.selected_patient_id:
            patient = patient_manager.get_patient(st.session_state.selected_patient_id)
            if patient:
                st.divider()
                st.caption("Current Patient")
                st.write(f"**{patient.get('name', 'Unknown')}**")
                st.write(f"{patient.get('age', 'N/A')} yrs · {patient.get('sex', 'N/A')}")
                if patient.get('symptoms'):
                    st.write(f"📋 {format_symptoms(patient.get('symptoms', []))}")
        
        st.divider()
        if st.button("🚪 Log Out", use_container_width=True):
            for key in ["user_id", "email", "name", "selected_patient_id", 
                        "selected_analysis_id", "analysis_type", "current_analysis",
                        "current_signal", "current_fs", "current_lead_names",
                        "current_ecg_data", "current_patient", "analysis_complete",
                        "show_analysis_input"]:
                st.session_state[key] = None
            st.session_state.page = "Dashboard"
            st.rerun()


def render_login():
    login_tab, register_tab = st.tabs(["Log In", "Register"])
    with login_tab:
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Log In", use_container_width=True, type="primary"):
            ok, msg = do_login(email, password)
            if ok:
                st.rerun()
            else:
                st.error(msg)
    with register_tab:
        reg_email = st.text_input("Email", key="reg_email")
        reg_pass = st.text_input("Choose a password", type="password", key="reg_pass")
        reg_name = st.text_input("Full name", key="reg_name")
        if st.button("Register", use_container_width=True):
            ok, msg = do_register(reg_email, reg_pass, reg_name)
            if ok:
                st.success(msg)
            else:
                st.error(msg)
    st.stop()


# ===== PAGE: DASHBOARD =====
def page_dashboard():
    st.markdown('<div class="main-header">❤️ CardioAgent</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subheader">Explainable ECG Clinical Decision Support</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📋 New Analysis", use_container_width=True, type="primary"):
            st.session_state.page = "New Analysis"
            st.rerun()
    with col2:
        if st.button("📊 History", use_container_width=True):
            st.session_state.page = "History"
            st.rerun()
    with col3:
        if st.button("📈 Compare", use_container_width=True):
            st.session_state.page = "Compare"
            st.rerun()
    
    st.divider()
    st.subheader("Recent Patients")
    patients = patient_manager.get_patients_for_user(st.session_state.user_id)
    
    if not patients:
        st.info("No patients yet. Start by creating a new patient in New Analysis.")
        return
    
    for patient in patients[:5]:
        with st.container(border=True):
            cols = st.columns([3, 2, 2, 1])
            cols[0].write(f"**{patient.get('name', 'Unknown')}**")
            cols[1].write(f"{patient.get('age', 'N/A')} yrs · {patient.get('sex', 'N/A')}")
            cols[2].write(f"{len(patient.get('analyses', [])) or 0} analyses")
            if cols[3].button("Select", key=f"select_{patient['id']}"):
                st.session_state.selected_patient_id = patient["id"]
                st.session_state.page = "New Analysis"
                st.rerun()


# ===== PAGE: NEW ANALYSIS (SIMPLIFIED) =====
def page_new_analysis():
    st.markdown('<div class="main-header">📋 New Analysis</div>', unsafe_allow_html=True)
    
    user_id = st.session_state.user_id
    
    # ---- Patient Selection ----
    existing_patients = patient_manager.get_patients_for_user(user_id)
    
    if existing_patients:
        patient_options = {p["id"]: get_patient_display_name(p) for p in existing_patients}
        patient_options["new"] = "+ Create New Patient"
        
        selected = st.selectbox(
            "Patient",
            list(patient_options.keys()),
            format_func=lambda x: patient_options[x],
            key="patient_select"
        )
        
        if selected == "new":
            patient_id = render_create_patient(user_id)
            if patient_id:
                st.session_state.selected_patient_id = patient_id
                st.rerun()
        else:
            st.session_state.selected_patient_id = selected
            patient = patient_manager.get_patient(selected)
            if patient:
                st.session_state.current_patient = patient
    else:
        st.info("No patients yet. Create one below.")
        patient_id = render_create_patient(user_id)
        if patient_id:
            st.session_state.selected_patient_id = patient_id
            st.rerun()
    
    patient = st.session_state.current_patient
    if not patient:
        st.info("Please create or select a patient to continue.")
        return
    
    # ---- Patient Summary ----
    st.markdown(f"""
    <div class="patient-summary">
        <b>{patient.get('name', 'Unknown')}</b> · 
        {patient.get('age', 'N/A')} yrs · 
        {patient.get('sex', 'N/A')} · 
        Symptoms: {format_symptoms(patient.get('symptoms', []))}
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # ---- Analysis Type (Only Signal + Report) ----
    st.markdown("### Choose Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container(border=True):
            st.markdown('<div style="text-align:center; padding:8px 0;">', unsafe_allow_html=True)
            st.markdown('<span style="font-size:36px;">📊</span>', unsafe_allow_html=True)
            st.markdown('<div style="font-weight:600; font-size:18px;">Signal Analysis</div>', unsafe_allow_html=True)
            st.markdown('<div style="color:#5a6a7a; font-size:13px;">Raw ECG signal (.hea + .dat)</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            if st.button("Select Signal Analysis", use_container_width=True, key="btn_signal"):
                st.session_state.analysis_type = "signal"
                st.session_state.show_analysis_input = True
                st.rerun()
    
    with col2:
        with st.container(border=True):
            st.markdown('<div style="text-align:center; padding:8px 0;">', unsafe_allow_html=True)
            st.markdown('<span style="font-size:36px;">📄</span>', unsafe_allow_html=True)
            st.markdown('<div style="font-weight:600; font-size:18px;">Medical Report</div>', unsafe_allow_html=True)
            st.markdown('<div style="color:#5a6a7a; font-size:13px;">ECG report text or PDF</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            if st.button("Select Medical Report", use_container_width=True, key="btn_report"):
                st.session_state.analysis_type = "report"
                st.session_state.show_analysis_input = True
                st.rerun()
    
    # ---- Input Area (shown after selection) ----
    if st.session_state.show_analysis_input:
        st.divider()
        analysis_type = st.session_state.analysis_type
        
        if analysis_type == "signal":
            render_signal_analysis(patient)
        elif analysis_type == "report":
            render_report_analysis(patient)
        else:
            st.info("Select an analysis type above.")


def render_create_patient(user_id):
    """Render the create patient form."""
    with st.container(border=True):
        st.markdown("#### New Patient")
        
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Patient Name")
            age = st.number_input("Age", min_value=0, max_value=150, value=50)
        with col2:
            sex = st.selectbox("Sex", ["Select...", "Male", "Female", "Other"])
        
        st.markdown("**Symptoms**")
        symptoms_options = [
            "Chest pain", "Chest discomfort", "Shortness of breath",
            "Palpitations", "Dizziness", "Fainting", "Fatigue",
            "No symptoms", "Other"
        ]
        symptoms = st.multiselect("Select symptoms", symptoms_options, default=[])
        
        other_symptom = st.text_input("Other symptoms (optional)")
        if other_symptom and other_symptom not in symptoms:
            symptoms.append(other_symptom)
        
        if st.button("Create Patient", type="primary", use_container_width=True):
            if not name.strip():
                st.error("Patient name is required.")
                return None
            if sex == "Select...":
                st.warning("Please select a sex.")
                return None
            
            patient_id = patient_manager.create_patient(
                user_id=user_id,
                name=name.strip(),
                age=age,
                sex=sex,
                symptoms=symptoms
            )
            st.success(f"Patient {name} created!")
            patient = patient_manager.get_patient(patient_id)
            st.session_state.current_patient = patient
            st.session_state.selected_patient_id = patient_id
            return patient_id
    
    return None


# ===== SIGNAL ANALYSIS =====
def render_signal_analysis(patient):
    """Render signal analysis interface (simplified)."""
    st.markdown("#### Upload ECG Signal")
    st.caption("Upload .hea and .dat files")
    
    if not os.path.exists(CHECKPOINT):
        st.error(f"No trained model found at '{CHECKPOINT}'.")
        return
    
    col1, col2 = st.columns(2)
    with col1:
        hea_file = st.file_uploader("Select .hea file", type=["hea"])
    with col2:
        dat_file = st.file_uploader("Select matching .dat file", type=["dat"])
    
    if not hea_file or not dat_file:
        st.info("Upload both .hea and .dat files to continue.")
        return
    
    if st.button("🔍 Analyze Signal", type="primary", use_container_width=True):
        with st.spinner("Analyzing ECG signal..."):
            try:
                raw_signal, fs, lead_names, _ = load_wfdb_pair(hea_file, dat_file)
                val_warnings, duration_sec = validate_signal(raw_signal, fs, min_duration_sec=2.0)
                
                # Heart rate
                hr_result = detect_heart_rate(raw_signal[:, 0], fs)
                
                # Model
                model = load_model()
                model_input, _ = prepare_for_model(raw_signal, fs)
                x_tensor = torch.from_numpy(model_input).unsqueeze(0)
                
                with torch.no_grad():
                    logits = model(x_tensor)
                    probs = model.calibrated_probs(logits)[0]
                
                pred_idx = int(torch.argmax(probs).item())
                pred_class = SUPERCLASSES[pred_idx]
                confidence = float(probs[pred_idx].item())
                
                # Grad-CAM
                cam = grad_cam_1d(model, x_tensor, pred_idx)
                start_sec, end_sec, _ = top_attributed_region(cam, fs=100)
                
                # Measurements
                measurements = extract_ecg_measurements(raw_signal[:, 0], fs)
                if hr_result.get("heart_rate"):
                    measurements["heart_rate"] = hr_result["heart_rate"]
                
                # RAG
                query = f"{pred_class} {CLASS_FULL_NAMES.get(pred_class, '')} ECG"
                passages = []
                if os.path.exists(STORE_PATH):
                    try:
                        raw = retrieve(query, store_path=STORE_PATH, top_k=3)
                        passages = [{"source": s, "text": t, "score": sc} for s, t, sc in raw]
                    except:
                        pass
                
                features = {"heart_rate": measurements.get("heart_rate"), "n_rpeaks": hr_result.get("n_rpeaks", 0)}
                xai = {"region_start_sec": start_sec, "region_end_sec": end_sec}
                
                explanation, severity = compose_response(
                    pred_class, confidence, features, xai,
                    [(p["source"], p["text"], p["score"]) for p in passages],
                    patient, measurements
                )
                
                # Store analysis
                analysis_id = cdb.create_analysis_with_patient(
                    patient_id=patient["id"],
                    user_id=st.session_state.user_id,
                    analysis_type="signal",
                    prediction=pred_class,
                    confidence=confidence,
                    features=features,
                    xai=xai,
                    rag_sources=passages,
                    explanation=explanation,
                    severity=severity,
                    summary=f"ECG pattern: {FRIENDLY_DESCRIPTIONS.get(pred_class, pred_class)}",
                    mode_type="research",
                    patient_context={"age": patient.get("age"), "sex": patient.get("sex"), "symptoms": patient.get("symptoms")},
                    clinical_reasoning={"severity": severity, "explanation": explanation}
                )
                
                st.session_state.current_analysis = {
                    "id": analysis_id,
                    "prediction": pred_class,
                    "confidence": confidence,
                    "friendly_name": FRIENDLY_DESCRIPTIONS.get(pred_class, pred_class),
                    "explanation": explanation,
                    "severity": severity,
                    "features": features,
                    "measurements": measurements,
                    "xai": xai,
                    "rag_sources": passages,
                    "analysis_type": "signal",
                    "patient": patient,
                    "signal": raw_signal,
                    "fs": fs,
                    "lead_names": lead_names,
                    "cam": cam
                }
                st.session_state.analysis_complete = True
                st.rerun()
                
            except ECGLoadError as e:
                st.error(f"❌ {str(e)}")
    
    if st.session_state.analysis_complete and st.session_state.current_analysis:
        render_concise_result(st.session_state.current_analysis)


# ===== MEDICAL REPORT ANALYSIS =====
def render_report_analysis(patient):
    """Render medical report analysis interface (simplified)."""
    st.markdown("#### Upload or Paste ECG Report")
    
    input_method = st.radio(
        "Input method",
        ["Paste Report Text", "Upload PDF Report"],
        horizontal=True
    )
    
    ecg_data = ECGReportData()
    
    if input_method == "Paste Report Text":
        report_text = st.text_area("Paste ECG report text here", height=150)
        if report_text:
            ecg_data = parse_report_text(report_text)
            if ecg_data and ecg_data.has_data():
                st.success("✅ Measurements extracted")
                with st.expander("Extracted Measurements"):
                    st.json(ecg_data.to_dict())
    
    elif input_method == "Upload PDF Report":
        pdf_file = st.file_uploader("Upload PDF report", type=["pdf"])
        if pdf_file:
            with st.spinner("Processing PDF..."):
                ecg_data = parse_pdf_file_object(pdf_file)
                if ecg_data and ecg_data.has_data():
                    st.success("✅ PDF processed successfully")
                    with st.expander("Extracted Measurements"):
                        st.json(ecg_data.to_dict())
                else:
                    st.warning("Could not extract measurements. Try pasting text.")
    
    # Manual fallback
    with st.expander("Manual Entry (if auto-extraction failed)"):
        col1, col2 = st.columns(2)
        with col1:
            ecg_data.heart_rate = st.number_input("Heart Rate (bpm)", min_value=0.0, max_value=300.0, value=None, step=1.0)
            ecg_data.pr_interval = st.number_input("PR Interval (ms)", min_value=0.0, max_value=500.0, value=None, step=1.0)
        with col2:
            ecg_data.qrs_duration = st.number_input("QRS Duration (ms)", min_value=0.0, max_value=300.0, value=None, step=1.0)
            ecg_data.qtc_interval = st.number_input("QTc Interval (ms)", min_value=0.0, max_value=800.0, value=None, step=1.0)
            ecg_data.rhythm = st.selectbox("Rhythm", ["", "Sinus", "Atrial Fibrillation", "Atrial Flutter"])
            ecg_data.st_segment = st.selectbox("ST Segment", ["", "Normal", "Elevation", "Depression"])
            ecg_data.t_wave = st.selectbox("T Wave", ["", "Normal", "Inversion", "Flat"])
    
    if st.button("🔍 Analyze Report", type="primary", use_container_width=True):
        if not ecg_data.has_data():
            st.warning("Please enter at least one ECG measurement or interpretation.")
            return
        
        with st.spinner("Analyzing report..."):
            patient_context = PatientContext(
                age=patient.get("age"),
                sex=patient.get("sex"),
                symptoms=", ".join(patient.get("symptoms", [])) if patient.get("symptoms") else None,
                history=None,
                vitals={}
            )
            
            result = clinical_report_pipeline(ecg_data, patient_context, True, STORE_PATH)
            
            severity = result.get("severity", {}).get("level", "routine review")
            analysis_id = cdb.create_analysis_with_patient(
                patient_id=patient["id"],
                user_id=st.session_state.user_id,
                analysis_type="report",
                prediction="Report-based",
                confidence=None,
                features=ecg_data.to_dict(),
                xai={},
                rag_sources=result.get("guidelines_used", []),
                explanation=result.get("llm_response", ""),
                severity=severity,
                summary="Report-based ECG analysis",
                mode_type="report",
                report_text=ecg_data.raw_report_text,
                patient_context={"age": patient.get("age"), "sex": patient.get("sex"), "symptoms": patient.get("symptoms")},
                clinical_reasoning=result.get("full_output", {})
            )
            
            st.session_state.current_analysis = {
                "id": analysis_id,
                "prediction": "Report-based",
                "confidence": None,
                "friendly_name": "Report-based interpretation",
                "explanation": result.get("llm_response", ""),
                "severity": severity,
                "features": ecg_data.to_dict(),
                "xai": {},
                "rag_sources": result.get("guidelines_used", []),
                "analysis_type": "report",
                "patient": patient,
                "clinical_reasoning": result.get("full_output", {})
            }
            st.session_state.analysis_complete = True
            st.rerun()
    
    if st.session_state.analysis_complete and st.session_state.current_analysis:
        render_concise_result(st.session_state.current_analysis)


# ===== CONCISE RESULT RENDERER =====
def render_concise_result(analysis):
    """Render concise, clinically-focused results."""
    patient = analysis.get("patient", {})
    severity = analysis.get("severity", "routine review")
    explanation = analysis.get("explanation", "")
    pred = analysis.get("prediction")
    friendly = analysis.get("friendly_name", "")
    
    # Emergency warning
    if patient and patient.get("symptoms"):
        symptoms_text = " ".join(patient.get("symptoms", []))
        emergency_keywords = ["chest pain", "shortness of breath", "fainting", "loss of consciousness"]
        if any(k in symptoms_text.lower() for k in emergency_keywords):
            st.markdown("""
            <div class="emergency-warning">
            ⚠️ <strong>Emergency symptoms reported — seek immediate professional care.</strong>
            </div>
            """, unsafe_allow_html=True)
    
    st.divider()
    st.markdown('<div class="main-header">📊 Analysis Result</div>', unsafe_allow_html=True)
    
    # Patient info
    st.markdown(f"""
    <div style="color:#5a6a7a; font-size:14px; margin-bottom:12px;">
        {patient.get('name', 'Unknown')} · {patient.get('age', 'N/A')} yrs · {patient.get('sex', 'N/A')}
    </div>
    """, unsafe_allow_html=True)
    
    # ---- FINDING ----
    if pred and pred != "Report-based":
        finding_text = friendly.capitalize() if friendly else f"ECG pattern: {pred}"
    else:
        finding_text = "Report-based interpretation"
    
    st.markdown(f"""
    <div class="result-finding">
        {finding_text}
    </div>
    """, unsafe_allow_html=True)
    
    # ---- SEVERITY ----
    severity_colors = {
        "routine review": "severity-low",
        "prompt clinical review": "severity-moderate",
        "urgent evaluation may be appropriate": "severity-high"
    }
    color_class = severity_colors.get(severity, "severity-moderate")
    
    st.markdown(f"""
    <div style="margin: 8px 0 16px 0;">
        <span style="font-weight:500;">Severity:</span>
        <span class="{color_class}" style="font-weight:600;">{severity}</span>
    </div>
    """, unsafe_allow_html=True)
    
    # ---- KEY FINDINGS ----
    features = analysis.get("features", {})
    if features and any(features.values()):
        st.markdown("#### Key Findings")
        
        findings = []
        if features.get("heart_rate"):
            findings.append(f"Heart Rate: {features['heart_rate']} bpm")
        if features.get("rhythm"):
            findings.append(f"Rhythm: {features['rhythm']}")
        if features.get("st_segment"):
            findings.append(f"ST Segment: {features['st_segment']}")
        if features.get("t_wave"):
            findings.append(f"T Wave: {features['t_wave']}")
        if features.get("abnormalities"):
            findings.append(f"Abnormalities: {', '.join(features['abnormalities'])}")
        if features.get("qtc_interval"):
            findings.append(f"QTc: {features['qtc_interval']} ms")
        if features.get("qrs_duration"):
            findings.append(f"QRS: {features['qrs_duration']} ms")
        
        for f in findings[:6]:
            st.write(f"• {f}")
    
    # ---- WHY? (Grad-CAM for Signal) ----
    if analysis.get("analysis_type") == "signal":
        st.markdown("#### Why did the model say this?")
        
        # Grad-CAM Visualization
        if "cam" in analysis and analysis["cam"] is not None:
            try:
                signal = analysis.get("signal")
                fs = analysis.get("fs")
                cam = analysis.get("cam")
                lead_names = analysis.get("lead_names", [])
                
                if signal is not None and cam is not None:
                    fig = create_gradcam_visualization(
                        signal, cam, fs if fs else 100,
                        lead_idx=0,
                        title=f"Model Attribution — {friendly.capitalize() if friendly else 'ECG Analysis'}"
                    )
                    st.pyplot(fig)
                    import matplotlib.pyplot as plt
                    plt.close(fig)
                    
                    xai = analysis.get("xai", {})
                    if xai.get("region_start_sec"):
                        st.caption(f"🔍 The model focused on approximately {xai['region_start_sec']:.1f}s to {xai['region_end_sec']:.1f}s")
            except Exception as e:
                st.caption("Explainability visualization unavailable for this analysis.")
        else:
            st.caption("Explainability visualization not available for this analysis.")
    
    # ---- WHY? (Medical Report) ----
    elif analysis.get("analysis_type") == "report":
        st.markdown("#### Why was this flagged?")
        st.markdown("The report contains findings that may warrant clinical review.")
        
        rag_sources = analysis.get("rag_sources", [])
        if rag_sources:
            with st.expander("Evidence / Sources"):
                for s in rag_sources[:2]:
                    st.markdown(f"**Source:** {s.get('source', 'Unknown')}")
                    st.write(s.get('text', '')[:200] + "...")
                    st.divider()
    
    # ---- WHAT TO REVIEW ----
    st.markdown("#### What to Review")
    review_items = []
    if features and features.get("qtc_interval") and features["qtc_interval"] > 480:
        review_items.append("QTc is prolonged — consider further evaluation")
    if features and features.get("st_segment") == "Elevation":
        review_items.append("ST elevation noted — urgent evaluation may be appropriate")
    if features and features.get("abnormalities"):
        for abn in features.get("abnormalities", [])[:3]:
            review_items.append(f"Review: {abn}")
    if not review_items:
        review_items.append("Review the ECG findings in clinical context")
        review_items.append("Compare with previous ECGs if available")
    
    for item in review_items[:4]:
        st.write(f"• {item}")
    
    # ---- EXPLANATION ----
    if explanation and len(explanation) > 50:
        with st.expander("Detailed Explanation"):
            st.markdown(explanation)
    
    # ---- TECHNICAL DETAILS (collapsed) ----
    with st.expander("Technical Details"):
        st.json({
            "Analysis Type": analysis.get("analysis_type", "unknown"),
            "Prediction": analysis.get("prediction"),
            "Severity": severity,
            "Features": features
        })
    
    # ---- DISCLAIMER ----
    st.divider()
    st.caption("⚠️ This is an AI-assisted research prototype. All findings require clinical confirmation.")
    
    # ---- ACTIONS ----
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("💬 Ask CardioAgent", use_container_width=True):
            st.session_state.show_chat = not st.session_state.get("show_chat", False)
            st.rerun()
    with col2:
        if st.button("📄 Generate PDF", use_container_width=True):
            generate_pdf_for_analysis(analysis)
    with col3:
        if st.button("📋 Save to History", use_container_width=True):
            st.success("Analysis saved to history.")
    
    # ---- CHAT ----
    if st.session_state.get("show_chat", False):
        st.divider()
        st.markdown("### 💬 Ask CardioAgent")
        chat_key = f"chat_{analysis.get('id', 'default')}"
        if chat_key not in st.session_state.chat_histories:
            st.session_state.chat_histories[chat_key] = []
        history = st.session_state.chat_histories[chat_key]
        
        for turn in history:
            with st.chat_message("user" if turn["role"] == "user" else "assistant"):
                st.write(turn["text"])
        
        q = st.chat_input("Ask about this analysis...")
        if q:
            history.append({"role": "user", "text": q})
            with st.chat_message("user"):
                st.write(q)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        context = f"Analysis Type: {analysis.get('analysis_type', 'N/A')}\nSeverity: {severity}\nPatient: {patient.get('name', 'N/A')}\nFindings: {friendly}"
                        answer = generate_chat_response(context, history[:-1], q)
                    except Exception as e:
                        answer = f"Could not generate response: {e}"
                st.write(answer)
            history.append({"role": "assistant", "text": answer})


def generate_pdf_for_analysis(analysis):
    """Generate and download PDF report."""
    with st.spinner("Generating report..."):
        try:
            os.makedirs(REPORTS_DIR, exist_ok=True)
            pdf_path = os.path.join(REPORTS_DIR, f"report_{analysis.get('id', uuid.uuid4())}.pdf")
            
            patient = analysis.get("patient", {})
            patient_data = {
                "name": patient.get("name", "Unknown"),
                "age": patient.get("age", "N/A"),
                "sex": patient.get("sex", "N/A"),
                "email": st.session_state.email,
            }
            ecg_record = {
                "filename": f"{analysis.get('analysis_type', 'unknown').capitalize()} Analysis",
                "source_type": analysis.get('analysis_type', 'unknown'),
                "sampling_rate": "N/A",
                "n_leads": "N/A",
                "duration_sec": "N/A",
                "upload_time": "N/A"
            }
            analysis_data = {
                "prediction": analysis.get("prediction", "N/A"),
                "confidence": analysis.get("confidence"),
                "features": analysis.get("features", {}),
                "xai": analysis.get("xai", {}),
                "rag_sources": analysis.get("rag_sources", []),
                "explanation": analysis.get("explanation", ""),
                "severity": analysis.get("severity", "routine review"),
                "summary": analysis.get("friendly_name", "ECG Analysis"),
                "mode_type": analysis.get("analysis_type", "unknown"),
                "patient_context": patient,
                "clinical_reasoning": analysis.get("clinical_reasoning", {})
            }
            
            generate_pdf_report(pdf_path, patient_data, ecg_record, analysis_data, CLASS_FULL_NAMES)
            
            with open(pdf_path, "rb") as f:
                st.download_button(
                    "📥 Download Report",
                    f,
                    file_name=os.path.basename(pdf_path),
                    mime="application/pdf"
                )
            st.success("Report generated!")
        except Exception as e:
            st.error(f"Report generation failed: {e}")


# ===== PAGE: HISTORY =====
def page_history():
    st.markdown('<div class="main-header">📋 History</div>', unsafe_allow_html=True)
    
    user_id = st.session_state.user_id
    history = cdb.get_history_for_user(user_id)
    
    if not history:
        st.info("No analyses yet.")
        return
    
    for item in history[:20]:
        patient = item.get("patient", {})
        with st.container(border=True):
            cols = st.columns([2, 1.5, 2, 1.5, 1])
            cols[0].write(f"**{patient.get('name', 'Unknown')}**")
            cols[1].write(item.get("created_at", "")[:16].replace("T", " "))
            cols[2].write(f"_{item.get('analysis_type', 'unknown').capitalize()}_")
            cols[3].write(item.get("severity", "N/A"))
            if cols[4].button("View", key=f"view_{item['id']}"):
                st.session_state.selected_analysis_id = item["id"]
                st.session_state.page = "History Detail"
                st.rerun()


def page_history_detail():
    st.markdown('<div class="main-header">📋 Analysis Detail</div>', unsafe_allow_html=True)
    
    analysis_id = st.session_state.selected_analysis_id
    if not analysis_id:
        st.warning("No analysis selected.")
        return
    
    analysis = cdb.get_analysis_by_id(analysis_id)
    if not analysis:
        st.error("Analysis not found.")
        return
    
    patient = analysis.get("patient", {})
    rendered = {
        "id": analysis["id"],
        "prediction": analysis.get("prediction", "N/A"),
        "confidence": analysis.get("confidence"),
        "friendly_name": analysis.get("summary", "ECG Analysis"),
        "explanation": analysis.get("explanation", ""),
        "severity": analysis.get("severity", "routine review"),
        "features": analysis.get("features", {}),
        "xai": analysis.get("xai", {}),
        "rag_sources": analysis.get("rag_sources", []),
        "analysis_type": analysis.get("analysis_type", "unknown"),
        "patient": patient,
        "clinical_reasoning": analysis.get("clinical_reasoning", {})
    }
    
    render_concise_result(rendered)


# ===== PAGE: COMPARE (SIMPLIFIED) =====
def page_compare():
    st.markdown('<div class="main-header">📈 Compare</div>', unsafe_allow_html=True)
    st.caption("Compare two analyses of the same patient")
    
    user_id = st.session_state.user_id
    history = cdb.get_history_for_user(user_id)
    
    if len(history) < 2:
        st.info("You need at least two analyses to compare.")
        return
    
    # Group by patient
    patient_groups = {}
    for item in history:
        patient = item.get("patient", {})
        patient_id = patient.get("id")
        if not patient_id:
            continue
        if patient_id not in patient_groups:
            patient_groups[patient_id] = {
                "patient": patient,
                "analyses": []
            }
        patient_groups[patient_id]["analyses"].append(item)
    
    # Find patients with at least 2 analyses
    comparable_patients = {pid: data for pid, data in patient_groups.items() if len(data["analyses"]) >= 2}
    
    if not comparable_patients:
        st.info("No patient has multiple analyses to compare.")
        return
    
    # Select patient
    patient_options = {pid: data["patient"].get("name", "Unknown") for pid, data in comparable_patients.items()}
    selected_patient_id = st.selectbox(
        "Select Patient",
        list(patient_options.keys()),
        format_func=lambda x: patient_options[x]
    )
    
    if selected_patient_id not in comparable_patients:
        st.warning("Patient not found.")
        return
    
    patient_data = comparable_patients[selected_patient_id]
    patient = patient_data["patient"]
    analyses = sorted(patient_data["analyses"], key=lambda x: x.get("created_at", ""))
    
    if len(analyses) < 2:
        st.warning("Not enough analyses for this patient.")
        return
    
    # Select two analyses
    options = {}
    for a in analyses:
        label = f"{a.get('created_at', '')[:16].replace('T', ' ')} - {a.get('analysis_type', 'unknown')}"
        options[label] = a
    
    labels = list(options.keys())
    
    col1, col2 = st.columns(2)
    with col1:
        label1 = st.selectbox("Earlier Analysis", labels, index=0, key="cmp1")
    with col2:
        label2 = st.selectbox("Later Analysis", labels, index=min(1, len(labels)-1), key="cmp2")
    
    if label1 == label2:
        st.warning("Please select two different analyses.")
        return
    
    earlier = options[label1]
    later = options[label2]
    
    # ---- COMPARISON DISPLAY ----
    st.divider()
    st.markdown(f"""
    <div style="font-size:20px; font-weight:600; color:#1a2332;">
        Patient: {patient.get('name', 'Unknown')}
    </div>
    <div style="color:#5a6a7a; font-size:14px;">
        {patient.get('age', 'N/A')} yrs · {patient.get('sex', 'N/A')}
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="compare-card">
            <div class="label">Earlier Analysis</div>
            <div class="date">{earlier.get('created_at', '')[:16].replace('T', ' ')}</div>
            <div style="color:#5a6a7a; font-size:13px;">{earlier.get('analysis_type', 'unknown').capitalize()}</div>
            <div class="result">
                <b>Result:</b> {earlier.get('summary', 'N/A')}
            </div>
            <div style="margin-top:8px;">
                <b>Severity:</b> {earlier.get('severity', 'N/A')}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="compare-card">
            <div class="label">Later Analysis</div>
            <div class="date">{later.get('created_at', '')[:16].replace('T', ' ')}</div>
            <div style="color:#5a6a7a; font-size:13px;">{later.get('analysis_type', 'unknown').capitalize()}</div>
            <div class="result">
                <b>Result:</b> {later.get('summary', 'N/A')}
            </div>
            <div style="margin-top:8px;">
                <b>Severity:</b> {later.get('severity', 'N/A')}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # ---- OVERALL SUMMARY ----
    st.divider()
    st.markdown("#### Overall Summary")
    
    earlier_sev = earlier.get("severity", "routine review")
    later_sev = later.get("severity", "routine review")
    
    # Compare severity
    severity_order = ["routine review", "prompt clinical review", "urgent evaluation may be appropriate"]
    earlier_idx = severity_order.index(earlier_sev) if earlier_sev in severity_order else 1
    later_idx = severity_order.index(later_sev) if later_sev in severity_order else 1
    
    if earlier_idx < later_idx:
        st.warning("The later analysis shows increased concern.")
    elif earlier_idx > later_idx:
        st.success("The later analysis shows decreased concern.")
    else:
        st.info("The level of concern is similar between analyses.")
    
    # Compare features if both have them
    earlier_features = earlier.get("features", {})
    later_features = later.get("features", {})
    
    changes = []
    for key in ["heart_rate", "qtc_interval", "qrs_duration"]:
        if key in earlier_features and key in later_features:
            if earlier_features[key] != later_features[key]:
                changes.append(f"{key}: {earlier_features[key]} → {later_features[key]}")
    
    if changes:
        st.write("**Measured changes:**")
        for c in changes:
            st.write(f"• {c}")
    else:
        st.write("No comparable measurements between these analyses.")
    
    if earlier.get("analysis_type") != later.get("analysis_type"):
        st.caption("ℹ️ Different analysis types were used. Comparison may be limited.")
    
    st.caption("⚠️ Comparison based on available data. Missing information is not estimated.")


# ===== PAGE ROUTER =====
PAGES = {
    "Dashboard": page_dashboard,
    "New Analysis": page_new_analysis,
    "History": page_history,
    "History Detail": page_history_detail,
    "Compare": page_compare,
}

# ===== MAIN =====
render_sidebar()

current_page = st.session_state.page
if current_page in PAGES:
    PAGES[current_page]()
else:
    page_dashboard()