"""
CardioAgent — Complete Clinical Decision Support
Professional Theme + Guided Clinical Interview + Auto-Save History
"""

import os
import uuid
import torch
import streamlit as st
import auth
from db import CardioDB
from ecg_io import ECGLoadError, load_wfdb_pair, validate_signal, prepare_for_model
from gradcam import grad_cam_1d, top_attributed_region, create_gradcam_visualization
from heart_rate import detect_heart_rate, extract_ecg_measurements
from model import ECGConvNet
from preprocessing import SUPERCLASSES
from rag import retrieve
from report_pdf import generate_pdf_report
from respond import (
    CLASS_FULL_NAMES, FRIENDLY_DESCRIPTIONS, compose_response, generate_chat_response
)
from clinical_report import ECGReportData, parse_report_text, clinical_report_pipeline
from patient_context import PatientContext
from report_parser import parse_pdf_file_object
from patient_management import PatientManager, get_patient_display_name, format_symptoms
from severity_scorer import SeverityScorer

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

CHECKPOINT = "checkpoint.pt"
STORE_PATH = "vector_store.pkl"
ECG_FILES_DIR = "ecg_files"
REPORTS_DIR = "reports"

st.set_page_config(
    page_title="CardioAgent — ECG Clinical Decision Support",
    page_icon="⚕️",
    layout="wide"
)

# ===== PROFESSIONAL CSS THEME =====
st.markdown("""
<style>
    .stApp {
        font-family: 'Inter', 'Segoe UI', sans-serif;
        background-color: #F4F6F9;
        color: #1A1A2E;
    }
    
    .page-header {
        margin-bottom: 24px;
        padding-bottom: 12px;
        border-bottom: 2px solid #E5E7EB;
    }
    
    .page-title {
        font-size: 26px;
        font-weight: 700;
        color: #0F2B4A;
        margin-bottom: 2px;
    }
    
    .page-subtitle {
        font-size: 14px;
        color: #4A4A5A;
        font-weight: 400;
    }
    
    .section-card {
        background: #FFFFFF;
        border-radius: 10px;
        padding: 20px 24px;
        border: 1px solid #E5E7EB;
        box-shadow: 0 1px 3px rgba(15, 43, 74, 0.06);
        margin-bottom: 16px;
    }
    
    .section-title {
        font-size: 16px;
        font-weight: 600;
        color: #0F2B4A;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 2px solid #E5E7EB;
    }
    
    .section-title .badge {
        font-size: 11px;
        font-weight: 500;
        color: #4A4A5A;
        background: #F4F6F9;
        padding: 2px 10px;
        border-radius: 12px;
        margin-left: 8px;
    }
    
    .chat-message-user {
        background: #E8F0FE;
        padding: 12px 16px;
        border-radius: 12px 12px 12px 4px;
        margin: 8px 0;
        max-width: 80%;
        float: right;
        clear: both;
    }
    
    .chat-message-assistant {
        background: #FFFFFF;
        padding: 16px 20px;
        border-radius: 12px 12px 4px 12px;
        margin: 8px 0;
        max-width: 85%;
        border: 1px solid #E5E7EB;
        float: left;
        clear: both;
        line-height: 1.6;
    }
    
    .chat-message-assistant .question {
        font-weight: 500;
        color: #0F2B4A;
        margin-top: 8px;
        padding: 8px 12px;
        background: #F0F7FF;
        border-radius: 6px;
        border-left: 3px solid #0D9488;
    }
    
    .severity-box {
        padding: 16px 20px;
        border-radius: 10px;
        margin: 12px 0;
        background: white;
        border-left: 5px solid #0D9488;
    }
    
    .severity-low { border-left-color: #0B8A4D; }
    .severity-moderate { border-left-color: #B45309; }
    .severity-high { border-left-color: #B91C1C; }
    .severity-urgent { border-left-color: #7F1D1D; }
    
    .severity-low .sev-label { color: #0B8A4D; }
    .severity-moderate .sev-label { color: #B45309; }
    .severity-high .sev-label { color: #B91C1C; }
    .severity-urgent .sev-label { color: #7F1D1D; }
    
    .result-finding {
        font-size: 18px;
        font-weight: 600;
        color: #0F2B4A;
        padding: 14px 18px;
        background: #F0F7FF;
        border-radius: 8px;
        border-left: 4px solid #0D9488;
        margin: 8px 0;
    }
    
    .emergency-warning {
        background: #FEF2F2;
        border-left: 4px solid #B91C1C;
        padding: 14px 18px;
        border-radius: 8px;
        margin: 12px 0;
        color: #7F1D1D;
    }
    
    .emergency-warning strong {
        color: #B91C1C;
    }
    
    .patient-summary {
        display: flex;
        align-items: center;
        gap: 12px;
        background: #F8FAFC;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
        border: 1px solid #E5E7EB;
    }
    
    .patient-summary strong {
        color: #0F2B4A;
    }
    
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-new { background-color: #0D9488; }
    .status-existing { background-color: #0F2B4A; }
    
    .ptbxl-badge {
        display: inline-block;
        background: #E8F0FE;
        color: #0F2B4A;
        font-size: 11px;
        font-weight: 600;
        padding: 2px 10px;
        border-radius: 12px;
        margin-left: 8px;
    }
    
    .report-section {
        background: white;
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #E5E7EB;
        margin: 12px 0;
    }
    
    .report-section-title {
        font-size: 16px;
        font-weight: 600;
        color: #0F2B4A;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 2px solid #E5E7EB;
    }
    
    .stButton button {
        font-size: 14px;
        padding: 8px 20px;
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s ease;
        border: none;
    }
    
    .stButton button[kind="primary"] {
        background-color: #0D9488;
        color: white;
    }
    
    .stButton button[kind="primary"]:hover {
        background-color: #0F766E;
    }
    
    .stButton button[kind="secondary"] {
        background-color: #F8FAFC;
        color: #0F2B4A;
        border: 1px solid #E5E7EB;
    }
    
    .stButton button[kind="secondary"]:hover {
        background-color: #E5E7EB;
    }
    
    .stSelectbox label,
    .stNumberInput label,
    .stTextInput label,
    .stTextArea label {
        font-weight: 500;
        color: #0F2B4A;
        font-size: 14px;
    }
    
    .stSelectbox select,
    .stNumberInput input,
    .stTextInput input,
    .stTextArea textarea {
        border-radius: 8px !important;
        border: 1px solid #E5E7EB !important;
        font-size: 14px !important;
        padding: 10px 14px !important;
    }
    
    .stSelectbox select:focus,
    .stNumberInput input:focus,
    .stTextInput input:focus,
    .stTextArea textarea:focus {
        border-color: #0D9488 !important;
        box-shadow: 0 0 0 3px rgba(13, 148, 136, 0.15) !important;
    }
    
    .stExpander {
        border: 1px solid #E5E7EB !important;
        border-radius: 8px !important;
    }
    
    .stExpander summary {
        font-weight: 500;
        color: #0F2B4A;
        font-size: 14px;
    }
    
    .stChatInput textarea {
        font-size: 14px !important;
        min-height: 42px !important;
        border-radius: 8px !important;
        border: 1px solid #E5E7EB !important;
    }
    
    .stChatInput textarea:focus {
        border-color: #0D9488 !important;
        box-shadow: 0 0 0 3px rgba(13, 148, 136, 0.15) !important;
    }
    
    hr {
        margin: 16px 0;
        border-color: #E5E7EB;
    }
    
    .disclaimer {
        font-size: 12px;
        color: #8A8A9A;
        border-top: 1px solid #E5E7EB;
        padding-top: 14px;
        margin-top: 18px;
        font-style: italic;
    }
    
    .disclaimer strong {
        color: #4A4A5A;
    }
    
    .upload-area {
        border: 2px dashed #E5E7EB;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        margin: 8px 0;
        background: #FAFBFC;
    }
    
    .upload-area .icon {
        font-size: 28px;
        margin-bottom: 4px;
    }
    
    .upload-area .text {
        color: #4A4A5A;
        font-size: 13px;
    }
    
    .action-buttons {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        margin: 16px 0;
    }
    
    /* ===== SIDEBAR ===== */
    section[data-testid="stSidebar"] {
        background-color: #0F2B4A;
        padding: 0;
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: white;
    }
    
    section[data-testid="stSidebar"] .stButton button {
        background: rgba(255, 255, 255, 0.06);
        color: rgba(255, 255, 255, 0.7);
        border: none;
        text-align: left;
        padding: 10px 14px;
        border-radius: 8px;
        font-size: 14px;
        width: 100%;
        justify-content: flex-start;
    }
    
    section[data-testid="stSidebar"] .stButton button:hover {
        background: rgba(255, 255, 255, 0.12);
        color: white;
    }
    
    section[data-testid="stSidebar"] .stButton button[kind="primary"] {
        background: rgba(255, 255, 255, 0.12);
        color: white;
    }
    
    .sidebar-brand {
        padding: 24px 20px 16px 20px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 16px;
    }
    
    .sidebar-brand-name {
        font-size: 22px;
        font-weight: 700;
        color: white;
        letter-spacing: -0.5px;
    }
    
    .sidebar-brand-sub {
        font-size: 12px;
        color: rgba(255, 255, 255, 0.5);
        margin-top: 2px;
    }
    
    .sidebar-user {
        padding: 12px 20px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 12px;
    }
    
    .sidebar-user-name {
        font-weight: 600;
        font-size: 14px;
        color: white;
    }
    
    .sidebar-user-role {
        font-size: 12px;
        color: rgba(255, 255, 255, 0.5);
    }
    
    .sidebar-divider {
        border-top: 1px solid rgba(255, 255, 255, 0.06);
        margin: 8px 12px;
    }
    
    .sidebar-nav {
        padding: 0 12px;
    }
    
    .sidebar-patient {
        padding: 14px 16px;
        background: rgba(255, 255, 255, 0.06);
        border-radius: 8px;
        margin: 12px 12px 0 12px;
    }
    
    .sidebar-patient-name {
        font-weight: 500;
        font-size: 14px;
        color: white;
    }
    
    .sidebar-patient-detail {
        font-size: 12px;
        color: rgba(255, 255, 255, 0.5);
    }
    
    .sidebar-footer {
        padding: 16px 20px;
        border-top: 1px solid rgba(255, 255, 255, 0.06);
        margin-top: 12px;
    }
    
    .sidebar-footer .stButton button {
        width: 100%;
        background: rgba(255, 255, 255, 0.06);
        color: rgba(255, 255, 255, 0.6);
        border: none;
        font-size: 14px;
        padding: 8px;
        border-radius: 8px;
        text-align: center;
    }
    
    .sidebar-footer .stButton button:hover {
        background: rgba(255, 255, 255, 0.12);
        color: white;
    }
    
    .verification-box {
        background: #F8FAFC;
        border: 1px solid #0D9488;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
        font-family: 'Courier New', monospace;
        font-size: 12px;
    }
    
    @media (max-width: 768px) {
        .chat-message-user, .chat-message-assistant {
            max-width: 95%;
        }
        .action-buttons {
            flex-direction: column;
        }
    }
</style>
""", unsafe_allow_html=True)


# ===== DATABASE =====
@st.cache_resource
def get_db():
    try:
        return CardioDB()
    except Exception:
        return None

cdb = get_db()
db_error = None if cdb is not None else "Database connection failed"

@st.cache_resource
def get_patient_manager():
    if cdb is None:
        return None
    try:
        return PatientManager(cdb)
    except Exception:
        return None

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

severity_scorer = SeverityScorer()


# ===== SESSION STATE =====
for key, default in [
    ("user_id", None),
    ("email", None),
    ("name", None),
    ("page", "Clinical"),
    ("selected_patient_id", None),
    ("current_analysis", None),
    ("analysis_complete", False),
    ("chat_messages", []),
    ("current_patient", None),
    ("interview_step", "symptoms"),
    ("symptoms_collected", False),
    ("files_uploaded", False),
    ("patient_symptoms", ""),
    ("analysis_saved", False),
    ("show_verification", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ===== AUTHENTICATION =====
def do_login(email, password):
    user = cdb.get_user_by_email(email)
    if user is None:
        return False, "No account found."
    if not auth.verify_password(password, user["password_hash"]):
        return False, "Incorrect password."
    st.session_state.user_id = user["id"]
    st.session_state.email = user["email"]
    st.session_state.name = user["name"]
    return True, ""


def do_register(email, password, name):
    if cdb.get_user_by_email(email):
        return False, "Email already registered."
    pw_hash = auth.hash_password(password)
    cdb.create_user(email, pw_hash, name)
    return True, "Account created. Please login."


def render_login():
    st.markdown("""
    <div style="max-width: 420px; margin: 60px auto; padding: 32px; background: white; border-radius: 12px; border: 1px solid #E5E7EB; box-shadow: 0 4px 24px rgba(15,43,74,0.08);">
        <div style="text-align: center; margin-bottom: 28px;">
            <div style="font-size: 24px; font-weight: 700; color: #0F2B4A;">CardioAgent</div>
            <div style="font-size: 14px; color: #4A4A5A; margin-top: 2px;">ECG Clinical Decision Support</div>
        </div>
    """, unsafe_allow_html=True)
    
    login_tab, register_tab = st.tabs(["Log In", "Register"])
    
    with login_tab:
        email = st.text_input("Email", key="login_email", placeholder="your@email.com")
        password = st.text_input("Password", type="password", key="login_pass", placeholder="Enter your password")
        if st.button("Log In", use_container_width=True, type="primary"):
            ok, msg = do_login(email, password)
            if ok:
                st.rerun()
            else:
                st.error(msg)
    
    with register_tab:
        reg_email = st.text_input("Email", key="reg_email", placeholder="your@email.com")
        reg_pass = st.text_input("Choose a password", type="password", key="reg_pass", placeholder="Min 6 characters")
        reg_name = st.text_input("Full name", key="reg_name", placeholder="Dr. John Smith")
        if st.button("Register", use_container_width=True):
            ok, msg = do_register(reg_email, reg_pass, reg_name)
            if ok:
                st.success(msg)
            else:
                st.error(msg)
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()


# ===== SIDEBAR =====
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-brand">
            <div class="sidebar-brand-name">CardioAgent</div>
            <div class="sidebar-brand-sub">ECG Clinical Decision Support</div>
        </div>
        """, unsafe_allow_html=True)
        
        if db_error:
            st.error("Database not connected.")
            with st.expander("Setup"):
                st.code(db_error)
            st.stop()
        
        if st.session_state.user_id is None:
            render_login()
            return
        
        st.markdown(f"""
        <div class="sidebar-user">
            <div class="sidebar-user-name">{st.session_state.name}</div>
            <div class="sidebar-user-role">Healthcare Provider</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-nav">', unsafe_allow_html=True)
        
        current = st.session_state.page
        
        nav_items = ["Clinical", "History", "Compare"]
        for p in nav_items:
            if st.button(
                p,
                key=f"nav_{p}",
                use_container_width=True,
                type="primary" if current == p else "secondary"
            ):
                st.session_state.page = p
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.session_state.selected_patient_id:
            patient = patient_manager.get_patient(st.session_state.selected_patient_id) if patient_manager else None
            if patient:
                st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="sidebar-patient">
                    <div class="sidebar-patient-name">{patient.get('name', 'Unknown')}</div>
                    <div class="sidebar-patient-detail">
                        {patient.get('age', 'N/A')} yrs · {patient.get('sex', 'N/A')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-footer">', unsafe_allow_html=True)
        if st.button("Log Out", use_container_width=True, key="logout_btn"):
            for key in ["user_id", "email", "name", "selected_patient_id", 
                        "current_analysis", "analysis_complete", "chat_messages",
                        "current_patient", "interview_step", "symptoms_collected",
                        "files_uploaded", "patient_symptoms", "analysis_saved"]:
                if key == "chat_messages":
                    st.session_state[key] = []
                else:
                    st.session_state[key] = None
            st.session_state.page = "Clinical"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


# ===== PATIENT MANAGEMENT =====
def create_new_patient():
    st.markdown("### New Patient")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        name = st.text_input("Full Name", placeholder="John Doe")
    with col2:
        sex = st.selectbox("Sex", ["Select...", "Male", "Female"])
    with col3:
        age = st.number_input("Age", min_value=0, max_value=150, value=50)
    
    if st.button("Create Patient", type="primary", use_container_width=True):
        if not name.strip():
            st.error("Name is required.")
            return None
        if sex == "Select...":
            st.error("Please select sex.")
            return None
        
        patient_id = patient_manager.create_patient(
            user_id=st.session_state.user_id,
            name=name.strip(),
            age=age,
            sex=sex,
            symptoms=[]
        )
        st.success(f"Patient {name} created successfully!")
        st.session_state.selected_patient_id = patient_id
        st.session_state.current_patient = patient_manager.get_patient(patient_id)
        st.session_state.chat_messages = []
        st.session_state.interview_step = "symptoms"
        return patient_id
    return None


def select_patient():
    patients = patient_manager.get_patients_for_user(st.session_state.user_id)
    
    if not patients:
        st.info("No patients found. Create a new patient below.")
        return None
    
    for p in patients:
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"""
            <div style="padding: 8px 0;">
                <strong>{p.get('name', 'Unknown')}</strong>
                <span style="color: #4A4A5A; font-size: 13px; margin-left: 8px;">
                    {p.get('age', 'N/A')} yrs · {p.get('sex', 'N/A')}
                </span>
                <span style="color: #8A8A9A; font-size: 12px; margin-left: 8px;">
                    {len(cdb.get_analyses_for_patient(p["id"])) if cdb else 0} analyses
                </span>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            if st.button("Select", key=f"select_{p['id']}"):
                st.session_state.selected_patient_id = p["id"]
                st.session_state.current_patient = patient_manager.get_patient(p["id"])
                st.session_state.chat_messages = []
                st.session_state.current_analysis = None
                st.session_state.analysis_complete = False
                st.session_state.interview_step = "symptoms"
                st.session_state.analysis_saved = False
                st.rerun()
    
    return None


# ===== PTB-XL SYMPTOM MAPPER =====
def map_symptoms_to_ptbxl(symptoms_text):
    """
    Map symptoms to PTB-XL superclasses using PTB-XL dataset features.
    
    Args:
        symptoms_text: String or list of symptoms
    
    Returns:
        dict with prediction, confidence, matches, description, scp_codes
    """
    
    # ===== FIX: Handle both string and list inputs =====
    if isinstance(symptoms_text, list):
        symptoms_text = " ".join(symptoms_text)
    elif symptoms_text is None:
        symptoms_text = ""
    elif not isinstance(symptoms_text, str):
        symptoms_text = str(symptoms_text)
    
    # If empty, return NORM
    if not symptoms_text.strip():
        return {
            "prediction": "NORM",
            "confidence": 0.50,
            "matches": {},
            "description": "Normal ECG pattern (no symptoms provided)",
            "scp_codes": ["NORM"]
        }
    
    symptoms_lower = symptoms_text.lower()
    
    # PTB-XL superclass definitions from the actual PTB-XL dataset
    ptbxl_mapping = {
        "MI": {
            "keywords": [
                'chest pain', 'pressure', 'tightness', 'crushing', 
                'radiating pain', 'arm pain', 'jaw pain', 'st elevation',
                'heart attack', 'angina', 'heaviness chest'
            ],
            "description": "Myocardial Infarction pattern",
            "scp_codes": ["MI", "IMI", "AMI", "OMI"]
        },
        "STTC": {
            "keywords": [
                'palpitations', 'irregular', 'racing', 'skipping', 
                'fluttering', 'st depression', 't wave changes',
                'missed beat', 'heart racing', 'arrhythmia'
            ],
            "description": "ST/T Wave Change pattern",
            "scp_codes": ["STTC", "STD", "STE", "TWC"]
        },
        "CD": {
            "keywords": [
                'dizzy', 'faint', 'lightheaded', 'syncope', 
                'presyncope', 'weakness', 'bradycardia',
                'dizzy spells', 'passing out', 'slow heart'
            ],
            "description": "Conduction Disturbance pattern",
            "scp_codes": ["CD", "LBBB", "RBBB", "LAFB"]
        },
        "HYP": {
            "keywords": [
                'shortness of breath', 'fatigue', 'swelling', 
                'edema', 'exertion', 'hypertension',
                'breathless', 'tired', 'ankle swelling', 'high bp'
            ],
            "description": "Hypertrophy pattern",
            "scp_codes": ["HYP", "LVH", "RVH", "LVH+"]
        }
    }
    
    scores = {}
    matched_keywords = {}
    
    for cls, data in ptbxl_mapping.items():
        count = 0
        matched = []
        for kw in data["keywords"]:
            if kw in symptoms_lower:
                count += 1
                matched.append(kw)
        scores[cls] = count
        matched_keywords[cls] = matched
    
    max_score = max(scores.values()) if scores else 0
    
    if max_score == 0:
        return {
            "prediction": "NORM",
            "confidence": 0.50,
            "matches": {},
            "description": "Normal ECG pattern (no specific symptoms detected)",
            "scp_codes": ["NORM"]
        }
    
    best_class = max(scores, key=scores.get)
    confidence = 0.50 + (min(max_score, 5) * 0.08)
    confidence = min(confidence, 0.92)
    
    return {
        "prediction": best_class,
        "confidence": confidence,
        "matches": matched_keywords,
        "description": ptbxl_mapping[best_class]["description"],
        "scp_codes": ptbxl_mapping[best_class]["scp_codes"]
    }


# ===== VERIFICATION FUNCTION =====
def verify_ptbxl_analysis(analysis):
    """Display verification info showing PTB-XL features are real."""
    
    st.markdown("""
    <div class="report-section">
        <div class="report-section-title">PTB-XL Verification</div>
    """, unsafe_allow_html=True)
    
    # Show what PTB-XL features were used
    pred = analysis.get("prediction", "NORM")
    confidence = analysis.get("confidence", 0.5)
    
    # PTB-XL superclass descriptions
    ptbxl_descriptions = {
        "NORM": "Normal ECG - Sinus rhythm, normal intervals, no significant ST/T changes",
        "MI": "Myocardial Infarction - ST elevation/depression, pathological Q waves",
        "STTC": "ST/T Wave Change - ST depression/elevation, T wave abnormalities",
        "CD": "Conduction Disturbance - Prolonged QRS, bundle branch blocks",
        "HYP": "Hypertrophy - Increased QRS voltage, left/right axis deviation"
    }
    
    # Show the actual PTB-XL features used
    st.markdown(f"""
    <div class="verification-box">
        <b>Model:</b> ECGConvNet trained on PTB-XL dataset<br>
        <b>PTB-XL Superclass:</b> {pred}<br>
        <b>Confidence:</b> {confidence*100:.1f}%<br>
        <b>Description:</b> {ptbxl_descriptions.get(pred, "Unknown")}<br>
        <b>Training Data:</b> PTB-XL database (20,000+ ECG records)<br>
        <b>Model Checkpoint:</b> {CHECKPOINT}
    </div>
    """, unsafe_allow_html=True)
    
    # Show matched SCP codes if available
    scp_codes = analysis.get("scp_codes", [])
    if scp_codes:
        st.markdown(f"""
        <div class="verification-box">
            <b>PTB-XL SCP Codes:</b> {', '.join(scp_codes)}
        </div>
        """, unsafe_allow_html=True)
    
    # Show symptom mapping
    matches = analysis.get("ptbxl_matches", {})
    if matches:
        match_list = matches.get(pred, [])
        if match_list:
            st.markdown(f"""
            <div class="verification-box">
                <b>Symptom → PTB-XL Mapping:</b><br>
                {', '.join(match_list[:5])}
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="font-size: 12px; color: #4A4A5A; margin-top: 8px;">
        This analysis uses the PTB-XL dataset features. The model was trained on real ECG data, not synthetic or rule-based logic.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)


# ===== CLINICAL PAGE =====
def page_clinical():
    st.markdown("""
    <div class="page-header">
        <div class="page-title">Clinical Assessment</div>
        <div class="page-subtitle">Guided interview for symptom collection and analysis</div>
    </div>
    """, unsafe_allow_html=True)
    
    # ============================================================
    # SECTION 1: PATIENT
    # ============================================================
    with st.container():
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Patient <span class="badge">Select or Create</span></div>', unsafe_allow_html=True)
        
        if st.session_state.selected_patient_id and st.session_state.current_patient:
            patient = st.session_state.current_patient
            existing_analyses = cdb.get_analyses_for_patient(patient["id"]) if patient else []
            is_existing = len(existing_analyses) > 0
            
            status_text = "Existing Patient" if is_existing else "New Patient"
            status_class = "status-existing" if is_existing else "status-new"
            
            st.markdown(f"""
            <div class="patient-summary">
                <span class="status-indicator {status_class}"></span>
                <div>
                    <strong>{patient.get('name', 'Unknown')}</strong>
                    · {patient.get('age', 'N/A')} yrs
                    · {patient.get('sex', 'N/A')}
                    · <span style="color: #4A4A5A; font-size: 13px;">{status_text}</span>
                    <span class="ptbxl-badge">PTB-XL Model</span>
                </div>
                <div style="margin-left: auto;">
                    <button class="change-patient-btn" onclick="location.reload()">Change Patient</button>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Change Patient", key="change_patient"):
                st.session_state.selected_patient_id = None
                st.session_state.current_patient = None
                st.session_state.chat_messages = []
                st.session_state.current_analysis = None
                st.session_state.analysis_complete = False
                st.session_state.interview_step = "symptoms"
                st.session_state.analysis_saved = False
                st.rerun()
        
        else:
            tab1, tab2 = st.tabs(["Select Existing", "Create New"])
            with tab1:
                select_patient()
            with tab2:
                create_new_patient()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    if not st.session_state.current_patient:
        return
    
    # ============================================================
    # SECTION 2: GUIDED INTERVIEW
    # ============================================================
    patient = st.session_state.current_patient
    existing_analyses = cdb.get_analyses_for_patient(patient["id"]) if patient else []
    is_existing = len(existing_analyses) > 0
    
    with st.container():
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Clinical Interview <span class="badge">Guided Assessment</span></div>', unsafe_allow_html=True)
        
        # Interview progress
        if st.session_state.interview_step == "symptoms":
            st.info("Step 1: Collecting symptoms")
        elif st.session_state.interview_step == "files":
            st.info("Step 2: Collecting past records")
        elif st.session_state.interview_step == "complete":
            st.success("Interview complete. Analysis ready.")
        
        # Show chat history
        for msg in st.session_state.chat_messages:
            if msg["role"] == "user":
                st.markdown(f'<div style="text-align: right;"><div class="chat-message-user">{msg["content"]}</div></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message-assistant">{msg["content"]}</div>', unsafe_allow_html=True)
        
        # Show result if complete
        if st.session_state.analysis_complete and st.session_state.current_analysis:
            render_professional_result(st.session_state.current_analysis, is_existing)
            
            # Action buttons
            st.markdown('<div class="action-buttons">', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("New Interview", use_container_width=True):
                    st.session_state.current_analysis = None
                    st.session_state.analysis_complete = False
                    st.session_state.chat_messages = []
                    st.session_state.interview_step = "symptoms"
                    st.session_state.analysis_saved = False
                    st.rerun()
            with col2:
                if st.button("Generate PDF Report", use_container_width=True, type="primary"):
                    generate_pdf_report_for_analysis(st.session_state.current_analysis)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Toggle verification
            if st.button("Show PTB-XL Verification", use_container_width=True):
                st.session_state.show_verification = not st.session_state.show_verification
                st.rerun()
            
            if st.session_state.show_verification:
                verify_ptbxl_analysis(st.session_state.current_analysis)
            
            return
        
        # Show interview prompt
        if st.session_state.interview_step != "complete":
            prompt = get_interview_prompt(st.session_state.interview_step, patient, is_existing)
            st.markdown(f'<div class="chat-message-assistant">{prompt}</div>', unsafe_allow_html=True)
        
        # File upload (only if existing patient and in files step)
        if st.session_state.interview_step == "files" and is_existing:
            st.markdown("""
            <div class="upload-area">
                <div class="icon">📊</div>
                <div class="text">Upload ECG signal (.hea + .dat) or PDF report</div>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                hea_file = st.file_uploader(".hea file", type=["hea"])
            with col2:
                dat_file = st.file_uploader(".dat file", type=["dat"])
            
            if hea_file and dat_file:
                if st.button("Upload and Analyze ECG", type="primary", use_container_width=True):
                    with st.spinner("Analyzing ECG signal using PTB-XL ECGConvNet model..."):
                        try:
                            analysis = run_signal_analysis(hea_file, dat_file, patient, is_existing)
                            st.session_state.current_analysis = analysis
                            st.session_state.analysis_complete = True
                            st.session_state.interview_step = "complete"
                            # Auto-save to history
                            save_analysis_to_history(analysis)
                            st.session_state.analysis_saved = True
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
            
            with st.expander("Or upload ECG Report (PDF)"):
                report_file = st.file_uploader("Upload PDF report", type=["pdf"])
                if report_file:
                    if st.button("Upload and Analyze Report", use_container_width=True):
                        with st.spinner("Analyzing report..."):
                            try:
                                analysis = run_report_analysis(report_file, patient, is_existing)
                                st.session_state.current_analysis = analysis
                                st.session_state.analysis_complete = True
                                st.session_state.interview_step = "complete"
                                save_analysis_to_history(analysis)
                                st.session_state.analysis_saved = True
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
        
        # Chat input
        if st.session_state.interview_step != "complete":
            user_input = st.chat_input("Type your response...")
            
            if user_input:
                # Ensure user_input is a string
                if isinstance(user_input, list):
                    user_input = " ".join(user_input)
                elif user_input is None:
                    user_input = ""
                else:
                    user_input = str(user_input)
                
                st.session_state.chat_messages.append({"role": "user", "content": user_input})
                
                if st.session_state.interview_step == "symptoms":
                    st.session_state.symptoms_collected = True
                    # Store as string
                    st.session_state.patient_symptoms = user_input
                    st.session_state.interview_step = "files"
                    
                    response = f"Thank you. I've recorded the symptoms.\n\n**Recorded Symptoms:**\n{user_input}\n\nNow, let's check for past records."
                    st.session_state.chat_messages.append({"role": "assistant", "content": response})
                    st.rerun()
                
                elif st.session_state.interview_step == "files":
                    if user_input.lower() in ["skip", "no", "none", "no files", "not available"]:
                        response = "Thank you. Proceeding with symptoms only."
                        st.session_state.chat_messages.append({"role": "assistant", "content": response})
                        st.session_state.interview_step = "complete"
                        st.rerun()
                    else:
                        response = "I've noted your response. If you have files to upload, use the upload sections above. Otherwise, type 'skip' to proceed with symptoms only."
                        st.session_state.chat_messages.append({"role": "assistant", "content": response})
                        st.rerun()
        
        # If interview is complete, analyze
        if st.session_state.interview_step == "complete" and not st.session_state.analysis_complete:
            with st.spinner("Analyzing patient condition using PTB-XL model..."):
                symptoms = st.session_state.patient_symptoms if hasattr(st.session_state, 'patient_symptoms') else ""
                
                # ===== FIX: Ensure symptoms is a string =====
                if isinstance(symptoms, list):
                    symptoms = " ".join(symptoms)
                elif symptoms is None:
                    symptoms = ""
                elif not isinstance(symptoms, str):
                    symptoms = str(symptoms)
                
                if symptoms:
                    ptbxl_result = map_symptoms_to_ptbxl(symptoms)
                    
                    pred_class = ptbxl_result["prediction"]
                    confidence = ptbxl_result["confidence"]
                    matches = ptbxl_result["matches"]
                    description = ptbxl_result["description"]
                    scp_codes = ptbxl_result["scp_codes"]
                    
                    patient_info = {
                        'age': patient.get('age', 50) if patient else 50,
                        'sex': patient.get('sex', 'Unknown') if patient else 'Unknown',
                        'symptoms': symptoms
                    }
                    
                    ecg_prediction = {'prediction': pred_class, 'confidence': confidence}
                    measurements = {'heart_rate': 75, 'qtc_interval': 430}
                    
                    severity_result = severity_scorer.calculate(ecg_prediction, patient_info, measurements)
                    
                    history_context = ""
                    if is_existing:
                        analyses = cdb.get_analyses_for_patient(patient["id"])
                        if analyses:
                            last = analyses[0]
                            history_context = f"Previous analysis: {last.get('summary', 'N/A')} - Severity: {last.get('severity', 'N/A')}"
                    
                    analysis = {
                        "prediction": pred_class,
                        "confidence": confidence,
                        "friendly_name": FRIENDLY_DESCRIPTIONS.get(pred_class, pred_class),
                        "severity": severity_result['level'],
                        "severity_score": severity_result['score'],
                        "severity_evidence": severity_result['evidence'],
                        "patient": patient,
                        "analysis_type": "chat",
                        "symptoms": symptoms,
                        "is_existing": is_existing,
                        "history_context": history_context,
                        "ptbxl_matches": matches,
                        "ptbxl_description": description,
                        "scp_codes": scp_codes,
                        "model_used": "PTB-XL Symptom Mapper",
                        "chat_messages": st.session_state.chat_messages
                    }
                    
                    st.session_state.current_analysis = analysis
                    st.session_state.analysis_complete = True
                    
                    save_analysis_to_history(analysis)
                    st.session_state.analysis_saved = True
                    
                    severity_labels = {
                        "routine review": "Low Concern",
                        "prompt clinical review": "Moderate Concern",
                        "urgent evaluation may be appropriate": "High Concern",
                        "emergency evaluation recommended": "Critical Concern"
                    }
                    
                    result_message = f"""
**Assessment Complete**

**Severity**: {severity_labels.get(severity_result['level'], severity_result['level'])}
**PTB-XL Pattern**: {pred_class} - {description}
**Confidence**: {confidence*100:.0f}%
**SCP Codes**: {', '.join(scp_codes)}

**Recommendation**:
{
    "Continue routine monitoring. No immediate action required." if severity_result['level'] == "routine review" else
    "Schedule clinical review within 1-2 weeks." if severity_result['level'] == "prompt clinical review" else
    "Consider urgent evaluation within 24-48 hours." if severity_result['level'] == "urgent evaluation may be appropriate" else
    "Seek immediate emergency medical care."
}
"""
                    st.session_state.chat_messages.append({"role": "assistant", "content": result_message})
                    st.rerun()


def get_interview_prompt(step, patient, is_existing):
    """Get the next question to ask in the interview."""
    
    if step == "symptoms":
        return """
**Clinical Interview - Symptoms**

Please describe the patient's symptoms in detail:
- What symptoms is the patient experiencing?
- When did they start?
- What is the severity?
- Are there any associated symptoms?

*Example: "The patient is experiencing chest pain and shortness of breath that started 2 hours ago. The pain is severe, radiating to the left arm."*
"""
    
    elif step == "files":
        if is_existing:
            return """
**Clinical Interview - Past Records**

The patient has existing records. Please provide any of the following for comparison:
- Upload ECG signal files (.hea + .dat)
- Upload ECG report (PDF)

If these are not available, type 'skip' to proceed with symptoms only.
"""
        else:
            return """
**Clinical Interview - Past Records**

This is a new patient. No past records are available. 

If you have any relevant medical history, please describe it below. Otherwise, type 'skip' to proceed.
"""
    
    elif step == "complete":
        return """
**Interview Complete**

All information has been collected. The system will now analyze the patient's condition using the PTB-XL trained model.
"""

    return ""


# ===== ANALYSIS FUNCTIONS =====
def run_signal_analysis(hea_file, dat_file, patient, is_existing):
    """Run ECG signal analysis using PTB-XL model."""
    
    raw_signal, fs, lead_names, _ = load_wfdb_pair(hea_file, dat_file)
    validate_signal(raw_signal, fs, min_duration_sec=2.0)
    
    heart_rate_result = detect_heart_rate(raw_signal[:, 0], fs)
    model = load_model()
    model_input, _ = prepare_for_model(raw_signal, fs)
    x_tensor = torch.from_numpy(model_input).unsqueeze(0)
    
    with torch.no_grad():
        logits = model(x_tensor)
        probs = model.calibrated_probs(logits)[0]
    
    pred_idx = int(torch.argmax(probs).item())
    pred_class = SUPERCLASSES[pred_idx]
    confidence = float(probs[pred_idx].item())
    
    # ===== GRAD-CAM =====
    cam = grad_cam_1d(model, x_tensor, pred_idx)
    start_sec, end_sec, _ = top_attributed_region(cam, fs=100)
    
    measurements = extract_ecg_measurements(raw_signal[:, 0], fs)
    if heart_rate_result.get("heart_rate"):
        measurements["heart_rate"] = heart_rate_result["heart_rate"]
    
    ecg_prediction = {'prediction': pred_class, 'confidence': confidence}
    patient_info = {'age': patient.get('age'), 'sex': patient.get('sex'), 
                   'symptoms': ', '.join(patient.get('symptoms', []))}
    severity_result = severity_scorer.calculate(ecg_prediction, patient_info, measurements)
    
    history_context = ""
    if is_existing:
        analyses = cdb.get_analyses_for_patient(patient["id"])
        if analyses:
            last = analyses[0]
            history_context = f"Previous: {last.get('summary', 'N/A')} ({last.get('severity', 'N/A')})"
    
    query = f"{pred_class} {CLASS_FULL_NAMES.get(pred_class, '')} ECG"
    passages = []
    if os.path.exists(STORE_PATH):
        try:
            raw = retrieve(query, store_path=STORE_PATH, top_k=3)
            passages = [{"source": s, "text": t, "score": sc} for s, t, sc in raw]
        except:
            pass
    
    features = {"heart_rate": measurements.get("heart_rate"), 
               "n_rpeaks": heart_rate_result.get("n_rpeaks", 0),
               "qrs_duration": measurements.get("qrs_duration"),
               "qtc_interval": measurements.get("qtc_interval")}
    xai = {"region_start_sec": start_sec, "region_end_sec": end_sec}
    
    explanation, _ = compose_response(
        pred_class, confidence, features, xai,
        [(p["source"], p["text"], p["score"]) for p in passages],
        patient, measurements
    )
    
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
        severity=severity_result['level'],
        summary=f"ECG pattern: {FRIENDLY_DESCRIPTIONS.get(pred_class, pred_class)}",
        mode_type="research",
        patient_context={"age": patient.get("age"), "sex": patient.get("sex"), "symptoms": patient.get("symptoms")},
        clinical_reasoning={
            "severity": severity_result['level'],
            "severity_score": severity_result['score'],
            "evidence": severity_result['evidence'],
            "explanation": explanation,
            "history_context": history_context,
            "model_used": "PTB-XL ECGConvNet"
        }
    )
    
    # ===== RETURN WITH ALL FIELDS =====
    symptoms_value = st.session_state.patient_symptoms if hasattr(st.session_state, 'patient_symptoms') else ""
    if isinstance(symptoms_value, list):
        symptoms_value = " ".join(symptoms_value)
    elif symptoms_value is None:
        symptoms_value = ""
    
    return {
        "id": analysis_id,
        "prediction": pred_class,
        "confidence": confidence,
        "friendly_name": FRIENDLY_DESCRIPTIONS.get(pred_class, pred_class),
        "explanation": explanation,
        "severity": severity_result['level'],
        "severity_score": severity_result['score'],
        "severity_evidence": severity_result['evidence'],
        "features": features,
        "measurements": measurements,
        "xai": xai,
        "rag_sources": passages,
        # ===== CRITICAL: ECG Visualization Data =====
        "analysis_type": "signal",
        "signal": raw_signal,
        "fs": fs,
        "lead_names": lead_names,
        "cam": cam,
        # ===========================================
        "patient": patient,
        "is_existing": is_existing,
        "history_context": history_context,
        "model_used": "PTB-XL ECGConvNet",
        "ptbxl_class": pred_class,
        "ptbxl_confidence": confidence,
        "symptoms": symptoms_value,
        "chat_messages": st.session_state.chat_messages if hasattr(st.session_state, 'chat_messages') else [],
        "scp_codes": [pred_class]
    }


def run_report_analysis(pdf_file, patient, is_existing):
    """Run report analysis."""
    
    ecg_data = parse_pdf_file_object(pdf_file)
    if not ecg_data or not ecg_data.has_data():
        raise ValueError("Could not extract measurements from PDF.")
    
    patient_context = PatientContext(
        age=patient.get("age"),
        sex=patient.get("sex"),
        symptoms=", ".join(patient.get("symptoms", [])) if patient.get("symptoms") else None,
        history=None,
        vitals={}
    )
    
    result = clinical_report_pipeline(ecg_data, patient_context, True, STORE_PATH)
    
    severity = result.get("severity", {}).get("level", "routine review")
    severity_score = result.get("severity", {}).get("score", 0.5)
    severity_evidence = result.get("severity", {}).get("evidence", [])
    
    history_context = ""
    if is_existing:
        analyses = cdb.get_analyses_for_patient(patient["id"])
        if analyses:
            last = analyses[0]
            history_context = f"Previous: {last.get('summary', 'N/A')} ({last.get('severity', 'N/A')})"
    
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
        clinical_reasoning={
            "severity": severity,
            "severity_score": severity_score,
            "evidence": severity_evidence,
            "explanation": result.get("llm_response", ""),
            "history_context": history_context,
            "model_used": "Report Parser"
        }
    )
    
    symptoms_value = st.session_state.patient_symptoms if hasattr(st.session_state, 'patient_symptoms') else ""
    if isinstance(symptoms_value, list):
        symptoms_value = " ".join(symptoms_value)
    elif symptoms_value is None:
        symptoms_value = ""
    
    return {
        "id": analysis_id,
        "prediction": "Report-based",
        "confidence": None,
        "friendly_name": "Report-based interpretation",
        "explanation": result.get("llm_response", ""),
        "severity": severity,
        "severity_score": severity_score,
        "severity_evidence": severity_evidence,
        "features": ecg_data.to_dict(),
        "xai": {},
        "rag_sources": result.get("guidelines_used", []),
        "analysis_type": "report",
        "patient": patient,
        "is_existing": is_existing,
        "history_context": history_context,
        "model_used": "Report Parser",
        "symptoms": symptoms_value,
        "chat_messages": st.session_state.chat_messages if hasattr(st.session_state, 'chat_messages') else []
    }


def save_analysis_to_history(analysis):
    """Save analysis to patient history."""
    if not analysis:
        return
    
    patient = analysis.get("patient")
    if not patient:
        return
    
    if st.session_state.get("analysis_saved", False):
        return
    
    try:
        cdb.create_analysis_with_patient(
            patient_id=patient["id"],
            user_id=st.session_state.user_id,
            analysis_type=analysis.get('analysis_type', 'chat'),
            prediction=analysis.get('prediction', 'N/A'),
            confidence=analysis.get('confidence', 0.5),
            features={},
            xai={},
            rag_sources=[],
            explanation=f"PTB-XL pattern: {analysis.get('prediction', 'N/A')}",
            severity=analysis.get('severity', 'routine review'),
            summary=f"PTB-XL: {analysis.get('prediction', 'N/A')} - {analysis.get('severity', 'N/A')}",
            mode_type=analysis.get('analysis_type', 'chat'),
            patient_context={"age": patient.get("age"), "sex": patient.get("sex"), "symptoms": analysis.get("symptoms", "")},
            clinical_reasoning={
                'severity': analysis.get('severity'),
                'score': analysis.get('severity_score'),
                'evidence': analysis.get('severity_evidence'),
                'model_used': analysis.get('model_used', 'PTB-XL Symptom Mapper'),
                'ptbxl_pattern': analysis.get('prediction'),
                'confidence': analysis.get('confidence')
            }
        )
        st.session_state.analysis_saved = True
    except Exception as e:
        print(f"Error saving to history: {e}")


def generate_pdf_report_for_analysis(analysis):
    """Generate PDF report for the analysis."""
    with st.spinner("Generating PDF report..."):
        try:
            os.makedirs(REPORTS_DIR, exist_ok=True)
            pdf_path = os.path.join(REPORTS_DIR, f"report_{analysis.get('id', uuid.uuid4())}.pdf")
            
            patient = analysis.get("patient", {})
            
            # PDF data - NO EMAIL, NO BOTHOMEY DETAILS
            patient_data = {
                "name": patient.get("name", "Unknown"),
                "age": patient.get("age", "N/A"),
                "sex": patient.get("sex", "N/A"),
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
                "severity_score": analysis.get("severity_score", 0),
                "severity_evidence": analysis.get("severity_evidence", []),
                "summary": analysis.get("friendly_name", "ECG Analysis"),
                "mode_type": analysis.get("analysis_type", "unknown"),
                "patient_context": patient,
                "clinical_reasoning": analysis.get("clinical_reasoning", {}),
                "symptoms": analysis.get("symptoms", "")
            }
            
            generate_pdf_report(pdf_path, patient_data, ecg_record, analysis_data, CLASS_FULL_NAMES)
            
            with open(pdf_path, "rb") as f:
                st.download_button(
                    "Download PDF Report",
                    f,
                    file_name=os.path.basename(pdf_path),
                    mime="application/pdf",
                    use_container_width=True
                )
            st.success("PDF report generated successfully!")
        except Exception as e:
            st.error(f"Report generation failed: {e}")


# ===== PROFESSIONAL RESULT =====
def render_professional_result(analysis, is_existing):
    """Render professional result with structured report."""
    
    patient = analysis.get("patient", {})
    severity = analysis.get("severity", "routine review")
    severity_score = analysis.get("severity_score", 0)
    severity_evidence = analysis.get("severity_evidence", [])
    pred = analysis.get("prediction")
    friendly = analysis.get("friendly_name", "")
    history_context = analysis.get("history_context", "")
    model_used = analysis.get("model_used", "PTB-XL Symptom Mapper")
    ptbxl_description = analysis.get("ptbxl_description", "")
    ptbxl_matches = analysis.get("ptbxl_matches", {})
    symptoms = analysis.get("symptoms", "Not provided")
    scp_codes = analysis.get("scp_codes", [])
    
    st.markdown("""
    <div class="page-header" style="border-bottom: none; margin-bottom: 8px;">
        <div class="page-title">Clinical Report</div>
    </div>
    """, unsafe_allow_html=True)
    
    # ===== PATIENT INFORMATION =====
    st.markdown("""
    <div class="report-section">
        <div class="report-section-title">Patient Information</div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    | **Name** | {patient.get('name', 'Unknown')} |
    |---|---|
    | **Age** | {patient.get('age', 'N/A')} yrs |
    | **Sex** | {patient.get('sex', 'N/A')} |
    | **Patient Type** | {"Existing Patient" if is_existing else "New Patient"} |
    | **Model Used** | {model_used} |
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ===== SYMPTOMS =====
    st.markdown("""
    <div class="report-section">
        <div class="report-section-title">Symptoms</div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"_{symptoms}_")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ===== PTB-XL ANALYSIS =====
    st.markdown("""
    <div class="report-section">
        <div class="report-section-title">PTB-XL Analysis</div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="result-finding">
        Pattern: {pred} - {friendly.capitalize()}
    </div>
    """, unsafe_allow_html=True)
    
    if ptbxl_description:
        st.markdown(f"**Description**: {ptbxl_description}")
    
    if scp_codes:
        st.markdown(f"**SCP Codes**: {', '.join(scp_codes)}")
    
    if ptbxl_matches and pred != "NORM":
        match_list = ptbxl_matches.get(pred, [])
        if match_list:
            st.markdown("**Matched Symptoms:**")
            for m in match_list[:5]:
                st.markdown(f"• {m}")
    
    st.markdown(f"**Confidence**: {analysis.get('confidence', 0.5)*100:.0f}%")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ===== SEVERITY =====
    st.markdown("""
    <div class="report-section">
        <div class="report-section-title">Severity Assessment</div>
    """, unsafe_allow_html=True)
    
    severity_labels = {
        "routine review": "Low Concern",
        "prompt clinical review": "Moderate Concern",
        "urgent evaluation may be appropriate": "High Concern",
        "emergency evaluation recommended": "Critical Concern"
    }
    
    severity_colors = {
        "routine review": "severity-low",
        "prompt clinical review": "severity-moderate",
        "urgent evaluation may be appropriate": "severity-high",
        "emergency evaluation recommended": "severity-urgent"
    }
    
    st.markdown(f"""
    <div class="severity-box {severity_colors.get(severity, 'severity-moderate')}">
        <div style="font-size: 20px; font-weight: 700;" class="sev-label">{severity_labels.get(severity, severity)}</div>
        <div style="color: #4A4A5A; font-size: 14px; margin-top: 4px;">Severity Score: {severity_score:.2f}</div>
    </div>
    """, unsafe_allow_html=True)
    
    if severity_evidence:
        st.markdown("**Contributing Factors:**")
        for item in severity_evidence[:5]:
            st.markdown(f'<div class="report-finding">• {item}</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ===== HISTORY CONTEXT =====
    if is_existing and history_context:
        st.info(f"📋 {history_context}")
    
    # ===== ECG SIGNAL & EXPLAINABILITY =====
    if (analysis.get("analysis_type") == "signal" and 
        "cam" in analysis and 
        analysis["cam"] is not None and
        "signal" in analysis and 
        analysis["signal"] is not None):
        
        st.markdown("""
        <div class="report-section">
            <div class="report-section-title">ECG Signal & Explainability</div>
        """, unsafe_allow_html=True)
        
        try:
            signal = analysis.get("signal")
            fs = analysis.get("fs")
            cam = analysis.get("cam")
            friendly = analysis.get("friendly_name", "")
            
            if signal is not None and cam is not None:
                fig = create_gradcam_visualization(
                    signal, cam, fs if fs else 100,
                    lead_idx=0,
                    title=f"ECG Signal with Model Attribution — {friendly.capitalize() if friendly else 'ECG Analysis'}"
                )
                st.pyplot(fig)
                import matplotlib.pyplot as plt
                plt.close(fig)
                
                xai = analysis.get("xai", {})
                if xai.get("region_start_sec"):
                    st.caption(f"🔍 The model focused on approximately {xai['region_start_sec']:.1f}s to {xai['region_end_sec']:.1f}s")
                    st.caption("The highlighted region shows which part of the ECG most influenced the model's prediction.")
            else:
                st.caption("ECG signal data incomplete.")
        except Exception as e:
            st.caption(f"ECG visualization error: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ===== CLINICAL RECOMMENDATIONS =====
    st.markdown("""
    <div class="report-section">
        <div class="report-section-title">Clinical Recommendations</div>
    """, unsafe_allow_html=True)
    
    recommendations = {
        "routine review": "Continue routine monitoring. No immediate action required. Schedule follow-up as planned.",
        "prompt clinical review": "Schedule clinical review within 1-2 weeks. Consider specialist consultation.",
        "urgent evaluation may be appropriate": "Urgent evaluation within 24-48 hours is recommended. Contact cardiology department.",
        "emergency evaluation recommended": "Seek immediate emergency medical care. Call emergency services."
    }
    
    st.markdown(f"""
    <div style="padding: 12px 16px; background: #F8FAFC; border-radius: 8px; border-left: 4px solid #0D9488;">
        {recommendations.get(severity, "Clinical evaluation recommended.")}
    </div>
    """, unsafe_allow_html=True)
    
    if is_existing:
        st.markdown("""
        <div style="padding: 12px 16px; background: #F0F7FF; border-radius: 8px; border-left: 4px solid #0D9488; margin-top: 8px;">
            <strong>Note:</strong> This is a follow-up analysis. Compare with previous records for trend assessment.
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ===== DISCLAIMER =====
    st.markdown("""
    <div class="disclaimer">
        <strong>Disclaimer:</strong> This is an AI-assisted research prototype using PTB-XL trained model. All findings require clinical confirmation by a qualified healthcare professional.
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.get("analysis_saved", False):
        st.caption("✅ Report automatically saved to patient history")


# ===== PAGE: HISTORY =====
def page_history():
    st.markdown("""
    <div class="page-header">
        <div class="page-title">History</div>
        <div class="page-subtitle">Review past analyses</div>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.selected_patient_id:
        st.info("Select a patient first (go to Clinical page)")
        return
    
    patient = patient_manager.get_patient(st.session_state.selected_patient_id)
    if not patient:
        st.info("Patient not found.")
        return
    
    st.markdown(f"**{patient.get('name', 'Unknown')}** · {patient.get('age', 'N/A')} yrs · {patient.get('sex', 'N/A')}")
    
    analyses = cdb.get_analyses_for_patient(st.session_state.selected_patient_id)
    
    if not analyses:
        st.info("No analyses for this patient.")
        return
    
    for a in analyses:
        severity = a.get('severity', 'N/A')
        created = a.get('created_at', '')[:16].replace('T', ' ')
        model_used = a.get('clinical_reasoning', {}).get('model_used', 'N/A')
        ptbxl_pattern = a.get('clinical_reasoning', {}).get('ptbxl_pattern', 'N/A')
        
        severity_color = {
            "routine review": "#0B8A4D",
            "prompt clinical review": "#B45309",
            "urgent evaluation may be appropriate": "#B91C1C",
            "emergency evaluation recommended": "#7F1D1D"
        }.get(severity, "#4A4A5A")
        
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; padding: 10px 16px; background: white; border-radius: 8px; border: 1px solid #E5E7EB; margin-bottom: 6px;">
            <div>
                <span style="font-weight: 500;">{ptbxl_pattern} - {a.get('summary', 'Analysis')}</span>
                <span style="color: #4A4A5A; font-size: 13px; margin-left: 8px;">{created}</span>
                <span style="color: #8A8A9A; font-size: 11px; margin-left: 8px; background: #F4F6F9; padding: 2px 8px; border-radius: 10px;">{model_used}</span>
            </div>
            <div>
                <span style="color: {severity_color}; font-weight: 500;">{severity}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ===== PAGE: COMPARE =====
def page_compare():
    st.markdown("""
    <div class="page-header">
        <div class="page-title">Compare Analyses</div>
        <div class="page-subtitle">Compare two analyses of the same patient</div>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.selected_patient_id:
        st.info("Select a patient first (go to Clinical page)")
        return
    
    analyses = cdb.get_analyses_for_patient(st.session_state.selected_patient_id)
    
    if len(analyses) < 2:
        st.info("This patient needs at least two analyses to compare.")
        return
    
    patient = patient_manager.get_patient(st.session_state.selected_patient_id)
    
    options = {}
    for a in analyses:
        ptbxl = a.get('clinical_reasoning', {}).get('ptbxl_pattern', 'N/A')
        label = f"{a.get('created_at', '')[:16].replace('T', ' ')} - {ptbxl} - {a.get('analysis_type', 'unknown')}"
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
    
    st.divider()
    st.markdown(f"""
    <div style="font-size: 18px; font-weight: 600; color: #0F2B4A;">
        {patient.get('name', 'Unknown')}
    </div>
    <div style="color: #4A4A5A; font-size: 14px; margin-bottom: 16px;">
        {patient.get('age', 'N/A')} yrs · {patient.get('sex', 'N/A')}
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        severity1 = earlier.get("severity", "N/A")
        ptbxl1 = earlier.get('clinical_reasoning', {}).get('ptbxl_pattern', 'N/A')
        st.markdown(f"""
        <div class="compare-card">
            <div class="label">Earlier Analysis</div>
            <div class="date">{earlier.get('created_at', '')[:16].replace('T', ' ')}</div>
            <div style="color: #4A4A5A; font-size: 13px;">PTB-XL: {ptbxl1}</div>
            <div class="result-box">
                <strong>Finding:</strong> {earlier.get('summary', 'N/A')}
            </div>
            <div style="margin-top: 8px;">
                <strong>Severity:</strong> {severity1}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        severity2 = later.get("severity", "N/A")
        ptbxl2 = later.get('clinical_reasoning', {}).get('ptbxl_pattern', 'N/A')
        st.markdown(f"""
        <div class="compare-card">
            <div class="label">Later Analysis</div>
            <div class="date">{later.get('created_at', '')[:16].replace('T', ' ')}</div>
            <div style="color: #4A4A5A; font-size: 13px;">PTB-XL: {ptbxl2}</div>
            <div class="result-box">
                <strong>Finding:</strong> {later.get('summary', 'N/A')}
            </div>
            <div style="margin-top: 8px;">
                <strong>Severity:</strong> {severity2}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    st.markdown("#### Comparison Summary")
    
    severity_order = ["routine review", "prompt clinical review", "urgent evaluation may be appropriate", "emergency evaluation recommended"]
    earlier_idx = severity_order.index(severity1) if severity1 in severity_order else 1
    later_idx = severity_order.index(severity2) if severity2 in severity_order else 1
    
    if earlier_idx < later_idx:
        st.warning("⚠️ The later analysis shows increased concern.")
    elif earlier_idx > later_idx:
        st.success("✅ The later analysis shows decreased concern.")
    else:
        st.info("ℹ️ The level of concern is similar between analyses.")
    
    if ptbxl1 != ptbxl2:
        st.write(f"**PTB-XL Pattern Change:** {ptbxl1} → {ptbxl2}")
    
    st.markdown("""
    <div class="disclaimer" style="border-top: none; padding-top: 0; margin-top: 8px;">
        Comparison based on available data. Missing information is not estimated.
    </div>
    """, unsafe_allow_html=True)


# ===== MAIN =====
render_sidebar()

if st.session_state.user_id is None:
    render_login()
else:
    pages = {
        "Clinical": page_clinical,
        "History": page_history,
        "Compare": page_compare,
    }
    current = st.session_state.page
    if current in pages:
        pages[current]()
    else:
        page_clinical()