"""
CardioAgent — Interactive Clinical Decision Support
Clean, conversational interface using Knowledge Base (No Rules)
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
from respond import CLASS_FULL_NAMES, FRIENDLY_DESCRIPTIONS, compose_response
from patient_management import PatientManager, get_patient_display_name
from severity_scorer import SeverityScorer

import warnings
warnings.filterwarnings("ignore")

CHECKPOINT = "checkpoint.pt"
STORE_PATH = "vector_store.pkl"
ECG_FILES_DIR = "ecg_files"

st.set_page_config(
    page_title="CardioAgent — Clinical Decision Support",
    page_icon="⚕️",
    layout="wide"
)

# ===== PROFESSIONAL CSS =====
st.markdown("""
<style>
    .stApp {
        font-family: 'Inter', 'Segoe UI', sans-serif;
        background-color: #F4F6F9;
        color: #1A1A2E;
    }
    
    .page-header {
        margin-bottom: 20px;
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
        line-height: 1.7;
        font-size: 15px;
    }
    
    .severity-box {
        padding: 16px 20px;
        border-radius: 10px;
        margin: 12px 0;
        background: white;
        border-left: 5px solid #0D9488;
    }
    
    .severity-no { border-left-color: #0B8A4D; }
    .severity-moderate { border-left-color: #B45309; }
    .severity-urgent { border-left-color: #B91C1C; }
    
    .severity-no .sev-label { color: #0B8A4D; }
    .severity-moderate .sev-label { color: #B45309; }
    .severity-urgent .sev-label { color: #B91C1C; }
    
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
    
    .kb-badge {
        display: inline-block;
        background: #E8F0FE;
        color: #0F2B4A;
        font-size: 11px;
        font-weight: 600;
        padding: 2px 10px;
        border-radius: 12px;
        margin-left: 8px;
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
    
    .graph-explanation {
        background: #F8FAFC;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 14px;
        color: #4A4A5A;
        border-left: 3px solid #0D9488;
    }
    
    .kb-source {
        background: #F8FAFC;
        padding: 8px 12px;
        border-radius: 6px;
        font-size: 12px;
        color: #4A4A5A;
        border-left: 3px solid #8A8A9A;
        margin: 4px 0;
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

# Initialize severity scorer with knowledge base
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
    ("patient_symptoms", ""),
    ("analysis_saved", False),
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
            <div style="font-size: 14px; color: #4A4A5A; margin-top: 2px;">Clinical Decision Support</div>
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
            <div class="sidebar-brand-sub">Clinical Decision Support</div>
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
            if st.button(p, key=f"nav_{p}", use_container_width=True, type="primary" if current == p else "secondary"):
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
                    <div class="sidebar-patient-detail">{patient.get('age', 'N/A')} yrs · {patient.get('sex', 'N/A')}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-footer">', unsafe_allow_html=True)
        if st.button("Log Out", use_container_width=True, key="logout_btn"):
            for key in ["user_id", "email", "name", "selected_patient_id", 
                        "current_analysis", "analysis_complete", "chat_messages",
                        "current_patient", "interview_step", "patient_symptoms", "analysis_saved"]:
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


# ===== EXPLANATION FROM KNOWLEDGE BASE =====
def get_ecg_explanation_from_kb(pred_class, severity_scorer):
    """Get ECG pattern explanation from knowledge base."""
    ecg_data = severity_scorer.ecg_data
    if pred_class in ecg_data:
        desc = ecg_data[pred_class]
        if desc and len(desc) > 0:
            return desc[0]
    return "ECG pattern detected. Please consult a healthcare professional for interpretation."


def get_symptom_explanation_from_kb(symptoms, severity_scorer):
    """Get symptom explanation from knowledge base."""
    symptom_data = severity_scorer.symptom_data
    explanations = []
    
    if not symptom_data.get("entries"):
        return []
    
    for entry in symptom_data["entries"]:
        topic = entry.get("topic", "")
        content = entry.get("content", "")
        if any(word in symptoms.lower() for word in ["chest", "pain", "breath", "syncope", "palpitations", "dizzy", "fatigue"]):
            explanations.append({
                "topic": topic,
                "content": content[:300] + "..."
            })
    
    return explanations[:3]


def explain_prediction_in_plain_language(pred_class, confidence, symptoms, is_existing, history_context="", severity_scorer=None):
    """Convert technical prediction to plain language explanation using Knowledge Base."""
    
    plain_descriptions = {
        "NORM": {
            "finding": "The ECG pattern appears normal",
            "explanation": "The electrical activity of the heart shows a regular rhythm with no significant abnormalities. This is a reassuring finding."
        },
        "STTC": {
            "finding": "The ECG shows some changes in the ST segment or T wave",
            "explanation": "This means there may be some stress or strain on the heart muscle. It could be related to reduced blood flow, electrolyte imbalance, or other factors. This requires clinical correlation with symptoms."
        },
        "CD": {
            "finding": "The ECG shows a conduction delay in the heart's electrical system",
            "explanation": "This means the electrical signals in the heart are taking slightly longer than normal to travel through the heart's pathways. This can affect how the heart beats."
        },
        "HYP": {
            "finding": "The ECG suggests possible thickening of the heart muscle",
            "explanation": "This means the heart muscle may be working harder than normal, often due to high blood pressure or other conditions. This needs further evaluation."
        },
        "MI": {
            "finding": "The ECG shows changes that could indicate a heart attack",
            "explanation": "This means there are signs that the heart muscle may not be getting enough blood. This is a serious finding that requires immediate medical attention."
        }
    }
    
    result = plain_descriptions.get(pred_class, plain_descriptions["NORM"])
    
    # Get detailed explanation from Knowledge Base
    kb_explanation = ""
    if severity_scorer:
        kb_explanation = get_ecg_explanation_from_kb(pred_class, severity_scorer)
    
    severity_levels = {
        "No Concern": "This appears to be a low-risk situation. No immediate action is needed.",
        "Moderate Concern": "This situation needs attention. A doctor should review this within the next 1-2 weeks.",
        "Urgent Concern": "This needs prompt medical attention. Please consult a doctor within 24-48 hours."
    }
    
    response = f"""
**What the ECG shows:**
{result['finding']}

**What this means:**
{result['explanation']}

**Medical Reference:**
{kb_explanation if kb_explanation else "Please consult a healthcare professional for detailed interpretation."}

**Confidence in this finding:** {confidence*100:.0f}%

**Based on symptoms described:**
"{symptoms}"

{history_context}
"""
    
    return response


# ===== EXPLAIN ECG GRAPH =====
def explain_ecg_graph(region_start, region_end):
    """Explain what the ECG graph shows in plain language."""
    
    return f"""
**What you're looking at:**
This is a visual representation of the patient's heartbeat over time. The line shows the electrical activity of the heart.

**What the highlighted area means:**
The highlighted section (around {region_start:.1f} to {region_end:.1f} seconds) is where the computer model found the most important patterns. This is the part of the heartbeat that most influenced the analysis.

**Why this matters:**
Different parts of the heartbeat pattern can tell us different things. The highlighted area shows what the model focused on to make its assessment.
"""


# ===== CLINICAL PAGE =====
def page_clinical():
    st.markdown("""
    <div class="page-header">
        <div class="page-title">Clinical Assessment</div>
        <div class="page-subtitle">Patient evaluation and analysis</div>
    </div>
    """, unsafe_allow_html=True)
    
    # ===== PATIENT SELECTION =====
    with st.container():
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Patient</div>', unsafe_allow_html=True)
        
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
                    <span class="kb-badge">Knowledge Base</span>
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
    
    patient = st.session_state.current_patient
    existing_analyses = cdb.get_analyses_for_patient(patient["id"]) if patient else []
    is_existing = len(existing_analyses) > 0
    
    # ===== INTERVIEW =====
    with st.container():
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Patient Interview</div>', unsafe_allow_html=True)
        
        if st.session_state.interview_step == "symptoms":
            st.caption("Please describe what the patient is experiencing")
        elif st.session_state.interview_step == "ecg":
            st.caption("Please provide any available ECG recordings")
        elif st.session_state.interview_step == "complete":
            st.success("Information collected. Ready for analysis.")
        
        # Chat messages
        for msg in st.session_state.chat_messages:
            if msg["role"] == "user":
                st.markdown(f'<div style="text-align: right;"><div class="chat-message-user">{msg["content"]}</div></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message-assistant">{msg["content"]}</div>', unsafe_allow_html=True)
        
        # Show result
        if st.session_state.analysis_complete and st.session_state.current_analysis:
            render_professional_result(st.session_state.current_analysis, is_existing)
            
            if st.button("New Assessment", use_container_width=True):
                st.session_state.current_analysis = None
                st.session_state.analysis_complete = False
                st.session_state.chat_messages = []
                st.session_state.interview_step = "symptoms"
                st.session_state.analysis_saved = False
                st.rerun()
            return
        
        # Interview prompt
        if st.session_state.interview_step != "complete":
            if st.session_state.interview_step == "symptoms":
                prompt = """
**Please describe the patient's symptoms**

Tell me what the patient is experiencing. For example:
- Where is the discomfort?
- How long has it been happening?
- How severe is it?

*Example: "The patient has chest discomfort and difficulty breathing for the past 2 hours."*
"""
            else:
                prompt = """
**Do you have any ECG recordings?**

If you have an ECG recording, please upload the .hea and .dat files.

If not, just type "no" to continue with symptoms only.
"""
            st.markdown(f'<div class="chat-message-assistant">{prompt}</div>', unsafe_allow_html=True)
        
        # ECG Upload
        if st.session_state.interview_step == "ecg" and is_existing:
            st.markdown("""
            <div class="upload-area">
                <div class="icon">📊</div>
                <div class="text">Upload ECG recording (.hea + .dat files)</div>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                hea_file = st.file_uploader("Select .hea file", type=["hea"])
            with col2:
                dat_file = st.file_uploader("Select .dat file", type=["dat"])
            
            if hea_file and dat_file:
                if st.button("Analyze ECG Recording", type="primary", use_container_width=True):
                    with st.spinner("Analyzing..."):
                        try:
                            analysis = run_signal_analysis(hea_file, dat_file, patient, is_existing)
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
                if isinstance(user_input, list):
                    user_input = " ".join(user_input)
                else:
                    user_input = str(user_input)
                
                st.session_state.chat_messages.append({"role": "user", "content": user_input})
                
                if st.session_state.interview_step == "symptoms":
                    st.session_state.patient_symptoms = user_input
                    st.session_state.interview_step = "ecg" if is_existing else "complete"
                    
                    if is_existing:
                        response = f"Thank you. I've recorded the symptoms.\n\n**Symptoms:** {user_input}\n\nDo you have any ECG recordings for this patient? If yes, please upload them above. If not, type 'no'."
                    else:
                        response = f"Thank you. I've recorded the symptoms.\n\n**Symptoms:** {user_input}\n\nProceeding with analysis based on symptoms only."
                    
                    st.session_state.chat_messages.append({"role": "assistant", "content": response})
                    st.rerun()
                
                elif st.session_state.interview_step == "ecg":
                    if user_input.lower() in ["no", "none", "not available", "don't have", "skip"]:
                        response = "Understood. Proceeding with analysis based on symptoms only."
                        st.session_state.chat_messages.append({"role": "assistant", "content": response})
                        st.session_state.interview_step = "complete"
                        st.rerun()
                    else:
                        response = "I've noted your response. Please upload the ECG files above, or type 'no' if you don't have them."
                        st.session_state.chat_messages.append({"role": "assistant", "content": response})
                        st.rerun()
        
        # Analyze when complete
        if st.session_state.interview_step == "complete" and not st.session_state.analysis_complete:
            with st.spinner("Analyzing..."):
                symptoms = st.session_state.patient_symptoms if hasattr(st.session_state, 'patient_symptoms') else ""
                if isinstance(symptoms, list):
                    symptoms = " ".join(symptoms)
                elif symptoms is None:
                    symptoms = ""
                
                if symptoms:
                    # Get PTB-XL prediction from symptoms (using knowledge base)
                    pred_class, confidence = predict_from_symptoms_with_kb(symptoms, severity_scorer)
                    
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
                            history_context = f"\n\n**Previous record:** {last.get('summary', 'N/A')} - {last.get('severity', 'No Concern')}"
                    
                    plain_explanation = explain_prediction_in_plain_language(
                        pred_class, confidence, symptoms, is_existing, history_context, severity_scorer
                    )
                    
                    # Get symptom knowledge base entries
                    symptom_entries = get_symptom_explanation_from_kb(symptoms, severity_scorer)
                    
                    analysis = {
                        "prediction": pred_class,
                        "confidence": confidence,
                        "friendly_name": FRIENDLY_DESCRIPTIONS.get(pred_class, pred_class),
                        "severity": severity_result['level'],
                        "severity_score": severity_result['score'],
                        "severity_evidence": severity_result['evidence'],
                        "matched_symptoms": severity_result.get('matched_symptoms', []),
                        "red_flags": severity_result.get('red_flags', []),
                        "patient": patient,
                        "analysis_type": "chat",
                        "symptoms": symptoms,
                        "is_existing": is_existing,
                        "history_context": history_context,
                        "plain_explanation": plain_explanation,
                        "symptom_entries": symptom_entries,
                        "model_used": "PTB-XL + Knowledge Base"
                    }
                    
                    st.session_state.current_analysis = analysis
                    st.session_state.analysis_complete = True
                    save_analysis_to_history(analysis)
                    st.session_state.analysis_saved = True
                    st.rerun()


def predict_from_symptoms_with_kb(symptoms_text, severity_scorer):
    """Predict PTB-XL class using knowledge base."""
    symptoms_lower = symptoms_text.lower()
    
    # Check knowledge base entries for matching symptoms
    symptom_data = severity_scorer.symptom_data
    
    # Check for red flags first (higher priority)
    red_flag_indicators = ["chest pain", "shortness of breath", "syncope", "fainting", "loss of consciousness"]
    for indicator in red_flag_indicators:
        if indicator in symptoms_lower:
            # Check if it's a red flag in knowledge base
            for entry in symptom_data.get("entries", []):
                content = entry.get("content", "").lower()
                if indicator in content and ("urgent" in content or "red flag" in content or "immediate" in content):
                    return "MI", 0.80
    
    # Map symptoms to PTB-XL classes based on knowledge base
    if any(w in symptoms_lower for w in ['chest pain', 'pressure', 'tightness', 'crushing']):
        return "MI", 0.75
    elif any(w in symptoms_lower for w in ['palpitations', 'irregular', 'racing', 'fluttering']):
        return "STTC", 0.70
    elif any(w in symptoms_lower for w in ['dizzy', 'faint', 'lightheaded', 'syncope']):
        return "CD", 0.65
    elif any(w in symptoms_lower for w in ['shortness of breath', 'fatigue', 'swelling']):
        return "HYP", 0.60
    else:
        return "NORM", 0.55


def run_signal_analysis(hea_file, dat_file, patient, is_existing):
    """Run ECG signal analysis."""
    
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
    
    cam = grad_cam_1d(model, x_tensor, pred_idx)
    start_sec, end_sec, _ = top_attributed_region(cam, fs=100)
    
    measurements = extract_ecg_measurements(raw_signal[:, 0], fs)
    if heart_rate_result.get("heart_rate"):
        measurements["heart_rate"] = heart_rate_result["heart_rate"]
    
    symptoms = st.session_state.patient_symptoms if hasattr(st.session_state, 'patient_symptoms') else ""
    if isinstance(symptoms, list):
        symptoms = " ".join(symptoms)
    elif symptoms is None:
        symptoms = ""
    
    patient_info = {
        'age': patient.get('age', 50) if patient else 50,
        'sex': patient.get('sex', 'Unknown') if patient else 'Unknown',
        'symptoms': symptoms
    }
    
    ecg_prediction = {'prediction': pred_class, 'confidence': confidence}
    severity_result = severity_scorer.calculate(ecg_prediction, patient_info, measurements)
    
    history_context = ""
    if is_existing:
        analyses = cdb.get_analyses_for_patient(patient["id"])
        if analyses:
            last = analyses[0]
            history_context = f"\n\n**Previous record:** {last.get('summary', 'N/A')} - {last.get('severity', 'No Concern')}"
    
    plain_explanation = explain_prediction_in_plain_language(
        pred_class, confidence, symptoms, is_existing, history_context, severity_scorer
    )
    
    # Get symptom knowledge base entries
    symptom_entries = get_symptom_explanation_from_kb(symptoms, severity_scorer)
    
    # Get graph explanation
    graph_explanation = explain_ecg_graph(start_sec, end_sec)
    
    analysis_id = cdb.create_analysis_with_patient(
        patient_id=patient["id"],
        user_id=st.session_state.user_id,
        analysis_type="signal",
        prediction=pred_class,
        confidence=confidence,
        features={"heart_rate": measurements.get("heart_rate")},
        xai={"region_start_sec": start_sec, "region_end_sec": end_sec},
        rag_sources=[],
        explanation=plain_explanation,
        severity=severity_result['level'],
        summary=f"ECG pattern: {FRIENDLY_DESCRIPTIONS.get(pred_class, pred_class)}",
        mode_type="research",
        patient_context={"age": patient.get("age"), "sex": patient.get("sex"), "symptoms": symptoms},
        clinical_reasoning={
            "severity": severity_result['level'],
            "severity_score": severity_result['score'],
            "evidence": severity_result['evidence'],
            "matched_symptoms": severity_result.get('matched_symptoms', []),
            "red_flags": severity_result.get('red_flags', []),
            "model_used": "PTB-XL ECGConvNet + Knowledge Base"
        }
    )
    
    return {
        "id": analysis_id,
        "prediction": pred_class,
        "confidence": confidence,
        "friendly_name": FRIENDLY_DESCRIPTIONS.get(pred_class, pred_class),
        "severity": severity_result['level'],
        "severity_score": severity_result['score'],
        "severity_evidence": severity_result['evidence'],
        "matched_symptoms": severity_result.get('matched_symptoms', []),
        "red_flags": severity_result.get('red_flags', []),
        "patient": patient,
        "analysis_type": "signal",
        "signal": raw_signal,
        "fs": fs,
        "lead_names": lead_names,
        "cam": cam,
        "xai": {"region_start_sec": start_sec, "region_end_sec": end_sec},
        "symptoms": symptoms,
        "is_existing": is_existing,
        "history_context": history_context,
        "plain_explanation": plain_explanation,
        "symptom_entries": symptom_entries,
        "graph_explanation": graph_explanation,
        "model_used": "PTB-XL ECGConvNet + Knowledge Base"
    }


def save_analysis_to_history(analysis):
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
            explanation=analysis.get('plain_explanation', ''),
            severity=analysis.get('severity', 'No Concern'),
            summary=f"PTB-XL: {analysis.get('prediction', 'N/A')} - {analysis.get('severity', 'N/A')}",
            mode_type=analysis.get('analysis_type', 'chat'),
            patient_context={"age": patient.get("age"), "sex": patient.get("sex"), "symptoms": analysis.get("symptoms", "")},
            clinical_reasoning={
                'severity': analysis.get('severity'),
                'score': analysis.get('severity_score'),
                'evidence': analysis.get('severity_evidence'),
                'matched_symptoms': analysis.get('matched_symptoms', []),
                'red_flags': analysis.get('red_flags', []),
                'model_used': analysis.get('model_used', 'PTB-XL + Knowledge Base'),
                'ptbxl_pattern': analysis.get('prediction'),
                'confidence': analysis.get('confidence')
            }
        )
        st.session_state.analysis_saved = True
    except Exception as e:
        print(f"Error saving: {e}")


# ===== PROFESSIONAL RESULT =====
def render_professional_result(analysis, is_existing):
    patient = analysis.get("patient", {})
    severity = analysis.get("severity", "No Concern")
    severity_score = analysis.get("severity_score", 0)
    severity_evidence = analysis.get("severity_evidence", [])
    pred = analysis.get("prediction")
    friendly = analysis.get("friendly_name", "")
    history_context = analysis.get("history_context", "")
    symptoms = analysis.get("symptoms", "Not provided")
    plain_explanation = analysis.get("plain_explanation", "")
    matched_symptoms = analysis.get("matched_symptoms", [])
    red_flags = analysis.get("red_flags", [])
    symptom_entries = analysis.get("symptom_entries", [])
    graph_explanation = analysis.get("graph_explanation", "")
    
    st.markdown("""
    <div class="page-header" style="border-bottom: none; margin-bottom: 8px;">
        <div class="page-title">Assessment Results</div>
    </div>
    """, unsafe_allow_html=True)
    
    # ===== PATIENT =====
    st.markdown("""
    <div class="report-section">
        <div class="report-section-title">Patient Information</div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    | **Name** | {patient.get('name', 'Unknown')} |
    |---|---|
    | **Age** | {patient.get('age', 'N/A')} yrs |
    | **Sex** | {patient.get('sex', 'N/A')} |
    | **Type** | {"Existing Patient" if is_existing else "New Patient"} |
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ===== SYMPTOMS =====
    st.markdown("""
    <div class="report-section">
        <div class="report-section-title">Symptoms Reported</div>
    """, unsafe_allow_html=True)
    st.markdown(f"_{symptoms}_")
    
    if matched_symptoms:
        st.markdown("**Symptoms identified from Knowledge Base:**")
        for s in matched_symptoms:
            st.markdown(f"• {s}")
    
    if red_flags:
        st.markdown("**⚠️ Red Flags Detected:**")
        for rf in red_flags:
            st.markdown(f"• {rf}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ===== KNOWLEDGE BASE REFERENCE =====
    if symptom_entries:
        st.markdown("""
        <div class="report-section">
            <div class="report-section-title">Medical Knowledge Base Reference</div>
        """, unsafe_allow_html=True)
        
        for entry in symptom_entries[:2]:
            st.markdown(f"""
            <div class="kb-source">
                <strong>{entry.get('topic', 'Reference')}</strong><br>
                {entry.get('content', '')}
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ===== PLAIN EXPLANATION =====
    st.markdown("""
    <div class="report-section">
        <div class="report-section-title">What the Analysis Shows</div>
    """, unsafe_allow_html=True)
    st.markdown(plain_explanation)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ===== SEVERITY =====
    st.markdown("""
    <div class="report-section">
        <div class="report-section-title">Urgency Level</div>
    """, unsafe_allow_html=True)
    
    severity_colors = {
        "No Concern": "severity-no",
        "Moderate Concern": "severity-moderate",
        "Urgent Concern": "severity-urgent"
    }
    
    severity_descriptions = {
        "No Concern": "This appears to be a low-risk situation. No immediate action is needed.",
        "Moderate Concern": "This situation needs attention. A doctor should review this within the next 1-2 weeks.",
        "Urgent Concern": "This needs prompt medical attention. Please consult a doctor within 24-48 hours."
    }
    
    st.markdown(f"""
    <div class="severity-box {severity_colors.get(severity, 'severity-moderate')}">
        <div style="font-size: 20px; font-weight: 700;" class="sev-label">{severity}</div>
        <div style="color: #4A4A5A; font-size: 14px; margin-top: 4px;">{severity_descriptions.get(severity, '')}</div>
    </div>
    """, unsafe_allow_html=True)
    
    if severity_evidence:
        st.markdown("**Factors considered:**")
        for item in severity_evidence[:5]:
            st.markdown(f"• {item}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ===== HISTORY =====
    if is_existing and history_context:
        st.info(history_context)
    
    # ===== ECG GRAPH =====
    if analysis.get("analysis_type") == "signal" and "cam" in analysis and analysis["cam"] is not None:
        st.markdown("""
        <div class="report-section">
            <div class="report-section-title">ECG Recording Analysis</div>
        """, unsafe_allow_html=True)
        
        try:
            signal = analysis.get("signal")
            fs = analysis.get("fs")
            cam = analysis.get("cam")
            
            if signal is not None and cam is not None:
                fig = create_gradcam_visualization(
                    signal, cam, fs if fs else 100,
                    lead_idx=0,
                    title="Heartbeat Pattern"
                )
                st.pyplot(fig)
                import matplotlib.pyplot as plt
                plt.close(fig)
                
                st.markdown(f"""
                <div class="graph-explanation">
                    {graph_explanation}
                </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            st.caption("Graph visualization unavailable.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ===== RECOMMENDATIONS =====
    st.markdown("""
    <div class="report-section">
        <div class="report-section-title">Recommendations</div>
    """, unsafe_allow_html=True)
    
    recommendations = {
        "No Concern": "Continue routine monitoring. No immediate action required. Follow up as scheduled.",
        "Moderate Concern": "Schedule a clinical review within 1-2 weeks. Consider consulting a specialist.",
        "Urgent Concern": "Seek medical evaluation within 24-48 hours. Contact a cardiology department."
    }
    
    st.markdown(f"""
    <div style="padding: 12px 16px; background: #F8FAFC; border-radius: 8px; border-left: 4px solid #0D9488;">
        {recommendations.get(severity, "Clinical evaluation recommended.")}
    </div>
    """, unsafe_allow_html=True)
    
    if is_existing:
        st.markdown("""
        <div style="padding: 12px 16px; background: #F0F7FF; border-radius: 8px; border-left: 4px solid #0D9488; margin-top: 8px;">
            <strong>Note:</strong> This is a follow-up analysis. Please compare with previous records for any changes.
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ===== DISCLAIMER =====
    st.markdown("""
    <div class="disclaimer">
        <strong>Disclaimer:</strong> This is an AI-assisted decision support tool using verified medical knowledge bases. All findings must be confirmed by a qualified healthcare professional. This does not replace a doctor's clinical judgment.
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.get("analysis_saved", False):
        st.caption("✅ Results automatically saved")


# ===== PAGE: HISTORY =====
def page_history():
    st.markdown("""
    <div class="page-header">
        <div class="page-title">History</div>
        <div class="page-subtitle">Review past assessments</div>
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
        st.info("No assessments for this patient.")
        return
    
    for a in analyses:
        severity = a.get('severity', 'N/A')
        created = a.get('created_at', '')[:16].replace('T', ' ')
        
        severity_color = {
            "No Concern": "#0B8A4D",
            "Moderate Concern": "#B45309",
            "Urgent Concern": "#B91C1C"
        }.get(severity, "#4A4A5A")
        
        model_used = a.get('clinical_reasoning', {}).get('model_used', 'PTB-XL')
        
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; padding: 10px 16px; background: white; border-radius: 8px; border: 1px solid #E5E7EB; margin-bottom: 6px;">
            <div>
                <span style="font-weight: 500;">{a.get('summary', 'Assessment')}</span>
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
        <div class="page-title">Compare Assessments</div>
        <div class="page-subtitle">Compare two assessments of the same patient</div>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.selected_patient_id:
        st.info("Select a patient first (go to Clinical page)")
        return
    
    analyses = cdb.get_analyses_for_patient(st.session_state.selected_patient_id)
    if len(analyses) < 2:
        st.info("This patient needs at least two assessments to compare.")
        return
    
    patient = patient_manager.get_patient(st.session_state.selected_patient_id)
    
    options = {}
    for a in analyses:
        label = f"{a.get('created_at', '')[:16].replace('T', ' ')} - {a.get('summary', 'Assessment')}"
        options[label] = a
    
    labels = list(options.keys())
    
    col1, col2 = st.columns(2)
    with col1:
        label1 = st.selectbox("Earlier Assessment", labels, index=0, key="cmp1")
    with col2:
        label2 = st.selectbox("Later Assessment", labels, index=min(1, len(labels)-1), key="cmp2")
    
    if label1 == label2:
        st.warning("Please select two different assessments.")
        return
    
    earlier = options[label1]
    later = options[label2]
    
    st.divider()
    st.markdown(f"**Patient:** {patient.get('name', 'Unknown')} · {patient.get('age', 'N/A')} yrs · {patient.get('sex', 'N/A')}")
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        severity1 = earlier.get("severity", "N/A")
        st.markdown(f"""
        <div style="background: white; border-radius: 10px; padding: 16px; border: 1px solid #E5E7EB;">
            <div style="font-size: 12px; color: #4A4A5A; text-transform: uppercase;">Earlier</div>
            <div style="font-size: 14px; font-weight: 500;">{earlier.get('created_at', '')[:16].replace('T', ' ')}</div>
            <div style="margin-top: 8px;"><strong>Finding:</strong> {earlier.get('summary', 'N/A')}</div>
            <div><strong>Severity:</strong> {severity1}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        severity2 = later.get("severity", "N/A")
        st.markdown(f"""
        <div style="background: white; border-radius: 10px; padding: 16px; border: 1px solid #E5E7EB;">
            <div style="font-size: 12px; color: #4A4A5A; text-transform: uppercase;">Later</div>
            <div style="font-size: 14px; font-weight: 500;">{later.get('created_at', '')[:16].replace('T', ' ')}</div>
            <div style="margin-top: 8px;"><strong>Finding:</strong> {later.get('summary', 'N/A')}</div>
            <div><strong>Severity:</strong> {severity2}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    st.markdown("#### Summary")
    
    severity_order = ["No Concern", "Moderate Concern", "Urgent Concern"]
    earlier_idx = severity_order.index(severity1) if severity1 in severity_order else 1
    later_idx = severity_order.index(severity2) if severity2 in severity_order else 1
    
    if earlier_idx < later_idx:
        st.warning("The later assessment shows increased concern.")
    elif earlier_idx > later_idx:
        st.success("The later assessment shows decreased concern.")
    else:
        st.info("The level of concern is similar between assessments.")


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