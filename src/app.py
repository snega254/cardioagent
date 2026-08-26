"""
CardioAgent - full application with conversational, assistant-style UI.

Run: streamlit run src/app.py

Navigation: Dashboard / New Analysis / History / Compare / Knowledge Base
The "Ask CardioAgent" chatbot lives embedded within the Analysis Result
view (both right after running a new analysis, and when reopening a past
report from History) rather than as a separate top-level page, since it
needs a loaded analysis to answer about.

Pipeline: ECG (.hea+.dat) -> validate -> preprocess -> ECGConvNet ->
prediction -> Grad-CAM + heart rate -> RAG retrieval -> Gemini explanation
-> chatbot follow-ups -> MongoDB + PDF report.
"""
import os
import uuid

import torch

import streamlit as st
import auth
from chat import SUGGESTED_QUESTIONS, ask_chatbot
from db import CardioDB
from document_processing import chunk_text, clean_text, extract_text
from ecg_io import ECGLoadError, load_wfdb_pair, prepare_for_model, validate_signal
from gradcam import grad_cam_1d, top_attributed_region
from heart_rate import detect_heart_rate
from ingest_documents import seed_default_kb
from model import ECGConvNet
from preprocessing import SUPERCLASSES
from rag import build_query, retrieve
from report_pdf import generate_pdf_report
from respond import CLASS_FULL_NAMES, FRIENDLY_DESCRIPTIONS, compose_response
from triage import generate_triage_assessment
from vector_store import VectorStore

CHECKPOINT = "checkpoint.pt"
STORE_PATH = "vector_store.pkl"
ECG_FILES_DIR = "ecg_files"
REPORTS_DIR = "reports"

st.set_page_config(page_title="CardioAgent", layout="wide", page_icon="\U00002764")


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
def load_model():
    model = ECGConvNet()
    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.log_temperature.data = ckpt["log_temperature"]
    model.eval()
    return model


for key, default in [("user_id", None), ("username", None), ("name", None),
                      ("age", None), ("sex", None), ("page", "Dashboard"),
                      ("current_ecg", None), ("open_report_id", None),
                      ("chat_histories", {})]:
    if key not in st.session_state:
        st.session_state[key] = default


def do_login(username, password):
    user = cdb.get_user_by_username(username)
    if user is None:
        return False, "No account with that username."
    if not auth.verify_password(password, user["password_hash"]):
        return False, "Incorrect password."
    st.session_state.user_id = user["id"]
    st.session_state.username = user["username"]
    st.session_state.name = user["name"]
    st.session_state.age = user["age"]
    st.session_state.sex = user["sex"]
    return True, ""


def do_register(username, password, name, age, sex):
    ok, msg = auth.validate_username(username)
    if not ok:
        return False, msg
    ok, msg = auth.validate_password(password)
    if not ok:
        return False, msg
    if cdb.get_user_by_username(username) is not None:
        return False, "That username is already taken."
    pw_hash = auth.hash_password(password)
    cdb.create_user(username, pw_hash, name, age, sex)
    return True, "Account created. Please log in."


with st.sidebar:
    st.markdown("## \U00002764 CardioAgent")
    st.caption("AI-Assisted ECG Analysis")

    if db_error:
        st.error("Database not connected.")
        with st.expander("Setup instructions"):
            st.code(db_error)
        st.stop()

    if st.session_state.user_id is None:
        login_tab, register_tab = st.tabs(["Log In", "Register"])
        with login_tab:
            u = st.text_input("Username", key="login_u")
            p = st.text_input("Password", type="password", key="login_p")
            if st.button("Log In", use_container_width=True):
                ok, msg = do_login(u, p)
                if ok:
                    st.rerun()
                else:
                    st.error(msg)
        with register_tab:
            ru = st.text_input("Choose a username", key="reg_u")
            rp = st.text_input("Choose a password", type="password", key="reg_p")
            rname = st.text_input("Full name", key="reg_name")
            rage = st.number_input("Age", min_value=0, max_value=120, value=30, key="reg_age")
            rsex = st.selectbox("Sex", ["Prefer not to say", "Female", "Male", "Other"], key="reg_sex")
            if st.button("Register", use_container_width=True):
                ok, msg = do_register(ru, rp, rname, int(rage), rsex)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)
        st.stop()
    else:
        st.success(f"**{st.session_state.name or st.session_state.username}**")
        st.divider()
        nav_options = ["Dashboard", "New Analysis", "History", "Compare", "Knowledge Base"]
        st.session_state.page = st.radio(
            "Navigation", nav_options,
            index=nav_options.index(st.session_state.page),
            label_visibility="collapsed",
        )
        st.divider()
        if st.button("Log Out", use_container_width=True):
            for key in ["user_id", "username", "name", "age", "sex",
                        "current_ecg", "open_report_id"]:
                st.session_state[key] = None
            st.session_state.chat_histories = {}
            st.session_state.page = "Dashboard"
            st.rerun()


def page_dashboard():
    st.title("Dashboard")
    st.caption(f"Welcome back, {st.session_state.name or st.session_state.username}.")

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("New ECG Analysis", use_container_width=True, type="primary"):
            st.session_state.page = "New Analysis"
            st.rerun()
    with c2:
        if st.button("Patient History", use_container_width=True):
            st.session_state.page = "History"
            st.rerun()
    with c3:
        if st.button("Knowledge Base", use_container_width=True):
            st.session_state.page = "Knowledge Base"
            st.rerun()

    st.divider()
    st.subheader("Recent Analyses")
    reports = cdb.get_reports_for_user(st.session_state.user_id)[:5]
    if not reports:
        st.info("No analyses yet. Start with 'New ECG Analysis' above.")
        return

    for r in reports:
        analysis = r["analysis"] or {}
        ecg = r["ecg_record"] or {}
        pred = analysis.get("prediction")
        friendly = FRIENDLY_DESCRIPTIONS.get(pred, "an unrecognized pattern") if pred else "N/A"
        with st.container(border=True):
            cols = st.columns([2, 2, 3, 1])
            cols[0].write(r["created_at"][:19].replace("T", " "))
            cols[1].write(ecg.get("filename", "N/A"))
            cols[2].write(friendly.capitalize())
            if cols[3].button("View", key=f"view_{r['id']}"):
                st.session_state.page = "History"
                st.session_state.open_report_id = r["id"]
                st.rerun()


def render_chat(analysis, ecg_record, chat_key):
    st.subheader("Ask CardioAgent")

    if chat_key not in st.session_state.chat_histories:
        st.session_state.chat_histories[chat_key] = []
    history = st.session_state.chat_histories[chat_key]

    for turn in history:
        with st.chat_message("user" if turn["role"] == "user" else "assistant"):
            st.write(turn["text"])

    st.caption("Suggested questions:")
    sq_cols = st.columns(3)
    clicked_question = None
    for i, q in enumerate(SUGGESTED_QUESTIONS):
        if sq_cols[i % 3].button(q, key=f"{chat_key}_sq_{i}", use_container_width=True):
            clicked_question = q

    typed_question = st.chat_input("Ask a question about this ECG analysis...")
    question = clicked_question or typed_question

    if question:
        history.append({"role": "user", "text": question})
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = ask_chatbot(analysis, ecg_record, history[:-1], question)
                except Exception as e:
                    answer = (f"I couldn't generate a response right now "
                              f"({e}). The analysis data is still available above.")
            st.write(answer)
        history.append({"role": "assistant", "text": answer})


def render_report_body(analysis, ecg_record, chat_key, show_chat=True):
    pred = analysis.get("prediction")
    friendly = FRIENDLY_DESCRIPTIONS.get(pred, "an unrecognized pattern") if pred else None

    st.subheader("AI Interpretation")
    if friendly:
        st.markdown(f"> \"Patterns in this recording were most consistent with **{friendly}**.\"")
    else:
        st.caption("No prediction available.")

    features = analysis.get("features") or {}
    m1, m2, m3 = st.columns(3)
    m1.metric("Heart Rate", f"{features['heart_rate']:.0f} bpm" if features.get("heart_rate") else "N/A")
    m2.metric("Leads", ecg_record.get("n_leads", "N/A"))
    m3.metric("Duration", f"{ecg_record.get('duration_sec', 0):.0f} sec")

    st.subheader("Why did the AI say this?")
    xai = analysis.get("xai") or {}
    if xai.get("region_start_sec") is not None:
        st.write("The model placed stronger importance on a specific portion of the "
                 "recording when making this prediction. This attribution shows what "
                 "the model focused on — it does not by itself prove a specific ECG "
                 "wave or interval is abnormal.")
    else:
        st.caption("No attribution information available.")

    explanation = analysis.get("explanation")
    st.subheader("CardioAgent Explanation")
    if explanation:
        st.write(explanation)
    else:
        st.warning("An explanation could not be generated for this analysis "
                   "(the underlying prediction and measurements above are still valid).")

    st.markdown(
        "> **This is an AI research prototype, not a confirmed medical diagnosis.** "
        "Professional clinical interpretation is required."
    )

    with st.expander("Technical details"):
        conf = analysis.get("confidence")
        rag_sources = analysis.get("rag_sources") or []
        st.json({
            "internal_class": pred,
            "calibrated_confidence": f"{conf*100:.1f}%" if conf is not None else None,
            "sampling_rate_hz": ecg_record.get("sampling_rate"),
            "n_rpeaks": features.get("n_rpeaks"),
            "region_start_sec": xai.get("region_start_sec"),
            "region_end_sec": xai.get("region_end_sec"),
            "retrieved_sources": [s.get("source") for s in rag_sources],
            "retrieval_scores": [round(s.get("score", 0), 3) for s in rag_sources],
        })

    with st.expander("Clinical Triage Assessment (optional, for emergency/urgent-care context)"):
        st.caption(
            "This produces an emergency-triage-style assessment using patient "
            "vitals, symptoms, and history you enter below, combined with this "
            "ECG analysis. Every recommendation requires physician confirmation "
            "before acting — this tool never issues autonomous orders."
        )
        tc1, tc2 = st.columns(2)
        with tc1:
            t_age = st.number_input("Patient age", min_value=0, max_value=120, value=50,
                                     key=f"{chat_key}_t_age")
            t_sex = st.selectbox("Sex", ["Male", "Female", "Other"], key=f"{chat_key}_t_sex")
            t_history = st.text_input("Relevant history (comma-separated)",
                                       key=f"{chat_key}_t_hist")
            t_meds = st.text_input("Current medications (comma-separated)",
                                    key=f"{chat_key}_t_meds")
            t_pde5_hours = st.number_input("Hours since last PDE-5 inhibitor dose (if any, else leave 0)",
                                            min_value=0, max_value=200, value=0,
                                            key=f"{chat_key}_t_pde5")
        with tc2:
            t_complaint = st.text_input("Chief complaint", key=f"{chat_key}_t_complaint")
            t_onset = st.text_input("Symptom onset", key=f"{chat_key}_t_onset")
            t_pain = st.slider("Pain severity (1-10)", 1, 10, 5, key=f"{chat_key}_t_pain")
            t_systolic = st.number_input("Systolic BP (mmHg)", min_value=0, max_value=300,
                                          value=120, key=f"{chat_key}_t_sbp")
            t_diastolic = st.number_input("Diastolic BP (mmHg)", min_value=0, max_value=200,
                                           value=80, key=f"{chat_key}_t_dbp")
            t_spo2 = st.number_input("SpO2 (%)", min_value=0, max_value=100, value=98,
                                      key=f"{chat_key}_t_spo2")

        if st.button("Generate Triage Assessment", key=f"{chat_key}_triage_btn"):
            with st.spinner("Generating triage assessment..."):
                patient = {
                    "age": t_age, "sex": t_sex,
                    "history": t_history or "None reported",
                    "medications": [m.strip() for m in t_meds.split(",") if m.strip()],
                    "hours_since_pde5_inhibitor": t_pde5_hours if t_pde5_hours > 0 else None,
                }
                symptoms = {
                    "chief_complaint": t_complaint or "Not provided",
                    "onset": t_onset or "Not provided",
                    "pain_severity": t_pain,
                    "associated": [],
                }
                vitals = {
                    "bp": f"{t_systolic}/{t_diastolic}", "systolic_bp": t_systolic,
                    "hr": features.get("heart_rate"), "spo2": t_spo2, "rr": None,
                }
                ecg_findings = {
                    "predicted_class": pred, "confidence": analysis.get("confidence"),
                    "heart_rate": features.get("heart_rate"),
                    "gradcam_leads": None,  # not tracked per-lead by current Grad-CAM implementation
                    "gradcam_window": (f"{xai.get('region_start_sec'):.2f}s-{xai.get('region_end_sec'):.2f}s"
                                        if xai.get("region_start_sec") is not None else None),
                }
                rag_guidelines = []
                if os.path.exists(STORE_PATH):
                    raw = retrieve(f"{pred} contraindications guidelines", store_path=STORE_PATH, top_k=3)
                    rag_guidelines = [{"source": s, "text": t} for s, t, sc in raw]

                try:
                    triage_text, flags = generate_triage_assessment(
                        patient, symptoms, vitals, ecg_findings, rag_guidelines)
                    st.markdown(triage_text)
                    if flags.nitrate_contraindicated or flags.beta_blocker_caution:
                        st.warning("Deterministic safety flags were raised — see Section 4 above.")
                except Exception as e:
                    st.error(f"Could not generate triage assessment: {e}")

        st.caption(
            "⚠️ Note: current Grad-CAM output identifies a time region, not "
            "per-lead attribution — the 'affected leads' concept from the "
            "original triage design isn't yet computed by this pipeline. "
            "This is a real gap, not hidden here: extending Grad-CAM to "
            "per-lead attribution would need a per-lead saliency method, "
            "which isn't implemented yet."
        )

    if show_chat:
        st.divider()
        render_chat(analysis, ecg_record, chat_key)


def page_new_analysis():
    st.title("New ECG Analysis")

    if not os.path.exists(CHECKPOINT):
        st.error(f"No trained model found at '{CHECKPOINT}'. Run `python src/train.py` first.")
        return

    st.markdown("**Supported ECG format: WFDB (.hea + .dat)**")
    st.caption("Upload the matching .hea and .dat files for one ECG recording.")

    col_a, col_b = st.columns(2)
    with col_a:
        hea_file = st.file_uploader("Select .hea file", type=["hea"])
    with col_b:
        dat_file = st.file_uploader("Select matching .dat file", type=["dat"])

    if not hea_file or not dat_file:
        st.info("Select both files to continue.")
        return

    hea_base = os.path.splitext(hea_file.name)[0]
    dat_base = os.path.splitext(dat_file.name)[0]
    if hea_base != dat_base:
        st.error(f"File name mismatch: '{hea_file.name}' and '{dat_file.name}' don't "
                 f"appear to belong to the same recording.")
        return

    try:
        raw_signal, fs, lead_names, load_warnings = load_wfdb_pair(hea_file, dat_file)
        val_warnings, duration_sec = validate_signal(raw_signal, fs)
    except ECGLoadError as e:
        st.error(f"Could not process this recording: {e}")
        return

    for w in load_warnings + val_warnings:
        st.warning(w)

    with st.container(border=True):
        st.subheader("ECG Uploaded Successfully")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Recording ID", hea_base)
        c2.metric("Sampling rate", f"{fs:.0f} Hz")
        c3.metric("Leads", len(lead_names))
        c4.metric("Duration", f"{duration_sec:.1f} s")

    with st.expander("View ECG"):
        lead_choice = st.selectbox("Lead", lead_names, key="preview_lead")
        st.line_chart(raw_signal[:, lead_names.index(lead_choice)])

    if st.button("Analyze ECG", type="primary"):
        with st.spinner("Running ECG analysis..."):
            hr_result = detect_heart_rate(raw_signal[:, 0], fs)
            model_input, prep_warnings = prepare_for_model(raw_signal, fs)
            for w in prep_warnings:
                st.warning(w)

            model = load_model()
            x_tensor = torch.from_numpy(model_input).unsqueeze(0)
            with torch.no_grad():
                logits = model(x_tensor)
                probs = model.calibrated_probs(logits)[0]
            pred_idx = int(torch.argmax(probs).item())
            pred_class = SUPERCLASSES[pred_idx]
            confidence = float(probs[pred_idx].item())

            cam = grad_cam_1d(model, x_tensor, pred_idx)
            start_sec, end_sec = top_attributed_region(cam, fs=100)

            query = build_query(pred_class, CLASS_FULL_NAMES[pred_class])
            passages = []
            if os.path.exists(STORE_PATH):
                raw_passages = retrieve(query, store_path=STORE_PATH, top_k=3)
                passages = [{"source": s, "text": t, "score": sc} for s, t, sc in raw_passages]

            try:
                explanation = compose_response(
                    pred_class, confidence, start_sec, end_sec,
                    [(p["source"], p["text"], p["score"]) for p in passages],
                    heart_rate=hr_result["heart_rate"], n_rpeaks=hr_result["n_rpeaks"],
                )
            except Exception as e:
                st.warning(f"Could not generate the AI explanation: {e}. "
                           f"The prediction and measurements below are still valid.")
                explanation = None

            record_uuid = str(uuid.uuid4())
            file_dir = os.path.join(ECG_FILES_DIR, record_uuid)
            os.makedirs(file_dir, exist_ok=True)
            with open(os.path.join(file_dir, hea_file.name), "wb") as f:
                f.write(hea_file.getbuffer())
            with open(os.path.join(file_dir, dat_file.name), "wb") as f:
                f.write(dat_file.getbuffer())

            ecg_id = cdb.create_ecg_record(
                st.session_state.user_id, hea_file.name, "wfdb", fs,
                len(lead_names), duration_sec, file_dir=file_dir,
            )
            features = {"heart_rate": hr_result["heart_rate"], "n_rpeaks": hr_result["n_rpeaks"],
                        "hr_reliable": hr_result["reliable"]}
            xai = {"region_start_sec": start_sec, "region_end_sec": end_sec}
            analysis_id = cdb.create_analysis(
                ecg_id, pred_class, confidence, features, xai, passages, explanation,
            )
            ecg_record = cdb.get_ecg_record(ecg_id)
            analysis = cdb.get_analysis(analysis_id)
            st.session_state.current_ecg = {"ecg_id": ecg_id, "analysis_id": analysis_id}

        render_report_body(analysis, ecg_record, chat_key=f"chat_{analysis_id}")

        st.divider()
        patient = {"name": st.session_state.name, "age": st.session_state.age,
                   "sex": st.session_state.sex, "username": st.session_state.username}
        os.makedirs(REPORTS_DIR, exist_ok=True)
        pdf_path = os.path.join(REPORTS_DIR, f"report_{analysis_id}.pdf")
        generate_pdf_report(pdf_path, patient, ecg_record, analysis, CLASS_FULL_NAMES)
        cdb.create_report(st.session_state.user_id, ecg_id, analysis_id,
                           {"generated_from": "New Analysis page"}, pdf_path)
        with open(pdf_path, "rb") as f:
            st.download_button("Download Report (PDF)", f,
                                file_name=os.path.basename(pdf_path), mime="application/pdf")
        st.success("Report generated and saved to your history.")


def page_history():
    st.title("Patient History")
    reports = cdb.get_reports_for_user(st.session_state.user_id)
    if not reports:
        st.info("No previous ECG analyses yet.")
        return

    open_id = st.session_state.get("open_report_id")
    st.subheader("Previous Reports")
    header = st.columns([2, 2, 3, 2, 1])
    for h, label in zip(header, ["Date", "ECG", "AI Interpretation", "Report", ""]):
        h.write(f"**{label}**")

    for r in reports:
        analysis = r["analysis"] or {}
        ecg = r["ecg_record"] or {}
        pred = analysis.get("prediction")
        friendly = FRIENDLY_DESCRIPTIONS.get(pred, "N/A").capitalize() if pred else "N/A"
        cols = st.columns([2, 2, 3, 2, 1])
        cols[0].write(r["created_at"][:19].replace("T", " "))
        cols[1].write(ecg.get("filename", "N/A"))
        cols[2].write(friendly)
        has_pdf = r.get("report_file_path") and os.path.exists(r["report_file_path"])
        cols[3].write("Available" if has_pdf else "Not generated")
        if cols[4].button("Open", key=f"open_{r['id']}"):
            st.session_state.open_report_id = r["id"]
            st.rerun()

    if open_id:
        report = cdb.get_report_by_id(open_id)
        if report:
            st.divider()
            st.subheader("Report Detail")
            render_report_body(report["analysis"] or {}, report["ecg_record"] or {},
                                chat_key=f"chat_{report['analysis_id']}")
            if report.get("report_file_path") and os.path.exists(report["report_file_path"]):
                with open(report["report_file_path"], "rb") as f:
                    st.download_button("Download PDF", f,
                                        file_name=os.path.basename(report["report_file_path"]),
                                        mime="application/pdf", key="hist_dl")


def page_compare():
    st.title("Compare ECG Records")
    ecg_records = cdb.get_ecg_records_for_user(st.session_state.user_id)
    if len(ecg_records) < 2:
        st.info("You need at least two ECG recordings to compare.")
        return

    options = {f"{r['upload_time'][:19].replace('T',' ')} - {r['filename']}": r for r in ecg_records}
    labels = list(options.keys())
    col_a, col_b = st.columns(2)
    with col_a:
        st.write("**Current ECG**")
        current_label = st.selectbox("Select ECG", labels, index=0, key="cmp_current")
    with col_b:
        st.write("**Previous ECG**")
        prev_label = st.selectbox("Select ECG", labels, index=min(1, len(labels) - 1), key="cmp_prev")

    if current_label == prev_label:
        st.warning("Select two different recordings to compare.")
        return

    ecg_current, ecg_prev = options[current_label], options[prev_label]

    def load_signal(ecg_record):
        file_dir = ecg_record.get("file_dir")
        if not file_dir or not os.path.isdir(file_dir):
            return None, None
        files = os.listdir(file_dir)
        hea = next((f for f in files if f.endswith(".hea")), None)
        dat = next((f for f in files if f.endswith(".dat")), None)
        if not hea or not dat:
            return None, None
        import wfdb
        record = wfdb.rdrecord(os.path.join(file_dir, os.path.splitext(hea)[0]))
        return record.p_signal, list(record.sig_name)

    sig_current, leads_current = load_signal(ecg_current)
    sig_prev, leads_prev = load_signal(ecg_prev)

    st.subheader("ECG Waveform Comparison")
    if sig_current is not None and sig_prev is not None:
        common_leads = [l for l in leads_current if l in leads_prev]
        if common_leads:
            lead_choice = st.selectbox("Lead", common_leads, key="cmp_lead")
            wc1, wc2 = st.columns(2)
            with wc1:
                st.caption("Current")
                st.line_chart(sig_current[:, leads_current.index(lead_choice)])
            with wc2:
                st.caption("Previous")
                st.line_chart(sig_prev[:, leads_prev.index(lead_choice)])
        else:
            st.caption("No common lead names found between these recordings.")
    else:
        st.caption("Original waveform data not available for one or both recordings.")

    all_reports = cdb.get_reports_for_user(st.session_state.user_id)
    analysis_current = next((r["analysis"] for r in all_reports if r["ecg_record_id"] == ecg_current["id"]), None)
    analysis_prev = next((r["analysis"] for r in all_reports if r["ecg_record_id"] == ecg_prev["id"]), None)

    st.subheader("Feature Comparison")
    if analysis_current and analysis_prev:
        f_cur = analysis_current.get("features") or {}
        f_prev = analysis_prev.get("features") or {}
        rows = []
        if f_cur.get("heart_rate") is not None and f_prev.get("heart_rate") is not None:
            change = f_cur["heart_rate"] - f_prev["heart_rate"]
            rows.append(["Heart rate (bpm)", f"{f_prev['heart_rate']:.1f}", f"{f_cur['heart_rate']:.1f}", f"{change:+.1f}"])
        if rows:
            st.table({"Feature": [r[0] for r in rows], "Previous": [r[1] for r in rows],
                      "Current": [r[2] for r in rows], "Change": [r[3] for r in rows]})
        else:
            st.caption("No comparable features available for both recordings.")
    else:
        st.caption("Analysis not available for one or both recordings.")

    st.subheader("AI Interpretation Comparison")
    pred_prev = analysis_prev.get("prediction") if analysis_prev else None
    pred_cur = analysis_current.get("prediction") if analysis_current else None
    pc1, pc2 = st.columns(2)
    pc1.metric("Previous", FRIENDLY_DESCRIPTIONS.get(pred_prev, "N/A").capitalize() if pred_prev else "N/A")
    pc2.metric("Current", FRIENDLY_DESCRIPTIONS.get(pred_cur, "N/A").capitalize() if pred_cur else "N/A")

    if pred_prev and pred_cur:
        if pred_prev != pred_cur:
            st.write(f"The model's interpretation changed from "
                     f"**{FRIENDLY_DESCRIPTIONS.get(pred_prev, pred_prev)}** to "
                     f"**{FRIENDLY_DESCRIPTIONS.get(pred_cur, pred_cur)}**.")
        else:
            st.write("The model produced the same interpretation for both recordings.")
    st.caption(
        "This describes how the model's output and measured features changed between "
        "recordings — it is not a clinical judgment about whether the patient's "
        "condition has improved or worsened. Clinical interpretation requires "
        "professional validation."
    )


def page_knowledge_base():
    st.title("Knowledge Base")
    st.caption("Medical reference documents used to ground CardioAgent's explanations. "
               "Kept separate from patient data.")

    if st.button("Build Default Knowledge Base"):
        with st.spinner("Indexing..."):
            store = VectorStore()
            ok = seed_default_kb(store)
            if ok:
                store.save(STORE_PATH)
                st.success(f"Indexed {len(store.chunks)} chunks.")
            else:
                st.error("Default knowledge base file not found.")

    st.divider()
    uploaded_files = st.file_uploader("Upload PDF/TXT", type=["pdf", "txt"], accept_multiple_files=True)
    if st.button("Add Documents"):
        if not uploaded_files:
            st.warning("No files selected.")
        else:
            with st.spinner("Processing..."):
                store = VectorStore()
                if os.path.exists(STORE_PATH):
                    store.load(STORE_PATH)
                os.makedirs("documents", exist_ok=True)
                for uf in uploaded_files:
                    save_path = os.path.join("documents", uf.name)
                    with open(save_path, "wb") as f:
                        f.write(uf.getbuffer())
                    try:
                        raw_text = extract_text(save_path)
                    except Exception as e:
                        st.error(f"Failed to process {uf.name}: {e}")
                        continue
                    cleaned = clean_text(raw_text)
                    if not cleaned:
                        st.warning(f"No extractable text in {uf.name}.")
                        continue
                    chunks = chunk_text(cleaned)
                    store.add_texts(chunks, source=uf.name)
                    st.success(f"{uf.name}: indexed")
                store.save(STORE_PATH)

    st.divider()
    st.subheader("Indexed Documents")
    if os.path.exists(STORE_PATH):
        store = VectorStore()
        store.load(STORE_PATH)
        docs = store.list_documents()
        if docs:
            for src, count in docs.items():
                with st.container(border=True):
                    c1, c2, c3 = st.columns([3, 1, 1])
                    c1.write(f"**{src}**")
                    c2.write("Indexed")
                    c3.write(f"{count} chunks")
        else:
            st.caption("No documents indexed yet.")
    else:
        st.caption("No knowledge base built yet.")


PAGES = {
    "Dashboard": page_dashboard,
    "New Analysis": page_new_analysis,
    "History": page_history,
    "Compare": page_compare,
    "Knowledge Base": page_knowledge_base,
}
PAGES[st.session_state.page]()
