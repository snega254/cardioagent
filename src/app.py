"""
CardioAgent frontend (Streamlit).

Run: streamlit run src/app.py

Two tabs:
1. ECG Analysis  - either pick a PTB-XL record OR upload your own ECG CSV,
   run the full pipeline, see every intermediate step.
2. Knowledge Base - build the small built-in knowledge base with one
   click, or upload PDF/TXT documents to add to it.
"""
import os

import torch

import streamlit as st
from document_processing import chunk_text, clean_text, extract_text
from gradcam import grad_cam_1d, top_attributed_region
from ingest_documents import seed_default_kb
from model import ECGConvNet
from preprocessing import SUPERCLASSES, load_and_preprocess_record, load_metadata
from rag import build_query, retrieve
from respond import CLASS_FULL_NAMES, compose_response
from user_upload import load_uploaded_csv
from vector_store import VectorStore

DATA_DIR = "data/ptbxl"
CHECKPOINT = "checkpoint.pt"
STORE_PATH = "vector_store.pkl"

st.set_page_config(page_title="CardioAgent", layout="wide")
st.title("CardioAgent - Explainable ECG Analysis")
st.caption("Research prototype. Not a medical device. Not for clinical use.")

tab_analysis, tab_kb = st.tabs(["ECG Analysis", "Knowledge Base"])


@st.cache_resource
def load_model():
    model = ECGConvNet()
    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.log_temperature.data = ckpt["log_temperature"]
    model.eval()
    return model


@st.cache_data
def load_meta():
    return load_metadata(DATA_DIR)


def run_analysis_and_display(x, ground_truth_label=None):
    """x: preprocessed signal [n_leads, n_samples]. Shared by both the
    PTB-XL path and the user-upload path so behavior stays identical."""
    st.subheader("1. ECG Waveform (Lead I, preprocessed)")
    st.line_chart(x[0])

    model = load_model()
    x_tensor = torch.from_numpy(x).unsqueeze(0)
    with torch.no_grad():
        logits = model(x_tensor)
        probs = model.calibrated_probs(logits)[0]
    pred_idx = int(torch.argmax(probs).item())
    pred_class = SUPERCLASSES[pred_idx]
    confidence = float(probs[pred_idx].item())

    st.subheader("2. Model Prediction")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predicted class", CLASS_FULL_NAMES[pred_class])
        st.metric("Calibrated confidence", f"{confidence*100:.1f}%")
        if ground_truth_label is not None:
            st.caption(f"Ground truth (reference only, not model output): "
                       f"{ground_truth_label}")
    with col2:
        st.bar_chart({SUPERCLASSES[i]: float(probs[i])
                      for i in range(len(SUPERCLASSES))})

    cam = grad_cam_1d(model, x_tensor, pred_idx)
    start_sec, end_sec = top_attributed_region(cam, fs=100)

    st.subheader("3. Explainability (Grad-CAM)")
    st.line_chart(cam)
    st.write(f"Most influential region of the signal: "
             f"~{start_sec:.2f}s to {end_sec:.2f}s")

    st.subheader("4. Retrieved Medical Knowledge (RAG)")
    query = build_query(pred_class, CLASS_FULL_NAMES[pred_class])
    if os.path.exists(STORE_PATH):
        passages = retrieve(query, store_path=STORE_PATH, top_k=3)
        if passages:
            for source, text, score in passages:
                st.markdown(f"**Source: `{source}`** (similarity: {score:.3f})")
                st.write(text)
        else:
            st.warning("Knowledge base is empty. Build it in the "
                       "Knowledge Base tab.")
    else:
        passages = []
        st.warning("No knowledge base found yet. Build one in the "
                   "Knowledge Base tab first (one click).")

    st.subheader("5. CardioAgent Final Explanation")
    response = compose_response(pred_class, confidence, start_sec, end_sec, passages)
    st.text(response)


with tab_analysis:
    st.header("Analyze an ECG")

    if not os.path.exists(CHECKPOINT):
        st.error(f"No trained model found at '{CHECKPOINT}'. Run "
                 f"`python src/train.py` first.")
    else:
        source_choice = st.radio(
            "ECG source",
            ["Upload my own ECG (CSV)", "Use a PTB-XL record"],
            horizontal=True,
        )

        if source_choice == "Upload my own ECG (CSV)":
            st.write("CSV format: each column is one ECG lead, each row is "
                     "one time sample. 1-12 leads supported (fewer than 12 "
                     "will be zero-padded, which reduces accuracy).")
            uploaded_ecg = st.file_uploader("Upload ECG CSV", type=["csv"])
            col_a, col_b = st.columns(2)
            with col_a:
                input_fs = st.number_input("Sampling rate of your file (Hz)",
                                            min_value=10, max_value=2000, value=250)
            with col_b:
                has_header = st.checkbox("File has a header row", value=True)

            if st.button("Run Analysis", type="primary", key="run_upload"):
                if uploaded_ecg is None:
                    st.error("Please upload a CSV file first.")
                else:
                    try:
                        x, warns = load_uploaded_csv(uploaded_ecg, input_fs, has_header)
                        for w in warns:
                            st.warning(w)
                        run_analysis_and_display(x, ground_truth_label=None)
                    except Exception as e:
                        st.error(f"Could not process the uploaded file: {e}")

        else:
            if not os.path.isdir(DATA_DIR):
                st.error(f"PTB-XL data not found at '{DATA_DIR}'. See "
                         f"README for download instructions.")
            else:
                df = load_meta()
                record_id = st.number_input(
                    "PTB-XL ecg_id (see src/list_valid_ids.py for examples)",
                    min_value=int(df.index.min()), max_value=int(df.index.max()),
                    value=int(df.index[0]), step=1,
                )
                if st.button("Run Analysis", type="primary", key="run_ptbxl"):
                    if record_id not in df.index:
                        st.error(f"ecg_id {record_id} has no mapped "
                                 f"diagnostic superclass. Try a different id.")
                    else:
                        row = df.loc[record_id]
                        x = load_and_preprocess_record(DATA_DIR, row.filename_lr)
                        run_analysis_and_display(x, ground_truth_label=row.superclasses)

with tab_kb:
    st.header("Knowledge Base")

    st.subheader("Quick start")
    st.write("Build the small, built-in ECG knowledge base with one click "
             "(21 hand-written entries covering the 5 diagnostic "
             "categories plus basic ECG terminology).")
    if st.button("Build / Rebuild Default Knowledge Base"):
        with st.spinner("Embedding and indexing... (first run downloads "
                         "the embedding model, ~80MB, needs internet once)"):
            store = VectorStore()
            ok = seed_default_kb(store)
            if ok:
                store.save(STORE_PATH)
                st.success(f"Built knowledge base with "
                           f"{len(store.chunks)} chunks.")
            else:
                st.error("knowledge_base/kb_texts.json not found.")

    st.divider()
    st.subheader("Add your own documents (optional)")
    st.write("Upload cardiology/ECG reference PDFs or TXT files to add to "
             "the knowledge base alongside the built-in entries.")
    uploaded_files = st.file_uploader(
        "Upload documents", type=["pdf", "txt"], accept_multiple_files=True
    )
    if st.button("Process and Add Uploaded Documents"):
        if not uploaded_files:
            st.warning("No files selected.")
        else:
            with st.spinner("Extracting, chunking, and embedding documents..."):
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
                        st.error(f"Failed to extract text from {uf.name}: {e}")
                        continue
                    cleaned = clean_text(raw_text)
                    if not cleaned:
                        st.warning(f"No extractable text in {uf.name} "
                                   f"(likely a scanned/image-only PDF).")
                        continue
                    chunks = chunk_text(cleaned)
                    store.add_texts(chunks, source=uf.name)
                    st.success(f"Indexed {uf.name}: {len(chunks)} chunks")
                store.save(STORE_PATH)

    st.divider()
    st.subheader("Indexed Documents")
    if os.path.exists(STORE_PATH):
        store = VectorStore()
        store.load(STORE_PATH)
        docs = store.list_documents()
        if docs:
            for src, count in docs.items():
                st.write(f"- **{src}**: {count} chunks")
            st.write(f"**Total chunks in index: {len(store.chunks)}**")
        else:
            st.write("Vector store exists but is empty.")
    else:
        st.write("No vector store built yet. Click the button above.")
