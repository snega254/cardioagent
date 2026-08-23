# CardioAgent — Research Prototype (PTB-XL, real document RAG + frontend)

## What's new since your last run (read this first)

You already have a working, trained model and real results:
- Macro AUROC 0.8855 on the official PTB-XL test fold
- Per-class AUROC: NORM 0.9306, MI 0.8862, STTC 0.9251, CD 0.9001, HYP 0.7857
- Faithfulness sanity check: top-attributed deletion drop 0.1335 vs random
  deletion drop 0.0275 (difference 0.1059)

**Keep using your existing `checkpoint.pt` — you do not need to retrain.**
`model.py`, `train.py`, `dataset.py`, `preprocessing.py`, `gradcam.py` are
unchanged. Only these are new/changed:

- `src/vector_store.py` (NEW) — real, persistent, growable document vector
  store using sentence embeddings (replaces the old hardcoded 15-entry
  TF-IDF index for retrieval, though `knowledge_base.py`/`kb_texts.json`
  are kept so you can still seed those 15 entries into the new store).
- `src/document_processing.py` (NEW) — PDF/TXT text extraction, cleaning, chunking.
- `src/ingest_documents.py` (NEW) — CLI to bulk-add documents to the store.
- `src/app.py` (NEW) — Streamlit frontend (ECG Analysis tab + Knowledge Base tab).
- `src/rag.py`, `src/respond.py`, `src/pipeline.py` — updated to use the
  real vector store and to show which source document each retrieved
  passage came from. Also fixed a Windows console encoding bug (em-dashes
  were displaying as `ù` in your terminal — replaced with plain hyphens).
- `src/list_valid_ids.py` (NEW) — prints valid `--record_id` values (your
  `12000` error was because that id has no mapped diagnostic label, not a bug).

## Steps to run now

```bash
pip install -r requirements.txt
```

### 1. Build your knowledge base (real documents)

Put any PDF/TXT cardiology reference material in a `documents/` folder,
then:

```bash
python src/ingest_documents.py --seed_default_kb
```

This seeds the original 15 hand-written entries AND ingests anything in
`documents/`. First run downloads the embedding model (~80MB, needs
internet once, then works offline). This creates `vector_store.pkl`.

If you have zero documents ready and no time to find any, running just
`--seed_default_kb` alone still gives you a real (if small) working
vector-embedding-backed store — better than nothing, and still real, not
simulated.

### 2. Try the end-to-end CLI demo

```bash
python src/list_valid_ids.py
python src/pipeline.py --record_id <one of the ids just printed>
```

### 3. Launch the frontend

```bash
streamlit run src/app.py
```

Use the "Knowledge Base" tab to upload more documents live, and the "ECG
Analysis" tab to run the full pipeline and see every step (waveform,
prediction, Grad-CAM, retrieved evidence with source names, final
explanation).

## For the paper

Use your real numbers above. For the RAG section, report:
- Number of source documents ingested and total chunk count (printed by
  `ingest_documents.py` and shown in the Knowledge Base tab).
- 1-2 qualitative examples of retrieved evidence + similarity scores for
  a specific prediction (from `pipeline.py` output).
- State honestly: retrieval quality was checked qualitatively (does the
  retrieved passage plausibly relate to the predicted class), not via a
  formal precision/recall protocol against labeled relevance judgments —
  that would need a labeled query-relevance set you don't have time to
  build. Say this explicitly as a limitation, don't imply more rigor than
  this.
