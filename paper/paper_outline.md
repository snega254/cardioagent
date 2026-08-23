# CardioAgent: An Evidence-Grounded, Explainable ECG Classification Pipeline
### (Short/Workshop Paper Outline — fill in only with numbers from your own runs)

## Abstract
[FILL IN LAST, after Results. 3-4 sentences: problem, method, what you
measured, one honest headline result.]

## 1. Introduction
- Motivate: ECG-based AI diagnostic support is increasingly explored, but
  systems combining prediction, explainability, and knowledge-grounded
  explanation are rarely evaluated for whether the explanation is
  *faithful* to the model, or whether it risks introducing unsupported
  claims.
- State your scope honestly: this is a lightweight, fully open-source,
  reproducible prototype and a preliminary faithfulness sanity check —
  not a large-scale clinical validation. Say this explicitly, in the
  introduction, not just the limitations section.

## 2. Related Work
- Summarize (in your own words, no verbatim quoting) the landscape you
  and I discussed: deep learning for ECG classification generally; GAF-
  based image representations for ECG (cite the specific paper you
  reviewed, if using it); RAG-grounded LLM explanation systems for ECG
  (ECG-Chat, CardioRAG, the MIT RAG-ECG paper — verify these citations
  yourself before submission, from the actual papers, not from this
  conversation).
- State plainly what already exists (ECG+XAI+RAG systems already exist)
  and scope your contribution narrowly and honestly (see Section 5).

## 3. Method
### 3.1 Data
PTB-XL, 100Hz version, official `strat_fold` split (folds 1-8 train, 9
val, 10 test). [FILL IN: exact number of records used in train/val/test
after your --max_records cap, from the printed output of train.py]

### 3.2 Preprocessing
Bandpass filter (0.5-40Hz, 4th order Butterworth), per-lead z-score
normalization. [State any deviations you made from the provided code.]

### 3.3 Model
Small 1D-CNN, 4 convolutional blocks, global average pooling, linear
classification head, multi-label (sigmoid) output over 5 PTB-XL
diagnostic superclasses (NORM/MI/STTC/CD/HYP). Post-hoc temperature
scaling fit on the validation fold for calibration.
[FILL IN: total parameter count, training time, number of epochs actually
run — from train.py output]

### 3.4 Explainability
Grad-CAM on the final convolutional layer, upsampled to input resolution,
used to identify the most-attributed time region of the waveform per
prediction.

### 3.5 Faithfulness Sanity Check
Deletion test: compare the drop in predicted probability from zeroing the
top-attributed region vs. a random region of equal size, averaged over
[FILL IN: n] test records. Explicitly note this is a lightweight sanity
check, not a comprehensive faithfulness validation (no stability testing
under noise, no comparison across multiple attribution methods).

### 3.6 Knowledge Retrieval (RAG)
Hand-authored knowledge base of [FILL IN: count] short passages covering
the 5 diagnostic superclasses and basic ECG terminology, written by the
authors for this project (not scraped) to ensure quality control and
avoid licensing concerns. TF-IDF vectorization with cosine-similarity
retrieval [+ FAISS indexing, if you enabled it]. Query is constructed
purely from the predicted class label and name — the retrieval component
never receives ECG signal data, enforcing a logical separation between
signal analysis and knowledge grounding.

### 3.7 Response Generation
Template-based composition of the final explanation from: predicted
class, calibrated confidence, Grad-CAM-identified time region, and
retrieved passages. State explicitly: this is a deliberate design choice
to guarantee the explanation contains no claims beyond what the pipeline
itself produced (i.e., no hallucination risk), traded off against the
more natural phrasing an LLM-based generator might provide.

## 4. Results
[FILL IN EVERYTHING BELOW FROM YOUR ACTUAL RUN OUTPUT — do not estimate
or invent numbers]
- Table: per-class and macro AUROC on the official test fold.
- Faithfulness sanity check: mean probability drop, top-attributed vs.
  random deletion, and the difference between them.
- 1-2 concrete end-to-end example outputs from pipeline.py (full text of
  the generated response for a real record ID), presented as qualitative
  examples, clearly labeled as such.

## 5. Discussion — Contribution and Honest Scoping
- State plainly: the individual components (1D-CNN for ECG, Grad-CAM,
  RAG-grounded explanation) are established techniques, not novel in
  themselves.
- State the actual contribution: a fully reproducible, offline-capable,
  hallucination-safe (template-grounded) pipeline with an explicit,
  logically separated architecture and an initial Grad-CAM faithfulness
  sanity check for ECG classification — positioned as a small step toward
  addressing the faithfulness-evaluation gap in current ECG-XAI-RAG
  systems, not a solved problem.
- Explicitly do NOT claim: first system to combine ECG+XAI+RAG; a
  clinically validated diagnostic tool; state-of-the-art classification
  performance.

## 6. Limitations
- Small model, small training subset (given time budget) — likely
  underperforms larger published PTB-XL baselines; report this honestly
  rather than omitting comparison.
- Faithfulness check is a single lightweight sanity check, not a full
  validation.
- No human/clinician evaluation of explanation usefulness.
- Template-based generation, not natural free-text explanation.
- Single dataset, no external validation cohort.

## 7. Conclusion
[FILL IN: 3-4 sentences summarizing what was built, what was measured,
and the one honest takeaway.]

## References
[FILL IN: your actual reference list — verify every citation against the
real paper before submission; do not carry over any citation from this
conversation without checking it yourself.]
