# CardioAgent — Conversational AI-Assisted ECG Analysis

## What's new in this version

- **Gemini-based explanation** (`src/respond.py`) replaces the old
  template system — the LLM turns structured pipeline output (prediction,
  Grad-CAM region, heart rate, RAG evidence) into plain-language text,
  under a strict prompt that forbids inventing findings.
- **Embedded chatbot** (`src/chat.py`) — "Ask CardioAgent" appears right
  below every analysis result (both fresh and reopened from History).
  Every answer is grounded in that specific analysis's real data; the
  prompt explicitly tells the model to say "not measured" rather than
  guess when asked about something the pipeline didn't produce.
- **Friendly-first UI** — the main view leads with a plain-language
  interpretation and real measurements (heart rate, leads, duration).
  Confidence, the internal class code, and retrieval similarity scores
  are moved into a collapsed "Technical details" expander, not shown by
  default, per your requirements doc.
- **Graceful Gemini failure handling** — if the API key is missing or a
  call fails, the app shows a clear warning and still saves/displays the
  real prediction and measurements; it never crashes or silently drops
  data.

## New: Clinical Triage Assessment (src/triage.py, src/triage_rules.py)

An optional expander inside every analysis result — "Clinical Triage
Assessment" — lets you enter patient vitals/symptoms/history/medications
and generates an emergency-triage-style 5-part assessment (urgency tier,
waveform audit, bedside protocol, contraindication alerts, guideline
citations), modeled on the ACC/AHA-style prompt you provided.

**Hard rule, enforced in the prompt itself, not just by convention:**
every medication/disposition item is framed as a recommendation requiring
physician confirmation — never as an autonomous order. This is stated as
a non-negotiable rule in `triage.py`'s system prompt regardless of
urgency tier.

**Safety-critical design choice:** contraindication checks (BP<90 for
nitrates, PDE-5 inhibitor timing window, bradycardia/tachycardia
thresholds, inferior-MI-pattern RV-infarction flag) are computed
**deterministically in Python** (`triage_rules.py`), not left to the LLM
to notice under time pressure. The LLM narrates these pre-computed flags
and grounds them in retrieved guidelines — it doesn't do the arithmetic
itself. This is fully unit-tested: 7 scenarios including your exact
patient case (BP 88/58, sildenafil 18h prior, inferior-lead MI pattern)
— all 3 nitrate contraindication reasons and the beta-blocker caution
were verified to fire correctly. A real bug was caught and fixed during
this testing (the original lead-detection logic missed "Myocardial
Infarction" as a full string vs. the short code "MI" — now checks both).

**Known, stated limitation:** the current Grad-CAM implementation
produces a time-window attribution across all leads combined, not
per-lead attribution. The original triage design references specific
leads (e.g., "II, III, aVF") — this isn't yet computed by the pipeline.
The UI says this explicitly rather than faking per-lead output. Extending
to per-lead Grad-CAM would need a different attribution method, since the
current model's first conv layer mixes all 12 leads together immediately.

I added 5 new knowledge-base entries (`GUIDELINES` category) covering
nitrate/beta-blocker contraindication concepts and inferior-MI/RV risk,
written in my own words — not copied from ACC/AHA text, which is
copyrighted. If you want the triage assessment grounded in the actual
guideline documents, upload real ACC/AHA/ESC PDFs via the Knowledge Base
tab's document upload.

## Model name check

You used `gemini-3.6-flash` — I checked this against Google's current
release notes (today, Aug 25 2026): it's a real, stable, generally-
available model. It's not the newest Flash model (Gemini 3.7 Flash
shipped Aug 13), but it's fully supported and not deprecated. No change
needed, but if you want the newest one, `gemini-3.7-flash` is the
current successor — swap `GEMINI_MODEL` in `src/respond.py` if desired.

## Setup

```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# edit secrets.toml: real MongoDB Atlas URI + real Gemini API key
# (get a key at https://aistudio.google.com/apikey)
```

## Run

```bash
streamlit run src/app.py
```

## Testing note — read before assuming something is broken

**What I verified directly:**
- `respond.py` / `chat.py`: prompt construction, safety-rule inclusion,
  missing-data fallback text ("Not available"), and the missing-API-key
  error path — all tested without a real API call.
- `app.py`: ran the full app via Streamlit's official `AppTest` framework
  with `mongomock` standing in for Atlas. Registered a real user, logged
  in, navigated all 5 pages, then seeded a real mock analysis+report and
  confirmed the History detail view renders correctly — real heart-rate/
  leads/duration metrics, the chat input box present, and the technical
  details JSON expander all confirmed rendering with zero exceptions.

**What I could NOT test (needs your machine and real credentials):**
- An actual Gemini API call — no network route to Google's API from my
  sandbox. The prompt-building and error-handling around it are verified;
  the network call itself is not.
- The real Atlas connection.
- The full "Analyze ECG" flow against your real trained `checkpoint.pt`
  and a real WFDB file.

If any of those three break, send me the exact error immediately —
that's precisely the boundary I couldn't push past from here.
