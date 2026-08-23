"""
Handles arbitrary user-uploaded ECG signals (a CSV file), so CardioAgent
can analyze someone's own ECG, not just a PTB-XL record picked by id.

Expected CSV format: rows = time samples, columns = leads (1 to 12
columns supported). If you have fewer than 12 leads (e.g. a single-lead
wearable recording), missing leads are zero-padded — this is flagged to
the user as reducing accuracy, since the model was trained on real 12-lead
data, not zero-padded data. This is an honest limitation, not something
to hide.
"""
import numpy as np
import pandas as pd
from scipy.signal import resample

from preprocessing import bandpass_filter, normalize

MODEL_N_LEADS = 12
MODEL_FS = 100  # Hz — matches the PTB-XL 100Hz version the model was trained on


def load_uploaded_csv(file_like, input_fs, has_header=True):
    """
    file_like: file path or file-like object (e.g. Streamlit's UploadedFile)
    input_fs: sampling rate in Hz of the uploaded signal, provided by the user
    Returns: (signal [n_leads, n_samples] float32, warnings: list[str])
    """
    warnings = []

    df = pd.read_csv(file_like, header=0 if has_header else None)
    raw = df.select_dtypes(include=[np.number]).values.astype(np.float32)

    if raw.size == 0:
        raise ValueError("No numeric data found in the uploaded CSV. "
                          "Make sure it contains ECG voltage values.")

    if raw.ndim == 1:
        raw = raw.reshape(-1, 1)

    n_samples_in, n_leads_in = raw.shape

    if n_leads_in < MODEL_N_LEADS:
        pad = np.zeros((n_samples_in, MODEL_N_LEADS - n_leads_in), dtype=np.float32)
        raw = np.hstack([raw, pad])
        warnings.append(
            f"Uploaded file had {n_leads_in} lead(s); the model expects 12. "
            f"Missing leads were zero-padded. Treat results as illustrative "
            f"only - this reduces accuracy compared to a real 12-lead recording."
        )
    elif n_leads_in > MODEL_N_LEADS:
        raw = raw[:, :MODEL_N_LEADS]
        warnings.append(
            f"Uploaded file had {n_leads_in} columns; only the first 12 "
            f"were used as ECG leads."
        )

    if input_fs != MODEL_FS:
        n_samples_out = int(round(n_samples_in * MODEL_FS / input_fs))
        if n_samples_out < 20:
            raise ValueError(
                "Uploaded signal is too short after resampling to be "
                "analyzed meaningfully. Please upload at least a few "
                "seconds of signal."
            )
        raw = resample(raw, n_samples_out, axis=0)
        warnings.append(f"Resampled from {input_fs}Hz to {MODEL_FS}Hz "
                         f"(the rate the model was trained on).")

    try:
        raw = bandpass_filter(raw, fs=MODEL_FS)
    except Exception as e:
        warnings.append(f"Skipped bandpass filtering — signal too short "
                         f"or filtering failed ({e}).")

    raw = normalize(raw)

    return raw.T.astype(np.float32), warnings  # [n_leads, n_samples]
