"""
Preprocessing for PTB-XL (100Hz version).

Loads metadata, maps SCP codes to the 5 official diagnostic superclasses
(NORM, MI, STTC, CD, HYP), loads raw waveforms via wfdb, and applies a
standard bandpass filter + per-lead normalization.
"""
import ast
import os

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def load_metadata(data_dir):
    """Load PTB-XL database CSV and SCP statement mapping."""
    db_path = os.path.join(data_dir, "ptbxl_database.csv")
    scp_path = os.path.join(data_dir, "scp_statements.csv")

    df = pd.read_csv(db_path, index_col="ecg_id")
    df.scp_codes = df.scp_codes.apply(ast.literal_eval)

    scp_df = pd.read_csv(scp_path, index_col=0)
    scp_df = scp_df[scp_df.diagnostic == 1]

    def codes_to_superclasses(scp_codes):
        classes = set()
        for code in scp_codes.keys():
            if code in scp_df.index:
                sc = scp_df.loc[code].diagnostic_class
                if isinstance(sc, str) and sc in SUPERCLASSES:
                    classes.add(sc)
        return list(classes)

    df["superclasses"] = df.scp_codes.apply(codes_to_superclasses)
    # Drop records with no mapped superclass (ambiguous/other) — standard practice
    df = df[df.superclasses.apply(len) > 0]
    return df


def multilabel_targets(df):
    """One-hot multi-label target matrix, aligned to df index order."""
    y = np.zeros((len(df), len(SUPERCLASSES)), dtype=np.float32)
    for i, classes in enumerate(df.superclasses.values):
        for c in classes:
            y[i, SUPERCLASSES.index(c)] = 1.0
    return y


def bandpass_filter(signal, fs=100, low=0.5, high=40.0, order=4):
    """4th-order Butterworth bandpass, applied per lead. signal: [n_samples, n_leads]"""
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    filtered = np.zeros_like(signal)
    for lead in range(signal.shape[1]):
        filtered[:, lead] = filtfilt(b, a, signal[:, lead])
    return filtered


def normalize(signal):
    """Per-lead z-score normalization."""
    mean = signal.mean(axis=0, keepdims=True)
    std = signal.std(axis=0, keepdims=True) + 1e-8
    return (signal - mean) / std


def preprocess_uploaded_signal(raw_signal, orig_fs, target_fs=100, target_leads=12):
    """
    Preprocesses a user-uploaded ECG signal so it matches what the trained
    model expects, as closely as possible.

    raw_signal: numpy array, shape [n_samples, n_leads_uploaded]
    orig_fs: sampling rate the uploaded signal was recorded at (user-provided)
    Returns: preprocessed array [target_leads, n_samples_at_target_fs]

    IMPORTANT LIMITATION (state this in your paper if you use this feature):
    the model was trained exclusively on real 12-lead PTB-XL recordings.
    If the uploaded signal has fewer than 12 leads, missing leads are
    zero-padded (not fabricated from the leads you do have) - this means
    predictions on non-12-lead uploads are a lower-confidence demonstration
    capability, not a validated one. If more than 12 leads are uploaded,
    only the first 12 are used.
    """
    import numpy as np
    from scipy.signal import resample

    if raw_signal.ndim == 1:
        raw_signal = raw_signal.reshape(-1, 1)

    n_samples_orig, n_leads_orig = raw_signal.shape

    # Resample to target_fs if needed
    if orig_fs != target_fs:
        n_samples_target = int(round(n_samples_orig * target_fs / orig_fs))
        resampled = resample(raw_signal, n_samples_target, axis=0)
    else:
        resampled = raw_signal

    # Adjust lead count
    n_samples, n_leads = resampled.shape
    if n_leads >= target_leads:
        adjusted = resampled[:, :target_leads]
    else:
        pad = np.zeros((n_samples, target_leads - n_leads), dtype=resampled.dtype)
        adjusted = np.concatenate([resampled, pad], axis=1)

    filtered = bandpass_filter(adjusted, fs=target_fs)
    normalized = normalize(filtered)
    return normalized.T.astype(np.float32)  # [n_leads, n_samples]


def load_and_preprocess_record(data_dir, filename_lr):
    """
    filename_lr: value from df.filename_lr (100Hz record path, relative to data_dir)
    Returns preprocessed signal, shape [n_leads, n_samples] (channels-first for PyTorch).
    """
    record_path = os.path.join(data_dir, filename_lr)
    signal, _ = wfdb.rdsamp(record_path)  # [n_samples, n_leads], fs=100
    signal = bandpass_filter(signal, fs=100)
    signal = normalize(signal)
    return signal.T.astype(np.float32)  # [n_leads, n_samples]
