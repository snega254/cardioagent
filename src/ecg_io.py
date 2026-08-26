"""
Unified ECG input loader supporting the formats CardioAgent actually
supports: WFDB (.hea+.dat), CSV, TXT, and NumPy (.npy). Each format is
handled according to what it can and can't tell us automatically:

- WFDB: sampling rate and lead names are read from the header file
  automatically — never guessed.
- CSV/TXT: no reliable metadata standard exists for these, so the
  sampling rate MUST be supplied by the caller (never assumed).
- NPY: same as CSV/TXT — no metadata, sampling rate must be supplied.

Every loader returns (raw_signal [n_samples, n_leads], fs, lead_names,
warnings) or raises a ValueError with a clear, specific message if the
file can't be processed — never silently produces a wrong-shaped result.
"""
import os
import tempfile

import numpy as np
import pandas as pd


class ECGLoadError(ValueError):
    """Raised when an ECG file can't be validly loaded — always caught
    and shown to the user as a clear error, never silently ignored."""
    pass


def load_csv_or_txt(file_like, fs, has_header=True):
    if fs is None or fs <= 0:
        raise ECGLoadError("Sampling rate must be provided for CSV/TXT files "
                            "(this format has no reliable metadata standard).")
    try:
        df = pd.read_csv(file_like, header=0 if has_header else None, sep=None,
                          engine="python")
    except Exception as e:
        raise ECGLoadError(f"Could not parse the file as CSV/TXT: {e}")

    raw = df.select_dtypes(include=[np.number]).values.astype(np.float64)
    if raw.size == 0:
        raise ECGLoadError("No numeric ECG data found in the file.")
    if raw.ndim == 1:
        raw = raw.reshape(-1, 1)

    lead_names = (list(df.columns[:raw.shape[1]]) if has_header
                  else [f"lead_{i+1}" for i in range(raw.shape[1])])
    return raw, float(fs), lead_names, []


def load_npy(file_like, fs, leads_first=None):
    if fs is None or fs <= 0:
        raise ECGLoadError("Sampling rate must be provided for .npy files "
                            "(NumPy arrays carry no metadata).")
    try:
        arr = np.load(file_like)
    except Exception as e:
        raise ECGLoadError(f"Could not load .npy file: {e}")

    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim != 2:
        raise ECGLoadError(f"Expected a 1D or 2D array, got shape {arr.shape}.")

    warnings = []
    n0, n1 = arr.shape
    if n0 < n1 and n0 <= 12:
        arr = arr.T
        warnings.append(f"Array shape {arr.T.shape} looked like [leads, samples] "
                         f"and was transposed to [samples, leads]. Verify this is "
                         f"correct for your file.")

    lead_names = [f"lead_{i+1}" for i in range(arr.shape[1])]
    return arr.astype(np.float64), float(fs), lead_names, warnings


def load_wfdb_pair(hea_file, dat_file):
    """
    hea_file, dat_file: file-like objects (e.g. Streamlit UploadedFile)
    with matching basenames. Sampling rate and lead names are read
    automatically from the header — this is WFDB's actual metadata, not
    a guess.
    """
    import wfdb

    hea_name = getattr(hea_file, "name", "record.hea")
    base_name = os.path.splitext(hea_name)[0]

    with tempfile.TemporaryDirectory() as tmpdir:
        hea_path = os.path.join(tmpdir, base_name + ".hea")
        dat_path = os.path.join(tmpdir, base_name + ".dat")
        with open(hea_path, "wb") as f:
            f.write(hea_file.getbuffer() if hasattr(hea_file, "getbuffer") else hea_file.read())
        with open(dat_path, "wb") as f:
            f.write(dat_file.getbuffer() if hasattr(dat_file, "getbuffer") else dat_file.read())

        try:
            record = wfdb.rdrecord(os.path.join(tmpdir, base_name))
        except Exception as e:
            raise ECGLoadError(
                f"Could not read WFDB record (check that the .hea and .dat "
                f"filenames match exactly except for extension): {e}"
            )

        signal = record.p_signal.astype(np.float64)  # [n_samples, n_leads]
        fs = float(record.fs)
        lead_names = list(record.sig_name)

    return signal, fs, lead_names, []


def validate_signal(raw_signal, fs, min_duration_sec=1.0, max_nan_fraction=0.05):
    """
    Runs required pre-analysis validation. Returns (warnings, duration_sec)
    or raises ECGLoadError for hard failures.
    """
    warnings = []
    n_samples, n_leads = raw_signal.shape

    if n_leads < 1:
        raise ECGLoadError("No leads found in the signal.")
    if n_leads > 12:
        warnings.append(f"{n_leads} columns found; only the first 12 will "
                         f"be used as ECG leads.")

    duration_sec = n_samples / fs
    if duration_sec < min_duration_sec:
        raise ECGLoadError(f"Signal is only {duration_sec:.2f}s long - too "
                            f"short to analyze (minimum {min_duration_sec}s).")

    nan_fraction = float(np.isnan(raw_signal).mean())
    if nan_fraction > max_nan_fraction:
        raise ECGLoadError(f"{nan_fraction*100:.1f}% of values are missing/NaN - "
                            f"too much missing data to analyze reliably "
                            f"(maximum allowed: {max_nan_fraction*100:.0f}%).")
    elif nan_fraction > 0:
        warnings.append(f"{nan_fraction*100:.2f}% of values were missing/NaN.")

    if np.isinf(raw_signal).any():
        raise ECGLoadError("Signal contains infinite values - the file may be corrupted.")

    if np.nanstd(raw_signal) < 1e-8:
        warnings.append("Signal has near-zero variance - this may be a flat/dead "
                         "channel rather than real ECG data.")

    return warnings, duration_sec


def prepare_for_model(raw_signal, fs, model_fs=100, model_n_leads=12):
    """
    Takes a validated raw_signal [n_samples, n_leads] at its real fs, and
    prepares it exactly the way the trained model expects: resampled to
    model_fs, lead count adjusted to model_n_leads (zero-padded or
    truncated), bandpass filtered, normalized.

    Returns (model_input [n_leads, n_samples], warnings).

    IMPORTANT: run R-peak/heart-rate detection on the ORIGINAL raw_signal
    at the ORIGINAL fs, BEFORE calling this — resampling here is for the
    classifier only and would reduce timing precision for heart-rate math.
    """
    from scipy.signal import resample

    from preprocessing import bandpass_filter, normalize

    warnings = []
    signal = raw_signal.copy()
    n_samples, n_leads = signal.shape

    if n_leads < model_n_leads:
        pad = np.zeros((n_samples, model_n_leads - n_leads), dtype=signal.dtype)
        signal = np.hstack([signal, pad])
        warnings.append(f"{n_leads} lead(s) provided; {model_n_leads - n_leads} "
                         f"missing lead(s) were zero-padded. This reduces model "
                         f"accuracy compared to a full {model_n_leads}-lead recording.")
    elif n_leads > model_n_leads:
        signal = signal[:, :model_n_leads]
        warnings.append(f"Only the first {model_n_leads} leads were used.")

    if fs != model_fs:
        n_out = int(round(n_samples * model_fs / fs))
        signal = resample(signal, n_out, axis=0)
        warnings.append(f"Resampled from {fs}Hz to {model_fs}Hz for the model "
                         f"(the rate it was trained on).")

    try:
        signal = bandpass_filter(signal, fs=model_fs)
    except Exception as e:
        warnings.append(f"Skipped bandpass filtering ({e}).")

    signal = normalize(signal)

    return signal.T.astype(np.float32), warnings
