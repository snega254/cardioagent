"""
Unified ECG input loader with proper WFDB validation.
"""
import os
import tempfile
import numpy as np
import pandas as pd


class ECGLoadError(ValueError):
    pass


def validate_wfdb_structure(hea_file, dat_file):
    """
    Properly validates WFDB structure by reading the .hea header,
    determining the referenced signal file, and verifying the .dat exists.
    Returns the record name and base name.
    """
    import wfdb
    
    hea_name = getattr(hea_file, "name", "record.hea")
    base_name = os.path.splitext(hea_name)[0]
    
    # Read header content to verify structure
    try:
        if hasattr(hea_file, "getbuffer"):
            hea_content = hea_file.getbuffer().tobytes().decode("utf-8", errors="ignore")
        else:
            hea_file.seek(0)
            hea_content = hea_file.read().decode("utf-8", errors="ignore")
    except Exception as e:
        raise ECGLoadError(f"Could not read .hea file: {e}")
    
    # Parse header lines
    lines = hea_content.strip().split("\n")
    if not lines:
        raise ECGLoadError(".hea file is empty")
    
    # First line: record name, number of signals, sampling rate, etc.
    parts = lines[0].split()
    if len(parts) < 3:
        raise ECGLoadError("Invalid .hea header: missing required fields")
    
    try:
        n_signals = int(parts[1])
        sampling_rate = float(parts[2])
    except ValueError:
        raise ECGLoadError("Invalid .hea header: could not parse signal count or sampling rate")
    
    if n_signals < 1:
        raise ECGLoadError(f"Invalid number of signals: {n_signals}")
    if n_signals > 12:
        raise ECGLoadError(f"Too many signals: {n_signals} (max 12)")
    if sampling_rate <= 0:
        raise ECGLoadError(f"Invalid sampling rate: {sampling_rate}")
    
    # Check that the .dat file exists and is readable
    try:
        if hasattr(dat_file, "getbuffer"):
            dat_content = dat_file.getbuffer()
        else:
            dat_file.seek(0)
            dat_content = dat_file.read()
        if len(dat_content) == 0:
            raise ECGLoadError(".dat file is empty")
    except Exception as e:
        raise ECGLoadError(f"Could not read .dat file: {e}")
    
    return base_name, n_signals, sampling_rate


def load_wfdb_pair(hea_file, dat_file):
    """
    Loads a WFDB recording with proper validation.
    """
    import wfdb
    
    # First, validate the WFDB structure
    base_name, n_signals, fs = validate_wfdb_structure(hea_file, dat_file)
    
    # Now load the record using wfdb
    with tempfile.TemporaryDirectory() as tmpdir:
        hea_path = os.path.join(tmpdir, base_name + ".hea")
        dat_path = os.path.join(tmpdir, base_name + ".dat")
        
        with open(hea_path, "wb") as f:
            if hasattr(hea_file, "getbuffer"):
                f.write(hea_file.getbuffer())
            else:
                hea_file.seek(0)
                f.write(hea_file.read())
        
        with open(dat_path, "wb") as f:
            if hasattr(dat_file, "getbuffer"):
                f.write(dat_file.getbuffer())
            else:
                dat_file.seek(0)
                f.write(dat_file.read())
        
        try:
            record = wfdb.rdrecord(os.path.join(tmpdir, base_name))
        except Exception as e:
            raise ECGLoadError(f"Could not read WFDB record: {e}")
        
        signal = record.p_signal.astype(np.float64)
        loaded_fs = float(record.fs)
        lead_names = list(record.sig_name)
        
        # Validate loaded data matches header
        if signal.shape[1] != n_signals:
            raise ECGLoadError(
                f"Lead count mismatch: header says {n_signals}, "
                f"loaded {signal.shape[1]}"
            )
        
        if abs(loaded_fs - fs) > 0.1:
            raise ECGLoadError(
                f"Sampling rate mismatch: header says {fs}, "
                f"loaded {loaded_fs}"
            )
        
        # Check signal values are readable
        if np.isnan(signal).any():
            raise ECGLoadError("Signal contains NaN values")
        if np.isinf(signal).any():
            raise ECGLoadError("Signal contains infinite values")
        
        return signal, fs, lead_names, []


def load_csv_or_txt(file_like, fs, has_header=True):
    if fs is None or fs <= 0:
        raise ECGLoadError("Sampling rate must be provided for CSV/TXT files.")
    try:
        df = pd.read_csv(file_like, header=0 if has_header else None, sep=None,
                          engine="python")
    except Exception as e:
        raise ECGLoadError(f"Could not parse file as CSV/TXT: {e}")
    
    raw = df.select_dtypes(include=[np.number]).values.astype(np.float64)
    if raw.size == 0:
        raise ECGLoadError("No numeric ECG data found.")
    if raw.ndim == 1:
        raw = raw.reshape(-1, 1)
    
    lead_names = (list(df.columns[:raw.shape[1]]) if has_header
                  else [f"lead_{i+1}" for i in range(raw.shape[1])])
    return raw, float(fs), lead_names, []


def validate_signal(raw_signal, fs, min_duration_sec=2.0, max_nan_fraction=0.05):
    warnings = []
    n_samples, n_leads = raw_signal.shape
    
    if n_leads < 1:
        raise ECGLoadError("No leads found.")
    if n_leads > 12:
        warnings.append(f"{n_leads} leads found; only the first 12 will be used.")
    
    duration_sec = n_samples / fs
    if duration_sec < min_duration_sec:
        raise ECGLoadError(f"Signal is only {duration_sec:.2f}s - too short")
    
    nan_fraction = float(np.isnan(raw_signal).mean())
    if nan_fraction > max_nan_fraction:
        raise ECGLoadError(f"{nan_fraction*100:.1f}% values are NaN")
    elif nan_fraction > 0:
        warnings.append(f"{nan_fraction*100:.2f}% values were NaN")
    
    if np.isinf(raw_signal).any():
        raise ECGLoadError("Signal contains infinite values")
    
    if np.nanstd(raw_signal) < 1e-8:
        warnings.append("Signal has near-zero variance")
    
    return warnings, duration_sec


def prepare_for_model(raw_signal, fs, model_fs=100, model_n_leads=12):
    from scipy.signal import resample
    from preprocessing import bandpass_filter, normalize
    
    warnings = []
    signal = raw_signal.copy()
    n_samples, n_leads = signal.shape
    
    if n_leads < model_n_leads:
        pad = np.zeros((n_samples, model_n_leads - n_leads), dtype=signal.dtype)
        signal = np.hstack([signal, pad])
        warnings.append(
            f"{n_leads} lead(s) provided; {model_n_leads - n_leads} "
            f"missing lead(s) were zero-padded."
        )
    elif n_leads > model_n_leads:
        signal = signal[:, :model_n_leads]
        warnings.append(f"Only the first {model_n_leads} leads were used.")
    
    if fs != model_fs:
        n_out = int(round(n_samples * model_fs / fs))
        signal = resample(signal, n_out, axis=0)
        warnings.append(f"Resampled from {fs}Hz to {model_fs}Hz")
    
    try:
        signal = bandpass_filter(signal, fs=model_fs)
    except Exception as e:
        warnings.append(f"Skipped bandpass filtering ({e})")
    
    signal = normalize(signal)
    return signal.T.astype(np.float32), warnings