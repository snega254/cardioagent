"""
R-peak detection, heart-rate calculation, and extended ECG measurements.
"""
import neurokit2 as nk
import numpy as np


def detect_heart_rate(signal_1lead, fs):
    """Detect R-peaks and calculate heart rate."""
    result = {
        "heart_rate": None,
        "n_rpeaks": 0,
        "reliable": False,
        "reason": "",
        "rr_intervals": [],
        "rr_variability": None,
    }
    
    duration_sec = len(signal_1lead) / fs
    if duration_sec < 2.0:
        result["reason"] = f"Signal too short ({duration_sec:.1f}s)"
        return result
    
    try:
        cleaned = nk.ecg_clean(signal_1lead, sampling_rate=fs)
        _, info = nk.ecg_peaks(cleaned, sampling_rate=fs)
        rpeak_indices = info["ECG_R_Peaks"]
    except Exception as e:
        result["reason"] = f"R-peak detection failed: {e}"
        return result
    
    n_rpeaks = len(rpeak_indices)
    result["n_rpeaks"] = int(n_rpeaks)
    
    if n_rpeaks < 3:
        result["reason"] = f"Only {n_rpeaks} R-peak(s) detected"
        return result
    
    rr_intervals_sec = np.diff(rpeak_indices) / fs
    mean_rr = float(np.mean(rr_intervals_sec))
    std_rr = float(np.std(rr_intervals_sec))
    
    result["rr_intervals"] = rr_intervals_sec.tolist()
    
    if mean_rr <= 0:
        result["reason"] = "Invalid RR intervals"
        return result
    
    heart_rate = 60.0 / mean_rr
    plausible = 30.0 <= heart_rate <= 220.0
    cv_rr = std_rr / mean_rr if mean_rr > 0 else float("inf")
    stable = cv_rr < 0.35
    
    result["rr_variability"] = cv_rr
    
    if not plausible:
        result["reason"] = f"Heart rate ({heart_rate:.1f} bpm) outside plausible range"
        return result
    
    if not stable:
        result["reason"] = f"RR intervals irregular (CV={cv_rr:.2f})"
        result["heart_rate"] = None
        return result
    
    result["heart_rate"] = round(heart_rate, 1)
    result["reliable"] = True
    result["reason"] = f"{n_rpeaks} R-peaks, stable rhythm (CV={cv_rr:.2f})"
    return result


def extract_ecg_measurements(signal, fs, rpeak_indices=None):
    """
    Extract extended ECG measurements:
    - Heart rate (from R-peaks)
    - RR intervals and variability
    - PR interval, QRS duration, QT interval where possible
    """
    measurements = {
        "heart_rate": None,
        "n_rpeaks": 0,
        "rr_intervals": [],
        "rr_variability": None,
        "pr_interval": None,
        "qrs_duration": None,
        "qt_interval": None,
        "qtc_interval": None,
        "rhythm_regularity": None,
    }
    
    # Use neurokit2 for comprehensive ECG analysis
    try:
        # Clean signal and detect peaks
        cleaned = nk.ecg_clean(signal, sampling_rate=fs)
        _, rpeaks_info = nk.ecg_peaks(cleaned, sampling_rate=fs)
        rpeaks = rpeaks_info["ECG_R_Peaks"]
        
        if len(rpeaks) >= 3:
            measurements["n_rpeaks"] = len(rpeaks)
            rr_intervals = np.diff(rpeaks) / fs
            measurements["rr_intervals"] = rr_intervals.tolist()
            mean_rr = np.mean(rr_intervals)
            measurements["heart_rate"] = round(60.0 / mean_rr, 1)
            measurements["rr_variability"] = round(float(np.std(rr_intervals) / mean_rr), 3)
            
            # Rhythm regularity
            cv_rr = measurements["rr_variability"]
            if cv_rr < 0.15:
                measurements["rhythm_regularity"] = "regular"
            elif cv_rr < 0.35:
                measurements["rhythm_regularity"] = "somewhat irregular"
            else:
                measurements["rhythm_regularity"] = "irregular"
        
        # Extract waveforms and intervals using neurokit2
        try:
            # Get waveform characteristics
            signal_df = nk.signal_distribution(cleaned, method="bins")
            
            # Find QRS complexes for duration estimation
            try:
                # This is a simplified approach - full ECG segmentation would use nk.ecg_delineate
                # For now, we estimate from the cleaned signal
                rpeaks = rpeaks_info["ECG_R_Peaks"]
                if len(rpeaks) > 1:
                    # Estimate QRS duration as the width of the QRS complex at half height
                    # This is a proxy - not as accurate as full delineation
                    qrs_durations = []
                    for rp in rpeaks[1:-1]:
                        # Look for QRS onset (before R) and offset (after R)
                        window = max(10, int(0.05 * fs))  # 50ms window
                        start = max(0, rp - window)
                        end = min(len(cleaned), rp + window)
                        segment = cleaned[start:end]
                        if len(segment) > 0:
                            # Estimate QRS width from signal derivative
                            derivative = np.diff(segment)
                            # Simple threshold-based detection
                            threshold = 0.3 * np.max(np.abs(derivative))
                            qrs_onset = np.where(np.abs(derivative) > threshold)[0]
                            if len(qrs_onset) > 2:
                                qrs_duration = (qrs_onset[-1] - qrs_onset[0]) / fs
                                qrs_durations.append(qrs_duration)
                    
                    if qrs_durations:
                        measurements["qrs_duration"] = round(np.mean(qrs_durations) * 1000, 1)  # ms
            
            except Exception:
                pass
            
            # Estimate PR and QT intervals from beat positions
            try:
                # Simple estimation from ECG waveform
                # This is a rough estimate - full delineation would be better
                if len(rpeaks) > 2:
                    # Find P waves (negative peaks before R)
                    pr_intervals = []
                    qt_intervals = []
                    
                    for i in range(1, len(rpeaks) - 1):
                        rp = rpeaks[i]
                        prev_rp = rpeaks[i-1]
                        
                        # PR interval: from previous R to current R, estimate P position
                        # Look for P wave in the PR segment
                        pr_segment_start = int(prev_rp + 0.12 * fs)  # 120ms after previous R
                        pr_segment_end = int(rp - 0.04 * fs)  # 40ms before current R
                        
                        if pr_segment_start < pr_segment_end and pr_segment_start < len(cleaned):
                            segment = cleaned[pr_segment_start:pr_segment_end]
                            if len(segment) > 0:
                                # Look for negative deflection (P wave)
                                p_idx = np.argmin(segment) if len(segment) > 0 else -1
                                if p_idx >= 0:
                                    p_pos = pr_segment_start + p_idx
                                    pr_interval = (rp - p_pos) / fs
                                    if 0.08 < pr_interval < 0.25:  # Normal PR range
                                        pr_intervals.append(pr_interval)
                        
                        # QT interval: from R to end of T wave
                        qt_segment_end = int(rp + 0.45 * fs)  # Up to 450ms after R
                        if qt_segment_end < len(cleaned):
                            segment = cleaned[rp:qt_segment_end]
                            if len(segment) > 0:
                                # Find T wave end (return to baseline)
                                t_end = np.argmin(np.abs(segment - np.mean(segment[-int(0.05*fs):])))
                                if t_end > 0:
                                    qt_interval = (rp + t_end - rp) / fs
                                    if 0.2 < qt_interval < 0.55:
                                        qt_intervals.append(qt_interval)
                    
                    if pr_intervals:
                        measurements["pr_interval"] = round(np.mean(pr_intervals) * 1000, 1)  # ms
                    
                    if qt_intervals and measurements["heart_rate"]:
                        qt_mean = np.mean(qt_intervals)
                        measurements["qt_interval"] = round(qt_mean * 1000, 1)  # ms
                        # QTc using Bazett's formula
                        hr = measurements["heart_rate"]
                        qtc = qt_mean / np.sqrt(60.0 / hr)
                        measurements["qtc_interval"] = round(qtc * 1000, 1)  # ms
            except Exception:
                pass
                
        except Exception:
            pass
            
    except Exception:
        pass
    
    return measurements


def extract_lead_measurements(signal, fs, lead_names=None):
    """Extract measurements per lead."""
    measurements_by_lead = {}
    
    for lead_idx in range(signal.shape[1]):
        lead_name = lead_names[lead_idx] if lead_names else f"lead_{lead_idx+1}"
        lead_signal = signal[:, lead_idx]
        
        try:
            cleaned = nk.ecg_clean(lead_signal, sampling_rate=fs)
            _, rpeaks_info = nk.ecg_peaks(cleaned, sampling_rate=fs)
            rpeaks = rpeaks_info["ECG_R_Peaks"]
            
            if len(rpeaks) >= 3:
                rr_intervals = np.diff(rpeaks) / fs
                mean_rr = np.mean(rr_intervals)
                hr = 60.0 / mean_rr
                
                measurements_by_lead[lead_name] = {
                    "heart_rate": round(hr, 1),
                    "n_rpeaks": len(rpeaks),
                    "rr_variability": round(float(np.std(rr_intervals) / mean_rr), 3),
                }
        except Exception:
            measurements_by_lead[lead_name] = {"error": "Could not analyze lead"}
    
    return measurements_by_lead