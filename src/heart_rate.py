"""
R-peak detection and heart-rate calculation.

Uses neurokit2's R-peak detector (a standard, established implementation)
on ONE lead (the first available lead) and converts sample-index RR
intervals into real time using the actual sampling rate — never assumes
a sampling rate, never estimates heart rate from sample counts alone.

Per the project requirement: if detection isn't reliable, no heart rate
is returned — the caller must not display a fabricated number.
"""
import neurokit2 as nk
import numpy as np


def detect_heart_rate(signal_1lead, fs):
    """
    signal_1lead: 1D numpy array, a single ECG lead, already preprocessed.
    fs: actual sampling rate in Hz of this signal (must be correct — this
        function trusts it completely and cannot verify it independently).

    Returns a dict:
        {
          "heart_rate": float or None,
          "n_rpeaks": int,
          "reliable": bool,
          "reason": str  (explains why, especially if unreliable)
        }
    """
    result = {"heart_rate": None, "n_rpeaks": 0, "reliable": False, "reason": ""}

    duration_sec = len(signal_1lead) / fs
    if duration_sec < 2.0:
        result["reason"] = (f"Signal too short ({duration_sec:.1f}s) for reliable "
                             f"R-peak detection; need at least ~2 seconds.")
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
        result["reason"] = (f"Only {n_rpeaks} R-peak(s) detected - too few for a "
                             f"reliable heart rate estimate.")
        return result

    # Convert sample-index RR intervals into real seconds using the actual fs.
    rr_intervals_sec = np.diff(rpeak_indices) / fs
    mean_rr = float(np.mean(rr_intervals_sec))
    std_rr = float(np.std(rr_intervals_sec))

    if mean_rr <= 0:
        result["reason"] = "Invalid RR intervals computed (non-positive mean)."
        return result

    heart_rate = 60.0 / mean_rr

    # Reliability heuristics — explicit and stated, not hidden:
    # 1) physiologically plausible range
    # 2) RR-interval variability not excessive (suggests noisy detection)
    plausible = 30.0 <= heart_rate <= 220.0
    cv_rr = std_rr / mean_rr if mean_rr > 0 else float("inf")
    stable = cv_rr < 0.35

    if not plausible:
        result["reason"] = (f"Computed heart rate ({heart_rate:.1f} bpm) is outside "
                             f"the physiologically plausible range - likely a "
                             f"detection error, not a real reading.")
        return result

    if not stable:
        result["reason"] = (f"RR intervals were too irregular (coefficient of "
                             f"variation {cv_rr:.2f}) to trust this as a reliable "
                             f"single heart-rate estimate - detection may be noisy, "
                             f"or the rhythm may be genuinely irregular.")
        result["heart_rate"] = None
        return result

    result["heart_rate"] = round(heart_rate, 1)
    result["reliable"] = True
    result["reason"] = f"{n_rpeaks} R-peaks detected, RR intervals stable (CV={cv_rr:.2f})."
    return result
