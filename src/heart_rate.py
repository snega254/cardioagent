"""
ECG measurement extraction using NeuroKit2.

Provides:
- R-peak detection
- Heart-rate calculation
- RR intervals
- RR variability
- Rhythm regularity
- PR interval
- QRS duration
- QT interval
- QTc interval
- Per-lead measurements

Note:
These measurements are algorithmically derived from the ECG signal.
They are intended for decision-support/research use and are not a
replacement for clinician interpretation.
"""

import neurokit2 as nk
import numpy as np


# ============================================================
# 1. HEART RATE + R-PEAK DETECTION
# ============================================================

def detect_heart_rate(signal_1lead, fs):
    """
    Detect R-peaks and calculate heart rate.

    Parameters
    ----------
    signal_1lead : numpy.ndarray
        Single ECG lead.
    fs : float
        Sampling frequency.

    Returns
    -------
    dict
        Heart rate, R-peaks, RR intervals and reliability information.
    """

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
        result["reason"] = (
            f"Signal too short ({duration_sec:.1f}s)"
        )
        return result

    try:
        # Clean ECG
        cleaned = nk.ecg_clean(
            signal_1lead,
            sampling_rate=fs
        )

        # Detect R-peaks
        _, info = nk.ecg_peaks(
            cleaned,
            sampling_rate=fs
        )

        rpeak_indices = info["ECG_R_Peaks"]

    except Exception as e:
        result["reason"] = (
            f"R-peak detection failed: {e}"
        )
        return result

    n_rpeaks = len(rpeak_indices)

    result["n_rpeaks"] = int(n_rpeaks)

    if n_rpeaks < 3:
        result["reason"] = (
            f"Only {n_rpeaks} R-peak(s) detected"
        )
        return result

    # --------------------------------------------------------
    # RR intervals
    # --------------------------------------------------------

    rr_intervals_sec = np.diff(rpeak_indices) / fs

    mean_rr = float(np.mean(rr_intervals_sec))
    std_rr = float(np.std(rr_intervals_sec))

    result["rr_intervals"] = (
        rr_intervals_sec.tolist()
    )

    if mean_rr <= 0:
        result["reason"] = "Invalid RR intervals"
        return result

    # --------------------------------------------------------
    # Heart rate
    # --------------------------------------------------------

    heart_rate = 60.0 / mean_rr

    # Coefficient of variation of RR intervals
    cv_rr = (
        std_rr / mean_rr
        if mean_rr > 0
        else float("inf")
    )

    result["rr_variability"] = round(
        float(cv_rr),
        3
    )

    # --------------------------------------------------------
    # Basic sanity checks
    # --------------------------------------------------------

    plausible = (
        30.0 <= heart_rate <= 220.0
    )

    if not plausible:
        result["reason"] = (
            f"Heart rate ({heart_rate:.1f} bpm) "
            "outside plausible range"
        )
        return result

    # Do NOT reject heart rate just because rhythm is irregular.
    # Irregular rhythm can itself be clinically meaningful.

    result["heart_rate"] = round(
        heart_rate,
        1
    )

    result["reliable"] = True

    if cv_rr < 0.15:
        rhythm = "regular"
    elif cv_rr < 0.35:
        rhythm = "somewhat irregular"
    else:
        rhythm = "irregular"

    result["reason"] = (
        f"{n_rpeaks} R-peaks, "
        f"{rhythm} rhythm "
        f"(RR CV={cv_rr:.2f})"
    )

    return result


# ============================================================
# 2. EXTENDED ECG MEASUREMENTS
# ============================================================

def extract_ecg_measurements(
    signal,
    fs,
    rpeak_indices=None
):
    """
    Extract ECG measurements.

    Measurements:
        - Heart rate
        - Number of R-peaks
        - RR intervals
        - RR variability
        - Rhythm regularity
        - PR interval
        - QRS duration
        - QT interval
        - QTc interval

    Parameters
    ----------
    signal : numpy.ndarray
        Single-lead ECG signal.

    fs : float
        Sampling frequency.

    rpeak_indices : array-like, optional
        Previously detected R-peaks.

    Returns
    -------
    dict
        ECG measurements.
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

    # --------------------------------------------------------
    # Clean ECG
    # --------------------------------------------------------

    try:

        cleaned = nk.ecg_clean(
            signal,
            sampling_rate=fs
        )

    except Exception:
        return measurements

    # --------------------------------------------------------
    # R-peak detection
    # --------------------------------------------------------

    try:

        if rpeak_indices is None:

            _, rpeaks_info = nk.ecg_peaks(
                cleaned,
                sampling_rate=fs
            )

            rpeaks = np.asarray(
                rpeaks_info["ECG_R_Peaks"]
            )

        else:

            rpeaks = np.asarray(
                rpeak_indices
            )

    except Exception:
        return measurements

    # --------------------------------------------------------
    # Heart rate / RR
    # --------------------------------------------------------

    if len(rpeaks) >= 3:

        measurements["n_rpeaks"] = int(
            len(rpeaks)
        )

        rr_intervals = (
            np.diff(rpeaks) / fs
        )

        mean_rr = float(
            np.mean(rr_intervals)
        )

        if mean_rr > 0:

            heart_rate = (
                60.0 / mean_rr
            )

            rr_cv = (
                float(np.std(rr_intervals))
                / mean_rr
            )

            measurements["rr_intervals"] = (
                rr_intervals.tolist()
            )

            measurements["heart_rate"] = round(
                heart_rate,
                1
            )

            measurements["rr_variability"] = round(
                rr_cv,
                3
            )

            # Rhythm classification
            if rr_cv < 0.15:
                measurements["rhythm_regularity"] = (
                    "regular"
                )

            elif rr_cv < 0.35:
                measurements["rhythm_regularity"] = (
                    "somewhat irregular"
                )

            else:
                measurements["rhythm_regularity"] = (
                    "irregular"
                )

    # ========================================================
    # ECG WAVE DELINEATION
    # ========================================================

    try:

        # NeuroKit2 identifies P-wave, QRS and T-wave
        # boundaries from the ECG waveform.

        signals, waves = nk.ecg_delineate(
            cleaned,
            rpeaks,
            sampling_rate=fs,
            method="dwt"
        )

    except Exception:
        # If delineation fails, keep measurements that
        # were already successfully calculated.
        return measurements

    # ========================================================
    # QRS DURATION
    # ========================================================

    try:

        qrs_onsets = waves.get(
            "ECG_R_Onsets"
        )

        qrs_offsets = waves.get(
            "ECG_R_Offsets"
        )

        if (
            qrs_onsets is not None
            and qrs_offsets is not None
        ):

            qrs_durations = []

            for onset, offset in zip(
                qrs_onsets,
                qrs_offsets
            ):

                if (
                    onset is None
                    or offset is None
                ):
                    continue

                if (
                    np.isnan(onset)
                    or np.isnan(offset)
                ):
                    continue

                duration = (
                    offset - onset
                ) / fs

                # Physiologically reasonable range
                if 0.04 <= duration <= 0.20:

                    qrs_durations.append(
                        duration
                    )

            if qrs_durations:

                measurements["qrs_duration"] = round(
                    np.mean(qrs_durations) * 1000,
                    1
                )

    except Exception:
        pass

    # ========================================================
    # PR INTERVAL
    # ========================================================

    try:

        p_onsets = waves.get(
            "ECG_P_Onsets"
        )

        if p_onsets is not None:

            pr_intervals = []

            for p_onset, r_peak in zip(
                p_onsets,
                rpeaks
            ):

                if (
                    p_onset is None
                    or r_peak is None
                ):
                    continue

                if np.isnan(p_onset):
                    continue

                pr = (
                    r_peak - p_onset
                ) / fs

                # Reasonable PR interval range
                if 0.08 <= pr <= 0.30:

                    pr_intervals.append(
                        pr
                    )

            if pr_intervals:

                measurements["pr_interval"] = round(
                    np.mean(pr_intervals) * 1000,
                    1
                )

    except Exception:
        pass

    # ========================================================
    # QT INTERVAL
    # ========================================================

    try:

        t_offsets = waves.get(
            "ECG_T_Offsets"
        )

        if t_offsets is not None:

            qt_intervals = []

            for r_peak, t_offset in zip(
                rpeaks,
                t_offsets
            ):

                if (
                    r_peak is None
                    or t_offset is None
                ):
                    continue

                if np.isnan(t_offset):
                    continue

                qt = (
                    t_offset - r_peak
                ) / fs

                # Reasonable QT range
                if 0.20 <= qt <= 0.60:

                    qt_intervals.append(
                        qt
                    )

            if qt_intervals:

                qt_mean = float(
                    np.mean(qt_intervals)
                )

                measurements["qt_interval"] = round(
                    qt_mean * 1000,
                    1
                )

                # ------------------------------------------------
                # QTc using Bazett's formula
                #
                # QTc = QT / sqrt(RR)
                # ------------------------------------------------

                if measurements["heart_rate"]:

                    hr = measurements[
                        "heart_rate"
                    ]

                    rr = 60.0 / hr

                    qtc = (
                        qt_mean
                        / np.sqrt(rr)
                    )

                    measurements[
                        "qtc_interval"
                    ] = round(
                        qtc * 1000,
                        1
                    )

    except Exception:
        pass

    return measurements


# ============================================================
# 3. PER-LEAD MEASUREMENTS
# ============================================================

def extract_lead_measurements(
    signal,
    fs,
    lead_names=None
):
    """
    Extract basic measurements independently
    for every ECG lead.

    Parameters
    ----------
    signal : numpy.ndarray
        Shape [n_samples, n_leads]

    fs : float
        Sampling frequency.

    lead_names : list, optional
        Names of ECG leads.

    Returns
    -------
    dict
        Measurements for each lead.
    """

    measurements_by_lead = {}

    for lead_idx in range(
        signal.shape[1]
    ):

        if lead_names:

            lead_name = lead_names[
                lead_idx
            ]

        else:

            lead_name = (
                f"lead_{lead_idx + 1}"
            )

        lead_signal = signal[
            :,
            lead_idx
        ]

        try:

            result = detect_heart_rate(
                lead_signal,
                fs
            )

            measurements_by_lead[
                lead_name
            ] = {
                "heart_rate": result[
                    "heart_rate"
                ],
                "n_rpeaks": result[
                    "n_rpeaks"
                ],
                "rr_variability": result[
                    "rr_variability"
                ],
                "reliable": result[
                    "reliable"
                ],
            }

        except Exception as e:

            measurements_by_lead[
                lead_name
            ] = {
                "error": (
                    f"Could not analyze lead: {e}"
                )
            }

    return measurements_by_lead