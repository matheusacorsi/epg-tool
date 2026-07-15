"""Peak-detection features.

Several waveforms in the Bonani et al. (2010) taxonomy are explicitly
described as having a "waves" component and a separate, sharper "peaks"
component (E2 waves/peaks, G waves/peaks -- Table 1), which plain
FFT/wavelet energy doesn't directly capture. These features summarize
the discrete peaks in a window directly: how many, how prominent, how
wide, and how regularly spaced.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from .base import register_feature


@register_feature("peaks")
def peak_features(window: np.ndarray, sample_rate_hz: float, context: dict) -> dict[str, float]:
    std = float(window.std())
    if std == 0 or len(window) < 3:
        return {
            "peaks_rate_per_s": 0.0,
            "peaks_mean_prominence": 0.0,
            "peaks_mean_width_s": 0.0,
            "peaks_interval_cv": 0.0,
        }

    # Prominence threshold scales with the window's own noise level so it
    # adapts across differing baseline noise/gain rather than using one
    # fixed voltage cutoff for every waveform/session.
    prominence_threshold = 0.5 * std
    peak_idx, properties = find_peaks(window, prominence=prominence_threshold, width=0)

    duration_s = len(window) / sample_rate_hz
    if len(peak_idx) == 0:
        return {
            "peaks_rate_per_s": 0.0,
            "peaks_mean_prominence": 0.0,
            "peaks_mean_width_s": 0.0,
            "peaks_interval_cv": 0.0,
        }

    mean_prominence = float(np.mean(properties["prominences"]))
    mean_width_s = float(np.mean(properties["widths"])) / sample_rate_hz

    if len(peak_idx) >= 2:
        intervals = np.diff(peak_idx) / sample_rate_hz
        interval_cv = float(np.std(intervals) / np.mean(intervals)) if np.mean(intervals) > 0 else 0.0
    else:
        interval_cv = 0.0

    return {
        "peaks_rate_per_s": len(peak_idx) / duration_s,
        "peaks_mean_prominence": mean_prominence,
        "peaks_mean_width_s": mean_width_s,
        "peaks_interval_cv": interval_cv,
    }
