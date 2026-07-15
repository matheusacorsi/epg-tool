"""Distribution-shape features.

Complement the raw min/max/std in amplitude.py with moments (skewness,
kurtosis) and a percentile-based spread that's robust to single-sample
spikes -- useful since several waveforms (e.g. G peaks) are defined by
sharp, asymmetric excursions rather than a symmetric spread around the
mean.
"""

from __future__ import annotations
import warnings

import numpy as np
from scipy import stats

from .base import register_feature


@register_feature("shape")
def shape_features(window: np.ndarray, sample_rate_hz: float, context: dict) -> dict[str, float]:
    p10, p25, p75, p90 = np.percentile(window, [10, 25, 75, 90])
    # A constant (or near-constant, e.g. a flat Np period at the ADC's
    # noise floor) window makes skew/kurtosis undefined or numerically
    # unstable -- scipy warns "precision loss" rather than erroring, but
    # the value is meaningless either way, so treat both as zero.
    is_constant = window.std() < 1e-6
    if is_constant:
        skewness, kurtosis = 0.0, 0.0
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            skewness = float(stats.skew(window))
            kurtosis = float(stats.kurtosis(window))
        if not np.isfinite(skewness):
            skewness = 0.0
        if not np.isfinite(kurtosis):
            kurtosis = 0.0
    return {
        "shape_skewness": skewness,
        "shape_kurtosis": kurtosis,
        "shape_iqr": float(p75 - p25),
        "shape_p10_p90_range": float(p90 - p10),
    }
