from __future__ import annotations

import numpy as np
import pytest

from epg_tool.io.session import EPGSession, LabeledSegment, normalize_samples, normalize_session


def _toy_session(samples: np.ndarray) -> EPGSession:
    return EPGSession(
        insect_id="toy",
        samples=samples.astype(np.float32),
        sample_rate_hz=10.0,
        source_files=[],
        segments=[LabeledSegment(code=1, start_s=0.0, end_s=1.0, start_idx=0, end_idx=len(samples))],
        recording_end_s=len(samples) / 10.0,
    )


def test_normalize_samples_maps_percentile_span_to_unit_interval():
    # A clean ramp: the 0.5/99.5 percentiles are ~the ends, so output ~[0,1].
    samples = np.linspace(2.0, 4.0, 1000)
    out = normalize_samples(samples)
    assert out.min() == pytest.approx(0.0, abs=0.02)
    assert out.max() == pytest.approx(1.0, abs=0.02)


def test_normalize_samples_is_robust_to_a_single_outlier():
    # The bulk sits in [0, 1]; one huge spike must not compress it. Under
    # naive min/max this bulk would collapse toward 0 -- percentile scaling
    # keeps the bulk spread out (that spread is the whole point).
    samples = np.concatenate([np.linspace(0.0, 1.0, 999), [1000.0]])
    out = normalize_samples(samples)
    bulk = out[:999]
    assert bulk.max() - bulk.min() > 0.9  # bulk still spans ~the full range


def test_normalize_samples_flat_trace_is_safe():
    out = normalize_samples(np.full(100, 3.0))
    assert np.all(np.isfinite(out))
    assert out.dtype == np.float32


def test_normalize_session_preserves_segments_and_timing():
    session = _toy_session(np.linspace(-1.0, 1.0, 100))
    norm = normalize_session(session)
    assert norm.segments == session.segments
    assert norm.sample_rate_hz == session.sample_rate_hz
    assert norm.recording_end_s == session.recording_end_s
    assert len(norm.samples) == len(session.samples)
    # values changed (scaled), not the original object
    assert not np.array_equal(norm.samples, session.samples)


def test_normalize_is_affine_so_relative_order_is_kept():
    samples = np.array([0.1, 0.2, 0.5, 0.3, 0.9, 0.4] * 100)
    out = normalize_samples(samples)
    assert np.array_equal(np.argsort(samples), np.argsort(out))
