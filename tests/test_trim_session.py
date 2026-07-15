from __future__ import annotations

import numpy as np
import pytest

from epg_tool.io.session import EPGSession, LabeledSegment, trim_session_start


def _toy_session() -> EPGSession:
    # 100 samples at 10 Hz = 10s. Segments: Np[0,3), C[3,6), D[6,10)
    segments = [
        LabeledSegment(code=1, start_s=0.0, end_s=3.0, start_idx=0, end_idx=30),
        LabeledSegment(code=2, start_s=3.0, end_s=6.0, start_idx=30, end_idx=60),
        LabeledSegment(code=3, start_s=6.0, end_s=10.0, start_idx=60, end_idx=100),
    ]
    return EPGSession(
        insect_id="toy",
        samples=np.arange(100, dtype=np.float32),
        sample_rate_hz=10.0,
        source_files=[],
        segments=segments,
        recording_end_s=10.0,
    )


def test_trim_zero_is_noop():
    session = _toy_session()
    trimmed = trim_session_start(session, 0.0)
    assert trimmed is session


def test_trim_drops_fully_covered_segments_and_shifts_remaining():
    session = _toy_session()
    trimmed = trim_session_start(session, 4.0)  # drop first 4s (40 samples)

    assert len(trimmed.samples) == 60
    np.testing.assert_array_equal(trimmed.samples, session.samples[40:])
    assert trimmed.duration_s == pytest.approx(6.0)
    assert trimmed.recording_end_s == pytest.approx(6.0)

    # Np[0,3) entirely dropped; C[3,6) straddles the cut -> truncated to [0,2)
    assert len(trimmed.segments) == 2
    c_seg, d_seg = trimmed.segments
    assert c_seg.code == 2
    assert c_seg.start_s == pytest.approx(0.0)
    assert c_seg.end_s == pytest.approx(2.0)
    assert c_seg.start_idx == 0
    assert c_seg.end_idx == 20

    assert d_seg.code == 3
    assert d_seg.start_s == pytest.approx(2.0)
    assert d_seg.end_s == pytest.approx(6.0)
    assert d_seg.start_idx == 20
    assert d_seg.end_idx == 60


def test_trim_exactly_at_segment_boundary():
    session = _toy_session()
    trimmed = trim_session_start(session, 3.0)  # exactly at Np/C boundary
    assert len(trimmed.segments) == 2
    assert trimmed.segments[0].code == 2
    assert trimmed.segments[0].start_s == pytest.approx(0.0)


def test_trim_entire_recording_raises():
    session = _toy_session()
    with pytest.raises(ValueError, match="entire recording"):
        trim_session_start(session, 10.0)


def test_trim_preserves_insect_id_and_sample_rate():
    session = _toy_session()
    trimmed = trim_session_start(session, 2.0)
    assert trimmed.insect_id == session.insect_id
    assert trimmed.sample_rate_hz == session.sample_rate_hz
