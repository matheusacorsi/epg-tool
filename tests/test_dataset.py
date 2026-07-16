from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import REAL_ANA, REAL_D01, make_ana_bytes, make_d0x_bytes, requires_real_data
from epg_tool.io.session import EPGSession, LabeledSegment
from epg_tool.species.profile import SpeciesProfile, WaveformDef, load_profile
from epg_tool.training.dataset import (
    RecordingRef,
    build_dataset,
    build_features_for_session,
    compute_class_sample_weights,
    discover_recordings,
    group_train_val_test_split,
)


@pytest.fixture()
def two_recordings(tmp_path):
    """Two tiny synthetic recordings under a Stylet+-like layout:
    arquivosDataAcquisition/{insect}.D01 + DataANA/{insect}_.ANA
    """
    acq_dir = tmp_path / "arquivosDataAcquisition"
    ana_dir = acq_dir / "DataANA"
    acq_dir.mkdir()
    ana_dir.mkdir()

    for insect in ["insectA", "insectB"]:
        samples = list(np.sin(np.linspace(0, 20, 200)) * 0.1)
        (acq_dir / f"{insect}.D01").write_bytes(make_d0x_bytes(samples, sample_rate_hz="100,000"))
        rows = [(1, "0,00", 0), (2, "1,00", 0), (99, "2,00", 0)]
        (ana_dir / f"{insect}_.ANA").write_bytes(make_ana_bytes(rows))

    return tmp_path


def _profile_with_trim(trim_start_s: float) -> SpeciesProfile:
    return SpeciesProfile(
        name="trim_test",
        common_name="trim_test",
        reference="",
        waveforms=[WaveformDef(code=1, label="Np"), WaveformDef(code=2, label="C")],
        sentinel_codes=frozenset({99}),
        trim_start_s=trim_start_s,
    )


def _toy_session_10s() -> EPGSession:
    # 100 samples at 10Hz = 10s: Np[0,4), C[4,10)
    segments = [
        LabeledSegment(code=1, start_s=0.0, end_s=4.0, start_idx=0, end_idx=40),
        LabeledSegment(code=2, start_s=4.0, end_s=10.0, start_idx=40, end_idx=100),
    ]
    return EPGSession(
        insect_id="toy",
        samples=np.arange(100, dtype=np.float32),
        sample_rate_hz=10.0,
        source_files=[],
        segments=segments,
        recording_end_s=10.0,
    )


def test_build_features_uses_profile_trim_start_by_default():
    profile = _profile_with_trim(trim_start_s=4.0)
    session = _toy_session_10s()
    X, y = build_features_for_session(session, profile, window_s=1.0)
    # Np[0,4) is entirely trimmed away by the profile default -> only C windows remain
    assert set(y) == {2}


def test_build_features_explicit_trim_overrides_profile_default():
    profile = _profile_with_trim(trim_start_s=4.0)
    session = _toy_session_10s()
    X, y = build_features_for_session(session, profile, window_s=1.0, trim_start_s=0.0)
    assert set(y) == {1, 2}


def test_build_features_applies_profile_normalization():
    session = _toy_session_10s()  # samples arange(100) -> amp_max scales with the range
    raw = SpeciesProfile(
        name="n", common_name="n", reference="",
        waveforms=[WaveformDef(code=1, label="Np"), WaveformDef(code=2, label="C")],
        sentinel_codes=frozenset({99}), normalize=False,
    )
    norm = SpeciesProfile(
        name="n", common_name="n", reference="",
        waveforms=[WaveformDef(code=1, label="Np"), WaveformDef(code=2, label="C")],
        sentinel_codes=frozenset({99}), normalize=True,
    )
    X_raw, _ = build_features_for_session(session, raw, window_s=1.0, trim_start_s=0.0)
    X_norm, _ = build_features_for_session(session, norm, window_s=1.0, trim_start_s=0.0)
    # Normalized amplitude is squeezed toward [0,1]; raw runs over the full arange span.
    assert X_norm["amp_max"].max() <= 1.5
    assert X_raw["amp_max"].max() > 5


def _profile_with_weight_multipliers(multipliers: dict) -> SpeciesProfile:
    return SpeciesProfile(
        name="weight_test",
        common_name="weight_test",
        reference="",
        waveforms=[WaveformDef(code=1, label="Np"), WaveformDef(code=3, label="D"), WaveformDef(code=5, label="E2")],
        sentinel_codes=frozenset({99}),
        class_weight_multipliers=multipliers,
    )


def test_compute_class_sample_weights_balanced_without_multipliers():
    profile = _profile_with_weight_multipliers({})
    y = np.array([1, 1, 1, 1, 5, 5, 5, 5, 3, 3])  # Np x4, E2 x4, D x2
    weights = compute_class_sample_weights(y, profile)
    # balanced: rarer classes get higher weight, but D isn't boosted beyond that
    np_weight = weights[y == 1][0]
    d_weight = weights[y == 3][0]
    e2_weight = weights[y == 5][0]
    assert np_weight == pytest.approx(e2_weight)
    assert d_weight > np_weight  # D is rarer (2 vs 4) -> balanced already upweights it some


def test_compute_class_sample_weights_applies_extra_multiplier():
    y = np.array([1, 1, 1, 1, 5, 5, 5, 5, 3, 3])
    unboosted = compute_class_sample_weights(y, _profile_with_weight_multipliers({}))
    boosted = compute_class_sample_weights(y, _profile_with_weight_multipliers({"D": 3.0}))

    # D's weight is exactly 3x what plain balanced gives it...
    np.testing.assert_allclose(boosted[y == 3], unboosted[y == 3] * 3.0)
    # ...while every other class is untouched.
    np.testing.assert_allclose(boosted[y == 1], unboosted[y == 1])
    np.testing.assert_allclose(boosted[y == 5], unboosted[y == 5])


def test_compute_class_sample_weights_ignores_unknown_label():
    y = np.array([1, 1, 3, 3])
    profile = _profile_with_weight_multipliers({"NotARealLabel": 5.0})
    weights = compute_class_sample_weights(y, profile)
    assert len(weights) == 4  # doesn't raise, just skips the unmatched label


def test_discover_recordings_finds_both(two_recordings):
    recordings = discover_recordings(two_recordings)
    assert {r.insect_id for r in recordings} == {"insectA", "insectB"}
    for rec in recordings:
        assert rec.d0x_paths[0].name == f"{rec.insect_id}.D01"
        assert rec.ana_path.name == f"{rec.insect_id}_.ANA"


def test_discover_recordings_raises_if_no_match(tmp_path):
    ana_dir = tmp_path / "DataANA"
    ana_dir.mkdir()
    (ana_dir / "orphan_.ANA").write_bytes(make_ana_bytes([(1, "0,00", 0), (99, "1,00", 0)]))
    with pytest.raises(FileNotFoundError):
        discover_recordings(tmp_path)


def test_build_dataset_groups_by_insect_id(two_recordings):
    profile = load_profile("diaphorina_citri")
    recordings = discover_recordings(two_recordings)
    # trim_start_s=0 overrides the profile's default 600s trim, which
    # would consume this whole 2s synthetic fixture otherwise.
    X, y, groups = build_dataset(recordings, profile, window_s=0.5, trim_start_s=0)
    assert len(X) == len(y) == len(groups)
    assert set(groups) == {"insectA", "insectB"}
    assert isinstance(X, pd.DataFrame)


def test_group_split_keeps_insects_disjoint():
    rng = np.random.default_rng(0)
    n_per_insect = 20
    insects = [f"insect{i}" for i in range(5)]
    X = pd.DataFrame({"f1": rng.normal(size=n_per_insect * len(insects))})
    y = rng.integers(1, 3, size=n_per_insect * len(insects))
    groups = np.repeat(insects, n_per_insect)

    split = group_train_val_test_split(X, y, groups, test_size=0.2, val_size=0.2, random_state=0)
    train_groups = set(split.train[2])
    val_groups = set(split.val[2])
    test_groups = set(split.test[2])

    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)
    assert train_groups | val_groups | test_groups == set(insects)


def test_group_split_supports_plain_two_way_split_when_val_size_zero():
    rng = np.random.default_rng(0)
    n_per_insect = 20
    insects = [f"insect{i}" for i in range(5)]
    X = pd.DataFrame({"f1": rng.normal(size=n_per_insect * len(insects))})
    y = rng.integers(1, 3, size=n_per_insect * len(insects))
    groups = np.repeat(insects, n_per_insect)

    split = group_train_val_test_split(X, y, groups, test_size=0.2, val_size=0.0, random_state=0)
    assert len(split.val[0]) == 0
    train_groups = set(split.train[2])
    test_groups = set(split.test[2])
    assert train_groups.isdisjoint(test_groups)
    assert train_groups | test_groups == set(insects)
    assert len(split.train[0]) + len(split.test[0]) == len(X)


def test_group_split_falls_back_and_warns_with_few_insects():
    X = pd.DataFrame({"f1": np.arange(100.0)})
    y = np.zeros(100, dtype=int)
    groups = np.array(["only_insect"] * 100)

    with pytest.warns(UserWarning, match="Only 1 insect"):
        split = group_train_val_test_split(X, y, groups, test_size=0.2, val_size=0.2)

    assert len(split.train[0]) + len(split.val[0]) + len(split.test[0]) == 100
    # chronological: train comes first, test comes last
    assert split.train[0]["f1"].max() < split.val[0]["f1"].min()
    assert split.val[0]["f1"].max() < split.test[0]["f1"].min()


@requires_real_data
def test_discover_recordings_against_real_data_dir():
    data_root = REAL_ANA.parent.parent.parent  # "Arquivo Ondas EPG Diaphorina citri"
    recordings = discover_recordings(data_root)
    assert any(r.insect_id == "PsilideoMudaT1-3-ch6" for r in recordings)
    rec = next(r for r in recordings if r.insect_id == "PsilideoMudaT1-3-ch6")
    assert rec.ana_path == REAL_ANA
    assert rec.d0x_paths[0] == REAL_D01
    assert len(rec.d0x_paths) == 9
