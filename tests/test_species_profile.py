from __future__ import annotations

import pytest

from epg_tool.species.profile import SpeciesProfile, WaveformDef, load_profile


def test_load_diaphorina_citri_profile():
    profile = load_profile("diaphorina_citri")
    assert profile.name == "diaphorina_citri"
    assert profile.code_to_label == {1: "Np", 2: "C", 3: "D", 4: "E1", 5: "E2", 7: "G"}
    assert profile.sentinel_codes == frozenset({99})
    assert profile.label_for_code(3) == "D"
    assert profile.label_for_code(99) is None
    assert profile.trim_start_s == 600
    assert profile.normalize is True
    assert profile.window_s == 4.0
    assert profile.class_weight_multipliers == {"D": 2.0}


def test_profile_without_preprocessing_section_defaults_to_no_trim():
    profile = SpeciesProfile(
        name="bare",
        common_name="bare",
        reference="",
        waveforms=[WaveformDef(code=1, label="Np")],
        sentinel_codes=frozenset({99}),
    )
    assert profile.trim_start_s == 0.0
    assert profile.class_weight_multipliers == {}


def test_unknown_profile_raises():
    with pytest.raises(FileNotFoundError, match="diaphorina_citri"):
        load_profile("not_a_real_species")


def test_duplicate_codes_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        SpeciesProfile(
            name="bad",
            common_name="bad",
            reference="",
            waveforms=[WaveformDef(code=1, label="A"), WaveformDef(code=1, label="B")],
            sentinel_codes=frozenset({99}),
        )


def test_sentinel_and_waveform_code_overlap_rejected():
    with pytest.raises(ValueError, match="both waveforms and sentinels"):
        SpeciesProfile(
            name="bad",
            common_name="bad",
            reference="",
            waveforms=[WaveformDef(code=99, label="A")],
            sentinel_codes=frozenset({99}),
        )
