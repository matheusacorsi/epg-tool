from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from epg_tool.models.registry import has_trained_model, load_model, model_path_for, save_model
from epg_tool.models.tabular import TabularModel, random_forest_model, xgboost_model
from epg_tool.species.profile import SpeciesProfile, WaveformDef


def _separable_dataset(n_per_class: int = 30, seed: int = 0):
    rng = np.random.default_rng(seed)
    X_a = rng.normal(loc=0.0, scale=0.1, size=(n_per_class, 2))
    X_b = rng.normal(loc=5.0, scale=0.1, size=(n_per_class, 2))
    X = np.vstack([X_a, X_b])
    y = np.array([1] * n_per_class + [4] * n_per_class)  # waveform codes, not 0/1
    return pd.DataFrame(X, columns=["f1", "f2"]), y


def _imbalanced_overlapping_dataset(seed: int = 0):
    """A majority class, a rare minority class that overlaps heavily with
    it, and a well-separated third class -- mirrors the real D-vs-E2
    situation (D is rare and gets swamped by the far more common E2
    unless something corrects for the imbalance)."""
    rng = np.random.default_rng(seed)
    majority = rng.normal(loc=0.0, scale=1.0, size=(500, 2))
    minority = rng.normal(loc=0.6, scale=1.0, size=(20, 2))
    other = rng.normal(loc=6.0, scale=0.3, size=(100, 2))
    X = np.vstack([majority, minority, other])
    y = np.array([5] * 500 + [3] * 20 + [1] * 100)
    return pd.DataFrame(X, columns=["f1", "f2"]), y


class _RecordingEstimator:
    """Fake estimator that just records what fit() received, to test
    TabularModel's sample_weight/class_weight plumbing in isolation from
    any real ML behavior."""

    def __init__(self):
        self.received_sample_weight = None

    def fit(self, X, y, sample_weight=None):
        self.received_sample_weight = sample_weight
        self.classes_ = np.unique(y)

    def predict(self, X):
        return np.zeros(len(X), dtype=int)


def test_no_class_weight_means_no_sample_weight_passed():
    X, y = _separable_dataset()
    est = _RecordingEstimator()
    TabularModel(estimator=est, class_weight=None).fit(X, y)
    assert est.received_sample_weight is None


def test_balanced_class_weight_computes_sample_weight_when_none_given():
    X, y = _separable_dataset()  # balanced 30/30 -> all weights should come out equal
    est = _RecordingEstimator()
    TabularModel(estimator=est, class_weight="balanced").fit(X, y)
    assert est.received_sample_weight is not None
    assert np.allclose(est.received_sample_weight, est.received_sample_weight[0])


def test_explicit_sample_weight_overrides_balanced_default():
    X, y = _separable_dataset()
    est = _RecordingEstimator()
    custom_weights = np.arange(len(y), dtype=float)
    TabularModel(estimator=est, class_weight="balanced").fit(X, y, sample_weight=custom_weights)
    np.testing.assert_array_equal(est.received_sample_weight, custom_weights)


def test_xgboost_balanced_class_weight_improves_minority_recall():
    X, y = _imbalanced_overlapping_dataset()
    unweighted = xgboost_model(class_weight=None, n_estimators=50).fit(X, y)
    balanced = xgboost_model(class_weight="balanced", n_estimators=50).fit(X, y)

    minority_mask = y == 3
    unweighted_recall = (unweighted.predict(X)[minority_mask] == 3).mean()
    balanced_recall = (balanced.predict(X)[minority_mask] == 3).mean()
    assert balanced_recall >= unweighted_recall


@pytest.mark.parametrize("factory", [random_forest_model, xgboost_model])
def test_tabular_model_fits_and_predicts_correct_codes(factory):
    X, y = _separable_dataset()
    model = factory()
    model.fit(X, y)
    preds = model.predict(X)
    assert (preds == y).mean() > 0.95
    assert set(model.classes_) == {1, 4}


def test_tabular_model_predict_proba_shape():
    X, y = _separable_dataset()
    model = random_forest_model(n_estimators=10).fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_tabular_model_save_and_load_roundtrip(tmp_path):
    X, y = _separable_dataset()
    model = random_forest_model(n_estimators=10).fit(X, y)
    path = tmp_path / "model.joblib"
    model.save(path)
    loaded = TabularModel.load(path)
    np.testing.assert_array_equal(loaded.predict(X), model.predict(X))


def test_tabular_model_load_from_bytes_roundtrip(tmp_path):
    X, y = _separable_dataset()
    model = random_forest_model(n_estimators=10).fit(X, y)
    path = tmp_path / "model.joblib"
    model.save(path)
    loaded = TabularModel.load_from_bytes(path.read_bytes())
    np.testing.assert_array_equal(loaded.predict(X), model.predict(X))


def test_feature_importances_indexed_by_name():
    X, y = _separable_dataset()
    model = random_forest_model(n_estimators=10).fit(X, y)
    importances = model.feature_importances()
    assert set(importances.index) == {"f1", "f2"}


def _dummy_profile(tmp_registry_path) -> SpeciesProfile:
    return SpeciesProfile(
        name="dummy",
        common_name="dummy",
        reference="",
        waveforms=[WaveformDef(code=1, label="Np")],
        sentinel_codes=frozenset({99}),
        model_registry={"random_forest": str(tmp_registry_path)},
    )


def test_registry_save_and_load(tmp_path):
    registry_path = tmp_path / "models" / "rf.joblib"
    profile = _dummy_profile(registry_path)
    X, y = _separable_dataset()
    model = random_forest_model(n_estimators=10).fit(X, y)

    assert not has_trained_model(profile, "random_forest")
    saved_path = save_model(model, profile, "random_forest")
    assert saved_path == registry_path
    assert has_trained_model(profile, "random_forest")

    loaded = load_model(profile, "random_forest")
    np.testing.assert_array_equal(loaded.predict(X), model.predict(X))


def test_registry_unregistered_model_type_raises(tmp_path):
    profile = _dummy_profile(tmp_path / "rf.joblib")
    with pytest.raises(KeyError, match="xgboost"):
        model_path_for(profile, "xgboost")


def test_registry_missing_file_raises(tmp_path):
    profile = _dummy_profile(tmp_path / "does_not_exist.joblib")
    with pytest.raises(FileNotFoundError):
        load_model(profile, "random_forest")


def test_model_path_for_anchors_relative_paths_to_project_root(monkeypatch, tmp_path):
    """model_registry paths in species YAML are relative to wherever the
    epg_tool package lives on disk, not the process's CWD -- regression
    test for a bug where a bare Path(raw) resolved against CWD and
    silently missed the model when deployed in a monorepo subfolder."""
    from epg_tool.models.registry import _PROJECT_ROOT

    profile = SpeciesProfile(
        name="dummy_relative",
        common_name="dummy_relative",
        reference="",
        waveforms=[WaveformDef(code=1, label="Np")],
        sentinel_codes=frozenset({99}),
        model_registry={"random_forest": "models/diaphorina_citri/random_forest.joblib"},
    )
    monkeypatch.chdir(tmp_path)  # simulate running from an unrelated CWD
    resolved = model_path_for(profile, "random_forest")
    assert resolved == _PROJECT_ROOT / "models/diaphorina_citri/random_forest.joblib"
    assert resolved != tmp_path / "models/diaphorina_citri/random_forest.joblib"
