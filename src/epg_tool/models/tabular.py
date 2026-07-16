"""Tabular ML model wrappers (Random Forest / XGBoost) with a consistent
fit/predict/save/load interface, operating on extracted-feature DataFrames
keyed by waveform *code* (int), not label string.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


@dataclass
class TabularModel:
    estimator: Any
    feature_names: list[str] = field(default_factory=list)
    label_encoder: LabelEncoder = field(default_factory=LabelEncoder)
    # "balanced" auto-computes per-sample weights (inverse class frequency)
    # at fit time when the caller doesn't supply explicit sample_weight --
    # XGBoost's sklearn wrapper has no built-in class_weight like RF's, so
    # this is how it gets the same rare-class boost (see DiscoEPG's
    # analogous oversampling fix for their rare/short "pd" waveform).
    class_weight: str | None = None
    # Optional log transition matrix (+ its class order) for Viterbi
    # sequence decoding at inference, learned from the training data at
    # `train` time and bundled with the model. None on models trained
    # before sequence decoding existed -- decoding silently falls back to
    # per-window argmax then.
    transition_log: np.ndarray | None = None
    decode_classes: list | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "TabularModel":
        self.feature_names = list(X.columns)
        y_enc = self.label_encoder.fit_transform(y)
        if sample_weight is None and self.class_weight == "balanced":
            from sklearn.utils.class_weight import compute_sample_weight

            sample_weight = compute_sample_weight("balanced", y_enc)
        fit_kwargs = {} if sample_weight is None else {"sample_weight": sample_weight}
        self.estimator.fit(X[self.feature_names], y_enc, **fit_kwargs)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        y_enc = self.estimator.predict(X[self.feature_names])
        return self.label_encoder.inverse_transform(y_enc)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict_proba(X[self.feature_names])

    @property
    def classes_(self) -> np.ndarray:
        return self.label_encoder.classes_

    def feature_importances(self) -> pd.Series | None:
        importances = getattr(self.estimator, "feature_importances_", None)
        if importances is None:
            return None
        return pd.Series(importances, index=self.feature_names).sort_values(ascending=False)

    def save(self, path: str | Path) -> None:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "TabularModel":
        import joblib

        return joblib.load(Path(path))

    @classmethod
    def load_from_bytes(cls, data: bytes) -> "TabularModel":
        """Load a model already in memory (e.g. a Streamlit upload
        buffer) instead of from a filesystem path."""
        import io

        import joblib

        return joblib.load(io.BytesIO(data))


def random_forest_model(**kwargs) -> TabularModel:
    from sklearn.ensemble import RandomForestClassifier

    params = dict(
        n_estimators=150,
        # Unbounded depth on a large dataset (hundreds of thousands of
        # windows) grows enormous, overfit trees -- multi-GB model files
        # that are both impractical to ship and worse on held-out data
        # than a properly regularized forest. These bounds are a general
        # regularization default, not tuned to one dataset's size.
        max_depth=12,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    params.update(kwargs)
    return TabularModel(estimator=RandomForestClassifier(**params))


def xgboost_model(class_weight: str | None = "balanced", **kwargs) -> TabularModel:
    from xgboost import XGBClassifier

    params = dict(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        eval_metric="mlogloss",
    )
    params.update(kwargs)
    return TabularModel(estimator=XGBClassifier(**params), class_weight=class_weight)
