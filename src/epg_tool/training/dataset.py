"""Discovers paired .D0x/.ANA recordings under a data folder, turns them
into a windowed, feature-extracted dataset, and splits it by insect
individual (not randomly) so validation never leaks a partial recording
of the same insect into training.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from epg_tool.features import extract_features, make_windows
from epg_tool.features.baseline import estimate_np_baseline
from epg_tool.features.windowing import build_sample_labels
from epg_tool.io.d0x import is_d0x_filename
from epg_tool.io.session import build_session, trim_session_start
from epg_tool.species.profile import SpeciesProfile


@dataclass
class RecordingRef:
    insect_id: str
    d0x_paths: list[Path]
    ana_path: Path


def _stem_for_ana(ana_path: Path) -> str:
    """'PsilideoMudaT1-3-ch6_.ANA' -> 'PsilideoMudaT1-3-ch6' (Stylet+
    convention: the .ANA file has the same stem as its .D0x series, plus
    a trailing underscore)."""
    stem = ana_path.stem
    return stem[:-1] if stem.endswith("_") else stem


def find_matching_d0x(ana_path: Path, search_root: Path) -> list[Path]:
    """Locate the .D0x series for one .ANA file by recursively searching
    ``search_root`` for files sharing the .ANA file's stem -- robust to
    whatever folder-naming convention a given data export used (a
    DataANA/ subfolder next to the .D0x files in one batch, unrelated
    sibling names like "ArquivosD0 Testemunha" / "Testemunha ANA" in
    another), since it doesn't assume any fixed parent/child relationship
    between the two, just a shared root somewhere above both."""
    stem = _stem_for_ana(ana_path)
    from epg_tool.io.d0x import find_d0x_series

    matches = sorted(p for p in search_root.rglob(f"{stem}.D*") if is_d0x_filename(p.name))
    if matches:
        return find_d0x_series(matches[0])
    raise FileNotFoundError(f"No .D0x series found for annotation file {ana_path} under {search_root}")


def discover_recordings(data_root: str | Path, ana_glob: str = "*.ANA") -> list[RecordingRef]:
    """Recursively find every .ANA file under ``data_root`` and pair it
    with its .D0x series. Insect/recording IDs come from the shared
    filename stem."""
    data_root = Path(data_root)
    recordings = []
    for ana_path in sorted(data_root.rglob(ana_glob)):
        d0x_paths = find_matching_d0x(ana_path, search_root=data_root)
        recordings.append(
            RecordingRef(insect_id=_stem_for_ana(ana_path), d0x_paths=d0x_paths, ana_path=ana_path)
        )
    return recordings


def build_features_for_session(
    session,
    profile: SpeciesProfile,
    window_s: float,
    step_s: float | None = None,
    min_purity: float = 0.0,
    extractors: list[str] | None = None,
    trim_start_s: float | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    session = trim_session_start(session, profile.trim_start_s if trim_start_s is None else trim_start_s)

    np_code = profile.label_to_code.get("Np")
    if np_code is not None:
        np_mask = build_sample_labels(len(session.samples), session.segments) == np_code
    else:
        np_mask = np.zeros(len(session.samples), dtype=bool)
    context = {"np_baseline_v": estimate_np_baseline(session.samples, np_mask)}

    windows = make_windows(session, window_s=window_s, step_s=step_s, min_purity=min_purity)
    rows = [extract_features(w.samples, session.sample_rate_hz, extractors, context) for w in windows]
    y = np.array([w.label_code for w in windows])
    return pd.DataFrame(rows), y


def build_dataset(
    recordings: list[RecordingRef],
    profile: SpeciesProfile,
    window_s: float,
    step_s: float | None = None,
    min_purity: float = 0.0,
    extractors: list[str] | None = None,
    trim_start_s: float | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Build (X, y, groups) across every recording, where ``groups`` is
    the insect_id -- the unit that train/val/test splitting must respect."""
    X_parts, y_parts, group_parts = [], [], []
    for rec in recordings:
        session = build_session(
            rec.d0x_paths[0], rec.ana_path, insect_id=rec.insect_id, sentinel_codes=profile.sentinel_codes
        )
        X, y = build_features_for_session(session, profile, window_s, step_s, min_purity, extractors, trim_start_s)
        X_parts.append(X)
        y_parts.append(y)
        group_parts.append(np.full(len(y), rec.insect_id))

    X = pd.concat(X_parts, ignore_index=True)
    y = np.concatenate(y_parts)
    groups = np.concatenate(group_parts)
    return X, y, groups


def compute_class_sample_weights(y: np.ndarray, profile: SpeciesProfile) -> np.ndarray:
    """Inverse-class-frequency ("balanced") sample weights, with an extra
    per-label multiplier from the species profile's
    ``training.class_weight_multipliers`` layered on top -- for labels
    that are both rare and easily confused with a much more common one,
    where plain balanced weighting alone under-corrects (see
    ``diaphorina_citri.yaml`` for the D-vs-E2 case this was tuned for)."""
    from sklearn.utils.class_weight import compute_sample_weight

    weights = compute_sample_weight("balanced", y)
    for label, multiplier in profile.class_weight_multipliers.items():
        code = profile.label_to_code.get(label)
        if code is not None:
            weights = weights * np.where(y == code, multiplier, 1.0)
    return weights


Split = tuple[pd.DataFrame, np.ndarray, np.ndarray]


@dataclass
class DatasetSplit:
    train: Split
    val: Split
    test: Split


def _subset(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, idx: np.ndarray) -> Split:
    return X.iloc[idx].reset_index(drop=True), y[idx], groups[idx]


def _chronological_split(
    X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, test_size: float, val_size: float
) -> DatasetSplit:
    n = len(X)
    train_end = int(n * (1 - test_size - val_size))
    val_end = int(n * (1 - test_size))
    idx = np.arange(n)
    return DatasetSplit(
        train=_subset(X, y, groups, idx[:train_end]),
        val=_subset(X, y, groups, idx[train_end:val_end]),
        test=_subset(X, y, groups, idx[val_end:]),
    )


def group_train_val_test_split(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
) -> DatasetSplit:
    """Split by insect individual so no train/val/test set shares an
    insect with another -- avoids the leakage that a random per-window
    split would introduce (adjacent windows of the same probe are highly
    correlated). Needs at least 3 distinct insects; with fewer, falls
    back to a chronological split and warns loudly that the resulting
    metrics are a sanity check, not a generalization estimate."""
    unique_groups = np.unique(groups)
    if len(unique_groups) < 3:
        warnings.warn(
            f"Only {len(unique_groups)} insect(s) in this dataset -- a per-individual "
            "train/val/test split needs at least 3 to guarantee no leakage. Falling "
            "back to a chronological split within the available recording(s); treat "
            "the resulting metrics as a pipeline sanity check, not a real "
            "generalization estimate. Add more labeled insects to get a meaningful split.",
            stacklevel=2,
        )
        return _chronological_split(X, y, groups, test_size, val_size)

    from sklearn.model_selection import GroupShuffleSplit

    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(gss_test.split(X, y, groups))

    if val_size <= 0:
        # Plain two-way calibration/validation split -- no separate
        # tuning set requested.
        return DatasetSplit(
            train=_subset(X, y, groups, train_val_idx),
            val=_subset(X, y, groups, np.array([], dtype=int)),
            test=_subset(X, y, groups, test_idx),
        )

    val_relative_size = val_size / (1 - test_size)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_relative_size, random_state=random_state)
    train_rel_idx, val_rel_idx = next(
        gss_val.split(X.iloc[train_val_idx], y[train_val_idx], groups[train_val_idx])
    )
    train_idx = train_val_idx[train_rel_idx]
    val_idx = train_val_idx[val_rel_idx]

    return DatasetSplit(
        train=_subset(X, y, groups, train_idx),
        val=_subset(X, y, groups, val_idx),
        test=_subset(X, y, groups, test_idx),
    )
