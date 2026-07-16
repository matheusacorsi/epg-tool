"""Command-line interface: parse/inspect, plot, train, predict, evaluate,
and export-parameters -- one entry point per stage of the pipeline
described in the project README.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import typer

app = typer.Typer(help="EPG waveform classification toolkit", no_args_is_help=True)


def _load_species(species: str):
    from epg_tool.species.profile import load_profile

    return load_profile(species)


def _build_model(model_type: str):
    from epg_tool.models.tabular import random_forest_model, xgboost_model

    if model_type == "random_forest":
        return random_forest_model()
    if model_type == "xgboost":
        return xgboost_model()
    raise typer.BadParameter(f"Unknown model type {model_type!r}. Use random_forest or xgboost.")


@app.command()
def inspect(
    d0x: Path = typer.Argument(..., help="Any .D0x file in the recording's series"),
    ana: Path = typer.Argument(..., help="Matching .ANA annotation file"),
    species: str = typer.Option("diaphorina_citri", help="Species profile name"),
):
    """Parse a .D0x/.ANA pair and print the labeled-segment table."""
    from epg_tool.io.session import build_session

    profile = _load_species(species)
    session = build_session(d0x, ana, insect_id=d0x.stem)
    df = session.to_dataframe()
    df["label"] = df["code"].map(profile.code_to_label)
    pd.set_option("display.width", 120)
    typer.echo(df.to_string(index=False))
    typer.echo(f"\nDuration: {session.duration_s:.1f}s, sample rate: {session.sample_rate_hz} Hz")


@app.command()
def plot(
    d0x: Path = typer.Argument(..., help="Any .D0x file in the recording's series"),
    ana: Path = typer.Argument(..., help="Matching .ANA annotation file"),
    species: str = typer.Option("diaphorina_citri"),
    start: float = typer.Option(0.0, help="Window start (s)"),
    end: Optional[float] = typer.Option(None, help="Window end (s); defaults to full recording"),
    out: Path = typer.Option(Path("qc_plot.png")),
):
    """Render a QC plot (raw trace + color-coded ground-truth segments)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from epg_tool.export.plotting import plot_session
    from epg_tool.io.session import build_session

    profile = _load_species(species)
    session = build_session(d0x, ana, insect_id=d0x.stem)
    fig, _ax = plot_session(session, profile, start_s=start, end_s=end)
    fig.savefig(out)
    plt.close(fig)
    typer.echo(f"Wrote {out}")


@app.command()
def train(
    data_root: Path = typer.Argument(..., help="Folder to recursively search for .D0x/.ANA pairs"),
    species: str = typer.Option("diaphorina_citri"),
    model: str = typer.Option("random_forest", help="random_forest or xgboost"),
    window_s: Optional[float] = typer.Option(None, help="Window length (s); defaults to the species profile's setting"),
    step_s: Optional[float] = typer.Option(None),
    min_purity: float = typer.Option(0.5, help="Drop windows where the majority label covers less than this"),
    test_size: float = typer.Option(0.2, help="Held-out validation fraction (by insect)"),
    val_size: float = typer.Option(
        0.0, help="Extra tuning-validation fraction on top of test_size; 0 for a plain calibration/validation split"
    ),
    trim_start_s: Optional[float] = typer.Option(
        None, help="Seconds to drop from the start of every recording; defaults to the species profile's setting"
    ),
    out: Optional[Path] = typer.Option(None, help="Where to save the model; defaults to the species registry path"),
):
    """Discover recordings under DATA_ROOT, build a windowed feature
    dataset split by insect individual (default: 80% calibration / 20%
    held-out validation), train a tabular model, and report validation
    metrics."""
    from epg_tool.models.postprocess import build_blended_transition_log
    from epg_tool.models.registry import model_path_for, save_model
    from epg_tool.training.dataset import (
        build_dataset,
        compute_class_sample_weights,
        discover_recordings,
        group_train_val_test_split,
        predict_postprocessed,
        sequences_by_group,
    )
    from epg_tool.training.evaluate import evaluate

    profile = _load_species(species)
    recordings = discover_recordings(data_root)
    if not recordings:
        raise typer.BadParameter(f"No .D0x/.ANA pairs found under {data_root}")
    typer.echo(f"Found {len(recordings)} recording(s): {[r.insect_id for r in recordings]}")

    effective_window = profile.window_s if window_s is None else window_s
    effective_trim = profile.trim_start_s if trim_start_s is None else trim_start_s
    typer.echo(f"Window {effective_window:.2f}s, normalize={profile.normalize}")
    typer.echo(f"Trimming first {effective_trim:.0f}s of each recording before windowing")
    X, y, groups = build_dataset(
        recordings, profile, window_s=effective_window, step_s=step_s, min_purity=min_purity, trim_start_s=trim_start_s
    )
    typer.echo(f"Built {len(X)} windows across {len(set(groups))} insect(s)")

    split = group_train_val_test_split(X, y, groups, test_size=test_size, val_size=val_size)
    typer.echo(
        f"Calibration: {len(split.train[0])} windows ({len(set(split.train[2]))} insects) -- "
        f"Validation: {len(split.test[0])} windows ({len(set(split.test[2]))} insects)"
        + (f" -- Tuning: {len(split.val[0])} windows ({len(set(split.val[2]))} insects)" if len(split.val[0]) else "")
    )

    clf = _build_model(model)
    train_X, train_y = split.train[0], split.train[1]
    # XGBoost has no built-in class_weight like RF -- compute balanced
    # (+ any species-specific extra multiplier, e.g. for D vs E2) sample
    # weights explicitly. RF is left alone; its own class_weight="balanced"
    # already handles this at the estimator level.
    sample_weight = compute_class_sample_weights(train_y, profile) if model == "xgboost" else None
    clf.fit(train_X, train_y, sample_weight=sample_weight)

    # Learn the blended transition matrix from the calibration insects only
    # (never the held-out ones) and bundle it with the model so `predict`,
    # `evaluate`, and the app can Viterbi-decode with it.
    if profile.decode_sequence:
        classes = list(clf.classes_)
        train_seqs = sequences_by_group(y, groups, keep_groups=set(split.train[2]))
        clf.transition_log = build_blended_transition_log(
            train_seqs, classes, profile.allowed_transitions, profile.code_to_label
        )
        clf.decode_classes = classes
        typer.echo("Learned blended transition matrix for Viterbi decoding")

    val_X, val_y, val_groups = split.val
    if len(val_X) > 0:
        val_pred = predict_postprocessed(clf, val_X, val_groups, profile, threshold=0.0)
        tuning_result = evaluate(val_y, val_pred, profile)
        typer.echo(f"\nTuning-validation accuracy: {tuning_result.accuracy:.3f}")
        typer.echo(f"Tuning-validation time-overlap agreement: {tuning_result.time_overlap_agreement:.3f}")
        typer.echo(tuning_result.classification_report.to_string())

    test_X, test_y, test_groups = split.test
    if len(test_X) > 0:
        # Report the shipped pipeline's numbers: decoded, but un-gated
        # (threshold=0) so accuracy isn't conflated with review coverage.
        test_pred = predict_postprocessed(clf, test_X, test_groups, profile, threshold=0.0)
        test_result = evaluate(test_y, test_pred, profile)
        decode_note = " (Viterbi-decoded)" if profile.decode_sequence and clf.transition_log is not None else ""
        typer.echo(f"\nHeld-out validation accuracy{decode_note}: {test_result.accuracy:.3f}")
        typer.echo(f"Held-out validation time-overlap agreement: {test_result.time_overlap_agreement:.3f}")
        typer.echo(test_result.classification_report.to_string())
        typer.echo("\nConfusion matrix (held-out validation):")
        typer.echo(test_result.confusion_matrix.to_string())

    if out is None:
        out_path = save_model(clf, profile, model)
    else:
        out_path = out
        clf.save(out_path)
    typer.echo(f"\nSaved model to {out_path}")


@app.command()
def evaluate(
    data_root: Path = typer.Argument(..., help="Folder to recursively search for .D0x/.ANA pairs"),
    species: str = typer.Option("diaphorina_citri"),
    model: str = typer.Option("random_forest"),
    window_s: Optional[float] = typer.Option(None, help="Window length (s); defaults to the species profile's setting"),
    step_s: Optional[float] = typer.Option(None),
    min_purity: float = typer.Option(0.5),
    trim_start_s: Optional[float] = typer.Option(
        None, help="Seconds to drop from the start of every recording; defaults to the species profile's setting"
    ),
):
    """Evaluate an already-trained model (or the rule-based baseline)
    against every recording under DATA_ROOT."""
    from epg_tool.models.registry import load_model
    from epg_tool.models.rules import RuleBasedClassifier
    from epg_tool.training.dataset import build_dataset, discover_recordings, predict_postprocessed
    from epg_tool.training.evaluate import evaluate as evaluate_predictions

    profile = _load_species(species)
    effective_window = profile.window_s if window_s is None else window_s
    recordings = discover_recordings(data_root)
    X, y, groups = build_dataset(
        recordings, profile, window_s=effective_window, step_s=step_s, min_purity=min_purity, trim_start_s=trim_start_s
    )

    if model == "rule_based":
        clf = RuleBasedClassifier(profile)
        y_pred = clf.predict(X)
    else:
        clf = load_model(profile, model)
        # Decoded but un-gated (threshold=0), matching how `train` reports.
        y_pred = predict_postprocessed(clf, X, groups, profile, threshold=0.0)
    result = evaluate_predictions(y, y_pred, profile)

    typer.echo(f"Accuracy: {result.accuracy:.3f}")
    typer.echo(f"Time-overlap agreement: {result.time_overlap_agreement:.3f}")
    typer.echo(result.classification_report.to_string())
    typer.echo("\nConfusion matrix:")
    typer.echo(result.confusion_matrix.to_string())


@app.command()
def predict(
    d0x: Path = typer.Argument(..., help="Any .D0x file in the recording's series (no .ANA needed)"),
    species: str = typer.Option("diaphorina_citri"),
    model: str = typer.Option("random_forest", help="random_forest, xgboost, or rule_based"),
    window_s: Optional[float] = typer.Option(None, help="Window length (s); defaults to the species profile's setting"),
    step_s: Optional[float] = typer.Option(None),
    confidence_threshold: Optional[float] = typer.Option(
        None, help="Windows below this top-posterior become 'unclassified'; defaults to the profile's setting"
    ),
    out: Path = typer.Option(Path("predicted_.ANA")),
):
    """Classify an unlabeled recording and write a Stylet+-compatible
    .ANA prediction file."""
    from epg_tool.export.ana_writer import write_ana_file
    from epg_tool.features import extract_features, make_inference_windows
    from epg_tool.features.baseline import estimate_np_baseline
    from epg_tool.io.session import EPGSession, load_d0x_session
    from epg_tool.models.postprocess import postprocess_predictions
    from epg_tool.models.registry import load_model
    from epg_tool.models.rules import RuleBasedClassifier

    profile = _load_species(species)
    window_s = profile.window_s if window_s is None else window_s
    threshold = profile.confidence_threshold if confidence_threshold is None else confidence_threshold
    samples, sample_rate_hz, source_files = load_d0x_session(d0x)
    session = EPGSession(
        insect_id=d0x.stem,
        samples=samples,
        sample_rate_hz=sample_rate_hz,
        source_files=source_files,
        segments=[],
        recording_end_s=len(samples) / sample_rate_hz,
    )

    windows = make_inference_windows(samples, sample_rate_hz, window_s=window_s, step_s=step_s)

    # Feature windows are cut from the (optionally) normalized trace for ML
    # models -- matching training -- but the exported .ANA stays aligned to
    # the original full-length recording. The rule-based classifier reasons
    # in absolute volts, so it always sees the raw trace.
    feature_samples = samples
    if model != "rule_based" and profile.normalize:
        from epg_tool.io.session import normalize_samples

        feature_samples = normalize_samples(samples)
    feature_windows = make_inference_windows(feature_samples, sample_rate_hz, window_s=window_s, step_s=step_s)
    context = {"np_baseline_v": estimate_np_baseline(feature_samples, np.zeros(len(feature_samples), dtype=bool))}
    X = pd.DataFrame([extract_features(w.samples, sample_rate_hz, context=context) for w in feature_windows])

    if model == "rule_based":
        clf = RuleBasedClassifier(profile)
        pred_codes = clf.predict(X)
    else:
        clf = load_model(profile, model)
        # One recording -> decode the whole window sequence in order, then
        # apply the confidence gate (unclassified for low-confidence windows).
        tl = getattr(clf, "transition_log", None)
        transition_log = tl if (profile.decode_sequence and tl is not None) else None
        pred_codes = postprocess_predictions(
            clf.predict_proba(X), list(clf.classes_), transition_log=transition_log,
            threshold=threshold, unclassified_code=profile.unclassified_code,
        )

    write_ana_file(out, windows, pred_codes, session, profile)
    n_unclassified = int(np.sum(pred_codes == profile.unclassified_code)) if model != "rule_based" else 0
    note = f" ({n_unclassified} below confidence {threshold:g}, marked unclassified)" if n_unclassified else ""
    typer.echo(f"Wrote predictions for {len(windows)} windows to {out}{note}")


@app.command(name="export-parameters")
def export_parameters_cmd(
    data_root: Path = typer.Argument(..., help="Folder to recursively search for .D0x/.ANA pairs"),
    species: str = typer.Option("diaphorina_citri"),
    out: Path = typer.Option(Path("parameters.xlsx")),
):
    """Compute non-sequential + sequential EPG behavioral parameters for
    every recording under DATA_ROOT and export them to an Excel workbook."""
    from epg_tool.export.parameters import export_parameters_excel
    from epg_tool.io.session import build_session
    from epg_tool.training.dataset import discover_recordings

    profile = _load_species(species)
    recordings = discover_recordings(data_root)
    sessions = [
        build_session(r.d0x_paths[0], r.ana_path, insect_id=r.insect_id, sentinel_codes=profile.sentinel_codes)
        for r in recordings
    ]
    export_parameters_excel(sessions, profile, out)
    typer.echo(f"Wrote parameters for {len(sessions)} insect(s) to {out}")


if __name__ == "__main__":
    app()
