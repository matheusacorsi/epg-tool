# EPG Waveform Classifier

Automated classification of insect probing/feeding behavior from
Electrical Penetration Graph (EPG-DC / Stylet+) recordings. Parses raw
`.D0x` traces and `.ANA` ground-truth annotations, extracts features,
classifies waveforms (rule-based or trained ML), and exports
Stylet+-compatible predictions plus Backus/Sarria-style behavioral
parameters.

The first populated species profile is *Diaphorina citri* (Asian citrus
psyllid) per Bonani et al. (2010, *Entomologia Experimentalis et
Applicata* 134:35-49) — a worked example, not a hardcoded assumption.
Every waveform label, threshold, and trained-model path lives in a
per-species YAML config (`src/epg_tool/species/profiles/`), so adding a
new insect (aphid, whitefly, leafhopper, ...) is a new config file, not
new code.

## Repo layout

```
src/epg_tool/
  io/            .D0x / .ANA parsers, multi-file session stitching
  species/       species profile schema + profiles/*.yaml
  features/      pluggable feature extraction + fixed-length windowing
  models/        rule-based classifier, tabular ML (RF/XGBoost), model registry
  training/      dataset discovery, group-based train/val/test split, metrics
  export/        .ANA writer, QC plotting, EPG behavioral parameters
  cli.py         typer CLI
streamlit_app.py Streamlit dashboard (upload -> classify -> review -> download)
tests/           pytest suite, including integration tests against real recordings
```

## Setup

```bash
conda create -n epg python=3.11 -y
conda activate epg
pip install -e .[dev]
pytest
```

**macOS + XGBoost:** the pip wheel needs an OpenMP runtime it doesn't
bundle. If you hit `Library not loaded: @rpath/libomp.dylib`, either
`conda install -c conda-forge libomp` (if your channel has it) or copy
an existing `libomp.dylib` (e.g. from a package that vendors one, such
as `sklearn/.dylibs/libomp.dylib`) into your env's `lib/` directory.
Not an issue on Streamlit Community Cloud (Linux).

## Species profiles

A profile is one YAML file: waveform codes/labels/colors, rule-based
thresholds (generalizing a Table-1-style expert chart), a model
registry (paths to trained artifacts), and which behavioral parameters
to compute. See `src/epg_tool/species/profiles/diaphorina_citri.yaml`
for the reference instance. To add a species: write a new YAML file
with that species' waveform vocabulary and thresholds — nothing in
`io/`, `features/`, `models/`, `cli.py`, or `streamlit_app.py` needs to
change.

Numeric waveform codes follow the Stylet+ mark convention exported in
your `.ANA` files; cross-check them against your annotation software's
legend (often an accompanying analysis worksheet) before writing a new
profile, the way `diaphorina_citri.yaml`'s header documents how codes
1/2/3/4/5/7 were resolved to Np/C/D/E1/E2/G for this dataset.

## Feature extraction

Registered extractors (`src/epg_tool/features/`): `amplitude`,
`spectral` (FFT dominant frequency/centroid/band energies/flatness/
entropy), `wavelet`, `slope`, `baseline` (voltage-level shift from the
session's own Np reference), `shape` (skewness/kurtosis/percentile
spread), and `peaks` (rate/prominence/width/spacing of discrete peaks —
targets the "waves vs peaks" distinction Bonani et al. describe for E2
and G). Add a new one by writing a function decorated with
`@register_feature("name")` returning a flat `dict[str, float]` —
nothing else needs to change, it's picked up automatically.

## CLI

```bash
epg-tool inspect  <d0x> <ana> --species diaphorina_citri     # print labeled segments
epg-tool plot     <d0x> <ana> --species diaphorina_citri --out qc.png
epg-tool train    <data_root> --species diaphorina_citri --model random_forest
epg-tool evaluate <data_root> --species diaphorina_citri --model random_forest
epg-tool predict  <d0x>       --species diaphorina_citri --model random_forest --confidence-threshold 0.55 --out predicted_.ANA
epg-tool export-parameters <data_root> --species diaphorina_citri --out parameters.xlsx
```

`<data_root>` is searched recursively for `.ANA` files, each matched to
its `.D0x` series by shared filename stem (handles both "everything in
one folder" and the Stylet+ convention of annotations living in a
`DataANA/` subfolder next to the raw files).

**Group-based splitting:** `train` and dataset-building split by insect
individual, never randomly by window — adjacent windows of the same
probe are highly correlated, so a random split would leak. This needs
**at least 3 labeled insects** to guarantee no leakage in every split;
with fewer, it falls back to a chronological split and warns loudly.
`train`'s default (`--test-size 0.2 --val-size 0.0`) is a plain 80%
calibration / 20% held-out validation split by insect; pass `--val-size`
if you also want a separate tuning split on top of that.

**Trimming a noisy warm-up period:** a species profile can set
`preprocessing.trim_start_s` (see `diaphorina_citri.yaml`, set to 600 —
the first ~10 min of acquisition on this rig is consistently noisy) to
drop the start of every recording before windowing, training, and
evaluation. Override per-run with `--trim-start-s`. This never affects
`predict`'s exported `.ANA` file, which always stays full-length and
time-aligned with the original recording.

**Per-recording normalization and window size** are also profile
settings (`preprocessing.normalize`, `preprocessing.window_s`), so
training and inference always use the same values — a model is only
valid for the window size and normalization it was trained under.
`normalize: true` applies per-recording amplitude normalization before
ML feature extraction (see the D-tuning note under "Known limitations").
Both are read from the profile by `train`, `evaluate`, `predict`, and the
Streamlit app; override the window size per-run with `--window-s`. The
rule-based classifier is never normalized (it reasons in absolute volts).

**Sequence decoding + confidence gate.** Because each window is scored
independently, the raw predictions can flip implausibly mid-probe. If a
profile sets `sequence.decode: true`, `train` learns a first-order
transition matrix from the calibration insects and bundles it with the
model; `predict`/`evaluate`/the app then **Viterbi-decode** each
recording's window sequence (in time order, never across recordings) to
enforce the waveform grammar and smooth spurious flips. The matrix is
*blended*: empirical everywhere, except transitions that are both
biologically impossible (`sequence.allowed_transitions`) and essentially
absent from the data, which are hard-zeroed — so it never contradicts the
dataset's real annotations. `sequence.confidence_threshold` (also a
per-run `--confidence-threshold` and a Streamlit slider, default 0.55)
relabels windows whose top posterior is below it as `unclassified` for
manual review instead of forcing a guess; those show up as a neutral band
in plots, a slice in the pies, and code-0 rows in the exported `.ANA`.

## Streamlit app

```bash
streamlit run streamlit_app.py
```

Upload a recording's `.D0x` files (+ optionally its `.ANA` ground
truth), pick a species profile and classifier, and get back: a QC plot,
predicted segments, accuracy metrics (if ground truth was supplied),
behavioral parameters, and downloads (predicted `.ANA`, parameters
Excel). Training itself is a CLI job, not something the app does live —
it's a batch operation over many recordings, a poor fit for a
request/response UI.

### Deploying to Streamlit Community Cloud

1. `git init`, commit, push to a GitHub repo. Note `.gitignore` excludes
   the attached paper PDF (copyrighted) and raw recordings under
   `Arquivo Ondas EPG Diaphorina citri/` / `data/` (they'll only grow —
   don't let them live in git history; the app takes recordings via
   upload, not from a bundled folder).
2. On [share.streamlit.io](https://share.streamlit.io), point at the
   repo and `streamlit_app.py`. `requirements.txt` lists plain
   dependencies (numpy, pandas, ..., streamlit) rather than an editable
   install of `epg_tool` itself; `streamlit_app.py` instead adds its own
   `src/` to `sys.path` relative to its own file location. Both choices
   matter specifically if this app ever lives in a subfolder of a larger
   monorepo rather than at a dedicated repo's root: `-e .` and bare
   `model_registry` paths resolve against the installer's/process's
   *working directory*, which Streamlit Cloud sets to the repo root
   regardless of where the app file or its `models/` folder actually
   sit — silently breaking both the install and model loading. Anchoring
   everything to each file's own on-disk location (`Path(__file__)` /
   the `epg_tool` package's location, see `models/registry.py`) instead
   works the same whether the app is at the repo root or nested.
3. Trained model artifacts under `models/<species>/*.joblib` **are**
   committed so the deployed app has something beyond the rule-based
   baseline out of the box (currently ~42MB for random_forest, ~7MB for
   xgboost — comfortably under GitHub's 100MB hard limit, see the
   hyperparameter note below). Retrain and commit again as more labeled
   data comes in; move to Git LFS if a future model grows past ~50-80MB.

## Known limitations / next steps

- **Development is focused on XGBoost going forward.** It leads Random
  Forest on overall held-out accuracy (~86% vs ~85% under the current
  normalized/4s config). RF's code/model stay in place, usable, and
  retrained under the same config (and it happens to do better on `D` —
  see below), but new tuning work targets XGBoost specifically.
- **`D` waveform is hard to detect, and this has been actively tuned,
  not just documented.** `D` is short-duration (mean ~46s per Bonani
  et al.), rare (~4.5% of windows), and easily confused with the far
  more common `E2`. Reviewing the DiscoEPG paper (Dinh et al. 2026,
  `papers/main.pdf` -- a Python package for automatic aphid EPG
  annotation) surfaced three applicable methodology changes, each
  ablated on the 20-insect group split. The combined effect took
  held-out accuracy **0.821 → 0.859** and `D`'s F1 **0.228 → 0.295**
  (+29% relative), while improving every other class too:
    - **Per-recording amplitude normalization** (their Eq. 1,
      `preprocessing.normalize`). Their pipeline min-max normalizes each
      recording to [0,1]; done naively that was *destructive* here
      (accuracy collapsed to 0.675 as one voltage transient compressed
      the whole trace), so `normalize_session` scales by the robust
      0.5–99.5 percentile span instead. That version was a clean win on
      its own (0.821 → 0.837), removing cross-insect gain differences.
    - **Longer analysis window** (`preprocessing.window_s`, now 4s).
      Sweeping 1–5s, accuracy and `D`-F1 both peaked at 4s before
      declining at 5s — the extra frequency resolution helps separate
      low-frequency `D` (1–3.5 Hz) from the higher-frequency waveforms
      it was confused with. (This *reverses* an earlier finding: shorter
      windows were ruled out, but longer ones, once normalization was in
      place, were the single biggest lever.)
    - **Per-label oversampling weight** for `D`
      (`training.class_weight_multipliers`, the same idea as DiscoEPG's
      oversampling of their rare `pd` waveform), re-swept jointly with
      the above and settled at `D×2`.
  Two DiscoEPG ideas were tried and *dropped*: feeding richer wavelet
  coefficients as features (their best XGBoost input for aphids) slightly
  *hurt* here, and raw min/max normalization (see above).
  Note that window size and the `D` multiplier were selected on the same
  held-out split they're reported on, so those specific figures are
  mildly optimistic; the *direction* of each change is consistent across
  the whole sweep, and the held-out recordings that were never used in
  training (see below) are the cleaner test.
- **Sequence structure helps overall accuracy, but not `D` recall.**
  EPG waveforms follow a strong first-order grammar (Bonani Fig. 6), so
  Viterbi-decoding the per-window probabilities against a learned
  transition matrix (`sequence.decode`) lifts held-out accuracy
  **0.859 → 0.870** and sharply improves rare-class *precision*
  (`D` 0.37 → 0.45, plus `E1`/`G`). It does **not** raise `D` *recall* —
  decoding can't recover a `D` event the model labeled `E2` throughout,
  because that's a feature-space confusion, not a sequence error. Three
  related ideas were measured and **not** shipped for exactly this
  reason: a **greedy t-1 transition mask** (what a naive reading of the
  transition network suggests) actively *cascaded* errors, collapsing
  accuracy to **0.730** — global Viterbi is used instead; **neighbor-context
  features** raised overall accuracy (~0.869) but not `D`; and a
  **hierarchical `D`-vs-`E2` specialist sub-model** gave ~zero gain over
  the flat model (same features, same information). Also note this
  dataset's annotations do *not* obey the paper's strict determinism
  (`D`→`E1` is ~55% at the event level, not 100%), so the biological
  grammar is used only to hard-zero transitions that are also empirically
  absent — the decoder stays data-driven otherwise.
- **Random Forest actually edges XGBoost on `D`** (F1 0.344 vs 0.295 on
  the same split) though it trails on overall accuracy (0.846 vs 0.859).
  Development still focuses on XGBoost per the note above, but if `D`
  recall is the priority for a given analysis, RF is worth a look — both
  bundled models are retrained under the identical normalization/window
  config so they're directly comparable.
- **Per-insect variability is real and large.** Individual held-out
  insects can score well below the aggregate validation accuracy (the
  paper itself notes highly variable probing behavior between
  individuals) — read single-recording accuracy in the Streamlit app as
  one data point, not a generalization estimate; the CLI's aggregate
  held-out metrics across all validation insects are the more meaningful
  number.
- **Random Forest hyperparameters are deliberately regularized**
  (`max_depth=12, min_samples_leaf=20, n_estimators=150` in
  `models/tabular.py`) — an unbounded-depth forest on ~450k calibration
  windows produced a 1.1GB file (unusable for git/GitHub) with no better
  held-out accuracy than the regularized version. Revisit these if
  retraining on substantially more data changes that trade-off.
- **Rule-based %amplitude calibration.** The Bonani et al. Table 1
  convention assumes voltage gained to within the 5V full-scale range;
  this dataset's raw `.D0x` values run much smaller (~0.1-0.3V), so the
  rule-based classifier's amplitude thresholds may need retuning for
  your acquisition gain settings — that's a YAML edit
  (`rule_based_thresholds` in the species profile), not a code change.
- **CNN upgrade path stubbed, not implemented** (`epg_tool.models.sequence`)
  behind the optional `torch` extra, for once there's enough data to
  benefit from a sequence model over hand-engineered features.
