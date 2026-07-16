"""Sequence post-processing for per-window waveform predictions.

Window-level classifiers score each fixed-length window independently, so
they can produce biologically implausible label sequences (an isolated E2
window inside a long D event, a single-window flip mid-probe). EPG waveforms
follow a strong first-order grammar (Bonani et al. 2010 Fig. 6): from any
state the insect almost always *stays* there, and when it moves it moves only
to a small set of successors. We exploit that with Viterbi decoding over the
per-window class probabilities and a transition matrix learned at train time.

Two design choices worth knowing:
  * Decoding is **global** (Viterbi), not greedy. A greedy "mask the next
    window given the previous prediction" rule was measured to *hurt*
    (held-out accuracy 0.86 -> 0.73 on this dataset) because one wrong window
    forces the next into a wrong state and errors cascade. Viterbi optimizes
    the whole path, so a locally-odd window can be overruled by strong
    evidence on both sides.
  * The transition matrix is **blended**: empirical (learned from training
    label sequences) everywhere, except transitions that are both flagged
    biologically-impossible by the species profile AND essentially absent
    from the data, which are hard-zeroed. This keeps decoding faithful to the
    dataset's real (sometimes messy) annotations while still forbidding the
    handful of jumps that are genuinely impossible.

Decoding must run on one recording's windows in time order -- never across a
concatenation of several recordings.
"""

from __future__ import annotations

import numpy as np


def build_blended_transition_log(
    sequences: list[np.ndarray],
    classes: list[int],
    allowed_transitions: dict[str, list[str]],
    code_to_label: dict[int, str],
    floor: float = 1.0,
    forbid_below_frac: float = 0.005,
) -> np.ndarray:
    """Learn a log transition matrix ``log P(next | prev)`` from per-recording
    label ``sequences`` (each a 1-D array of waveform codes in time order),
    then hard-zero (``-inf``) any transition that is *both* biologically
    impossible per ``allowed_transitions`` *and* empirically rare (row
    fraction < ``forbid_below_frac``). Self-transitions are always allowed.

    ``floor`` is Laplace smoothing so unseen-but-permitted transitions keep a
    small nonzero probability. Rows are returned in the order of ``classes``.
    """
    idx = {c: i for i, c in enumerate(classes)}
    k = len(classes)
    counts = np.zeros((k, k), dtype=float)
    for seq in sequences:
        seq = np.asarray(seq)
        for a, b in zip(seq[:-1], seq[1:]):
            if a in idx and b in idx:
                counts[idx[a], idx[b]] += 1.0

    # Biologically-permitted successor set per class (self always allowed).
    label_to_code = {v: k_ for k_, v in code_to_label.items()}
    permitted = np.zeros((k, k), dtype=bool)
    np.fill_diagonal(permitted, True)
    for prev_label, succ_labels in allowed_transitions.items():
        pc = label_to_code.get(prev_label)
        if pc is None or pc not in idx:
            continue
        for sl in succ_labels:
            sc = label_to_code.get(sl)
            if sc is not None and sc in idx:
                permitted[idx[pc], idx[sc]] = True

    row_totals = counts.sum(axis=1, keepdims=True)
    row_frac = np.divide(counts, row_totals, out=np.zeros_like(counts), where=row_totals > 0)
    # Forbid where impossible AND empirically negligible.
    forbid = (~permitted) & (row_frac < forbid_below_frac)

    smoothed = counts + floor
    smoothed[forbid] = 0.0
    row_sums = smoothed.sum(axis=1, keepdims=True)
    probs = np.divide(smoothed, row_sums, out=np.full_like(smoothed, 1.0 / k), where=row_sums > 0)
    with np.errstate(divide="ignore"):
        return np.log(probs)


def viterbi_decode(proba: np.ndarray, transition_log: np.ndarray, classes: list[int]) -> np.ndarray:
    """Most-likely label path for one recording's windows (in time order).

    ``proba`` is (n_windows, n_classes) emission probabilities aligned to
    ``classes``; ``transition_log`` is the (n_classes, n_classes) log matrix
    from :func:`build_blended_transition_log`. Returns an array of waveform
    codes. Falls back to per-window argmax for a single window or empty input.
    """
    classes = list(classes)
    n = len(proba)
    if n == 0:
        return np.array([], dtype=int)
    log_e = np.log(np.clip(proba, 1e-12, None))
    if n == 1:
        return np.array([classes[int(np.argmax(log_e[0]))]])

    k = len(classes)
    dp = np.full((n, k), -np.inf)
    back = np.zeros((n, k), dtype=int)
    dp[0] = log_e[0]
    for t in range(1, n):
        scores = dp[t - 1][:, None] + transition_log  # (prev, next)
        back[t] = np.argmax(scores, axis=0)
        dp[t] = log_e[t] + np.max(scores, axis=0)

    path = np.zeros(n, dtype=int)
    path[-1] = int(np.argmax(dp[-1]))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return np.array([classes[i] for i in path])


def postprocess_predictions(
    proba: np.ndarray,
    classes: list[int],
    *,
    transition_log: np.ndarray | None = None,
    threshold: float = 0.0,
    unclassified_code: int = 0,
) -> np.ndarray:
    """One recording's windows -> final predicted codes: Viterbi-decode if a
    ``transition_log`` is available (else per-window argmax), then apply the
    confidence gate. This is the single entry point the CLI and app share so
    training, ``predict``, ``evaluate`` and Streamlit stay consistent."""
    classes = list(classes)
    if transition_log is not None:
        codes = viterbi_decode(proba, transition_log, classes)
    else:
        codes = np.array([classes[int(i)] for i in np.argmax(proba, axis=1)]) if len(proba) else np.array([], dtype=int)
    return apply_confidence_threshold(codes, proba, threshold, unclassified_code)


def apply_confidence_threshold(
    codes: np.ndarray,
    proba: np.ndarray,
    threshold: float,
    unclassified_code: int,
) -> np.ndarray:
    """Relabel windows whose top posterior is below ``threshold`` as
    ``unclassified_code`` (for user review) instead of forcing a guess.
    A no-op when ``threshold <= 0``. The confidence is the model's own max
    posterior per window -- decoding may pick a different final label, but a
    window the model was unsure about is flagged regardless of what the
    sequence prior nudged it toward."""
    if threshold <= 0:
        return codes
    top = proba.max(axis=1)
    return np.where(top < threshold, unclassified_code, codes)
