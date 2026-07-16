from __future__ import annotations

import numpy as np

from epg_tool.models.postprocess import (
    apply_confidence_threshold,
    build_blended_transition_log,
    postprocess_predictions,
    viterbi_decode,
)

# codes: 1=Np 2=C 3=D 4=E1 5=E2 ; a tiny grammar for tests
CLASSES = [1, 2, 3, 4, 5]
CODE_TO_LABEL = {1: "Np", 2: "C", 3: "D", 4: "E1", 5: "E2"}
ALLOWED = {"Np": ["C"], "C": ["Np", "D"], "D": ["E1"], "E1": ["E2"], "E2": ["C"]}


def _uniform_log_trans(forbid=()):
    k = len(CLASSES)
    T = np.ones((k, k))
    idx = {c: i for i, c in enumerate(CLASSES)}
    for a, b in forbid:
        T[idx[a], idx[b]] = 0.0
    T = T / T.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore"):
        return np.log(T)


def test_viterbi_overrules_isolated_forbidden_window():
    # Sequence of D windows with one window that argmax would call E2, but
    # D->E2 and E2->D are forbidden -> Viterbi should keep it D.
    proba = np.array([
        [0.0, 0.0, 0.9, 0.0, 0.1],
        [0.0, 0.0, 0.45, 0.0, 0.55],  # argmax = E2
        [0.0, 0.0, 0.9, 0.0, 0.1],
    ])
    log_trans = _uniform_log_trans(forbid=[(3, 5), (5, 3)])
    out = viterbi_decode(proba, log_trans, CLASSES)
    assert list(out) == [3, 3, 3]


def test_viterbi_argmax_when_transitions_uniform_and_allowed():
    proba = np.array([[0.8, 0.2, 0, 0, 0], [0.1, 0.9, 0, 0, 0]])
    out = viterbi_decode(proba, _uniform_log_trans(), CLASSES)
    assert list(out) == [1, 2]  # Np then C, both allowed


def test_viterbi_single_and_empty():
    assert list(viterbi_decode(np.array([[0.1, 0.9, 0, 0, 0]]), _uniform_log_trans(), CLASSES)) == [2]
    assert len(viterbi_decode(np.empty((0, 5)), _uniform_log_trans(), CLASSES)) == 0


def test_confidence_threshold_relabels_low_confidence():
    codes = np.array([1, 2, 3])
    proba = np.array([[0.9, 0.1, 0, 0, 0], [0.4, 0.35, 0.25, 0, 0], [0.0, 0.0, 0.8, 0.2, 0.0]])
    out = apply_confidence_threshold(codes, proba, threshold=0.55, unclassified_code=0)
    assert list(out) == [1, 0, 3]  # middle window (max 0.4) -> unclassified


def test_confidence_threshold_zero_is_noop():
    codes = np.array([1, 2])
    proba = np.array([[0.3, 0.7, 0, 0, 0], [0.6, 0.4, 0, 0, 0]])
    np.testing.assert_array_equal(apply_confidence_threshold(codes, proba, 0.0, 0), codes)


def test_blended_matrix_forbids_impossible_and_absent_transition():
    # Np->E2 is impossible and never appears -> must be -inf.
    seqs = [np.array([1, 2, 3, 4, 5, 2, 1])]
    log_trans = build_blended_transition_log(seqs, CLASSES, ALLOWED, CODE_TO_LABEL)
    idx = {c: i for i, c in enumerate(CLASSES)}
    assert log_trans[idx[1], idx[5]] == -np.inf   # Np -> E2 forbidden
    assert np.isfinite(log_trans[idx[1], idx[2]])  # Np -> C allowed & seen


def test_blended_matrix_keeps_impossible_but_observed_transition():
    # D->C is "impossible" per grammar, but if the DATA shows it a lot, the
    # blend must NOT zero it (annotations win over the idealized grammar).
    seqs = [np.array([3, 2] * 50)]  # D->C repeated
    log_trans = build_blended_transition_log(seqs, CLASSES, ALLOWED, CODE_TO_LABEL)
    idx = {c: i for i, c in enumerate(CLASSES)}
    assert np.isfinite(log_trans[idx[3], idx[2]])  # kept because empirically common


def test_postprocess_predictions_argmax_fallback_without_transition():
    proba = np.array([[0.2, 0.8, 0, 0, 0], [0.7, 0.3, 0, 0, 0]])
    out = postprocess_predictions(proba, CLASSES, transition_log=None, threshold=0.0)
    assert list(out) == [2, 1]
