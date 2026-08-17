"""Grouped cross-validation: the guarantee that no subject is ever split."""

from __future__ import annotations

import numpy as np
import pytest

from src import config
from src.evaluate import (
    compute_report,
    evaluate_model,
    iter_group_splits,
    normalised_confusion,
    out_of_fold_predictions,
)
from src.model import MODELS, build_model


# --------------------------------------------------------------------------- #
# The central requirement
# --------------------------------------------------------------------------- #


def test_no_subject_appears_in_both_train_and_test(synthetic_dataset):
    groups = synthetic_dataset.groups

    n_folds = 0
    for train_idx, test_idx in iter_group_splits(len(groups), groups, n_splits=3):
        train_subjects = set(groups[train_idx])
        test_subjects = set(groups[test_idx])

        assert train_subjects, "empty training fold"
        assert test_subjects, "empty test fold"
        assert train_subjects.isdisjoint(test_subjects)
        n_folds += 1

    assert n_folds == 3


def test_every_epoch_is_tested_exactly_once(synthetic_dataset):
    groups = synthetic_dataset.groups
    seen = np.zeros(len(groups), dtype=int)

    for _, test_idx in iter_group_splits(len(groups), groups, n_splits=3):
        seen[test_idx] += 1

    np.testing.assert_array_equal(seen, 1)


def test_every_subject_is_held_out_exactly_once(synthetic_dataset):
    groups = synthetic_dataset.groups
    held_out = []

    for _, test_idx in iter_group_splits(len(groups), groups, n_splits=3):
        held_out.extend(np.unique(groups[test_idx]).tolist())

    assert sorted(held_out) == sorted(np.unique(groups).tolist())


def test_a_subjects_epochs_are_never_divided_across_folds(synthetic_dataset):
    """Every epoch of a subject must land in the same test fold."""
    groups = synthetic_dataset.groups
    fold_of_epoch = np.full(len(groups), -1)

    for k, (_, test_idx) in enumerate(iter_group_splits(len(groups), groups, n_splits=3)):
        fold_of_epoch[test_idx] = k

    for subject in np.unique(groups):
        assert len(set(fold_of_epoch[groups == subject])) == 1


def test_more_folds_than_subjects_is_rejected(synthetic_dataset):
    n_subjects = len(synthetic_dataset.subjects)
    with pytest.raises(ValueError, match="exceeds the number of subjects"):
        list(iter_group_splits(len(synthetic_dataset.y), synthetic_dataset.groups, n_subjects + 1))


# --------------------------------------------------------------------------- #
# Out-of-fold prediction plumbing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model_name", sorted(MODELS))
def test_out_of_fold_predictions_cover_every_epoch(synthetic_dataset, model_name):
    y_pred, fold_id = out_of_fold_predictions(
        synthetic_dataset, lambda: build_model(model_name, config.SEED), n_splits=3
    )

    assert y_pred.shape == synthetic_dataset.y.shape
    assert (y_pred >= 0).all() and (fold_id >= 0).all()
    assert set(np.unique(fold_id)) == {0, 1, 2}
    assert set(np.unique(y_pred)).issubset(set(range(len(config.STAGE_NAMES))))


def test_predictions_are_reproducible(synthetic_dataset):
    def run():
        return out_of_fold_predictions(
            synthetic_dataset, lambda: build_model("gbdt", config.SEED), n_splits=3
        )[0]

    np.testing.assert_array_equal(run(), run())


def test_models_learn_something_on_separable_data(synthetic_dataset):
    """A sanity floor, not a performance claim: the synthetic classes are
    spectrally distinct, so grouped CV should beat chance comfortably."""
    y_pred, fold_id = out_of_fold_predictions(
        synthetic_dataset, lambda: build_model("logreg", config.SEED), n_splits=3
    )
    report = compute_report(
        synthetic_dataset.y,
        y_pred,
        fold_id,
        synthetic_dataset.groups,
        model="logreg",
        n_splits=3,
    )

    assert report.macro_f1 > 0.5
    assert report.kappa > 0.5


# --------------------------------------------------------------------------- #
# Metrics and figures
# --------------------------------------------------------------------------- #


def test_report_covers_all_five_classes(synthetic_dataset):
    y_pred, fold_id = out_of_fold_predictions(
        synthetic_dataset, lambda: build_model("logreg", config.SEED), n_splits=3
    )
    report = compute_report(
        synthetic_dataset.y, y_pred, fold_id, synthetic_dataset.groups, "logreg", 3
    )

    assert set(report.per_class_f1) == set(config.STAGE_NAMES)
    assert sum(report.support.values()) == len(synthetic_dataset.y)
    assert len(report.fold_macro_f1) == len(report.fold_kappa) == 3
    assert np.array(report.confusion).shape == (5, 5)


def test_normalised_confusion_is_row_normalised_not_column():
    """The diagonal must read as recall, which is what the axis label claims.

    Asserted on a hand-built asymmetric matrix rather than on a model's output:
    a well-separated fixture predicts perfectly, and a diagonal confusion matrix
    normalises identically by row or by column, so it cannot tell the two apart.

    W: 8 of 10 correct, 2 called N1.   N1: 1 of 2 correct, 1 called W.
    Row-normalised (recall)    -> [W,W] = 8/10 = 0.80, [N1,N1] = 1/2 = 0.50
    Column-normalised (precision) -> [W,W] = 8/9  = 0.89, [N1,N1] = 1/3 = 0.33
    """
    w, n1 = config.STAGE_TO_INT["W"], config.STAGE_TO_INT["N1"]
    y_true = np.array([w] * 10 + [n1] * 2)
    y_pred = np.array([w] * 8 + [n1] * 2 + [n1] + [w])

    cm = normalised_confusion(y_true, y_pred)

    assert cm.shape == (5, 5)
    assert cm[w, w] == pytest.approx(0.80), "diagonal is not recall"
    assert cm[n1, n1] == pytest.approx(0.50), "diagonal is not recall"
    assert cm[w, n1] == pytest.approx(0.20)

    rows = cm.sum(axis=1)
    np.testing.assert_allclose(rows[~np.isnan(rows)], 1.0)
    # Classes with no annotated epochs stay NaN rather than reading as zero.
    for stage in ("REM", "N2", "N3"):
        assert np.isnan(cm[config.STAGE_TO_INT[stage]]).all()


def test_normalised_confusion_rows_sum_to_one(synthetic_dataset):
    """Same invariant, but on the real out-of-fold path end to end."""
    y_pred, _ = out_of_fold_predictions(
        synthetic_dataset, lambda: build_model("logreg", config.SEED), n_splits=3
    )
    cm = normalised_confusion(synthetic_dataset.y, y_pred)

    assert cm.shape == (5, 5)
    rows = cm.sum(axis=1)
    np.testing.assert_allclose(rows[~np.isnan(rows)], 1.0)


def test_evaluate_model_writes_both_figures(synthetic_dataset, tmp_path):
    report = evaluate_model(
        synthetic_dataset,
        model_name="logreg",
        n_splits=3,
        plot_subject=int(synthetic_dataset.subjects[0]),
        figures=tmp_path,
    )

    written = {p.name for p in tmp_path.glob("*.png")}
    assert "confusion_matrix_logreg.png" in written
    assert any(n.startswith("hypnogram_logreg_subject") for n in written)
    assert report.n_subjects == len(synthetic_dataset.subjects)
