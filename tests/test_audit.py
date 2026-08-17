"""Two invariants the rest of the suite asserted only indirectly.

Both gaps were found by mutation testing: breaking the pipeline in these two
specific ways left all other tests green.

1. ``epoch_raw`` was only ever exercised with ``crop=False``, or with a
   recording shorter than ``CROP_MARGIN_MIN``, so the crop could be disabled
   entirely without a single failure.  ``sleep_period_mask`` was well tested as
   a pure function; the *wiring* was not.
2. Nothing checked that the estimators actually carry balanced class weights.
   Dropping the weighting silently collapses N1 while leaving overall accuracy
   almost unchanged, which is exactly the failure the weighting prevents.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import config
from src.data import epoch_raw
from src.evaluate import compute_report, out_of_fold_predictions
from src.model import MODELS, build_model


# --------------------------------------------------------------------------- #
# 1. The sleep-period crop is actually applied
# --------------------------------------------------------------------------- #


def test_crop_is_wired_into_epoch_raw(annotated_raw, monkeypatch):
    """``crop=True`` must drop the wake tails, not just be accepted as a kwarg.

    The fixture is 29 epochs long, far shorter than the production 30 min
    margin, so the margin is shortened to 1 min (2 epochs) to make the crop
    observable.  Sleep spans onsets 4 to 22 (in 30 s units), so the retained
    window is 2 to 24: two wake epochs are trimmed from the head and four from
    the tail.
    """
    monkeypatch.setattr(config, "CROP_MARGIN_MIN", 1.0)

    uncropped = epoch_raw(annotated_raw[0], subject=0, crop=False)
    cropped = epoch_raw(annotated_raw[0], subject=0, crop=True)

    assert uncropped.data.shape[0] == 27
    assert cropped.data.shape[0] == 21, "crop=True did not remove the wake tails"

    # Every epoch carrying sleep must survive, and only wake may be trimmed.
    wake = config.STAGE_TO_INT["W"]
    dropped = set(uncropped.onsets_sec) - set(cropped.onsets_sec)
    dropped_labels = {
        int(lab)
        for lab, on in zip(uncropped.labels, uncropped.onsets_sec)
        if on in dropped
    }
    assert dropped_labels == {wake}


def test_crop_never_discards_a_sleep_epoch(annotated_raw, monkeypatch):
    """Whatever the margin, no non-wake epoch may be lost."""
    wake = config.STAGE_TO_INT["W"]
    uncropped = epoch_raw(annotated_raw[0], subject=0, crop=False)
    n_sleep = int((uncropped.labels != wake).sum())

    for margin in (0.0, 0.5, 1.0, 5.0):
        monkeypatch.setattr(config, "CROP_MARGIN_MIN", margin)
        cropped = epoch_raw(annotated_raw[0], subject=0, crop=True)
        assert int((cropped.labels != wake).sum()) == n_sleep


# --------------------------------------------------------------------------- #
# 2. Class weighting is present and effective
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model_name", sorted(MODELS))
def test_every_model_carries_balanced_class_weights(model_name):
    """Structural check: the weighting cannot be dropped from src/model.py."""
    pipeline = build_model(model_name, config.SEED)
    classifier = pipeline.named_steps["clf"]

    assert classifier.get_params()["class_weight"] == "balanced", (
        f"{model_name} lost its balanced class weights; the rare stages will "
        "not be predicted"
    )


def test_rare_class_is_still_predicted_under_imbalance(rng):
    """Behavioural check: a 3 % class must not vanish from the predictions.

    Without balanced weights both estimators stop predicting the rare class
    altogether while overall accuracy barely moves, which is precisely why
    macro F1 and not accuracy is the headline metric.
    """
    from src.data import Dataset
    from src.features import extract_features, feature_names
    from tests.conftest import SFREQ, make_epochs

    proportions = {"W": 0.22, "REM": 0.18, "N1": 0.03, "N2": 0.42, "N3": 0.15}

    X, y, g, t = [], [], [], []
    for subject in range(6):
        labels = []
        for stage, share in proportions.items():
            labels.extend([stage] * max(1, round(share * 200)))
        rng.shuffle(labels)
        X.append(extract_features(make_epochs(labels, rng), sfreq=SFREQ))
        y.append(np.array([config.STAGE_TO_INT[s] for s in labels]))
        g.append(np.full(len(labels), subject, dtype=int))
        t.append(np.arange(len(labels), dtype=float) * config.EPOCH_SEC)

    dataset = Dataset(
        X=np.concatenate(X),
        y=np.concatenate(y),
        groups=np.concatenate(g),
        onsets_sec=np.concatenate(t),
        feature_names=feature_names(),
    )

    rare = config.STAGE_TO_INT["N1"]
    assert (dataset.y == rare).mean() < 0.05, "fixture is not imbalanced enough"

    y_pred, fold_id = out_of_fold_predictions(
        dataset, lambda: build_model("logreg", config.SEED), n_splits=3
    )
    report = compute_report(
        dataset.y, y_pred, fold_id, dataset.groups, model="logreg", n_splits=3
    )

    recall = (y_pred[dataset.y == rare] == rare).mean()
    assert recall > 0.2, (
        f"rare stage recall collapsed to {recall:.2f}; balanced class weights "
        "are probably not being applied"
    )
    assert report.per_class_f1["N1"] > 0.1
