"""Stage mapping, epoch alignment and epoch-count consistency."""

from __future__ import annotations

import numpy as np
import pytest

from src import config
from src.data import (
    Dataset,
    SubjectEpochs,
    epoch_raw,
    map_stage,
    parse_subject_night,
    sleep_period_mask,
)
from src.features import extract_features, feature_names


# --------------------------------------------------------------------------- #
# Stage label mapping
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "description, expected",
    [
        ("Sleep stage W", "W"),
        ("Sleep stage 1", "N1"),
        ("Sleep stage 2", "N2"),
        ("Sleep stage 3", "N3"),
        ("Sleep stage 4", "N3"),
        ("Sleep stage R", "REM"),
    ],
)
def test_map_stage(description, expected):
    assert map_stage(description) == expected


def test_stages_3_and_4_merge_into_one_class():
    assert map_stage("Sleep stage 3") == map_stage("Sleep stage 4") == "N3"


@pytest.mark.parametrize("description", ["Sleep stage ?", "Movement time", "", "Sleep stage 5"])
def test_unscorable_annotations_are_dropped(description):
    assert map_stage(description) is None


def test_stage_map_produces_exactly_five_classes():
    assert set(config.STAGE_MAP.values()) == set(config.STAGE_NAMES)
    assert len(config.STAGE_NAMES) == 5
    assert sorted(config.STAGE_TO_INT.values()) == [0, 1, 2, 3, 4]


def test_parse_subject_night():
    assert parse_subject_night("SC4001E0-PSG.edf") == (0, 1)
    assert parse_subject_night("SC4112E0-Hypnogram.edf") == (11, 2)
    with pytest.raises(ValueError):
        parse_subject_night("ST7011J0-PSG.edf")  # sleep-telemetry, not cassette


# --------------------------------------------------------------------------- #
# Epoching
# --------------------------------------------------------------------------- #


def test_epoch_count_matches_scorable_annotations(annotated_raw):
    """One epoch per scorable annotation -- no more, no fewer."""
    raw, descriptions = annotated_raw
    expected = sum(map_stage(d) is not None for d in descriptions)

    epochs = epoch_raw(raw, subject=0, crop=False)

    assert epochs.data.shape[0] == expected
    assert len(epochs.labels) == expected
    assert len(epochs.onsets_sec) == expected


def test_epoch_labels_follow_the_annotation_order(annotated_raw):
    raw, descriptions = annotated_raw
    expected = [config.STAGE_TO_INT[s] for d in descriptions if (s := map_stage(d)) is not None]

    epochs = epoch_raw(raw, subject=0, crop=False)

    np.testing.assert_array_equal(epochs.labels, expected)


def test_epochs_are_contiguous_and_30s_long(annotated_raw):
    raw, _ = annotated_raw
    epochs = epoch_raw(raw, subject=0, crop=False)

    assert epochs.data.shape[1] == len(config.CHANNELS)
    assert epochs.data.shape[2] == int(config.EPOCH_SEC * epochs.sfreq)
    # Annotation-aligned epochs never overlap.
    assert (np.diff(epochs.onsets_sec) >= config.EPOCH_SEC - 1e-9).all()


def test_epoch_count_survives_feature_extraction(annotated_raw):
    """The feature matrix must have exactly one row per retained epoch."""
    raw, _ = annotated_raw
    epochs = epoch_raw(raw, subject=0, crop=False)

    X = extract_features(epochs.data, sfreq=epochs.sfreq)

    assert X.shape[0] == epochs.data.shape[0] == len(epochs.labels)
    assert X.shape[1] == len(feature_names())


def test_subject_epochs_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="inconsistent epoch counts"):
        SubjectEpochs(
            subject=0,
            data=np.zeros((3, 2, 10)),
            labels=np.zeros(2, dtype=int),
            onsets_sec=np.zeros(3),
            sfreq=100.0,
        )


# --------------------------------------------------------------------------- #
# Sleep-period crop
# --------------------------------------------------------------------------- #


def test_crop_keeps_all_sleep_and_trims_wake_tails():
    w, n2 = config.STAGE_TO_INT["W"], config.STAGE_TO_INT["N2"]
    # 200 wake epochs, 100 sleep epochs, 200 wake epochs.
    labels = np.array([w] * 200 + [n2] * 100 + [w] * 200)
    onsets = np.arange(len(labels)) * config.EPOCH_SEC

    keep = sleep_period_mask(labels, onsets, margin_sec=30 * 60)  # 60 epochs

    assert keep.sum() == 60 + 100 + 60
    assert keep[200:300].all()          # no sleep epoch is ever discarded
    assert not keep[:140].any()
    assert not keep[360:].any()


def test_crop_is_a_no_op_when_the_subject_never_sleeps():
    labels = np.full(50, config.STAGE_TO_INT["W"])
    onsets = np.arange(50) * config.EPOCH_SEC

    assert sleep_period_mask(labels, onsets, margin_sec=1800).all()


def test_crop_reduces_epoch_count_on_a_real_epochs_object(annotated_raw):
    raw, _ = annotated_raw
    uncropped = epoch_raw(raw, subject=0, crop=False)
    cropped = epoch_raw(raw, subject=0, crop=True)

    # This recording is far shorter than the 30 min margin, so nothing is lost;
    # the invariant that matters is that cropping never adds epochs or reorders.
    assert cropped.data.shape[0] <= uncropped.data.shape[0]
    assert set(cropped.onsets_sec).issubset(set(uncropped.onsets_sec))


def test_sleep_period_mask_validates_shapes():
    with pytest.raises(ValueError, match="same shape"):
        sleep_period_mask(np.zeros(5, dtype=int), np.zeros(4), margin_sec=0.0)


# --------------------------------------------------------------------------- #
# Dataset container
# --------------------------------------------------------------------------- #


def test_dataset_rejects_inconsistent_arrays():
    with pytest.raises(ValueError, match="number of epochs"):
        Dataset(
            X=np.zeros((10, 12)),
            y=np.zeros(9, dtype=int),
            groups=np.zeros(10, dtype=int),
            onsets_sec=np.zeros(10),
            feature_names=feature_names(),
        )


def test_dataset_rejects_wrong_feature_name_count():
    with pytest.raises(ValueError, match="feature_names"):
        Dataset(
            X=np.zeros((10, 11)),
            y=np.zeros(10, dtype=int),
            groups=np.zeros(10, dtype=int),
            onsets_sec=np.zeros(10),
            feature_names=feature_names(),
        )


def test_dataset_roundtrips_through_disk(synthetic_dataset, tmp_path):
    path = tmp_path / "features.npz"
    synthetic_dataset.save(path)
    restored = Dataset.load(path)

    np.testing.assert_allclose(restored.X, synthetic_dataset.X)
    np.testing.assert_array_equal(restored.y, synthetic_dataset.y)
    np.testing.assert_array_equal(restored.groups, synthetic_dataset.groups)
    assert restored.feature_names == synthetic_dataset.feature_names
