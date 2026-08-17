"""Temporal context, per-recording normalisation and feature-cache coherence.

These are the invariants the three new pipeline stages depend on.  The dangerous
failure modes here are all silent -- smoothing across a subject boundary, a
trailing window that quietly reaches into the future, a normalisation computed
over the pooled matrix instead of per night, or a stale cache -- so each one gets
an explicit test rather than being left to the end-to-end numbers.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import config
from src.data import (
    Dataset,
    StaleCacheError,
    SubjectEpochs,
    assemble_dataset,
    cache_path,
    prepare_recording,
)
from src.features import (
    add_temporal_context,
    base_feature_names,
    centred_window,
    extract_features,
    feature_names,
    normalise_per_recording,
    trailing_window,
)
from tests.conftest import SFREQ, make_bouts, make_epochs


# --------------------------------------------------------------------------- #
# Window definitions
# --------------------------------------------------------------------------- #


def test_centred_window_is_symmetric_and_triangular():
    """7.5 min / 30 s = 15 epochs, centred on the current one with no rounding."""
    weights, offsets = centred_window()

    assert len(weights) == len(offsets) == 15
    np.testing.assert_array_equal(offsets, np.arange(-7, 8))
    # Symmetric, peaking on the current epoch.
    np.testing.assert_allclose(weights, weights[::-1])
    assert weights.argmax() == 7
    assert offsets[weights.argmax()] == 0
    assert (weights > 0).all()


def test_trailing_window_never_looks_into_the_future():
    """2 min / 30 s = 4 epochs: three past epochs and the current one."""
    weights, offsets = trailing_window()

    assert len(weights) == len(offsets) == 4
    np.testing.assert_array_equal(offsets, np.arange(-3, 1))
    assert offsets.max() == 0, "a trailing window must not include future epochs"
    np.testing.assert_allclose(weights, 1.0)


# --------------------------------------------------------------------------- #
# Rolling behaviour
# --------------------------------------------------------------------------- #


def _contiguous(n_epochs: int) -> np.ndarray:
    return np.arange(n_epochs, dtype=float) * config.EPOCH_SEC


def test_smoothing_preserves_a_constant_column():
    """A weighted average of identical values is that value, edges included."""
    X = np.full((40, 3), 7.0)

    smoothed = add_temporal_context(X, _contiguous(40))

    np.testing.assert_allclose(smoothed, 7.0)


def test_trailing_average_uses_only_past_and_present():
    """Inject a step: no epoch before it may move."""
    n = 40
    X = np.zeros((n, 1))
    X[20:] = 1.0
    step = 20

    smoothed = add_temporal_context(X, _contiguous(n))
    n_base = X.shape[1]
    trailing = smoothed[:, 2 * n_base : 3 * n_base]

    # Everything strictly before the step is still exactly zero.
    np.testing.assert_allclose(trailing[:step], 0.0)
    # The step epoch itself sees one of four epochs raised.
    assert trailing[step, 0] == pytest.approx(0.25)
    assert trailing[step + 3, 0] == pytest.approx(1.0)


def test_centred_average_does_look_both_ways():
    """The centred window is the one allowed to see the future, up to 7 epochs.

    With a step at epoch 20, epoch 13 is exactly at the edge of reach (13 + 7 =
    20) and picks up the smallest triangular weight, 1/8 of a total weight of 8.
    Epoch 12 is one step beyond reach and must be untouched.
    """
    n = 40
    step = 20
    X = np.zeros((n, 1))
    X[step:] = 1.0
    n_base = X.shape[1]

    smoothed = add_temporal_context(X, _contiguous(n))
    centred = smoothed[:, n_base : 2 * n_base]

    assert centred[19, 0] > 0.0, "centred window should already see the step"
    assert centred[13, 0] == pytest.approx(1.0 / 64.0), "faintest weight at full reach"
    assert centred[12, 0] == pytest.approx(0.0), "nothing beyond a 7-epoch reach"


def test_smoothing_is_not_carried_across_a_gap():
    """A dropped epoch leaves a hole; the window must not bridge it.

    Sleep-EDF drops "Sleep stage ?" and "Movement time" epochs, which can fall
    inside the sleep period, so the epoch sequence is not always contiguous.
    """
    # Two blocks of 8 epochs, 100 epochs of empty lattice between them.
    onsets = np.concatenate(
        [np.arange(8, dtype=float), 108 + np.arange(8, dtype=float)]
    ) * config.EPOCH_SEC
    X = np.zeros((16, 1))
    X[8:] = 1.0

    smoothed = add_temporal_context(X, onsets)
    centred = smoothed[:, 1:2]

    # The blocks are far further apart than the window, so neither leaks.
    np.testing.assert_allclose(centred[:8], 0.0)
    np.testing.assert_allclose(centred[8:], 1.0)


def test_weights_are_renormalised_at_the_recording_edges():
    """A truncated window must average what exists, not pad with zeros."""
    X = np.ones((30, 1))

    smoothed = add_temporal_context(X, _contiguous(30))

    # If missing neighbours were treated as zeros the first epochs would sag.
    np.testing.assert_allclose(smoothed, 1.0)


def test_rolling_rejects_unsorted_onsets():
    X = np.zeros((5, 2))
    onsets = np.array([0.0, 30.0, 90.0, 60.0, 120.0])

    with pytest.raises(ValueError, match="strictly increasing"):
        add_temporal_context(X, onsets)


# --------------------------------------------------------------------------- #
# Per-recording normalisation
# --------------------------------------------------------------------------- #


def test_normalisation_centres_each_column_independently(rng):
    """Median 0 and unit 5-95 spread, per column -- not over the whole matrix."""
    X = rng.standard_normal((300, 6)) * np.array([1.0, 10.0, 100.0, 0.1, 5.0, 50.0])
    X += np.array([0.0, -20.0, 300.0, 1.0, -5.0, 80.0])

    Z = normalise_per_recording(X)

    np.testing.assert_allclose(np.median(Z, axis=0), 0.0, atol=1e-12)
    lo, hi = np.percentile(Z, config.NORM_PERCENTILES, axis=0)
    np.testing.assert_allclose(hi - lo, 1.0, rtol=1e-12)


def test_normalisation_leaves_a_constant_column_finite():
    """The spread floor must stop a near-constant column exploding."""
    X = np.column_stack([np.full(50, 3.0), np.linspace(0, 1, 50)])

    Z = normalise_per_recording(X)

    assert np.isfinite(Z).all()
    np.testing.assert_allclose(Z[:, 0], 0.0)


def test_normalisation_is_per_recording_not_pooled(rng):
    """Two nights with different gains must normalise to the same distribution.

    This is the whole point of doing it per recording: a subject whose electrode
    impedance doubles every power reading should still land on a comparable
    scale, which pooled normalisation would not deliver.
    """
    labels = make_bouts({"W": 0.3, "REM": 0.2, "N2": 0.5}, 60, rng)
    onsets = np.arange(len(labels), dtype=float) * config.EPOCH_SEC
    quiet = make_epochs(labels, rng)

    a = prepare_recording(extract_features(quiet, SFREQ), onsets)
    b = prepare_recording(extract_features(quiet * 40.0, SFREQ), onsets)

    np.testing.assert_allclose(a, b, rtol=1e-7, atol=1e-9)

    # Pooling the two nights before normalising would not give this.
    pooled = normalise_per_recording(
        np.concatenate(
            [
                add_temporal_context(extract_features(quiet, SFREQ), onsets),
                add_temporal_context(extract_features(quiet * 40.0, SFREQ), onsets),
            ]
        )
    )
    half = len(labels)
    assert not np.allclose(pooled[:half], pooled[half:], rtol=1e-3), (
        "pooled normalisation should NOT equalise the two gains; if it does, "
        "this test has stopped discriminating"
    )


# --------------------------------------------------------------------------- #
# Subject independence of the whole per-recording stage
# --------------------------------------------------------------------------- #


def test_prepare_recording_gives_each_subject_the_same_answer_alone(rng):
    """Per-subject features must not depend on who else is in the dataset.

    ``build_dataset`` calls ``prepare_recording`` inside its per-subject loop.
    If smoothing or normalisation were ever hoisted out onto the concatenated
    matrix, a subject's own features would start depending on its neighbours,
    and grouped cross-validation would no longer isolate subjects.
    """
    per_subject = []
    for gain in (1.0, 17.0, 0.3):
        labels = make_bouts({"W": 0.3, "N2": 0.5, "REM": 0.2}, 60, rng)
        onsets = np.arange(len(labels), dtype=float) * config.EPOCH_SEC
        raw = extract_features(make_epochs(labels, rng) * gain, SFREQ)
        per_subject.append((raw, onsets, prepare_recording(raw, onsets)))

    # Recomputing one subject in isolation reproduces it exactly.
    for raw, onsets, expected in per_subject:
        np.testing.assert_allclose(prepare_recording(raw, onsets), expected, rtol=1e-12)

    # And a pooled application differs, i.e. the distinction is observable.
    pooled_raw = np.concatenate([raw for raw, _, _ in per_subject])
    pooled_onsets = np.arange(pooled_raw.shape[0], dtype=float) * config.EPOCH_SEC
    pooled = prepare_recording(pooled_raw, pooled_onsets)
    stacked = np.concatenate([done for _, _, done in per_subject])

    assert not np.allclose(pooled, stacked, rtol=1e-3), (
        "pooling should visibly differ from per-recording processing"
    )


# --------------------------------------------------------------------------- #
# Assembly: the wiring, not just the functions
# --------------------------------------------------------------------------- #


def _synthetic_recordings(rng, sizes=(40, 55, 48), gains=(1.0, 25.0, 0.2)):
    """Recordings with deliberately different amplitudes and stage content.

    Different gains make pooled normalisation diverge sharply from per-recording
    normalisation; contiguous bouts make cross-boundary smoothing observable.
    """
    recordings, expected = [], []
    for subject, (n, gain) in enumerate(zip(sizes, gains)):
        labels = make_bouts({"W": 0.3, "N2": 0.4, "N3": 0.2, "REM": 0.1}, n, rng)
        onsets = np.arange(len(labels), dtype=float) * config.EPOCH_SEC
        data = make_epochs(labels, rng) * gain

        recordings.append(
            SubjectEpochs(
                subject=subject,
                data=data,
                labels=np.array([config.STAGE_TO_INT[s] for s in labels]),
                onsets_sec=onsets,
                sfreq=SFREQ,
            )
        )
        # What this recording must look like when processed entirely alone.
        expected.append(prepare_recording(extract_features(data, SFREQ), onsets))
    return recordings, expected


def test_assemble_dataset_processes_each_recording_independently(rng):
    """Each subject's rows must equal that subject processed on its own.

    This is the wiring test for the whole per-recording stage.  ``prepare_recording``
    being correct in isolation proves nothing about whether the assembly actually
    applies it per recording, which is exactly the gap the ``smooth_pooled`` and
    ``norm_pooled_recordings`` mutations exploit.
    """
    recordings, expected = _synthetic_recordings(rng)

    dataset = assemble_dataset(recordings)

    assert dataset.X.shape[1] == len(feature_names())
    for subject, block in enumerate(expected):
        rows = dataset.X[dataset.groups == subject]
        assert rows.shape == block.shape
        np.testing.assert_allclose(
            rows,
            block,
            rtol=1e-10,
            atol=1e-12,
            err_msg=(
                f"subject {subject} differs from being processed alone; temporal "
                "context or normalisation is being applied across recordings"
            ),
        )


def test_assemble_dataset_does_not_smooth_across_a_subject_boundary(rng):
    """The epochs either side of a boundary are where leakage would show first.

    A 15-epoch centred window reaches 7 epochs, so if the recordings were
    concatenated before smoothing, the last 7 rows of one subject and the first 7
    of the next would both be contaminated.
    """
    recordings, expected = _synthetic_recordings(rng)
    reach = len(centred_window()[1]) // 2

    dataset = assemble_dataset(recordings)

    for subject, block in enumerate(expected):
        rows = dataset.X[dataset.groups == subject]
        np.testing.assert_allclose(
            rows[:reach], block[:reach], rtol=1e-10, atol=1e-12,
            err_msg=f"subject {subject}: leading epochs contaminated by the previous recording",
        )
        np.testing.assert_allclose(
            rows[-reach:], block[-reach:], rtol=1e-10, atol=1e-12,
            err_msg=f"subject {subject}: trailing epochs contaminated by the next recording",
        )


def test_assemble_dataset_keeps_labels_and_groups_aligned(rng):
    recordings, _ = _synthetic_recordings(rng)

    dataset = assemble_dataset(recordings)

    assert len(dataset.y) == sum(len(r.labels) for r in recordings)
    for recording in recordings:
        mask = dataset.groups == recording.subject
        np.testing.assert_array_equal(dataset.y[mask], recording.labels)
        np.testing.assert_allclose(dataset.onsets_sec[mask], recording.onsets_sec)


def test_assemble_dataset_consumes_a_lazy_iterable(rng):
    """``build_dataset`` passes a generator so one recording is loaded at a time."""
    recordings, expected = _synthetic_recordings(rng, sizes=(40, 45), gains=(1.0, 8.0))

    dataset = assemble_dataset(r for r in recordings)

    np.testing.assert_allclose(dataset.X[dataset.groups == 0], expected[0], rtol=1e-10, atol=1e-12)


def test_assemble_dataset_rejects_an_empty_input():
    with pytest.raises(ValueError, match="no recordings"):
        assemble_dataset([])


def test_dataset_columns_match_the_context_feature_names(synthetic_dataset):
    assert synthetic_dataset.X.shape[1] == len(feature_names())
    assert synthetic_dataset.X.shape[1] == 3 * len(base_feature_names())
    assert np.isfinite(synthetic_dataset.X).all()


# --------------------------------------------------------------------------- #
# Feature-cache coherence
# --------------------------------------------------------------------------- #


def test_fingerprint_is_stable_across_calls():
    assert config.feature_fingerprint() == config.feature_fingerprint()
    assert len(config.feature_fingerprint()) == 10


@pytest.mark.parametrize(
    "attribute, value",
    [
        ("SMOOTH_CENTRED_MIN", 5.0),
        ("SMOOTH_TRAILING_MIN", 3.0),
        ("NORM_PERCENTILES", (10.0, 90.0)),
        ("WELCH_SEG_SEC", 2.0),
        ("HIGUCHI_KMAX", 8),
        ("CROP_MARGIN_MIN", 15.0),
        ("L_FREQ", 0.5),
    ],
)
def test_fingerprint_changes_with_every_spec_constant(attribute, value, monkeypatch):
    """Anything that changes the numbers must change the cache identity."""
    before = config.feature_fingerprint()
    monkeypatch.setattr(config, attribute, value)

    assert config.feature_fingerprint() != before, (
        f"changing {attribute} left the fingerprint untouched, so a stale cache "
        "would be reused"
    )


def test_cache_path_carries_the_fingerprint():
    path = cache_path(n_subjects=20, night=1, crop=True)

    assert config.feature_fingerprint() in path.name
    assert "20subj" in path.name and "night1" in path.name


def test_cache_path_distinguishes_cropped_from_uncropped():
    assert cache_path(crop=True) != cache_path(crop=False)


def test_loading_a_cache_from_another_spec_is_refused(synthetic_dataset, tmp_path, monkeypatch):
    """A renamed or hand-copied cache must not be trusted on filename alone."""
    path = tmp_path / "features.npz"
    synthetic_dataset.save(path)

    assert Dataset.load(path).X.shape == synthetic_dataset.X.shape

    monkeypatch.setattr(config, "SMOOTH_CENTRED_MIN", 12.5)
    with pytest.raises(StaleCacheError, match="feature spec"):
        Dataset.load(path)
