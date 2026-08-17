"""Feature matrix shape, naming and numerical behaviour."""

from __future__ import annotations

import numpy as np
import pytest

from src import config
from src.features import (
    LOG_AMPLITUDE_FEATURES,
    LOG_POWER_FEATURES,
    add_temporal_context,
    base_feature_names,
    compute_psd,
    extract_features,
    feature_names,
    higuchi_fd,
    normalise_per_recording,
    permutation_entropy,
    petrosian_fd,
    relative_band_powers,
    spectral_entropy,
)
from tests.conftest import SFREQ, make_epochs


# --------------------------------------------------------------------------- #
# Shape and naming
# --------------------------------------------------------------------------- #


def test_feature_matrix_shape(epoch_data):
    """39 per-epoch features: 14 per EEG channel, 9 for EOG, 2 for EMG."""
    data, _ = epoch_data
    X = extract_features(data, sfreq=SFREQ)

    n_eeg = len(config.EEG_CHANNELS) * (len(config.BANDS) + 5 + len(config.BAND_RATIOS))
    n_eog = len(config.EOG_CHANNELS) * (len(config.BANDS) + 4)
    n_emg = len(config.EMG_CHANNELS) * 2

    assert X.shape == (data.shape[0], n_eeg + n_eog + n_emg)
    assert X.shape[1] == 39
    assert len(base_feature_names()) == 39


def test_temporal_context_triples_the_column_count(epoch_data):
    """Raw, centred-smoothed and trailing-smoothed copies of every feature."""
    data, _ = epoch_data
    X = extract_features(data, sfreq=SFREQ)
    onsets = np.arange(X.shape[0], dtype=float) * config.EPOCH_SEC

    X_ctx = add_temporal_context(X, onsets)

    assert X_ctx.shape == (X.shape[0], 3 * X.shape[1])
    assert X_ctx.shape[1] == 117
    assert len(feature_names()) == 117
    # The raw block is left untouched at the front.
    np.testing.assert_allclose(X_ctx[:, : X.shape[1]], X)


def test_feature_names_match_columns(epoch_data):
    data, _ = epoch_data
    X = extract_features(data, sfreq=SFREQ)
    names = base_feature_names()

    assert len(names) == X.shape[1]
    assert len(set(names)) == len(names)
    assert len(set(feature_names())) == len(feature_names())
    # Channel-major layout: channel 0's six features come first.
    assert names[: len(config.BANDS) + 1] == [
        "fpz_cz_delta_rel",
        "fpz_cz_theta_rel",
        "fpz_cz_alpha_rel",
        "fpz_cz_sigma_rel",
        "fpz_cz_beta_rel",
        "fpz_cz_spectral_entropy",
    ]


def test_extract_features_rejects_non_epoched_input(rng):
    with pytest.raises(ValueError, match="n_epochs"):
        extract_features(rng.standard_normal((2, 3000)), sfreq=SFREQ)


def test_features_are_finite(epoch_data):
    data, _ = epoch_data
    assert np.isfinite(extract_features(data, sfreq=SFREQ)).all()


# --------------------------------------------------------------------------- #
# Numerical properties
# --------------------------------------------------------------------------- #


def test_relative_powers_sum_to_one(epoch_data):
    data, _ = epoch_data
    freqs, psd = compute_psd(data, SFREQ)
    powers = relative_band_powers(freqs, psd)

    assert powers.shape == (data.shape[0], len(config.CHANNELS), len(config.BANDS))
    np.testing.assert_allclose(powers.sum(axis=-1), 1.0, rtol=1e-10)
    assert (powers >= 0).all()


def test_welch_grid_aligns_with_band_edges():
    """Every band edge must fall on a bin boundary, not inside a bin."""
    freqs, _ = compute_psd(np.zeros((1, 1, int(config.EPOCH_SEC * SFREQ))), SFREQ)
    resolution = freqs[1] - freqs[0]

    edges = {config.TOTAL_BAND[0], config.TOTAL_BAND[1]}
    edges.update(edge for _, lo, hi in config.BANDS for edge in (lo, hi))
    for edge in edges:
        assert np.isclose(edge % resolution, 0.0, atol=1e-9) or np.isclose(
            edge % resolution, resolution, atol=1e-9
        ), f"{edge} Hz does not land on the {resolution} Hz grid"


@pytest.mark.parametrize(
    "freq, expected_band",
    [(1.5, "delta"), (6.0, "theta"), (10.0, "alpha"), (14.0, "sigma"), (22.0, "beta")],
)
def test_dominant_band_matches_injected_frequency(freq, expected_band, rng):
    """A pure tone must put most relative power in the band containing it."""
    t = np.arange(int(config.EPOCH_SEC * SFREQ)) / SFREQ
    signal = np.sin(2 * np.pi * freq * t)[np.newaxis, np.newaxis, :]

    freqs, psd = compute_psd(signal, SFREQ)
    powers = relative_band_powers(freqs, psd)[0, 0]

    winner = config.BANDS[int(np.argmax(powers))][0]
    assert winner == expected_band
    assert powers.max() > 0.8


def test_entropy_is_bounded_and_ordered(rng):
    """A tone is spectrally concentrated; white noise is not."""
    n_times = int(config.EPOCH_SEC * SFREQ)
    t = np.arange(n_times) / SFREQ

    tone = np.sin(2 * np.pi * 10.0 * t)[np.newaxis, np.newaxis, :]
    noise = rng.standard_normal((1, 1, n_times))

    h_tone = spectral_entropy(*compute_psd(tone, SFREQ))[0, 0]
    h_noise = spectral_entropy(*compute_psd(noise, SFREQ))[0, 0]

    assert 0.0 <= h_tone <= 1.0
    assert 0.0 <= h_noise <= 1.0
    assert h_tone < 0.5 < h_noise


def test_flat_channel_does_not_produce_nan():
    """A dead electrode must degrade gracefully instead of dividing by zero."""
    flat = np.zeros((1, 1, int(config.EPOCH_SEC * SFREQ)))
    freqs, psd = compute_psd(flat, SFREQ)

    powers = relative_band_powers(freqs, psd)
    entropy = spectral_entropy(freqs, psd)

    assert np.isfinite(powers).all() and np.isfinite(entropy).all()
    np.testing.assert_allclose(powers.sum(axis=-1), 1.0)
    np.testing.assert_allclose(entropy, 1.0)


def test_scale_free_features_are_amplitude_invariant(rng):
    """Scaling a recording must not change a *scale-free* feature.

    Relative powers, entropies, fractal dimensions and log band-power ratios all
    normalise away amplitude, so they must be bit-for-bit unaffected.  The
    absolute-power and amplitude columns are excluded here and pinned by the two
    tests below.
    """
    data = make_epochs(["N2"] * 3, rng)
    names = base_feature_names()
    scale_free = [
        i
        for i, n in enumerate(names)
        if n not in LOG_POWER_FEATURES and n not in LOG_AMPLITUDE_FEATURES
    ]
    assert len(scale_free) == 33  # 39 - 3 log_abspow - 3 log_std/log_iqr

    base = extract_features(data, sfreq=SFREQ)
    scaled = extract_features(data * 1000.0, sfreq=SFREQ)

    np.testing.assert_allclose(base[:, scale_free], scaled[:, scale_free], rtol=1e-9, atol=1e-12)


def test_log_absolute_power_shifts_by_two_log_k(rng):
    """Power goes as amplitude squared, so log power shifts by 2*ln(k).

    All logarithms in :mod:`src.features` are natural; ``np.log`` here is too.
    """
    k = 1000.0
    data = make_epochs(["N2", "N3", "REM"], rng)
    names = base_feature_names()
    columns = [names.index(n) for n in LOG_POWER_FEATURES]
    assert len(columns) == 3  # two EEG channels plus EOG

    base = extract_features(data, sfreq=SFREQ)
    scaled = extract_features(data * k, sfreq=SFREQ)

    shift = scaled[:, columns] - base[:, columns]
    np.testing.assert_allclose(shift, 2.0 * np.log(k), rtol=1e-9)


def test_log_amplitude_features_shift_by_one_log_k(rng):
    """A standard deviation and an IQR are amplitudes, so they shift by 1*ln(k).

    Deliberately asserted apart from the power columns above: the two groups
    carry different constants and conflating them would hide a squared-versus-
    linear mix-up.
    """
    k = 1000.0
    data = make_epochs(["N2", "N3", "REM"], rng)
    names = base_feature_names()
    columns = [names.index(n) for n in LOG_AMPLITUDE_FEATURES]
    assert len(columns) == 3  # eog_log_std, emg_log_std, emg_log_iqr

    base = extract_features(data, sfreq=SFREQ)
    scaled = extract_features(data * k, sfreq=SFREQ)

    shift = scaled[:, columns] - base[:, columns]
    np.testing.assert_allclose(shift, 1.0 * np.log(k), rtol=1e-9)


def test_normalisation_makes_the_whole_matrix_scale_invariant(rng):
    """After per-recording normalisation even the amplitude columns are scale-free.

    A robust z-score subtracts the median and divides by the 5-95 spread, and a
    constant shift of a log column cancels in both, so an arbitrary gain on the
    recording leaves the final feature matrix unchanged.
    """
    labels = ["N2", "N3", "REM", "W", "N1"] * 6
    data = make_epochs(labels, rng)
    onsets = np.arange(len(labels), dtype=float) * config.EPOCH_SEC

    def pipeline(signal):
        return normalise_per_recording(
            add_temporal_context(extract_features(signal, sfreq=SFREQ), onsets)
        )

    np.testing.assert_allclose(pipeline(data), pipeline(data * 250.0), rtol=1e-7, atol=1e-9)


# --------------------------------------------------------------------------- #
# Non-linear features
# --------------------------------------------------------------------------- #


def _reference_signals(rng):
    n = int(config.EPOCH_SEC * SFREQ)
    t = np.arange(n) / SFREQ
    return {
        "constant": np.zeros(n),
        "ramp": 1e-3 * np.arange(n),
        "sine": np.sin(2 * np.pi * 1.0 * t),
        "noise": rng.standard_normal(n),
    }


def test_higuchi_fd_on_reference_signals(rng):
    """1.0 for a line, close to 2.0 for white noise -- the defining behaviour."""
    sig = _reference_signals(rng)

    assert higuchi_fd(sig["constant"]) == pytest.approx(1.0)
    assert higuchi_fd(sig["ramp"]) == pytest.approx(1.0, abs=1e-6)
    assert higuchi_fd(sig["noise"]) == pytest.approx(2.0, abs=0.05)
    assert higuchi_fd(sig["ramp"]) < higuchi_fd(sig["sine"]) < higuchi_fd(sig["noise"])


def test_higuchi_fd_stays_in_the_valid_range(rng):
    """A near-periodic signal aliases at large k; the clip must contain it.

    A 10 Hz tone at 100 Hz decimates to a constant at k=10, which sends the
    unclipped slope far above 2.  Real broadband EEG never does this, but the
    guard has to hold when it happens.
    """
    n = int(config.EPOCH_SEC * SFREQ)
    tone = np.sin(2 * np.pi * 10.0 * np.arange(n) / SFREQ)

    assert 1.0 <= higuchi_fd(tone) <= 2.0


def test_petrosian_fd_on_reference_signals(rng):
    """Exactly 1.0 with no sign changes, and monotone in signal irregularity."""
    sig = _reference_signals(rng)

    assert petrosian_fd(sig["constant"]) == pytest.approx(1.0)
    assert petrosian_fd(sig["ramp"]) == pytest.approx(1.0)
    assert petrosian_fd(sig["sine"]) < petrosian_fd(sig["noise"])


def test_permutation_entropy_bounds_and_ordering(rng):
    """0 when one ordinal pattern repeats, ~1 for white noise."""
    sig = _reference_signals(rng)

    assert permutation_entropy(sig["constant"]) == pytest.approx(0.0)
    assert permutation_entropy(sig["ramp"]) == pytest.approx(0.0)
    assert permutation_entropy(sig["noise"]) == pytest.approx(1.0, abs=0.01)
    assert 0.0 < permutation_entropy(sig["sine"]) < permutation_entropy(sig["noise"])


@pytest.mark.parametrize("feature", [higuchi_fd, petrosian_fd, permutation_entropy])
def test_nonlinear_features_are_batched_consistently(feature, rng):
    """The vectorised path must agree with the single-signal path exactly."""
    batch = rng.standard_normal((5, len(config.CHANNELS), int(config.EPOCH_SEC * SFREQ)))

    result = feature(batch)

    assert result.shape == (5, len(config.CHANNELS))
    np.testing.assert_allclose(result[3, 2], feature(batch[3, 2]), rtol=1e-12)


def test_all_features_finite_on_a_dead_channel(rng):
    """A flat channel must not produce NaN or -inf anywhere in the matrix."""
    data = make_epochs(["N2"] * 4, rng)
    data[:, 0] = 0.0  # Fpz-Cz disconnected

    X = extract_features(data, sfreq=SFREQ)

    assert np.isfinite(X).all()
