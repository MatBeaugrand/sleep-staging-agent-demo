"""Feature matrix shape, naming and numerical behaviour."""

from __future__ import annotations

import numpy as np
import pytest

from src import config
from src.features import (
    compute_psd,
    extract_features,
    feature_names,
    relative_band_powers,
    spectral_entropy,
)
from tests.conftest import SFREQ, make_epochs


# --------------------------------------------------------------------------- #
# Shape and naming
# --------------------------------------------------------------------------- #


def test_feature_matrix_shape(epoch_data):
    data, _ = epoch_data
    X = extract_features(data, sfreq=SFREQ)

    n_expected = len(config.CHANNELS) * (len(config.BANDS) + 1)
    assert X.shape == (data.shape[0], n_expected)
    assert X.shape[1] == 12  # 2 channels x (5 bands + entropy)


def test_feature_names_match_columns(epoch_data):
    data, _ = epoch_data
    X = extract_features(data, sfreq=SFREQ)
    names = feature_names()

    assert len(names) == X.shape[1]
    assert len(set(names)) == len(names)
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


def test_relative_powers_are_amplitude_invariant(rng):
    """Scaling a recording must not change a *relative* feature."""
    data = make_epochs(["N2"] * 3, rng)

    base = extract_features(data, sfreq=SFREQ)
    scaled = extract_features(data * 1000.0, sfreq=SFREQ)

    np.testing.assert_allclose(base, scaled, rtol=1e-9, atol=1e-12)
