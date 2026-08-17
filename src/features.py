"""Spectral features: relative band powers and spectral entropy.

One feature vector per (epoch, channel):

* five **relative band powers** -- delta, theta, alpha, sigma, beta -- each the
  band's share of total power over :data:`config.TOTAL_BAND`;
* **spectral entropy** -- the normalised Shannon entropy of the power spectrum
  over the same support, in ``[0, 1]``.

Powers are obtained by integrating the Welch PSD with Simpson's rule rather
than by summing bins, so the values do not depend on the frequency grid.  The
total used as the denominator is the sum of the five band powers, which makes
the relative powers of a channel sum to exactly 1.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import simpson
from scipy.signal import welch

from . import config

#: Below this total power a channel is treated as flat (disconnected electrode,
#: clipped segment) and given a uniform spectrum rather than dividing by ~0.
_POWER_FLOOR = 1e-20


def feature_names(
    channel_slugs: list[str] | None = None,
    bands: list[tuple[str, float, float]] | None = None,
) -> list[str]:
    """Column names matching the layout produced by :func:`extract_features`."""
    channel_slugs = channel_slugs or config.CHANNEL_SLUGS
    bands = bands or config.BANDS

    names = []
    for slug in channel_slugs:
        names.extend(f"{slug}_{band}_rel" for band, _, _ in bands)
        names.append(f"{slug}_spectral_entropy")
    return names


def compute_psd(data: np.ndarray, sfreq: float) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD along the last axis.

    A 4 s Hann segment at 100 Hz gives a 0.25 Hz grid -- every band edge lands
    on a bin boundary -- and ~14 half-overlapping segments per 30 s epoch.
    """
    n_times = data.shape[-1]
    nperseg = min(int(round(config.WELCH_SEG_SEC * sfreq)), n_times)
    noverlap = int(nperseg * config.WELCH_OVERLAP)

    freqs, psd = welch(
        data,
        fs=sfreq,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        axis=-1,
    )
    return freqs, psd


def _integrate(freqs: np.ndarray, psd: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Integrate the PSD over ``[lo, hi]`` inclusive of both edges."""
    mask = (freqs >= lo) & (freqs <= hi)
    if mask.sum() < 2:
        raise ValueError(
            f"band [{lo}, {hi}] Hz covers {mask.sum()} bin(s) on a "
            f"{freqs[1] - freqs[0]:.3g} Hz grid; lengthen the Welch segment"
        )
    return simpson(psd[..., mask], x=freqs[mask], axis=-1)


def relative_band_powers(
    freqs: np.ndarray,
    psd: np.ndarray,
    bands: list[tuple[str, float, float]] | None = None,
) -> np.ndarray:
    """Share of total power in each band; the last axis sums to 1.

    Returns an array shaped ``psd.shape[:-1] + (n_bands,)``.
    """
    bands = bands or config.BANDS
    absolute = np.stack([_integrate(freqs, psd, lo, hi) for _, lo, hi in bands], axis=-1)

    total = absolute.sum(axis=-1, keepdims=True)
    flat = total < _POWER_FLOOR
    relative = np.divide(absolute, np.where(flat, 1.0, total))
    # Flat channels get a uniform spectrum instead of 0/0.
    return np.where(flat, 1.0 / len(bands), relative)


def spectral_entropy(
    freqs: np.ndarray, psd: np.ndarray, band: tuple[float, float] | None = None
) -> np.ndarray:
    """Shannon entropy of the normalised PSD, scaled to ``[0, 1]``.

    0 means all power sits in a single frequency bin, 1 means the spectrum is
    flat across the band.
    """
    lo, hi = band or config.TOTAL_BAND
    mask = (freqs >= lo) & (freqs <= hi)
    n_bins = int(mask.sum())
    if n_bins < 2:
        raise ValueError(f"entropy support [{lo}, {hi}] Hz covers {n_bins} bin(s)")

    power = psd[..., mask]
    total = power.sum(axis=-1, keepdims=True)
    flat = total < _POWER_FLOOR
    p = np.where(flat, 1.0 / n_bins, np.divide(power, np.where(flat, 1.0, total)))

    # 0 * log(0) == 0 by convention.
    terms = np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0)
    return -terms.sum(axis=-1) / np.log(n_bins)


def extract_features(data: np.ndarray, sfreq: float) -> np.ndarray:
    """Feature matrix for a stack of epochs.

    Parameters
    ----------
    data
        ``(n_epochs, n_channels, n_times)`` epoched signal.
    sfreq
        Sampling frequency in Hz.

    Returns
    -------
    ``(n_epochs, n_channels * (n_bands + 1))``, channel-major: all of channel
    0's features, then all of channel 1's.  See :func:`feature_names`.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 3:
        raise ValueError(f"expected (n_epochs, n_channels, n_times), got shape {data.shape}")

    freqs, psd = compute_psd(data, sfreq)          # (n_epochs, n_channels, n_freqs)
    powers = relative_band_powers(freqs, psd)      # (n_epochs, n_channels, n_bands)
    entropy = spectral_entropy(freqs, psd)         # (n_epochs, n_channels)

    per_channel = np.concatenate([powers, entropy[..., np.newaxis]], axis=-1)
    n_epochs, n_channels, n_feat = per_channel.shape
    return per_channel.reshape(n_epochs, n_channels * n_feat)
