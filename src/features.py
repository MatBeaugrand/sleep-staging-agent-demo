"""Per-epoch features, temporal context and per-recording normalisation.

Three stages, applied in this order by :func:`src.data.build_dataset`:

1. :func:`extract_features` -- 39 features per 30 s epoch.  The feature set
   differs by modality, because the channels do:

   * **EEG** (Fpz-Cz, Pz-Oz), 14 each: five relative band powers, spectral
     entropy, log absolute power, permutation entropy, Higuchi and Petrosian
     fractal dimension, and four log band-power ratios.
   * **EOG** (horizontal), 9: five relative band powers, spectral entropy, log
     absolute power, permutation entropy, log standard deviation.
   * **EMG** (submental), 2: log standard deviation and log IQR only.  The
     channel is recorded at 1 Hz, so it has no usable spectrum at all.

2. :func:`add_temporal_context` -- each of the 39 is additionally smoothed with a
   7.5 min centred triangular window and a 2 min trailing window, giving 117
   columns.  Nine of the twenty most important features in Vallat & Walker
   (2021) are smoothed ones.

3. :func:`normalise_per_recording` -- robust z-score per night.

Powers are obtained by integrating the Welch PSD with Simpson's rule rather
than by summing bins, so the values do not depend on the frequency grid.  The
total used as the denominator is the sum of the five band powers, which makes
the relative powers of a channel sum to exactly 1.

Scaling a recording by ``k`` leaves most columns untouched, shifts the three
``log_abspow`` columns by ``2*ln k`` (power goes as amplitude squared) and the
three ``log_std``/``log_iqr`` columns by ``1*ln k``.  All logarithms here are
natural.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import simpson
from scipy.signal import welch
from scipy.signal.windows import triang
from scipy.stats import iqr as _iqr

from . import config

#: Below this total power a channel is treated as flat (disconnected electrode,
#: clipped segment) and given a uniform spectrum rather than dividing by ~0.
_POWER_FLOOR = 1e-20

#: Floor applied inside every logarithm, so a flat channel yields a large
#: negative number rather than -inf.  Well below any real EEG power in V^2/Hz.
_LOG_FLOOR = 1e-20

#: Suffixes appended to a base feature name by :func:`add_temporal_context`.
CENTRED_SUFFIX = "_smooth_centred"
TRAILING_SUFFIX = "_smooth_trailing"


def _log(x: np.ndarray) -> np.ndarray:
    """Natural log with a floor, so flat channels stay finite."""
    return np.log(np.maximum(x, _LOG_FLOOR))


# --------------------------------------------------------------------------- #
# Feature naming
# --------------------------------------------------------------------------- #


def _eeg_feature_names(slug: str) -> list[str]:
    """The five band powers and spectral entropy come first, in that order.

    Downstream code and tests slice ``names[:len(BANDS) + 1]``; keep it stable.
    """
    names = [f"{slug}_{band}_rel" for band, _, _ in config.BANDS]
    names.append(f"{slug}_spectral_entropy")
    names.append(f"{slug}_log_abspow")
    names.append(f"{slug}_perm_entropy")
    names.append(f"{slug}_higuchi_fd")
    names.append(f"{slug}_petrosian_fd")
    names.extend(f"{slug}_log_ratio_{num}_{den}" for num, den in config.BAND_RATIOS)
    return names


def _eog_feature_names(slug: str) -> list[str]:
    names = [f"{slug}_{band}_rel" for band, _, _ in config.BANDS]
    names.append(f"{slug}_spectral_entropy")
    names.append(f"{slug}_log_abspow")
    names.append(f"{slug}_perm_entropy")
    names.append(f"{slug}_log_std")
    return names


def _emg_feature_names(slug: str) -> list[str]:
    return [f"{slug}_log_std", f"{slug}_log_iqr"]


def base_feature_names() -> list[str]:
    """Names of the 39 per-epoch features, before temporal context."""
    n_eeg = len(config.EEG_CHANNELS)
    n_eog = len(config.EOG_CHANNELS)
    slugs = config.CHANNEL_SLUGS

    names: list[str] = []
    for slug in slugs[:n_eeg]:
        names.extend(_eeg_feature_names(slug))
    for slug in slugs[n_eeg : n_eeg + n_eog]:
        names.extend(_eog_feature_names(slug))
    for slug in slugs[n_eeg + n_eog :]:
        names.extend(_emg_feature_names(slug))
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
    """39 features per epoch, laid out channel-major.

    Parameters
    ----------
    data
        ``(n_epochs, n_channels, n_times)`` epoched signal, with channels in
        :data:`config.CHANNELS` order.  The EMG channel is expected to have
        bypassed the band-pass (see :func:`src.data.load_raw`).
    sfreq
        Sampling frequency in Hz.

    Returns
    -------
    ``(n_epochs, 39)`` in the order given by :func:`base_feature_names`.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 3:
        raise ValueError(f"expected (n_epochs, n_channels, n_times), got shape {data.shape}")

def total_power(freqs: np.ndarray, psd: np.ndarray) -> np.ndarray:
    """Absolute power over :data:`config.TOTAL_BAND`, in signal units squared.

    Unlike the relative powers this keeps amplitude information, which is what
    makes it useful (N3 has far more absolute delta than N1) and also what makes
    it subject-dependent -- hence the per-recording normalisation downstream.
    """
    lo, hi = config.TOTAL_BAND
    return _integrate(freqs, psd, lo, hi)


# --------------------------------------------------------------------------- #
# Non-linear features
# --------------------------------------------------------------------------- #


def permutation_entropy(
    x: np.ndarray, order: int | None = None, delay: int | None = None
) -> np.ndarray:
    """Normalised permutation entropy along the last axis, in ``[0, 1]``.

    Each length-``order`` window is reduced to the *rank order* of its samples;
    the entropy of that pattern distribution measures how irregular the signal
    is.  Because only ranks matter the result is invariant to any monotonic
    rescaling of the signal, including amplitude scaling.

    0 means one ordinal pattern occurs throughout, 1 means all ``order!``
    patterns are equally frequent.
    """
    order = config.PERM_ENTROPY_ORDER if order is None else order
    delay = config.PERM_ENTROPY_DELAY if delay is None else delay
    if order < 2:
        raise ValueError(f"order must be >= 2, got {order}")

    n_windows = x.shape[-1] - (order - 1) * delay
    if n_windows < 2:
        raise ValueError(f"need >= 2 windows, got {n_windows} for order={order}, delay={delay}")

    # Ranks of each window, encoded as one integer in base `order`.
    embedded = np.stack([x[..., i * delay : i * delay + n_windows] for i in range(order)], axis=-1)
    ranks = np.argsort(embedded, axis=-1, kind="stable")
    codes = np.zeros(ranks.shape[:-1], dtype=np.int64)
    for i in range(order):
        codes = codes * order + ranks[..., i]

    # One histogram per leading index, via offset bincount.
    n_codes = order**order
    flat = codes.reshape(-1, n_windows)
    offsets = np.arange(flat.shape[0], dtype=np.int64)[:, None] * n_codes
    counts = np.bincount(
        (flat + offsets).ravel(), minlength=flat.shape[0] * n_codes
    ).reshape(flat.shape[0], n_codes)

    p = counts / n_windows
    terms = np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0)
    entropy = -terms.sum(axis=-1) / math.log(math.factorial(order))
    return entropy.reshape(x.shape[:-1])


def higuchi_fd(x: np.ndarray, kmax: int | None = None) -> np.ndarray:
    """Higuchi fractal dimension along the last axis.

    Curve length is measured at a range of sampling intervals ``k``; the fractal
    dimension is the slope of ``ln L(k)`` against ``ln(1/k)``.  Scaling the
    signal multiplies every ``L(k)`` by the same factor, which shifts the
    intercept and leaves the slope -- so this feature is amplitude-invariant.

    Roughly 1.0 for a straight line and approaching 2.0 for white noise.  A
    constant signal has zero curve length at every ``k`` and returns 1.0.

    The result is clipped to ``[1, 2]``, the range a curve's fractal dimension
    can occupy.  The clip is a guard, not a routine correction: it only binds on
    near-periodic input, where decimating by some ``k`` aliases the signal to a
    constant, ``L(k)`` collapses towards zero and the fitted slope diverges.  A
    pure 10 Hz tone sampled at 100 Hz does this at ``k = 10``; broadband EEG does
    not.
    """
    kmax = config.HIGUCHI_KMAX if kmax is None else kmax
    n = x.shape[-1]
    if kmax < 2 or kmax >= n:
        raise ValueError(f"need 2 <= kmax < n_times, got kmax={kmax}, n_times={n}")

    lengths = np.empty(x.shape[:-1] + (kmax,), dtype=float)
    for k in range(1, kmax + 1):
        total = np.zeros(x.shape[:-1], dtype=float)
        for m in range(k):
            idx = np.arange(m, n, k)
            n_intervals = idx.size - 1
            if n_intervals < 1:
                continue
            # Normalisation from Higuchi (1988): rescale the sub-series to the
            # full length, then divide by the interval k.
            scale = (n - 1) / (n_intervals * k) / k
            total += np.abs(np.diff(x[..., idx], axis=-1)).sum(axis=-1) * scale
        lengths[..., k - 1] = total / k

    ks = np.arange(1, kmax + 1, dtype=float)
    log_lengths = np.log(np.maximum(lengths, _LOG_FLOOR)).reshape(-1, kmax).T
    slope = np.polyfit(np.log(1.0 / ks), log_lengths, 1)[0].reshape(x.shape[:-1])

    # A flat signal has L(k) == 0 at every k; report the straight-line value.
    degenerate = lengths.max(axis=-1) <= _LOG_FLOOR
    return np.where(degenerate, 1.0, np.clip(slope, 1.0, 2.0))


def petrosian_fd(x: np.ndarray) -> np.ndarray:
    """Petrosian fractal dimension along the last axis.

    Built from the number of sign changes in the first difference, so it depends
    only on the shape of the signal and not on its amplitude.  A constant or
    monotonic signal has no sign changes and returns exactly 1.0.
    """
    n = x.shape[-1]
    if n < 3:
        raise ValueError(f"need >= 3 samples, got {n}")

    diff = np.diff(x, axis=-1)
    n_delta = (np.diff(np.sign(diff), axis=-1) != 0).sum(axis=-1)

    log_n = math.log10(n)
    return log_n / (log_n + np.log10(n / (n + 0.4 * n_delta)))


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #


    freqs, psd = compute_psd(data, sfreq)          # (n_epochs, n_channels, n_freqs)
    powers = relative_band_powers(freqs, psd)      # (n_epochs, n_channels, n_bands)
    entropy = spectral_entropy(freqs, psd)         # (n_epochs, n_channels)

    per_channel = np.concatenate([powers, entropy[..., np.newaxis]], axis=-1)
    n_epochs, n_channels, n_feat = per_channel.shape
    return per_channel.reshape(n_epochs, n_channels * n_feat)
    if data.shape[1] != len(config.CHANNELS):
        raise ValueError(
            f"expected {len(config.CHANNELS)} channels {config.CHANNELS}, "
            f"got {data.shape[1]}"
        )

    n_eeg = len(config.EEG_CHANNELS)
    n_spectral = len(config.SPECTRAL_CHANNELS)

    # The 1 Hz EMG is excluded here: it has no meaningful spectrum.
    freqs, psd = compute_psd(data[:, :n_spectral], sfreq)
    rel = relative_band_powers(freqs, psd)          # (n_epochs, n_spectral, n_bands)
    abspow = total_power(freqs, psd)                # (n_epochs, n_spectral)
    spec_ent = spectral_entropy(freqs, psd)         # (n_epochs, n_spectral)
    perm_ent = permutation_entropy(data[:, :n_spectral])

    band_index = {band: i for i, (band, _, _) in enumerate(config.BANDS)}

    blocks = []
    for ch in range(n_eeg):
        ratios = [
            _log(rel[:, ch, band_index[num]]) - _log(rel[:, ch, band_index[den]])
            for num, den in config.BAND_RATIOS
        ]
        blocks.append(
            np.column_stack(
                [
                    rel[:, ch, :],
                    spec_ent[:, ch],
                    _log(abspow[:, ch]),
                    perm_ent[:, ch],
                    higuchi_fd(data[:, ch]),
                    petrosian_fd(data[:, ch]),
                    *ratios,
                ]
            )
        )

    for ch in range(n_eeg, n_spectral):
        blocks.append(
            np.column_stack(
                [
                    rel[:, ch, :],
                    spec_ent[:, ch],
                    _log(abspow[:, ch]),
                    perm_ent[:, ch],
                    _log(data[:, ch].std(axis=-1)),
                ]
            )
        )

    for ch in range(n_spectral, data.shape[1]):
        signal = data[:, ch]
        blocks.append(
            np.column_stack(
                [_log(signal.std(axis=-1)), _log(_iqr(signal, axis=-1))]
            )
        )

    return np.concatenate(blocks, axis=1)

