"""Project-wide constants and path resolution.

All paths are derived from the repository root, which is located relative to
this file.  Nothing here is absolute or machine-specific; the data cache can be
redirected with the SLEEP_DATA_DIR environment variable.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #

SEED = 42

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def data_dir() -> Path:
    """Root of the data cache (raw downloads + derivatives)."""
    override = os.environ.get("SLEEP_DATA_DIR")
    return Path(override).expanduser().resolve() if override else PROJECT_ROOT / "data"


def raw_dir() -> Path:
    """Where MNE stores the downloaded PhysioNet files."""
    return data_dir() / "raw"


def derivatives_dir() -> Path:
    """Where the cached feature matrix lives."""
    return data_dir() / "derivatives"


def figures_dir() -> Path:
    return PROJECT_ROOT / "figures"


def ensure_dirs() -> None:
    for path in (raw_dir(), derivatives_dir(), figures_dir()):
        path.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Recording / preprocessing
# --------------------------------------------------------------------------- #

#: Channels are grouped by modality because they get different feature sets and
#: different filtering.  ``CHANNELS`` is the concatenation, and its order defines
#: the channel axis of the epoch array everywhere downstream.
EEG_CHANNELS = ["EEG Fpz-Cz", "EEG Pz-Oz"]
EOG_CHANNELS = ["EOG horizontal"]

#: Sleep-EDF's submental EMG is recorded at 1 Hz (MNE interpolates it up to the
#: file's 100 Hz).  Its Nyquist limit is 0.5 Hz, so every band in ``BANDS`` would
#: be pure interpolation artefact: it gets time-domain amplitude features only,
#: and it bypasses the band-pass entirely (see ``EOG_CHANNELS`` note in data.py).
EMG_CHANNELS = ["EMG submental"]

CHANNELS = EEG_CHANNELS + EOG_CHANNELS + EMG_CHANNELS

#: Short, filesystem- and column-safe names for the channels above, same order.
CHANNEL_SLUGS = ["fpz_cz", "pz_oz", "eog", "emg"]

#: Channels whose spectrum is meaningful, i.e. everything but the 1 Hz EMG.
#: These are the leading channels of the epoch array, which lets the feature code
#: slice rather than search.
SPECTRAL_CHANNELS = EEG_CHANNELS + EOG_CHANNELS

L_FREQ = 0.3   # Hz, band-pass low edge
H_FREQ = 35.0  # Hz, band-pass high edge

EPOCH_SEC = 30.0

#: Minutes of wake retained on either side of the sleep period.  Sleep-EDF
#: cassette recordings span ~20 h, the large majority of which is wake with the
#: subject out of bed; keeping all of it turns the task into wake detection.
CROP_MARGIN_MIN = 30.0

# --------------------------------------------------------------------------- #
# Sleep stages
# --------------------------------------------------------------------------- #

#: Raw hypnogram annotation description -> 5-class label.
#: Stages 3 and 4 are merged into N3 following the AASM re-scoring of the older
#: R&K annotations used by Sleep-EDF.  "Sleep stage ?" and "Movement time" are
#: deliberately absent: those epochs are dropped rather than assigned a class.
STAGE_MAP = {
    "Sleep stage W": "W",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N3",
    "Sleep stage R": "REM",
}

#: Canonical class order, used for label encoding, the confusion matrix axes and
#: the hypnogram y-axis.  Ordered as a hypnogram is conventionally drawn.
STAGE_NAMES = ["W", "REM", "N1", "N2", "N3"]

STAGE_TO_INT = {name: i for i, name in enumerate(STAGE_NAMES)}

# --------------------------------------------------------------------------- #
# Spectral features
# --------------------------------------------------------------------------- #

#: (name, f_low, f_high) in Hz.  The bands tile [0.5, 30] contiguously, so the
#: five relative powers of a channel sum to exactly 1 (see README).
BANDS = [
    ("delta", 0.5, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 12.0),
    ("sigma", 12.0, 16.0),
    ("beta", 16.0, 30.0),
]

#: Frequency support used both as the relative-power denominator and as the
#: domain of the spectral entropy.
TOTAL_BAND = (0.5, 30.0)

#: Welch segment length in seconds.  4 s gives a 0.25 Hz grid, so every band
#: edge above falls exactly on a bin boundary, and ~14 half-overlapping
#: segments are averaged within a 30 s epoch.
WELCH_SEG_SEC = 4.0
WELCH_OVERLAP = 0.5

#: (numerator, denominator) band-power ratios, stored as natural logs.  Ratios
#: of powers are scale-free, so these carry no amplitude information.
BAND_RATIOS = [
    ("delta", "beta"),
    ("delta", "theta"),
    ("theta", "alpha"),
    ("alpha", "sigma"),
]

# --------------------------------------------------------------------------- #
# Non-linear features
# --------------------------------------------------------------------------- #

PERM_ENTROPY_ORDER = 3   # embedding dimension; 3! = 6 ordinal patterns
PERM_ENTROPY_DELAY = 1   # in samples
HIGUCHI_KMAX = 10        # largest sub-series interval

# --------------------------------------------------------------------------- #
# Temporal context
# --------------------------------------------------------------------------- #

#: Every per-epoch feature is additionally smoothed two ways, following Vallat &
#: Walker (2021).  7.5 min / 30 s = 15 epochs exactly; 2 min / 30 s = 4 epochs.
SMOOTH_CENTRED_MIN = 7.5   # centred, triangular-weighted
SMOOTH_TRAILING_MIN = 2.0  # trailing, uniform, past epochs and the current one

# --------------------------------------------------------------------------- #
# Per-recording normalisation
# --------------------------------------------------------------------------- #

#: Robust z-score: (x - median) / (p_hi - p_lo), computed per recording so that
#: a night is normalised only against itself and never against other subjects.
NORM_PERCENTILES = (5.0, 95.0)

#: If a column's 5-95 spread is below this, centre it but do not divide.
#: Dividing a near-constant column by a near-zero spread amplifies pure noise.
NORM_SPREAD_FLOOR = 1e-12

# --------------------------------------------------------------------------- #
# Cross-validation
# --------------------------------------------------------------------------- #

N_SPLITS = 4


# --------------------------------------------------------------------------- #
# Feature-cache coherence
# --------------------------------------------------------------------------- #


def feature_spec() -> dict:
    """Everything that changes the meaning of a cached feature matrix.

    Read at call time, so monkeypatching a constant in a test is reflected here.
    """
    return {
        "channels": list(CHANNELS),
        "channel_slugs": list(CHANNEL_SLUGS),
        "bands": [list(b) for b in BANDS],
        "band_ratios": [list(r) for r in BAND_RATIOS],
        "total_band": list(TOTAL_BAND),
        "welch_seg_sec": WELCH_SEG_SEC,
        "welch_overlap": WELCH_OVERLAP,
        "perm_entropy": [PERM_ENTROPY_ORDER, PERM_ENTROPY_DELAY],
        "higuchi_kmax": HIGUCHI_KMAX,
        "smooth_centred_min": SMOOTH_CENTRED_MIN,
        "smooth_trailing_min": SMOOTH_TRAILING_MIN,
        "norm_percentiles": list(NORM_PERCENTILES),
        "norm_spread_floor": NORM_SPREAD_FLOOR,
        "epoch_sec": EPOCH_SEC,
        "l_freq": L_FREQ,
        "h_freq": H_FREQ,
        "crop_margin_min": CROP_MARGIN_MIN,
        "stage_map": dict(STAGE_MAP),
        "stage_names": list(STAGE_NAMES),
    }


def feature_fingerprint() -> str:
    """Short digest of :func:`feature_spec`.

    Goes into the cache filename *and* inside the ``.npz``, so a feature matrix
    computed under a different specification can never be silently reused.
    """
    payload = json.dumps(feature_spec(), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:10]
