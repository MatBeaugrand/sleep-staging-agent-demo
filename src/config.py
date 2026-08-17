"""Project-wide constants and path resolution.

All paths are derived from the repository root, which is located relative to
this file.  Nothing here is absolute or machine-specific; the data cache can be
redirected with the SLEEP_DATA_DIR environment variable.
"""

from __future__ import annotations

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

CHANNELS = ["EEG Fpz-Cz", "EEG Pz-Oz"]

#: Short, filesystem- and column-safe names for the channels above.
CHANNEL_SLUGS = ["fpz_cz", "pz_oz"]

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

# --------------------------------------------------------------------------- #
# Cross-validation
# --------------------------------------------------------------------------- #

N_SPLITS = 4
