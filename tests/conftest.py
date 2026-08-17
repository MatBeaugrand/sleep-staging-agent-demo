"""Synthetic fixtures.

The whole test suite runs without touching the network or the Sleep-EDF
download: every fixture here is generated from a seeded RNG.
"""

from __future__ import annotations

import mne
import numpy as np
import pytest

from src import config
from src.data import Dataset
from src.features import extract_features, feature_names

SFREQ = 100.0
N_TIMES = int(config.EPOCH_SEC * SFREQ)

#: A dominant frequency per class, chosen so each class sits in a different
#: band.  This makes the synthetic data separable enough for the end-to-end
#: cross-validation tests without any real EEG.
CLASS_FREQ = {"W": 20.0, "REM": 6.0, "N1": 5.0, "N2": 13.5, "N3": 1.5}


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(config.SEED)


def make_epochs(
    labels: list[str], rng: np.random.Generator, sfreq: float = SFREQ, n_times: int = N_TIMES
) -> np.ndarray:
    """(n_epochs, 2, n_times) signal whose spectrum depends on the label."""
    t = np.arange(n_times) / sfreq
    data = np.empty((len(labels), len(config.CHANNELS), n_times))
    for i, label in enumerate(labels):
        freq = CLASS_FREQ[label]
        for ch in range(len(config.CHANNELS)):
            phase = rng.uniform(0, 2 * np.pi)
            data[i, ch] = np.sin(2 * np.pi * freq * t + phase) + 0.3 * rng.standard_normal(n_times)
    return data


@pytest.fixture
def epoch_data(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """A small labelled stack of synthetic epochs."""
    labels = [s for s in config.STAGE_NAMES for _ in range(4)]
    data = make_epochs(labels, rng)
    y = np.array([config.STAGE_TO_INT[s] for s in labels])
    return data, y


@pytest.fixture
def synthetic_dataset(rng: np.random.Generator) -> Dataset:
    """Six subjects, unequal epoch counts, realistic-ish class imbalance."""
    proportions = {"W": 0.20, "REM": 0.18, "N1": 0.05, "N2": 0.40, "N3": 0.17}

    X_blocks, y_blocks, g_blocks, t_blocks = [], [], [], []
    for subject, n_epochs in zip(range(6), [60, 72, 68, 80, 64, 76]):
        labels = []
        for stage, share in proportions.items():
            labels.extend([stage] * max(1, round(share * n_epochs)))
        rng.shuffle(labels)

        data = make_epochs(labels, rng)
        X_blocks.append(extract_features(data, sfreq=SFREQ))
        y_blocks.append(np.array([config.STAGE_TO_INT[s] for s in labels]))
        g_blocks.append(np.full(len(labels), subject, dtype=int))
        t_blocks.append(np.arange(len(labels), dtype=float) * config.EPOCH_SEC)

    return Dataset(
        X=np.concatenate(X_blocks),
        y=np.concatenate(y_blocks),
        groups=np.concatenate(g_blocks),
        onsets_sec=np.concatenate(t_blocks),
        feature_names=feature_names(),
    )


@pytest.fixture
def annotated_raw(rng: np.random.Generator) -> tuple[mne.io.RawArray, list[str]]:
    """A short synthetic recording with a hypnogram-shaped annotation set.

    The stage sequence includes both an unscorable epoch and a movement-time
    epoch, which the pipeline must drop rather than classify.
    """
    descriptions = (
        ["Sleep stage W"] * 4
        + ["Sleep stage 1"] * 2
        + ["Sleep stage 2"] * 6
        + ["Sleep stage 3"] * 2
        + ["Sleep stage 4"] * 2
        + ["Sleep stage R"] * 3
        + ["Movement time"]
        + ["Sleep stage 2"] * 3
        + ["Sleep stage ?"]
        + ["Sleep stage W"] * 5
    )
    n_epochs = len(descriptions)
    n_times = int(n_epochs * config.EPOCH_SEC * SFREQ)

    info = mne.create_info(list(config.CHANNELS), sfreq=SFREQ, ch_types="eeg")
    # Volt-scale amplitudes so MNE's internal unit handling stays realistic.
    raw = mne.io.RawArray(rng.standard_normal((len(config.CHANNELS), n_times)) * 2e-5, info,
                          verbose="error")
    raw.set_annotations(
        mne.Annotations(
            onset=np.arange(n_epochs) * config.EPOCH_SEC,
            duration=np.full(n_epochs, config.EPOCH_SEC),
            description=descriptions,
        ),
        verbose="error",
    )
    return raw, descriptions
