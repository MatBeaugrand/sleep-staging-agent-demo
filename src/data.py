"""Download, preprocess and epoch the Sleep-EDF Expanded recordings.

The public surface is :func:`build_dataset`, which turns a set of subjects into
a feature matrix.  The lower-level helpers are deliberately split so that the
parts carrying methodological risk -- the stage mapping and the sleep-period
crop -- are pure functions that can be tested without downloading anything.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np

from . import config
from .features import (
    add_temporal_context,
    extract_features,
    feature_names,
    normalise_per_recording,
)

logger = logging.getLogger(__name__)

#: MNE dislikes an event code of 0, so integer labels are stored offset by one
#: in the events array and shifted back immediately afterwards.
_EVENT_OFFSET = 1

_SUBJECT_RE = re.compile(r"SC4(\d{2})(\d)", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #


def parse_subject_night(path: str | Path) -> tuple[int, int]:
    """Extract ``(subject, night)`` from a Sleep-EDF cassette filename.

    Cassette files are named ``SC4ssNEO-PSG.edf`` where ``ss`` is the subject
    number and ``N`` the night.
    """
    match = _SUBJECT_RE.search(Path(path).name)
    if match is None:
        raise ValueError(f"Not a sleep-cassette filename: {path}")
    return int(match.group(1)), int(match.group(2))


def map_stage(description: str) -> str | None:
    """Map a raw hypnogram annotation to one of the five classes.

    Returns ``None`` for annotations that carry no stage information
    (``"Sleep stage ?"``, ``"Movement time"``); those epochs are dropped rather
    than folded into a class.
    """
    return config.STAGE_MAP.get(description)


def sleep_period_mask(
    labels: np.ndarray,
    onsets_sec: np.ndarray,
    margin_sec: float,
    wake_label: int = config.STAGE_TO_INT["W"],
) -> np.ndarray:
    """Boolean mask keeping the sleep period plus ``margin_sec`` on each side.

    The sleep period runs from the first to the last non-wake epoch.  If the
    recording contains no sleep at all, everything is kept (there is nothing
    sensible to centre the window on).
    """
    labels = np.asarray(labels)
    onsets_sec = np.asarray(onsets_sec, dtype=float)
    if labels.shape != onsets_sec.shape:
        raise ValueError("labels and onsets_sec must have the same shape")

    asleep = np.flatnonzero(labels != wake_label)
    if asleep.size == 0:
        return np.ones(labels.shape, dtype=bool)

    start = onsets_sec[asleep[0]] - margin_sec
    stop = onsets_sec[asleep[-1]] + margin_sec
    return (onsets_sec >= start) & (onsets_sec <= stop)


# --------------------------------------------------------------------------- #
# Fetching
# --------------------------------------------------------------------------- #


def fetch_recordings(n_subjects: int = 8, night: int = 1) -> list[tuple[int, Path, Path]]:
    """Download (or reuse the cache for) ``n_subjects`` sleep-cassette records.

    Returns ``(subject_id, psg_path, hypnogram_path)`` sorted by subject.
    """
    config.ensure_dirs()
    paths = mne.datasets.sleep_physionet.age.fetch_data(
        subjects=list(range(n_subjects)),
        recording=[night],
        path=str(config.raw_dir()),
        on_missing="raise",
        verbose="error",
    )

    recordings = []
    for psg, hypno in paths:
        subject, _ = parse_subject_night(psg)
        recordings.append((subject, Path(psg), Path(hypno)))
    return sorted(recordings, key=lambda r: r[0])


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SubjectEpochs:
    """Epoched, labelled data for a single recording."""

    subject: int
    data: np.ndarray          # (n_epochs, n_channels, n_times)
    labels: np.ndarray        # (n_epochs,) int, indexes config.STAGE_NAMES
    onsets_sec: np.ndarray    # (n_epochs,) epoch start, seconds from record start
    sfreq: float

    def __post_init__(self) -> None:
        n = self.data.shape[0]
        if not (self.labels.shape[0] == self.onsets_sec.shape[0] == n):
            raise ValueError(
                "inconsistent epoch counts: "
                f"data={n}, labels={self.labels.shape[0]}, onsets={self.onsets_sec.shape[0]}"
            )


def load_raw(psg_path: Path, hypno_path: Path) -> mne.io.BaseRaw:
    """Read one recording, keep the four channels of interest, filter selectively.

    The band-pass is applied to the EEG and EOG channels only.  The submental
    EMG is recorded at 1 Hz, so almost all of its content lies below the 0.3 Hz
    high-pass edge: filtering it removes about 90 % of its variance and leaves
    little but interpolation artefact.  It is therefore left unfiltered, and
    :func:`src.features.extract_features` takes only time-domain amplitude
    features from it.

    Filtering happens on the full continuous recording, before any cropping, so
    the filter's edge artefacts sit in the discarded wake tails rather than at
    the boundary of the sleep period.
    """
    raw = mne.io.read_raw_edf(psg_path, preload=False, verbose="error")
    missing = [ch for ch in config.CHANNELS if ch not in raw.ch_names]
    if missing:
        raise ValueError(f"{psg_path.name} is missing channels {missing}")

    raw.pick(config.CHANNELS)
    raw.load_data(verbose="error")
    raw.set_annotations(mne.read_annotations(hypno_path), emit_warning=False, verbose="error")
    raw.filter(
        l_freq=config.L_FREQ,
        h_freq=config.H_FREQ,
        picks=config.SPECTRAL_CHANNELS,
        fir_design="firwin",
        verbose="error",
    )
    return raw


def epoch_raw(raw: mne.io.BaseRaw, subject: int, crop: bool = True) -> SubjectEpochs:
    """Cut ``raw`` into annotation-aligned 30 s epochs with integer stage labels.

    Epoch boundaries come from the hypnogram itself (``chunk_duration=30 s`` on
    the annotations), so an epoch never straddles two scored stages.
    """
    sfreq = float(raw.info["sfreq"])
    n_samples = int(round(config.EPOCH_SEC * sfreq))

    desc_to_id = {
        desc: config.STAGE_TO_INT[stage] + _EVENT_OFFSET
        for desc, stage in config.STAGE_MAP.items()
    }
    events, _ = mne.events_from_annotations(
        raw, event_id=desc_to_id, chunk_duration=config.EPOCH_SEC, verbose="error"
    )
    if events.size == 0:
        raise ValueError(f"subject {subject}: no scored epochs found")

    # The PSG signal can end before the hypnogram does; drop events whose full
    # 30 s window would run past the recording instead of letting MNE silently
    # discard them later.
    last_start = raw.first_samp + raw.n_times - n_samples
    events = events[events[:, 0] <= last_start]

    name_to_id = {
        name: config.STAGE_TO_INT[name] + _EVENT_OFFSET
        for name in config.STAGE_NAMES
        if config.STAGE_TO_INT[name] + _EVENT_OFFSET in set(events[:, 2])
    }
    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=name_to_id,
        tmin=0.0,
        tmax=config.EPOCH_SEC - 1.0 / sfreq,
        baseline=None,
        preload=True,
        on_missing="ignore",
        verbose="error",
    )

    labels = epochs.events[:, 2] - _EVENT_OFFSET
    onsets = (epochs.events[:, 0] - raw.first_samp) / sfreq
    data = epochs.get_data(copy=False)

    if crop:
        keep = sleep_period_mask(labels, onsets, config.CROP_MARGIN_MIN * 60.0)
        data, labels, onsets = data[keep], labels[keep], onsets[keep]

    return SubjectEpochs(
        subject=subject, data=data, labels=labels, onsets_sec=onsets, sfreq=sfreq
    )


# --------------------------------------------------------------------------- #
# Dataset assembly
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Dataset:
    """Feature matrix plus everything needed for grouped validation and plots."""

    X: np.ndarray            # (n_epochs, n_features)
    y: np.ndarray            # (n_epochs,) int labels
    groups: np.ndarray       # (n_epochs,) subject id -- the CV grouping key
    onsets_sec: np.ndarray   # (n_epochs,) epoch start within its own recording
    feature_names: list[str]

    def __post_init__(self) -> None:
        n = self.X.shape[0]
        if not (len(self.y) == len(self.groups) == len(self.onsets_sec) == n):
            raise ValueError("Dataset arrays disagree on the number of epochs")
        if self.X.shape[1] != len(self.feature_names):
            raise ValueError("feature_names does not match the number of columns")

    @property
    def subjects(self) -> np.ndarray:
        return np.unique(self.groups)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            X=self.X,
            y=self.y,
            groups=self.groups,
            onsets_sec=self.onsets_sec,
            feature_names=np.array(self.feature_names, dtype=object),
        )

    @classmethod
    def load(cls, path: Path) -> "Dataset":
        with np.load(path, allow_pickle=True) as f:
            return cls(
                X=f["X"],
                y=f["y"],
                groups=f["groups"],
                onsets_sec=f["onsets_sec"],
                feature_names=[str(n) for n in f["feature_names"]],
            )


def prepare_recording(X_raw: np.ndarray, onsets_sec: np.ndarray) -> np.ndarray:
    """Add temporal context and normalise, for **one** recording.

    Kept as its own function because both steps are only correct within a single
    night: the smoothers walk that recording's epoch lattice, and the robust
    z-score uses that recording's own percentiles.  Calling this on several
    concatenated nights would average one subject's epochs into another's and
    normalise every subject against a pooled distribution.
    """
    return normalise_per_recording(add_temporal_context(X_raw, onsets_sec))


def assemble_dataset(recordings: Iterable[SubjectEpochs]) -> Dataset:
    """Turn per-recording epochs into one feature matrix.

    This function exists separately from :func:`build_dataset` so that the step
    carrying the leakage risk is reachable from the test suite without
    downloading anything.  ``build_dataset`` can only run against real EDF files,
    so anything expressed only there is, in practice, unguarded.

    ``recordings`` is consumed lazily: only one recording's signal is held in
    memory at a time, and only its 39 per-epoch features are retained.

    The two per-recording steps below are spelled out rather than folded into
    :func:`prepare_recording` so that each is independently guarded by its own
    mutation (``smooth_pooled``, ``norm_pooled_recordings``).  Hoisting either
    one onto the concatenated matrix would smooth one subject's epochs into the
    next and normalise every subject against a pooled distribution.
    """
    blocks_raw, blocks_y, blocks_g, blocks_t = [], [], [], []
    for epochs in recordings:
        X_raw = extract_features(epochs.data, sfreq=epochs.sfreq)
        logger.info(
            "subject %02d: %d epochs, %d per-epoch features",
            epochs.subject,
            X_raw.shape[0],
            X_raw.shape[1],
        )
        blocks_raw.append(X_raw)
        blocks_y.append(epochs.labels)
        blocks_g.append(np.full(X_raw.shape[0], epochs.subject, dtype=int))
        blocks_t.append(epochs.onsets_sec)

    if not blocks_raw:
        raise ValueError("no recordings supplied")

    smoothed = [add_temporal_context(X, t) for X, t in zip(blocks_raw, blocks_t)]
    prepared = [normalise_per_recording(X) for X in smoothed]

    return Dataset(
        X=np.concatenate(prepared),
        y=np.concatenate(blocks_y),
        groups=np.concatenate(blocks_g),
        onsets_sec=np.concatenate(blocks_t),
        feature_names=feature_names(),
    )


def build_dataset(n_subjects: int = 8, night: int = 1, crop: bool = True) -> Dataset:
    """Full path from PhysioNet download to feature matrix."""
    recordings = fetch_recordings(n_subjects=n_subjects, night=night)

    def load_each() -> Iterator[SubjectEpochs]:
        for subject, psg, hypno in recordings:
            logger.info("subject %02d: loading %s", subject, psg.name)
            raw = load_raw(psg, hypno)
            yield epoch_raw(raw, subject=subject, crop=crop)

    return assemble_dataset(load_each())


def load_or_build(
    cache: Path | None = None, n_subjects: int = 8, night: int = 1, crop: bool = True
) -> Dataset:
    """Return the cached feature matrix, computing and caching it if absent."""
    if cache is None:
        suffix = "" if crop else "-full"
        cache = config.derivatives_dir() / f"features-{n_subjects}subj-night{night}{suffix}.npz"

    if cache.exists():
        logger.info("loading cached features from %s", cache)
        return Dataset.load(cache)

    dataset = build_dataset(n_subjects=n_subjects, night=night, crop=crop)
    dataset.save(cache)
    logger.info("cached features to %s", cache)
    return dataset
