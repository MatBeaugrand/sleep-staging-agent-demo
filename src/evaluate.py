"""Subject-wise cross-validation, metrics and figures.

Every number and every figure in this module comes from *out-of-fold*
predictions: each epoch is predicted by a model that never saw any epoch from
that epoch's subject.  Epochs are never split randomly -- consecutive epochs
from one night are near-duplicates, so a random split leaks the answer and
inflates every metric.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupKFold

from . import config
from .data import Dataset
from .model import MODEL_LABELS, build_model

logger = logging.getLogger(__name__)

_LABELS = list(range(len(config.STAGE_NAMES)))


# --------------------------------------------------------------------------- #
# Cross-validation
# --------------------------------------------------------------------------- #


def iter_group_splits(
    n_samples: int, groups: np.ndarray, n_splits: int = config.N_SPLITS
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(train_idx, test_idx)`` from a subject-grouped K-fold.

    ``GroupKFold`` is deterministic (it does not shuffle), so the folds are
    reproducible without depending on the random seed.  Group disjointness is
    re-checked here rather than assumed, so a future change to the splitter
    cannot silently introduce leakage.
    """
    groups = np.asarray(groups)
    n_groups = len(np.unique(groups))
    if n_splits > n_groups:
        raise ValueError(f"n_splits={n_splits} exceeds the number of subjects ({n_groups})")

    cv = GroupKFold(n_splits=n_splits)
    for train_idx, test_idx in cv.split(np.zeros(n_samples), groups=groups):
        overlap = set(groups[train_idx]) & set(groups[test_idx])
        if overlap:
            raise AssertionError(f"subject(s) {sorted(overlap)} appear in both train and test")
        yield train_idx, test_idx


def out_of_fold_predictions(
    dataset: Dataset,
    model_factory: Callable[[], BaseEstimator],
    n_splits: int = config.N_SPLITS,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict every epoch from a model trained without that epoch's subject.

    Returns ``(y_pred, fold_id)``, both aligned with ``dataset.y``.
    """
    y_pred = np.full(dataset.y.shape, -1, dtype=int)
    fold_id = np.full(dataset.y.shape, -1, dtype=int)

    for k, (train_idx, test_idx) in enumerate(
        iter_group_splits(len(dataset.y), dataset.groups, n_splits)
    ):
        estimator = model_factory()
        estimator.fit(dataset.X[train_idx], dataset.y[train_idx])
        y_pred[test_idx] = estimator.predict(dataset.X[test_idx])
        fold_id[test_idx] = k
        logger.info(
            "fold %d/%d: trained on %d epochs from subjects %s, tested on %s",
            k + 1,
            n_splits,
            len(train_idx),
            sorted(set(dataset.groups[train_idx])),
            sorted(set(dataset.groups[test_idx])),
        )

    if (y_pred < 0).any():
        raise RuntimeError("some epochs were never in a test fold")
    return y_pred, fold_id


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


@dataclass
class Report:
    model: str
    n_epochs: int
    n_subjects: int
    n_splits: int
    accuracy: float
    macro_f1: float
    kappa: float
    per_class_f1: dict[str, float]
    support: dict[str, int]
    fold_macro_f1: list[float] = field(default_factory=list)
    fold_kappa: list[float] = field(default_factory=list)
    #: Out-of-fold kappa computed within each held-out subject, keyed by subject
    #: id as a string so the report is JSON-serialisable.  The median of these is
    #: the quantity Vallat & Walker (2021) report, so it is what to compare
    #: against their 0.80 -- not the pooled kappa above.
    subject_kappa: dict[str, float] = field(default_factory=dict)
    confusion: list[list[int]] = field(default_factory=list)

    @property
    def median_subject_kappa(self) -> float:
        return float(np.median(list(self.subject_kappa.values())))

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["fold_macro_f1_mean"] = float(np.mean(self.fold_macro_f1))
        d["fold_macro_f1_std"] = float(np.std(self.fold_macro_f1))
        d["fold_kappa_mean"] = float(np.mean(self.fold_kappa))
        d["fold_kappa_std"] = float(np.std(self.fold_kappa))
        d["median_subject_kappa"] = self.median_subject_kappa
        return d

    def summary(self) -> str:
        subject_kappas = np.array(list(self.subject_kappa.values()))
        lines = [
            f"{MODEL_LABELS.get(self.model, self.model)}"
            f"  ({self.n_epochs} epochs, {self.n_subjects} subjects,"
            f" {self.n_splits}-fold grouped CV)",
            f"  accuracy        {self.accuracy:.3f}",
            f"  macro F1        {self.macro_f1:.3f}"
            f"   (per fold {np.mean(self.fold_macro_f1):.3f}"
            f" +/- {np.std(self.fold_macro_f1):.3f})",
            f"  Cohen's kappa   {self.kappa:.3f}"
            f"   (per fold {np.mean(self.fold_kappa):.3f}"
            f" +/- {np.std(self.fold_kappa):.3f})",
            f"  median per-subject kappa   {self.median_subject_kappa:.3f}"
            f"   (range {subject_kappas.min():.3f}-{subject_kappas.max():.3f})",
            "  per-class F1:",
        ]
        for stage in config.STAGE_NAMES:
            lines.append(
                f"    {stage:<4} {self.per_class_f1[stage]:.3f}"
                f"   (n = {self.support[stage]})"
            )
        return "\n".join(lines)


def compute_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fold_id: np.ndarray,
    groups: np.ndarray,
    model: str,
    n_splits: int,
) -> Report:
    per_class = f1_score(y_true, y_pred, labels=_LABELS, average=None, zero_division=0)
    counts = np.bincount(y_true, minlength=len(config.STAGE_NAMES))

    fold_f1, fold_kappa = [], []
    for k in np.unique(fold_id):
        m = fold_id == k
        fold_f1.append(
            float(f1_score(y_true[m], y_pred[m], labels=_LABELS, average="macro", zero_division=0))
        )
        fold_kappa.append(float(cohen_kappa_score(y_true[m], y_pred[m], labels=_LABELS)))

    subject_kappa = {
        str(int(subject)): float(
            cohen_kappa_score(y_true[groups == subject], y_pred[groups == subject], labels=_LABELS)
        )
        for subject in np.unique(groups)
    }

    return Report(
        model=model,
        n_epochs=int(len(y_true)),
        n_subjects=int(len(np.unique(groups))),
        n_splits=n_splits,
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(
            f1_score(y_true, y_pred, labels=_LABELS, average="macro", zero_division=0)
        ),
        kappa=float(cohen_kappa_score(y_true, y_pred, labels=_LABELS)),
        per_class_f1={s: float(v) for s, v in zip(config.STAGE_NAMES, per_class)},
        support={s: int(c) for s, c in zip(config.STAGE_NAMES, counts)},
        fold_macro_f1=fold_f1,
        fold_kappa=fold_kappa,
        subject_kappa=subject_kappa,
        confusion=confusion_matrix(y_true, y_pred, labels=_LABELS).tolist(),
    )


def normalised_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Row-normalised confusion matrix; row i is the recall profile of class i.

    Rows with no support are left as NaN rather than silently shown as zeros.
    """
    cm = confusion_matrix(y_true, y_pred, labels=_LABELS).astype(float)
    totals = cm.sum(axis=1, keepdims=True)
    return np.divide(cm, totals, out=np.full_like(cm, np.nan), where=totals > 0)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, model: str, path: Path) -> Path:
    cm = normalised_confusion(y_true, y_pred)
    support = np.bincount(y_true, minlength=len(config.STAGE_NAMES))

    fig, ax = plt.subplots(figsize=(6.8, 5.6), constrained_layout=True)
    im = ax.imshow(cm, cmap="Blues", vmin=0.0, vmax=1.0)

    ax.set_xticks(range(len(config.STAGE_NAMES)), config.STAGE_NAMES)
    ax.set_yticks(
        range(len(config.STAGE_NAMES)),
        [f"{s}\n(n={n})" for s, n in zip(config.STAGE_NAMES, support)],
    )
    ax.set_xlabel("Predicted stage")
    ax.set_ylabel("Annotated stage")
    ax.set_title(
        f"{MODEL_LABELS.get(model, model)}\n"
        "row-normalised confusion matrix, out-of-fold",
        fontsize=11,
    )

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if np.isnan(cm[i, j]):
                continue
            ax.text(
                j,
                i,
                f"{cm[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > 0.5 else "black",
                fontsize=10,
            )

    fig.colorbar(im, ax=ax, label="fraction of annotated epochs")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_hypnogram(
    dataset: Dataset, y_pred: np.ndarray, subject: int, model: str, path: Path
) -> Path:
    """Annotated vs predicted hypnogram for one subject, held out in its fold."""
    mask = dataset.groups == subject
    if not mask.any():
        raise ValueError(f"subject {subject} is not in this dataset")

    order = np.argsort(dataset.onsets_sec[mask])
    hours = dataset.onsets_sec[mask][order] / 3600.0
    truth = dataset.y[mask][order]
    pred = y_pred[mask][order]
    agree = truth == pred

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(11, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 3, 0.7]},
        constrained_layout=True,
    )

    for ax, series, title, colour in (
        (axes[0], truth, "Annotated (expert hypnogram)", "#333333"),
        (axes[1], pred, f"Predicted -- {MODEL_LABELS.get(model, model)}", "#1f77b4"),
    ):
        ax.step(hours, series, where="post", linewidth=1.0, color=colour)
        ax.set_yticks(range(len(config.STAGE_NAMES)), config.STAGE_NAMES)
        ax.set_ylim(len(config.STAGE_NAMES) - 0.5, -0.5)  # W at the top
        ax.set_ylabel("Stage")
        ax.set_title(title, fontsize=10, loc="left")
        ax.grid(axis="y", alpha=0.3)

    axes[2].fill_between(
        hours, 0, 1, where=~agree, step="post", color="#d62728", alpha=0.8, linewidth=0
    )
    axes[2].set_ylim(0, 1)
    axes[2].set_yticks([])
    axes[2].set_ylabel("Error", rotation=0, ha="right", va="center", fontsize=9)
    axes[2].set_xlabel("Time from start of recording (h)")
    axes[2].grid(axis="x", alpha=0.3)

    fig.suptitle(
        f"Subject {subject:02d} -- out-of-fold prediction "
        f"(epoch agreement {agree.mean():.1%}, "
        f"kappa {cohen_kappa_score(truth, pred, labels=_LABELS):.2f})",
        fontsize=11,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_subject_kappa(report: Report, path: Path, reference: float | None = 0.80) -> Path:
    """Out-of-fold kappa for each held-out subject, with the median marked.

    This is the figure to read when comparing against a published median
    per-subject kappa; the pooled kappa is a different quantity.
    """
    subjects = sorted(report.subject_kappa, key=lambda s: report.subject_kappa[s])
    values = [report.subject_kappa[s] for s in subjects]
    median = report.median_subject_kappa

    fig, ax = plt.subplots(figsize=(9.4, max(3.2, 0.28 * len(subjects) + 1.6)),
                           constrained_layout=True)
    positions = np.arange(len(subjects))
    ax.hlines(positions, 0, values, color="#c6d9ec", linewidth=3)
    ax.plot(values, positions, "o", color="#1f77b4", markersize=6, label="subject")

    ax.axvline(median, color="#d62728", linewidth=1.6,
               label=f"median {median:.3f}")
    if reference is not None:
        ax.axvline(reference, color="#666666", linestyle="--", linewidth=1.2,
                   label=f"Vallat & Walker 2021 median {reference:.2f}")

    ax.set_yticks(positions, [f"{int(s):02d}" for s in subjects])
    ax.set_ylabel("Subject")
    ax.set_xlabel("Cohen's kappa (out-of-fold, within subject)")
    ax.set_xlim(0, 1)
    ax.set_title(
        f"{MODEL_LABELS.get(report.model, report.model)} -- per-subject agreement",
        fontsize=11,
    )
    ax.grid(axis="x", alpha=0.3)
    # Outside the axes: the lollipop stems run from 0, so any in-axes corner
    # would sit on top of a subject's marker.
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, frameon=False)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def evaluate_model(
    dataset: Dataset,
    model_name: str,
    n_splits: int = config.N_SPLITS,
    seed: int = config.SEED,
    plot_subject: int | None = None,
    figures: Path | None = None,
) -> Report:
    """Cross-validate one model, write its two figures, return its report."""
    y_pred, fold_id = out_of_fold_predictions(
        dataset, lambda: build_model(model_name, seed), n_splits=n_splits
    )
    report = compute_report(
        dataset.y, y_pred, fold_id, dataset.groups, model=model_name, n_splits=n_splits
    )

    figures = figures or config.figures_dir()
    subject = int(dataset.subjects[0] if plot_subject is None else plot_subject)

    cm_path = plot_confusion_matrix(
        dataset.y, y_pred, model_name, figures / f"confusion_matrix_{model_name}.png"
    )
    hyp_path = plot_hypnogram(
        dataset,
        y_pred,
        subject,
        model_name,
        figures / f"hypnogram_{model_name}_subject{subject:02d}.png",
    )
    kappa_path = plot_subject_kappa(
        report, figures / f"subject_kappa_{model_name}.png"
    )
    logger.info("wrote %s, %s and %s", cm_path.name, hyp_path.name, kappa_path.name)
    return report


def write_reports(reports: list[Report], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stages": config.STAGE_NAMES,
        "seed": config.SEED,
        "models": {r.model: r.to_dict() for r in reports},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
