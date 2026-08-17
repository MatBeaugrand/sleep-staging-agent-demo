"""Command line entry point.

    python -m src.cli run                 # full pipeline, both models
    python -m src.cli fetch --subjects 8  # download only
    python -m src.cli features            # build and cache the feature matrix
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np

from . import config
from .data import cache_path, fetch_recordings, load_or_build
from .evaluate import evaluate_model, write_reports
from .model import MODELS


def set_seed(seed: int) -> None:
    """Seed the two RNGs the pipeline can reach.

    The estimators are seeded separately through their ``random_state``; the
    cross-validation splitter is deterministic by construction.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sleep-staging",
        description="Sleep staging on Sleep-EDF Expanded (sleep-cassette).",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="run",
        choices=["run", "fetch", "features"],
        help="run: full pipeline (default); fetch: download only; features: cache features",
    )
    parser.add_argument("--subjects", type=int, default=20, help="number of subjects (default 20)")
    parser.add_argument("--night", type=int, default=1, choices=[1, 2], help="recording night")
    parser.add_argument(
        "--folds", type=int, default=config.N_SPLITS, help="GroupKFold splits (default 5)"
    )
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODELS),
        choices=list(MODELS),
        help="models to evaluate (default: all)",
    )
    parser.add_argument(
        "--plot-subject",
        type=int,
        default=None,
        help="subject id for the hypnogram figure (default: lowest id)",
    )
    parser.add_argument(
        "--no-crop",
        action="store_true",
        help="keep the whole ~20 h recording instead of the sleep period +/- 30 min",
    )
    parser.add_argument(
        "--force", action="store_true", help="recompute features even if a cache exists"
    )
    parser.add_argument(
        "--figures-dir", type=Path, default=None, help="override the figures output directory"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    set_seed(args.seed)
    config.ensure_dirs()

    if args.folds > args.subjects:
        raise SystemExit(
            f"--folds ({args.folds}) cannot exceed --subjects ({args.subjects}): "
            "each fold must hold out at least one whole subject"
        )

    if args.command == "fetch":
        recordings = fetch_recordings(n_subjects=args.subjects, night=args.night)
        print(f"{len(recordings)} recordings cached under {config.raw_dir()}")
        for subject, psg, _ in recordings:
            print(f"  subject {subject:02d}  {psg.name}")
        return 0

    crop = not args.no_crop
    cache = cache_path(n_subjects=args.subjects, night=args.night, crop=crop)
    if args.force and cache.exists():
        cache.unlink()

    dataset = load_or_build(
        cache=cache, n_subjects=args.subjects, night=args.night, crop=crop
    )
    print(
        f"{dataset.X.shape[0]} epochs x {dataset.X.shape[1]} features "
        f"from {len(dataset.subjects)} subjects"
    )
    counts = np.bincount(dataset.y, minlength=len(config.STAGE_NAMES))
    print(
        "class balance: "
        + ", ".join(
            f"{s} {c} ({c / counts.sum():.1%})" for s, c in zip(config.STAGE_NAMES, counts)
        )
    )

    if args.command == "features":
        print(f"features cached at {cache}")
        return 0

    figures = args.figures_dir or config.figures_dir()
    reports = []
    for name in args.models:
        print()
        report = evaluate_model(
            dataset,
            model_name=name,
            n_splits=args.folds,
            seed=args.seed,
            plot_subject=args.plot_subject,
            figures=figures,
        )
        reports.append(report)
        print(report.summary())

    out = write_reports(reports, config.PROJECT_ROOT / "results" / "metrics.json")
    print(f"\nfigures -> {figures}\nmetrics -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
