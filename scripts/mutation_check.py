"""Mutation testing: does the test suite actually detect a broken pipeline?

A green test suite is a hypothesis, not a guarantee.  This script breaks the
pipeline in five specific, methodologically meaningful ways and checks that the
suite goes red for each one.  A mutation that leaves the suite green marks an
invariant nobody is testing.

    python scripts/mutation_check.py            # all mutations
    python scripts/mutation_check.py --list     # names only
    python scripts/mutation_check.py leak       # one mutation

Each mutation is applied to a scratch copy of the repository, so the working
tree is never touched.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: Directories that never need copying into the scratch tree.
_SKIP = {".git", ".venv", "venv", "data", "figures", "results", "__pycache__", ".pytest_cache"}


@dataclass(frozen=True)
class Mutation:
    """A single deliberate defect."""

    name: str
    file: str
    old: str
    new: str
    rationale: str


MUTATIONS = [
    Mutation(
        name="leak",
        file="src/evaluate.py",
        old="    cv = GroupKFold(n_splits=n_splits)",
        new="    from sklearn.model_selection import KFold\n"
            "    cv = KFold(n_splits=n_splits, shuffle=True, random_state=0)",
        rationale="Split epochs at random instead of by subject. Consecutive epochs "
                  "from one night are near-duplicates, so this leaks the answer.",
    ),
    Mutation(
        name="unscorable",
        file="src/config.py",
        old='STAGE_MAP = {\n    "Sleep stage W": "W",',
        new='STAGE_MAP = {\n    "Movement time": "W",\n    "Sleep stage ?": "W",\n'
            '    "Sleep stage W": "W",',
        rationale="Fold movement-time and unscorable epochs into wake instead of "
                  "dropping them, inventing labels the scorer never assigned.",
    ),
    Mutation(
        name="crop",
        file="src/data.py",
        old="        keep = sleep_period_mask(labels, onsets, config.CROP_MARGIN_MIN * 60.0)",
        new="        keep = np.ones(labels.shape, dtype=bool)",
        rationale="Keep the whole ~20 h cassette recording. Wake rises from 17 % to "
                  "71 % of epochs and the task degenerates into wake detection.",
    ),
    Mutation(
        name="weights",
        file="src/model.py",
        old='class_weight="balanced",',
        new="class_weight=None,",
        rationale="Drop balanced class weighting. Accuracy barely moves while the "
                  "rare N1 stage stops being predicted at all.",
    ),
    Mutation(
        name="confusion",
        file="src/evaluate.py",
        old="    totals = cm.sum(axis=1, keepdims=True)",
        new="    totals = cm.sum(axis=0, keepdims=True)",
        rationale="Normalise the confusion matrix by column instead of by row, so "
                  "the diagonal reads as precision while being labelled recall.",
    ),
]


def _scratch_copy(destination: Path) -> Path:
    """Copy the source tree, tests included, into ``destination``."""
    shutil.copytree(
        PROJECT_ROOT,
        destination,
        ignore=shutil.ignore_patterns(*_SKIP),
        dirs_exist_ok=True,
    )
    return destination


def apply_mutation(tree: Path, mutation: Mutation) -> None:
    """Rewrite one file in ``tree``, refusing to no-op silently."""
    target = tree / mutation.file
    source = target.read_text(encoding="utf-8")
    occurrences = source.count(mutation.old)
    if occurrences == 0:
        raise SystemExit(
            f"mutation {mutation.name!r} no longer applies: its target text is "
            f"absent from {mutation.file}. The mutation needs updating."
        )
    target.write_text(source.replace(mutation.old, mutation.new), encoding="utf-8")


def run_suite(tree: Path) -> tuple[bool, list[str]]:
    """Run pytest in ``tree``; return ``(passed, failing_test_ids)``."""
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=tree,
        capture_output=True,
        text=True,
    )
    failures = [
        line.split(" ")[1]
        for line in completed.stdout.splitlines()
        if line.startswith("FAILED ")
    ]
    return completed.returncode == 0, failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*", help="mutations to run (default: all)")
    parser.add_argument("--list", action="store_true", help="list mutation names and exit")
    args = parser.parse_args(argv)

    if args.list:
        for mutation in MUTATIONS:
            print(f"{mutation.name:<12} {mutation.rationale}")
        return 0

    selected = MUTATIONS
    if args.names:
        by_name = {m.name: m for m in MUTATIONS}
        unknown = set(args.names) - set(by_name)
        if unknown:
            raise SystemExit(f"unknown mutation(s): {sorted(unknown)}")
        selected = [by_name[n] for n in args.names]

    print("Baseline: the unmutated suite must be green before anything else.\n")
    with tempfile.TemporaryDirectory() as tmp:
        passed, failures = run_suite(_scratch_copy(Path(tmp) / "baseline"))
    if not passed:
        print(f"  BASELINE IS RED ({len(failures)} failing) -- fix that first.")
        return 2
    print("  baseline green\n")

    survivors = []
    for mutation in selected:
        print(f"[{mutation.name}] {mutation.rationale}")
        with tempfile.TemporaryDirectory() as tmp:
            tree = _scratch_copy(Path(tmp) / "mutant")
            apply_mutation(tree, mutation)
            passed, failures = run_suite(tree)

        if passed:
            survivors.append(mutation.name)
            print("  SURVIVED -- the suite stayed green. This invariant is untested.\n")
        else:
            shown = ", ".join(f.split("::")[-1] for f in failures[:3])
            more = f" (+{len(failures) - 3} more)" if len(failures) > 3 else ""
            print(f"  caught by {len(failures)} test(s): {shown}{more}\n")

    print("-" * 72)
    if survivors:
        print(f"{len(survivors)}/{len(selected)} mutation(s) survived: {survivors}")
        print("Each survivor is a gap in the test suite, not a bug in the pipeline.")
        return 1
    print(f"All {len(selected)} mutations were detected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
