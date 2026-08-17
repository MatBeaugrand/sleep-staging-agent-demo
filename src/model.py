"""Model definitions.

Both estimators are ``Pipeline`` objects so that every fitted preprocessing
step (currently the scaler) is fitted on the training fold only.  Fitting a
scaler on the whole feature matrix before cross-validation would let held-out
subjects influence the training data, which is exactly what the grouped split
is meant to prevent.

Both use balanced class weights: N1 accounts for only a few percent of epochs
and is invisible to an unweighted fit.
"""

from __future__ import annotations

from typing import Callable

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import SEED


def build_logistic_regression(seed: int = SEED) -> Pipeline:
    """Multinomial logistic regression -- the interpretable linear baseline."""
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=5000,
                    solver="lbfgs",
                    random_state=seed,
                ),
            ),
        ]
    )


def build_gradient_boosting(seed: int = SEED) -> Pipeline:
    """Histogram gradient boosting.

    ``early_stopping=False`` is deliberate: the built-in early stopping carves
    a random validation slice out of the training fold, which would put
    neighbouring epochs of the same subject on both sides of that split and
    make the run non-deterministic.  The tree count is fixed instead.
    """
    return Pipeline(
        [
            (
                "clf",
                HistGradientBoostingClassifier(
                    class_weight="balanced",
                    learning_rate=0.1,
                    max_iter=200,
                    max_leaf_nodes=31,
                    l2_regularization=1.0,
                    early_stopping=False,
                    random_state=seed,
                ),
            )
        ]
    )


#: Name -> factory, used by the CLI and by :mod:`src.evaluate`.
MODELS: dict[str, Callable[[int], Pipeline]] = {
    "logreg": build_logistic_regression,
    "gbdt": build_gradient_boosting,
}

MODEL_LABELS = {
    "logreg": "Logistic regression",
    "gbdt": "Gradient boosting",
}


def build_model(name: str, seed: int = SEED) -> Pipeline:
    """Instantiate a model by name."""
    try:
        factory = MODELS[name]
    except KeyError:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(MODELS)}") from None
    return factory(seed)
