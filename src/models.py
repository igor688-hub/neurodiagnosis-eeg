"""Classifier: median imputation, standardisation, L2 logistic regression.

Model for subject ``i`` with feature vector ``x_i`` (length p)::

    z_i   = (impute(x_i) - mu) / sigma            per feature, statistics of the training set
    P_i   = 1 / (1 + exp(-(w . z_i + b)))          probability of the PTSD cohort, in (0, 1)

``w, b`` minimise the class-weighted logistic loss with penalty ||w||^2 / (2C).
Every statistic (medians, mu, sigma, w, b) is estimated on training subjects
only; ``sklearn.pipeline.Pipeline`` enforces this inside cross-validation.

The fitted model is exported as plain arrays (JSON), so inference needs only
numpy: no pickle, no dependence on the scikit-learn version of the organiser.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config

# Baseline hyperparameters, fixed before any validation; tuned inside the
# nested cross-validation of the next stage.
BASELINE_C: Final[float] = 0.1
BASELINE_CLASS_WEIGHT: Final[str] = "balanced"


def make_pipeline(c: float = BASELINE_C, class_weight: str | None = BASELINE_CLASS_WEIGHT) -> Pipeline:
    """Unfitted pipeline; every step is fitted on the training fold only."""
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=c, class_weight=class_weight, max_iter=5000)),
        ]
    )


@dataclass(frozen=True)
class LogisticModel:
    """Fitted pipeline as plain parameters."""

    feature_names: tuple[str, ...]
    impute_values: npt.NDArray[np.float64]  # shape: (p,), training medians
    mean: npt.NDArray[np.float64]  # shape: (p,)
    scale: npt.NDArray[np.float64]  # shape: (p,)
    coef: npt.NDArray[np.float64]  # shape: (p,), log-odds per standard deviation
    intercept: float
    metadata: dict[str, object] = field(default_factory=dict)

    def decision_function(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Log-odds. ``x`` shape: (n_subjects, p) in ``feature_names`` order; NaN allowed."""
        x = np.where(np.isnan(x), self.impute_values, x)
        z = (x - self.mean) / self.scale
        return z @ self.coef + self.intercept

    def predict_proba(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Probability of the PTSD cohort, shape (n_subjects,)."""
        return 1.0 / (1.0 + np.exp(-self.decision_function(x)))

    @classmethod
    def from_pipeline(
        cls, pipeline: Pipeline, feature_names: tuple[str, ...], metadata: dict[str, object] | None = None
    ) -> LogisticModel:
        imputer: SimpleImputer = pipeline.named_steps["impute"]
        scaler: StandardScaler = pipeline.named_steps["scale"]
        clf: LogisticRegression = pipeline.named_steps["clf"]
        return cls(
            feature_names=tuple(feature_names),
            impute_values=imputer.statistics_.astype(float),
            mean=scaler.mean_.astype(float),
            scale=scaler.scale_.astype(float),
            coef=clf.coef_.ravel().astype(float),
            intercept=float(clf.intercept_[0]),
            metadata=dict(metadata or {}),
        )

    def to_json(self, path: Path) -> None:
        payload = asdict(self)
        for key in ("impute_values", "mean", "scale", "coef"):
            payload[key] = getattr(self, key).tolist()
        payload["feature_names"] = list(self.feature_names)
        path.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> LogisticModel:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            feature_names=tuple(payload["feature_names"]),
            impute_values=np.asarray(payload["impute_values"], dtype=float),
            mean=np.asarray(payload["mean"], dtype=float),
            scale=np.asarray(payload["scale"], dtype=float),
            coef=np.asarray(payload["coef"], dtype=float),
            intercept=float(payload["intercept"]),
            metadata=payload.get("metadata", {}),
        )


def fit_logistic(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.int_],
    feature_names: tuple[str, ...],
    c: float = BASELINE_C,
    class_weight: str | None = BASELINE_CLASS_WEIGHT,
    metadata: dict[str, object] | None = None,
) -> LogisticModel:
    """Fit the pipeline on all given subjects. ``x`` shape: (n_subjects, p); ``y``: 1 = PTSD."""
    pipeline = make_pipeline(c, class_weight).fit(x, y)
    meta = {"C": c, "class_weight": class_weight, "random_state": config.RANDOM_STATE, **(metadata or {})}
    return LogisticModel.from_pipeline(pipeline, feature_names, meta)
