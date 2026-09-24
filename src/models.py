"""Classifier: median imputation, standardisation, L2 logistic regression.

Model for subject ``i`` with feature vector ``x_i`` (length p)::

    z_i   = (impute(x_i) - mu) / sigma            per feature, statistics of the training set
    P_i   = 1 / (1 + exp(-(w . z_i + b)))          probability of the PTSD cohort, in (0, 1)

``w, b`` minimise the class-weighted logistic loss with penalty ||w||^2 / (2C).
Every statistic (medians, mu, sigma, w, b) is estimated on training subjects
only; ``sklearn.pipeline.Pipeline`` enforces this inside cross-validation.

The fitted model is exported as plain arrays (JSON), so inference needs only
numpy: no pickle, no dependence on the scikit-learn version of the organiser.

The training procedure includes model selection: ``select_candidate`` picks C,
the negative class and the preprocessing variant by grouped inner
cross-validation on the training subjects only, with a criterion fixed in
advance. The same procedure runs inside every outer fold of the evaluation
and once on all training subjects for the final weights.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config, dataset, features

# Defaults of a single fit. C is chosen by ``select_candidate``. Class weights
# stay balanced by design: P = 0.5 is then the point of equal PTSD and
# non-PTSD error rates, so specificity at the organisers' threshold has to come
# from discrimination, not from shrinking every probability towards zero.
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


# ---------------------------------------------------------------------------
# Model selection inside the training data (pre-declared grid and criterion)
# ---------------------------------------------------------------------------

COHORT_CONTROL: Final[int] = 0
COHORT_PTSD: Final[int] = 1
COHORT_SOMATOFORM: Final[int] = 2
COHORT_CODES: Final[dict[str, int]] = {
    config.GROUP_CONTROL: COHORT_CONTROL,
    config.GROUP_PTSD: COHORT_PTSD,
    config.GROUP_SOMATOFORM: COHORT_SOMATOFORM,
}

C_GRID: Final[tuple[float, ...]] = (0.01, 0.03, 0.1, 0.3, 1.0)
NEGATIVES: Final[tuple[str, ...]] = ("control+somatoform", "control")
FEATURE_VARIANTS: Final[tuple[str, ...]] = ("native", "requantized")  # keys of src.features.PREPROCESSING_VARIANTS
INNER_SPLITS: Final[int] = 5
DECISION_THRESHOLD: Final[float] = 0.5  # the organisers' threshold


@dataclass(frozen=True)
class Candidate:
    """One configuration of the training procedure."""

    c: float  # inverse L2 strength
    negatives: str  # cohorts used as the negative class in training
    features: str  # preprocessing variant of the feature table

    @property
    def key(self) -> str:
        return f"C={self.c}|neg={self.negatives}|feat={self.features}"


CANDIDATES: Final[tuple[Candidate, ...]] = tuple(
    Candidate(c, negatives, variant) for variant in FEATURE_VARIANTS for negatives in NEGATIVES for c in C_GRID
)


def selection_score(p: npt.NDArray[np.float64], cohort: npt.NDArray[np.int_]) -> float:
    """Criterion that mirrors the objective part of the scoring rules.

        score = 30 * max(0, (AUC - 0.5) / 0.5) + 20 * (1 - FPR_somatoform)

    AUC is PTSD vs control; FPR_somatoform is the share of somatoform
    subjects with P >= 0.5. Range 0-50.
    """
    pair = (cohort == COHORT_PTSD) | (cohort == COHORT_CONTROL)
    auc = roc_auc_score(cohort[pair] == COHORT_PTSD, p[pair])
    soma = cohort == COHORT_SOMATOFORM
    fpr = float(np.mean(p[soma] >= DECISION_THRESHOLD)) if soma.any() else 0.0
    return 30.0 * max(0.0, (auc - 0.5) / 0.5) + 20.0 * (1.0 - fpr)


def training_mask(cohort: npt.NDArray[np.int_], negatives: str) -> npt.NDArray[np.bool_]:
    """Subjects a candidate is trained on; evaluation always covers every cohort."""
    if negatives == "control":
        return cohort != COHORT_SOMATOFORM
    return np.ones(cohort.shape, dtype=bool)


def fit_candidate(
    x: npt.NDArray[np.float64], cohort: npt.NDArray[np.int_], candidate: Candidate
) -> Pipeline:
    """Fit the pipeline of ``candidate`` on the subjects it trains on. ``x`` shape: (n, p)."""
    mask = training_mask(cohort, candidate.negatives)
    return make_pipeline(candidate.c).fit(x[mask], (cohort[mask] == COHORT_PTSD).astype(int))


def inner_splits(cohort: npt.NDArray[np.int_], groups: npt.NDArray[np.int_], seed: int = config.RANDOM_STATE):
    """Group-aware folds stratified by cohort, shared by all candidates for a fair comparison."""
    cv = StratifiedGroupKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=seed)
    return list(cv.split(np.zeros(cohort.size), cohort, groups))


def select_candidate(
    tables: Mapping[str, npt.NDArray[np.float64]],
    cohort: npt.NDArray[np.int_],
    groups: npt.NDArray[np.int_],
    candidates: Sequence[Candidate] = CANDIDATES,
) -> tuple[Candidate, dict[str, float]]:
    """Choose a candidate by inner group cross-validation on the given subjects only.

    ``tables`` maps a feature variant to its matrix (n, p), rows aligned with
    ``cohort`` and ``groups``. Out-of-fold predictions of all inner folds are
    pooled and scored with ``selection_score``. Ties go to the stronger
    penalty (smaller C), then to the earlier entry of the grid.

    Returns
    -------
    (best candidate, {candidate.key: score}).
    """
    folds = inner_splits(cohort, groups)
    scores: dict[str, float] = {}
    for candidate in candidates:
        x = tables[candidate.features]
        oof = np.empty(cohort.size)
        for train, test in folds:
            oof[test] = fit_candidate(x[train], cohort[train], candidate).predict_proba(x[test])[:, 1]
        scores[candidate.key] = selection_score(oof, cohort)
    order = sorted(range(len(candidates)), key=lambda i: (-round(scores[candidates[i].key], 10), candidates[i].c, i))
    return candidates[order[0]], scores


# ---------------------------------------------------------------------------
# Training data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingData:
    """Training subjects (hold-out excluded) with aligned feature tables and labels."""

    subject_keys: tuple[str, ...]
    tables: dict[str, npt.NDArray[np.float64]]  # variant -> shape (n_subjects, n_features)
    cohort: npt.NDArray[np.int_]  # shape: (n_subjects,), models.COHORT_* codes
    groups: npt.NDArray[np.int_]  # shape: (n_subjects,), split_group
    export_family: npt.NDArray[np.str_]  # shape: (n_subjects,), diagnostic only
    metadata: npt.NDArray[np.float64]  # shape: (n_subjects, 5), export descriptors, diagnostic only

    @property
    def y(self) -> npt.NDArray[np.int_]:
        return (self.cohort == COHORT_PTSD).astype(int)


def _export_descriptors(registry: pd.DataFrame, subject_keys: Sequence[str]) -> npt.NDArray[np.float64]:
    """Per subject: one-hot format A/B/C, share of files with BS:50, share at 125 Hz."""
    ok = registry[registry["status"] == "ok"].groupby("subject_key")
    fam = ok["export_family"].first().reindex(subject_keys)
    notch = ok["notch_50"].mean().reindex(subject_keys)
    at_125 = ok["sfreq"].apply(lambda s: float(np.mean(s == config.TARGET_SFREQ))).reindex(subject_keys)
    return np.column_stack([fam == "A", fam == "B", fam == "C", notch, at_125]).astype(float)


def load_training_data(data_dir: Path = config.DATA_DIR) -> TrainingData:
    """Registry, split groups and both feature tables of the training subjects."""
    registry, _, subjects = dataset.scan_dataset(data_dir)
    registry, subjects = registry[~registry["holdout"]], subjects[~subjects["holdout"]]
    tables = {
        variant: features.build_feature_table(registry, cfg, data_dir)
        for variant, cfg in features.PREPROCESSING_VARIANTS.items()
    }
    keys = tuple(tables["native"].index)
    fam = registry[registry["status"] == "ok"].groupby("subject_key")["export_family"].first()
    return TrainingData(
        subject_keys=keys,
        tables={variant: table.loc[list(keys)].to_numpy(dtype=float) for variant, table in tables.items()},
        cohort=subjects.loc[list(keys), "group"].map(COHORT_CODES).to_numpy(dtype=int),
        groups=subjects.loc[list(keys), "split_group"].to_numpy(dtype=int),
        export_family=fam.reindex(keys).to_numpy(dtype=str),
        metadata=_export_descriptors(registry, keys),
    )
