from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.dataset import find_record_files, resolve_ambiguous_records
from src.evaluation import outer_folds
from src.features import _preprocess_many
from src.models import LogisticModel, TrainingData
from src.preprocessing import PreprocessingConfig, Reject

AGE_RIDGE_ALPHAS: Final[tuple[float, ...]] = (0.1, 1.0, 10.0, 100.0, 1000.0)


def quality_descriptors(
    subject_keys: Sequence[str], cfg: PreprocessingConfig = PreprocessingConfig(), data_dir: Path = config.DATA_DIR
) -> pd.DataFrame:
    """Rest-record quality per subject."""
    rows: dict[str, dict[str, float]] = {}
    for key in subject_keys:
        usable, _ = resolve_ambiguous_records(find_record_files(data_dir / key, strict=False))
        records = _preprocess_many([usable[config.REST_STEM]], cfg) if config.REST_STEM in usable else []
        row: dict[str, float] = {}
        for idx, channel in enumerate(config.CHANNELS):
            if records and records[0].n_windows:
                flags = records[0].reject[:, idx]
                row[f"good_{channel}"] = float(np.mean(flags == 0))
                row[f"copy_{channel}"] = float(np.any(flags & Reject.COPY))
            else:
                row[f"good_{channel}"], row[f"copy_{channel}"] = np.nan, 0.0
        rows[key] = row
    return pd.DataFrame.from_dict(rows, orient="index")


def diagnostic_oof(data: TrainingData, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Leave-one-group-out probabilities of a logistic model on diagnostic inputs ``x`` (n, q)."""
    p = np.empty(data.cohort.size)
    for train, test in outer_folds(data.groups):
        med = np.nanmedian(x[train], axis=0)
        med = np.where(np.isnan(med), 0.0, med)
        fill = lambda a: np.where(np.isnan(a), med, a)  # noqa: E731
        mu, sd = fill(x[train]).mean(axis=0), fill(x[train]).std(axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000)
        clf.fit((fill(x[train]) - mu) / sd, data.y[train])
        p[test] = clf.predict_proba((fill(x[test]) - mu) / sd)[:, 1]
    return p


def feature_contributions(model: LogisticModel, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Terms w_j * z_j of the log-odds, shape (n_subjects, p)."""
    filled = np.where(np.isnan(x), model.impute_values, x)
    return (filled - model.mean) / model.scale * model.coef


def age_prediction_oof(
    x: npt.NDArray[np.float64],
    age: npt.NDArray[np.float64],
    groups: npt.NDArray[np.int_],
    alphas: Sequence[float] = AGE_RIDGE_ALPHAS,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Leave-one-group-out prediction of age from features ``x`` (n, p)."""
    predicted, reference = np.empty(age.size), np.empty(age.size)
    for train, test in outer_folds(groups):
        pipeline = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler()),
                ("ridge", RidgeCV(alphas=tuple(alphas))),
            ]
        ).fit(x[train], age[train])
        predicted[test] = pipeline.predict(x[test])
        reference[test] = age[train].mean()
    return predicted, reference


def schulte_times(registry: pd.DataFrame, data_dir: Path = config.DATA_DIR) -> pd.DataFrame:
    """Schulte solve times per subject."""
    rows: dict[str, dict[str, float]] = {}
    for key, recs in registry[registry["status"] == "ok"].groupby("subject_key", sort=True):
        usable, _ = resolve_ambiguous_records({s: data_dir / r for s, r in zip(recs["stem"], recs["relpath"])})
        duration = dict(zip(recs["stem"], recs["active_duration_s"]))
        d = np.array([duration[s] if s in usable else np.nan for s in config.TASK_STEMS], dtype=float)
        finite = d[np.isfinite(d)]
        if finite.size >= 2 and np.ptp(finite) == 0.0:
            d[:] = np.nan
        rows[key] = {
            "trial1_s": float(d[0]),
            "total_s": float(d.sum()) if np.isfinite(d).all() else np.nan,
            "n_trials": float(np.isfinite(d).sum()),
        }
    return pd.DataFrame.from_dict(rows, orient="index")
