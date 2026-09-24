"""Confounder diagnostics of a trained procedure. Never part of the model.

Each function answers one question about what drives the predictions:

* ``quality_descriptors`` + ``diagnostic_oof``: can missingness of features
  and recording quality alone (no EEG content) separate the groups?
* ``feature_contributions``: which standardized terms w_j * z_j push a
  subject's log-odds, e.g. for false positives;
* the ``mask_at_test`` option of ``evaluation.nested_oof`` (dependence of the
  fitted model on features) and ``models.restricted_protocol`` (whether a
  model can be built without them) complement these.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src import config
from src.dataset import find_record_files, resolve_ambiguous_records
from src.evaluation import outer_folds
from src.features import _preprocess_many
from src.models import LogisticModel, TrainingData
from src.preprocessing import PreprocessingConfig, Reject


def quality_descriptors(
    subject_keys: Sequence[str], cfg: PreprocessingConfig = PreprocessingConfig(), data_dir: Path = config.DATA_DIR
) -> pd.DataFrame:
    """Rest-record quality per subject: share of retained windows and copy flag per channel.

    Returns
    -------
    DataFrame indexed by subject_key with ``good_{channel}`` (0-1, NaN without
    a usable rest record) and ``copy_{channel}`` (0/1) for the six channels.
    """
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
    """Leave-one-group-out probabilities of a logistic model on diagnostic inputs ``x`` (n, q).

    Missing values are replaced by the training-fold median; C = 1, balanced
    weights, all cohorts in training. Shape of the result: (n,).
    """
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
    """Terms w_j * z_j of the log-odds, shape (n_subjects, p); their sum plus the intercept is the logit."""
    filled = np.where(np.isnan(x), model.impute_values, x)
    return (filled - model.mean) / model.scale * model.coef
