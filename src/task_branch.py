"""Protocol 4: Schulte task branch - behaviour, task EEG and their combination.

Two analyses, each on a fixed population where every subject has the data
the analysis needs (no imputation of structurally missing trials):

* **Trial 1** (analysis A): the first Schulte trial, available for the
  original release and for the controls aged 65+. Behaviour = log10 of the
  active duration of trial 1 (time to solve the table, recording padding
  excluded). EEG = relative power 4-20 Hz (theta, alpha, low beta) on O1,
  Fp1, Fp2 in the first 20 s of the trial, and its change from rest.
* **Dynamics** (analysis B): all five trials, original release only.
  Behaviour = mean log10 duration, slope of log10 duration over trials 1-5,
  first-trial excess over trials 2-5. EEG = slope over trials of the alpha
  relative power on O1 and of the frontal theta relative power, each trial
  measured on its first 20 s.

EEG is always measured on the same time segment (first 20 s = 9 windows of
4 s with 2-s step, at least 5 retained), so a slower subject does not get
more data. Durations of format-C controls are templated (five identical
values per subject) and are not solve times; a within-subject rule marks
five identical durations as unreliable, and format C is excluded from the
main populations and reported separately.

For each analysis three models - behaviour only, EEG only, both - are fitted
on the same subjects with the same outer (leave-one-group-out) and inner
(grouped 5-fold) splits. The inner criterion is the AUC of PTSD vs all
non-PTSD subjects (threshold-free); calibration is reported separately.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

from src import config, dataset, models
from src.evaluation import auc_with_ci, outer_folds
from src.features import (
    BANDS_20,
    PREPROCESSING_VARIANTS,
    TOTAL_BAND_20,
    _preprocess_many,
    condition_spectrum,
    log_relative_powers,
)
from src.preprocessing import EpochedRecord

CHANNELS: Final[tuple[str, ...]] = ("O1", "Fp1", "Fp2")
SEGMENT_WINDOWS: Final[int] = 9  # first 20 s of a trial: windows starting at 0, 2, ..., 16 s
C_GRID: Final[tuple[float, ...]] = models.C_GRID


def first_segment(epoched: EpochedRecord, n_windows: int = SEGMENT_WINDOWS) -> EpochedRecord:
    """The first ``n_windows`` windows (time-ordered) of a record."""
    return EpochedRecord(epoched.windows[:n_windows], epoched.reject[:n_windows], epoched.sfreq, epoched.channels)


def active_duration_s(path: Path) -> float:
    """Recording length without constant edges (export padding), seconds; NaN if unreadable."""
    try:
        header = dataset.read_edf_header(path)
        digital = dataset.read_digital(header)
    except (dataset.EdfFormatError, OSError, ValueError):
        return float("nan")
    sfreq = header.sfreq()
    head, tail = dataset.edge_constant_samples(
        dataset.constant_stretch_mask(digital, max(2, round(dataset.MIN_CONSTANT_S * sfreq)))
    )
    return (digital.shape[1] - head - tail) / sfreq


def _relpow20(records: Sequence[EpochedRecord]) -> dict[str, float]:
    """log10 relative power 4-20 Hz per band on CHANNELS; NaN below 5 retained windows."""
    freqs, spectrum = condition_spectrum(records)  # shape: (6, n_freqs)
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = log_relative_powers(freqs, spectrum, BANDS_20, TOTAL_BAND_20)
    return {f"{band}_{ch}": float(rel[band][config.CHANNELS.index(ch)]) for band in BANDS_20 for ch in CHANNELS}


def subject_task_features(files: Mapping[str, Path]) -> dict[str, float]:
    """Behaviour and task-EEG features of one subject (same code in training and inference).

    Returns a dict with keys ``beh_*`` and ``eeg_*``; missing inputs give NaN.
    """
    usable, _ = dataset.resolve_ambiguous_records(files)
    out: dict[str, float] = {}

    durations = np.array([active_duration_s(usable[s]) if s in usable else np.nan for s in config.TASK_STEMS])
    finite = durations[np.isfinite(durations)]
    templated = finite.size >= 2 and np.ptp(finite) == 0.0  # five identical values: not solve times
    if templated:
        durations[:] = np.nan
    log_d = np.log10(durations)  # shape: (5,)
    out["beh_log_duration_trial1"] = float(log_d[0])
    if np.isfinite(log_d).all():
        trials = np.arange(1, 6)
        out["beh_log_duration_mean"] = float(log_d.mean())
        out["beh_log_duration_slope"] = float(np.polyfit(trials, log_d, 1)[0])
        out["beh_first_trial_excess"] = float(log_d[0] - log_d[1:].mean())
    else:
        out.update(beh_log_duration_mean=np.nan, beh_log_duration_slope=np.nan, beh_first_trial_excess=np.nan)
    out["beh_templated"] = float(templated)

    cfg = PREPROCESSING_VARIANTS["native"]
    rest = _preprocess_many([usable[config.REST_STEM]], cfg) if config.REST_STEM in usable else []
    rest_rel = _relpow20(rest)
    trial_rel: list[dict[str, float]] = []
    for stem in config.TASK_STEMS:
        recs = _preprocess_many([usable[stem]], cfg) if stem in usable else []
        trial_rel.append(_relpow20([first_segment(r) for r in recs]))
    for key, value in trial_rel[0].items():
        out[f"eeg_trial1_{key}"] = value
        out[f"eeg_react1_{key}"] = value - rest_rel[key]
    alpha_o1 = np.array([t["alpha_O1"] for t in trial_rel])
    theta_fp = np.array([np.nanmean([t["theta_Fp1"], t["theta_Fp2"]]) for t in trial_rel])
    trials = np.arange(1, 6)
    for name, series in (("alpha_O1", alpha_o1), ("theta_Fp", theta_fp)):
        out[f"eeg_slope_{name}"] = float(np.polyfit(trials, series, 1)[0]) if np.isfinite(series).all() else np.nan
    return out


TRIAL1_BEHAVIOUR: Final[tuple[str, ...]] = ("beh_log_duration_trial1",)
TRIAL1_EEG: Final[tuple[str, ...]] = tuple(
    f"eeg_{kind}_{band}_{ch}" for kind in ("trial1", "react1") for band in BANDS_20 for ch in CHANNELS
)
DYNAMICS_BEHAVIOUR: Final[tuple[str, ...]] = ("beh_log_duration_mean", "beh_log_duration_slope", "beh_first_trial_excess")
DYNAMICS_EEG: Final[tuple[str, ...]] = ("eeg_slope_alpha_O1", "eeg_slope_theta_Fp")


def build_task_table(registry: pd.DataFrame, data_dir: Path = config.DATA_DIR) -> pd.DataFrame:
    """``subject_task_features`` for every subject of ``registry``, indexed by subject_key."""
    rows = {}
    for key, recs in registry.groupby("subject_key", sort=True):
        usable = recs[recs["status"] == "ok"]
        rows[key] = subject_task_features({s: data_dir / r for s, r in zip(usable["stem"], usable["relpath"])})
    return pd.DataFrame.from_dict(rows, orient="index")


@dataclass(frozen=True)
class Population:
    """Subjects of one analysis with aligned labels, groups and strata."""

    name: str
    subject_keys: tuple[str, ...]
    table: pd.DataFrame
    cohort: npt.NDArray[np.int_]
    groups: npt.NDArray[np.int_]
    stratum: npt.NDArray[np.str_]

    @property
    def y(self) -> npt.NDArray[np.int_]:
        return (self.cohort == models.COHORT_PTSD).astype(int)


def make_population(
    name: str, keys: Sequence[str], table: pd.DataFrame, subjects: pd.DataFrame, stratum: Mapping[str, str]
) -> Population:
    keys = tuple(sorted(keys))
    return Population(
        name=name,
        subject_keys=keys,
        table=table.loc[list(keys)],
        cohort=subjects.loc[list(keys), "group"].map(models.COHORT_CODES).to_numpy(dtype=int),
        groups=subjects.loc[list(keys), "split_group"].to_numpy(dtype=int),
        stratum=np.array([stratum[k] for k in keys]),
    )


def _fit_select_predict(
    x: npt.NDArray[np.float64], y: npt.NDArray[np.int_], groups: npt.NDArray[np.int_], stratum: npt.NDArray[np.str_],
    train: npt.NDArray[np.int_], test: npt.NDArray[np.int_],
) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
    """Inner selection of C by pooled inner OOF AUC, fit, Platt on inner OOF; predict ``test``."""
    folds = models.inner_splits(stratum[train], groups[train])
    best_c, best_auc, best_oof = C_GRID[0], -np.inf, None
    for c in C_GRID:
        oof = np.empty(train.size)
        for tr, te in folds:
            oof[te] = models.make_pipeline(c).fit(x[train][tr], y[train][tr]).predict_proba(x[train][te])[:, 1]
        auc = roc_auc_score(y[train], oof)
        if auc > best_auc + 1e-12:
            best_c, best_auc, best_oof = c, auc, oof
    raw = models.make_pipeline(best_c).fit(x[train], y[train]).predict_proba(x[test])[:, 1]
    calibrated = models.apply_platt(raw, models.fit_platt(best_oof, y[train]))
    return test, raw, calibrated, best_c


def nested_oof_task(population: Population, features: Sequence[str], n_jobs: int = -1) -> pd.DataFrame:
    """LOGO out-of-fold raw and calibrated probabilities for one feature list."""
    x = population.table[list(features)].to_numpy(dtype=float)
    results = Parallel(n_jobs=n_jobs)(
        delayed(_fit_select_predict)(x, population.y, population.groups, population.stratum, train, test)
        for train, test in outer_folds(population.groups)
    )
    raw, cal, chosen = np.empty(x.shape[0]), np.empty(x.shape[0]), np.empty(x.shape[0])
    for test, r, c, best_c in results:
        raw[test], cal[test], chosen[test] = r, c, best_c
    return pd.DataFrame(
        {"stratum": population.stratum, "p_raw": raw, "p": cal, "C": chosen},
        index=pd.Index(population.subject_keys, name="subject_key"),
    )


def paired_auc_difference(
    y: npt.NDArray[np.int_], p_a: npt.NDArray[np.float64], p_b: npt.NDArray[np.float64],
    groups: npt.NDArray[np.int_], n_boot: int = 2000, seed: int = config.RANDOM_STATE,
) -> dict[str, float]:
    """AUC(p_a) - AUC(p_b) on the same subjects, class-stratified group bootstrap CI."""
    rng = np.random.default_rng(seed)
    pos = [np.flatnonzero((groups == g) & (y == 1)) for g in np.unique(groups[y == 1])]
    neg = [np.flatnonzero((groups == g) & (y == 0)) for g in np.unique(groups[y == 0])]
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate(
            [pos[i] for i in rng.integers(0, len(pos), len(pos))] + [neg[i] for i in rng.integers(0, len(neg), len(neg))]
        )
        diffs[b] = roc_auc_score(y[idx], p_a[idx]) - roc_auc_score(y[idx], p_b[idx])
    return {
        "value": float(roc_auc_score(y, p_a) - roc_auc_score(y, p_b)),
        "ci_low": float(np.quantile(diffs, 0.025)),
        "ci_high": float(np.quantile(diffs, 0.975)),
    }


def summarize_task(population: Population, oofs: Mapping[str, pd.DataFrame]) -> dict[str, object]:
    """AUCs with CI per model and stratum slices, paired differences, threshold metrics."""
    y, g, st = population.y, population.groups, population.stratum
    ptsd = st == models.STRATUM_PTSD
    report: dict[str, object] = {"n_by_stratum": {s: int(np.sum(st == s)) for s in np.unique(st)}}
    for model_name, oof in oofs.items():
        entry: dict[str, object] = {}
        for col in ("p_raw", "p"):
            p = oof[col].to_numpy()
            entry[col] = {
                "auc_ptsd_vs_all_non_ptsd": auc_with_ci(y, p, g),
                "auc_ptsd_vs_stratum": {
                    s: float(roc_auc_score(ptsd[ptsd | (st == s)], p[ptsd | (st == s)])) for s in np.unique(st) if s != "ptsd"
                },
                "sensitivity": float(np.mean(p[ptsd] >= 0.5)),
                "specificity_by_stratum": {s: float(np.mean(p[st == s] < 0.5)) for s in np.unique(st) if s != "ptsd"},
                "balanced_accuracy_vs_all_non_ptsd": 0.5 * (float(np.mean(p[ptsd] >= 0.5)) + float(np.mean(p[~ptsd] < 0.5))),
            }
        entry["C_per_fold"] = oof.groupby(population.groups)["C"].first().value_counts().to_dict()
        report[model_name] = entry
    if {"behaviour", "combined"} <= set(oofs):
        report["combined_minus_behaviour_auc"] = paired_auc_difference(
            y, oofs["combined"]["p_raw"].to_numpy(), oofs["behaviour"]["p_raw"].to_numpy(), g
        )
    if {"eeg", "behaviour"} <= set(oofs):
        report["eeg_minus_behaviour_auc"] = paired_auc_difference(
            y, oofs["eeg"]["p_raw"].to_numpy(), oofs["behaviour"]["p_raw"].to_numpy(), g
        )
    return report


RESULTS_DIR: Final[Path] = config.REPO_ROOT / "results" / "validation" / "protocol4"
MODELS_TRIAL1: Final[dict[str, tuple[str, ...]]] = {
    "behaviour": TRIAL1_BEHAVIOUR, "eeg": TRIAL1_EEG, "combined": TRIAL1_BEHAVIOUR + TRIAL1_EEG,
}
MODELS_DYNAMICS: Final[dict[str, tuple[str, ...]]] = {
    "behaviour": DYNAMICS_BEHAVIOUR, "eeg": DYNAMICS_EEG, "combined": DYNAMICS_BEHAVIOUR + DYNAMICS_EEG,
}


def build_populations(data_dir: Path = config.DATA_DIR) -> tuple[dict[str, Population], pd.DataFrame]:
    """Populations of protocol 4 and the full task table (subjects with Schulte files)."""
    registry, _, subjects = dataset.scan_dataset(data_dir)
    keep = subjects.index[subjects["has_task_files"]]
    registry, subjects = registry[registry["subject_key"].isin(keep)], subjects.loc[keep]
    table = build_task_table(registry, data_dir)
    family = registry[registry["status"] == "ok"].groupby("subject_key")["export_family"].first().reindex(table.index)
    stratum = dict(zip(table.index, models._strata(subjects.loc[table.index], family)))
    not_c = family != "C"
    trial1 = table.index[not_c & table["beh_log_duration_trial1"].notna()]
    trial1_with_c = table.index[table["beh_log_duration_trial1"].notna() | (family == "C")]
    dynamics = table.index[not_c & table[list(DYNAMICS_BEHAVIOUR)].notna().all(axis=1) & ~subjects.loc[table.index, "holdout"]]
    pops = {
        "trial1": make_population("trial1", trial1, table, subjects, stratum),
        "trial1_eeg_with_format_C": make_population("trial1_eeg_with_format_C", trial1_with_c, table, subjects, stratum),
        "dynamics": make_population("dynamics", dynamics, table, subjects, stratum),
    }
    return pops, table


def main(n_jobs: int = -1) -> None:
    import json

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pops, table = build_populations()
    table.to_csv(RESULTS_DIR / "task_features.csv")
    plan = {
        "trial1": MODELS_TRIAL1,
        "trial1_eeg_with_format_C": {"eeg": TRIAL1_EEG},
        "dynamics": MODELS_DYNAMICS,
    }
    report = {}
    for pop_name, model_specs in plan.items():
        pop = pops[pop_name]
        oofs = {name: nested_oof_task(pop, feats, n_jobs) for name, feats in model_specs.items()}
        pd.concat({k: v for k, v in oofs.items()}, axis=1).to_csv(RESULTS_DIR / f"oof_{pop_name}.csv")
        report[pop_name] = summarize_task(pop, oofs)
        missing = pop.table[[f for feats in model_specs.values() for f in feats]].isna()
        report[pop_name]["share_missing_eeg_by_stratum"] = {
            s: float(missing.loc[pop.stratum == s].any(axis=1).mean()) for s in np.unique(pop.stratum)
        }
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"protocol4 -> {RESULTS_DIR}")


if __name__ == "__main__":
    main()
