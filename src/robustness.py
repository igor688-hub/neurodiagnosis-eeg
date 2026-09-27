"""Robustness of the submitted procedure to recording batch."""
from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import roc_auc_score

from src import config, models
from src.evaluation import RESULTS_DIR, auc_with_ci, nested_oof

OUT_DIR: Final[Path] = RESULTS_DIR / "protocol8_robustness"
HELD_OUT: Final[tuple[str, ...]] = ("control_A", "control_C", "control_B_supplement", "control_ageing", "somatoform")
FORMAT_B_CONTROLS: Final[tuple[str, ...]] = ("control_B", "control_B_supplement", "control_ageing")


def subset(data: models.TrainingData, mask: npt.NDArray[np.bool_]) -> models.TrainingData:
    """Training data restricted to ``mask``."""
    return replace(
        data,
        subject_keys=tuple(np.asarray(data.subject_keys)[mask]),
        tables={k: v[mask] for k, v in data.tables.items()},
        cohort=data.cohort[mask],
        groups=data.groups[mask],
        stratum=data.stratum[mask],
        export_family=data.export_family[mask],
        metadata=data.metadata[mask],
    )


def fit_procedure(data: models.TrainingData, protocol: models.Protocol) -> tuple[object, str, tuple[float, float]]:
    """The training procedure of ``protocol`` on all subjects of ``data``."""
    inner_oof: dict[str, npt.NDArray[np.float64]] = {}
    best, _ = models.select_candidate(data.tables, data.cohort, data.groups, data.stratum, protocol, oof_out=inner_oof)
    pipeline = models.fit_candidate(data.tables[best.features], data.cohort, best, data.stratum)
    calibration = models.fit_calibration(inner_oof[best.key], data.y, data.stratum, protocol)
    return pipeline, best.features, calibration


def unseen_batch(
    data: models.TrainingData, protocol: models.Protocol, reference: pd.DataFrame, held_out: str, n_jobs: int = -1
) -> dict[str, float]:
    """AUC of PTSD against one group left out of the whole procedure, next to the in-training value."""
    out_mask = data.stratum == held_out
    train = subset(data, ~out_mask)
    oof = nested_oof(train, protocol, n_jobs=n_jobs)
    pipeline, feature_set, calibration = fit_procedure(train, protocol)
    p_out = pipeline.predict_proba(data.tables[feature_set][out_mask])[:, 1]
    ptsd = oof["stratum"].to_numpy() == models.STRATUM_PTSD
    p_ptsd = oof["p_raw"].to_numpy()[ptsd]
    y = np.r_[np.ones(p_ptsd.size), np.zeros(p_out.size)]
    groups = np.r_[oof["split_group"].to_numpy()[ptsd], data.groups[out_mask]]
    unseen = auc_with_ci(y.astype(int), np.r_[p_ptsd, p_out], groups)
    ref_st = reference["stratum"].to_numpy()
    ref_mask = (ref_st == models.STRATUM_PTSD) | (ref_st == held_out)
    return {
        "n": int(out_mask.sum()),
        "auc_unseen": unseen["value"],
        "auc_unseen_ci_low": unseen["ci_low"],
        "auc_unseen_ci_high": unseen["ci_high"],
        "auc_in_training": float(roc_auc_score(ref_st[ref_mask] == models.STRATUM_PTSD, reference["p_raw"].to_numpy()[ref_mask])),
        "share_p_ge_05_unseen": float(np.mean(models.apply_platt(p_out, calibration) >= models.DECISION_THRESHOLD)),
        "share_p_ge_05_in_training": float(np.mean(reference["p"].to_numpy()[ref_st == held_out] >= models.DECISION_THRESHOLD)),
    }


def within_format_b_auc(oof: pd.DataFrame, stratum: npt.NDArray[np.str_]) -> float:
    """AUC of PTSD against format-B controls, raw probability."""
    mask = np.isin(stratum, (models.STRATUM_PTSD, *FORMAT_B_CONTROLS))
    return float(roc_auc_score(stratum[mask] == models.STRATUM_PTSD, oof["p_raw"].to_numpy()[mask]))


def permute_within(
    data: models.TrainingData, mask: npt.NDArray[np.bool_], rng: np.random.Generator
) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.str_]]:
    """Permute (cohort, stratum) pairs between the split groups inside ``mask`` only."""
    cohort, stratum = data.cohort.copy(), data.stratum.copy()
    groups = np.unique(data.groups[mask])
    first = np.array([np.flatnonzero(data.groups == g)[0] for g in groups])
    source = dict(zip(groups, first[rng.permutation(groups.size)]))
    idx = np.flatnonzero(mask)
    donors = np.array([source[g] for g in data.groups[idx]])
    cohort[idx], stratum[idx] = data.cohort[donors], data.stratum[donors]
    return cohort, stratum


def permutation_within_format_b(
    data: models.TrainingData, protocol: models.Protocol, observed: float, n_perm: int, n_jobs: int = -1,
    log: Path | None = None,
) -> dict[str, object]:
    """Null distribution of the within-format-B AUC when labels are permuted inside format B."""
    rng = np.random.default_rng(config.RANDOM_STATE)
    mask = np.isin(data.stratum, (models.STRATUM_PTSD, *FORMAT_B_CONTROLS))
    null = []
    for i in range(n_perm):
        cohort, stratum = permute_within(data, mask, rng)
        oof = nested_oof(data, protocol, cohort=cohort, stratum=stratum, n_jobs=n_jobs)
        null.append(within_format_b_auc(oof, stratum))
        if log is not None:
            log.write_text(f"{i + 1}/{n_perm}\n", encoding="utf-8")
    arr = np.asarray(null)
    return {"observed_auc": observed, "n_permutations": n_perm, "null_auc": null, "null_mean": float(arr.mean()),
            "null_q95": float(np.quantile(arr, 0.95)), "p_value": float((1 + np.sum(arr >= observed)) / (1 + n_perm))}


def main() -> None:
    """Run the pre-registered robustness checks of protocol 8."""
    parser = argparse.ArgumentParser(description="Batch robustness of protocol 8")
    parser.add_argument("--check", choices=("unseen", "permutation"), required=True)
    parser.add_argument("--n-perm", type=int, default=60)
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    protocol = models.PROTOCOL_8
    data = models.load_training_data(protocol=protocol)
    reference = pd.read_csv(RESULTS_DIR / "protocol8" / "nested_oof.csv", index_col=0).loc[list(data.subject_keys)]
    if args.check == "unseen":
        result = {g: unseen_batch(data, protocol, reference, g, args.n_jobs) for g in HELD_OUT}
        (OUT_DIR / "unseen_batch.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    else:
        observed = within_format_b_auc(reference, data.stratum)
        result = permutation_within_format_b(data, protocol, observed, args.n_perm, args.n_jobs, OUT_DIR / "permutation_progress.txt")
        (OUT_DIR / "permutation_within_format_b.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result if args.check == "unseen" else {k: v for k, v in result.items() if k != "null_auc"}, indent=1))


if __name__ == "__main__":
    main()
