"""Protocol 5: the submitted model - trial-1 solve time plus rest EEG.

Rationale from protocols 1-4 (development results on the same data):

* the Schulte solve time separates PTSD from comparable controls (formats A
  and B, somatoform) and does not depend on the export format;
* EEG added no information about PTSD beyond behaviour, but rest EEG
  (4-20 Hz, O1/Fp1/Fp2, alpha peak frequency) separates the controls aged
  65+, who solve the first table even slower than the PTSD group, so a
  behaviour-only model would call them PTSD;
* only the first trial is available for every training group (the 65+
  controls were published with rest and trial 1), so behaviour is the trial-1
  solve time and no structurally missing trials are imputed.

Model: median imputation, standardisation, L2 logistic regression with
balanced class weights on 1 + 11 features; C chosen in inner grouped CV by a
threshold-free criterion weighted like the objective scoring,

    score = 30 * max(0, (AUC_controls - 0.5) / 0.5) + 20 * max(0, (AUC_specificity - 0.5) / 0.5),

AUC_controls: PTSD vs controls of formats A and B; AUC_specificity: PTSD vs
controls aged 65+ and somatoform subjects. The submitted probability is
Platt-calibrated on inner out-of-fold predictions (training class mix).

Training population: original release without format C (its solve times are
templated) and the controls aged 65+, all with a valid trial-1 recording.
Format-C controls and the rest-only supplement never enter training; their
predictions are out-of-sample checks (with behaviour missing).
"""
from __future__ import annotations

import dataclasses
import json
import platform
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import roc_auc_score

from src import config, dataset, features, models, task_branch
from src.evaluation import auc_with_ci

BEHAVIOUR: Final[tuple[str, ...]] = ("beh_log_duration_trial1",)
REST_EEG: Final[tuple[str, ...]] = models.PROTOCOL_3.feature_sets["rest20"][1]
MODELS: Final[dict[str, tuple[str, ...]]] = {
    "behaviour": BEHAVIOUR,
    "rest_eeg": REST_EEG,
    "combined": BEHAVIOUR + REST_EEG,
}
SUBMITTED: Final[str] = "combined"
CONTROL_STRATA: Final[tuple[str, ...]] = (models.STRATUM_CONTROL_A, models.STRATUM_CONTROL_B)
SPECIFICITY_STRATA: Final[tuple[str, ...]] = (models.STRATUM_CONTROL_AGEING, models.STRATUM_SOMATOFORM)
RESULTS_DIR: Final[Path] = config.REPO_ROOT / "results" / "validation" / "protocol5"


def criterion(y: npt.NDArray[np.int_], p: npt.NDArray[np.float64], stratum: npt.NDArray[np.str_]) -> float:
    """Threshold-free criterion weighted like the objective scoring (see module docstring)."""
    ptsd = y == 1

    def auc(negatives: tuple[str, ...]) -> float:
        mask = ptsd | np.isin(stratum, negatives)
        return float(roc_auc_score(ptsd[mask], p[mask]))

    return 30.0 * max(0.0, (auc(CONTROL_STRATA) - 0.5) / 0.5) + 20.0 * max(0.0, (auc(SPECIFICITY_STRATA) - 0.5) / 0.5)


def build_tables(data_dir: Path = config.DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str], pd.Series]:
    """Rest and task features of every subject, subject table, strata and export format."""
    registry, _, subjects = dataset.scan_dataset(data_dir)
    rest = features.build_feature_table(registry, data_dir=data_dir)
    task = task_branch.build_task_table(registry, data_dir)
    table = rest.join(task, how="left")
    family = registry[registry["status"] == "ok"].groupby("subject_key")["export_family"].first().reindex(table.index)
    stratum = dict(zip(table.index, models._strata(subjects.loc[table.index], family)))
    return table, subjects, stratum, family


def training_population(
    table: pd.DataFrame, subjects: pd.DataFrame, stratum: dict[str, str], family: pd.Series
) -> task_branch.Population:
    """Original release without format C plus controls aged 65+, with a valid trial-1 solve time."""
    eligible = (family != "C") & subjects.loc[table.index, "has_task_files"] & table["beh_log_duration_trial1"].notna()
    return task_branch.make_population("protocol5", table.index[eligible], table, subjects, stratum)


def fit_final(population: task_branch.Population, feature_names: tuple[str, ...]) -> models.LogisticModel:
    """Select C, fit on the whole population, calibrate on inner OOF predictions."""
    x = population.table[list(feature_names)].to_numpy(dtype=float)
    best_c, inner_oof = task_branch.select_c(x, population.y, population.groups, population.stratum, criterion)
    pipeline = models.make_pipeline(best_c).fit(x, population.y)
    calibration = models.fit_platt(inner_oof, population.y)
    metadata = {
        "protocol": "protocol5",
        "C": best_c,
        "feature_set": "trial1_behaviour+rest20",
        "preprocessing_variant": "native",
        "class_weight": models.BASELINE_CLASS_WEIGHT,
        "calibration": "platt on inner out-of-fold predictions",
        "n_subjects_trained": int(population.y.size),
        "n_ptsd": int(population.y.sum()),
        "strata": {str(k): int(v) for k, v in zip(*np.unique(population.stratum, return_counts=True))},
        "python": platform.python_version(),
    }
    fitted = models.LogisticModel.from_pipeline(pipeline, feature_names, metadata)
    return dataclasses.replace(fitted, calibration=calibration)


def summarize(population: task_branch.Population, oofs: dict[str, pd.DataFrame]) -> dict[str, object]:
    """Protocol 5 report: per model and probability, AUC slices and threshold metrics."""
    y, g, st = population.y, population.groups, population.stratum
    ptsd = y == 1
    report: dict[str, object] = {"n_by_stratum": {s: int(np.sum(st == s)) for s in np.unique(st)}}
    for name, oof in oofs.items():
        entry: dict[str, object] = {"C_per_fold": oof.groupby(g)["C"].first().value_counts().to_dict()}
        for col in ("p_raw", "p"):
            p = oof[col].to_numpy()

            def auc_ci(negatives: tuple[str, ...]) -> dict[str, float]:
                mask = ptsd | np.isin(st, negatives)
                return auc_with_ci(ptsd[mask].astype(int), p[mask], g[mask])

            spec_mask = np.isin(st, SPECIFICITY_STRATA)
            fpr_pooled = float(np.mean(p[spec_mask] >= 0.5))
            auc_controls = auc_ci(CONTROL_STRATA)
            entry[col] = {
                "auc_ptsd_vs_controls_AB": auc_controls,
                "auc_ptsd_vs_ageing_and_somatoform": auc_ci(SPECIFICITY_STRATA),
                "auc_ptsd_vs_all_non_ptsd": auc_with_ci(y, p, g),
                "auc_ptsd_vs_stratum": {
                    s: float(roc_auc_score(ptsd[ptsd | (st == s)], p[ptsd | (st == s)])) for s in np.unique(st) if s != "ptsd"
                },
                "sensitivity": float(np.mean(p[ptsd] >= 0.5)),
                "specificity_by_stratum": {s: float(np.mean(p[st == s] < 0.5)) for s in np.unique(st) if s != "ptsd"},
                "specificity_ageing_and_somatoform_pooled": 1.0 - fpr_pooled,
                "balanced_accuracy_vs_all_non_ptsd": 0.5 * (float(np.mean(p[ptsd] >= 0.5)) + float(np.mean(p[~ptsd] < 0.5))),
                "score_estimate": {
                    "auc_points_vs_controls_AB": 30.0 * max(0.0, (auc_controls["value"] - 0.5) / 0.5),
                    "specificity_points_ageing_and_somatoform": 20.0 * (1.0 - fpr_pooled),
                },
            }
        report[name] = entry
    for other in ("behaviour", "rest_eeg"):
        report[f"combined_minus_{other}_auc_all"] = task_branch.paired_auc_difference(
            y, oofs["combined"]["p_raw"].to_numpy(), oofs[other]["p_raw"].to_numpy(), g
        )
    return report


def main(n_jobs: int = -1) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table, subjects, stratum, family = build_tables()
    population = training_population(table, subjects, stratum, family)
    oofs = {
        name: task_branch.nested_oof_task(population, feats, n_jobs, score=criterion) for name, feats in MODELS.items()
    }
    pd.concat(oofs, axis=1).to_csv(RESULTS_DIR / "nested_oof.csv")
    report = summarize(population, oofs)

    final = fit_final(population, MODELS[SUBMITTED])
    unseen = table.index.difference(population.subject_keys)
    unseen = unseen[~pd.Series(subjects.loc[unseen, "group"] == config.GROUP_PTSD, index=unseen)]
    x_unseen = table.loc[unseen, list(final.feature_names)].to_numpy(dtype=float)
    p_unseen = pd.DataFrame(
        {"stratum": [stratum[k] for k in unseen], "p": final.predict_proba(x_unseen), "p_raw": final.predict_proba_raw(x_unseen)},
        index=unseen,
    )
    p_unseen.to_csv(RESULTS_DIR / "unseen_controls.csv")
    report["unseen_controls_final_model"] = {
        s: {
            "n": int(len(d)),
            "share_p_ge_0.5": float(np.mean(d["p"] >= 0.5)),
            "share_p_raw_ge_0.5": float(np.mean(d["p_raw"] >= 0.5)),
            "median_p": float(d["p"].median()),
        }
        for s, d in p_unseen.groupby("stratum")
    }
    report["final_model"] = {"C": final.metadata["C"], "calibration": list(final.calibration or ())}
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"protocol5 -> {RESULTS_DIR}")


if __name__ == "__main__":
    main()
