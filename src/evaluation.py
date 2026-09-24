"""Nested evaluation of the whole training procedure, metrics and null models.

Outer loop: leave-one-group-out over the duplicate-aware ``split_group``
(158 groups for 166 subjects). In every outer fold the complete training
procedure of ``src.models`` runs on the remaining subjects: candidate
selection by inner grouped cross-validation, then a fit of the chosen
candidate. The held-out group is predicted once, so every subject gets one
out-of-fold (OOF) probability from a model that never saw it or its
duplicates, and the selection step is inside the evaluation.

Metrics are computed on the pooled OOF probabilities:

* ROC-AUC PTSD vs control with a percentile 95% CI from 2000 bootstrap
  resamples of split groups, stratified by class (each resample keeps the
  number of PTSD and control groups; a group enters with all its subjects);
* balanced accuracy, sensitivity and specificity at P = 0.5;
* specificity on somatoform subjects (1 - share with P >= 0.5) with a group
  bootstrap CI;
* the same quantities by export format, reported descriptively.

The CI describes the uncertainty of fixed OOF predictions over subjects; it
does not include the variability of re-training and re-selection.

Null models: (1) a logistic model on export metadata only, evaluated by the
same outer loop, measures how much the recording format alone reveals the
label; (2) a permutation test re-runs the entire nested procedure with cohort
labels permuted between split groups.

The held-out ageing group (``holdout``) is never loaded here.
"""
from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from src import config, models
from src.models import (
    COHORT_CODES,
    COHORT_CONTROL,
    COHORT_PTSD,
    COHORT_SOMATOFORM,
    PROTOCOL_1,
    PROTOCOLS,
    Protocol,
    TrainingData,
    load_training_data,
)

N_BOOTSTRAP: Final[int] = 2000
CI_LEVEL: Final[float] = 0.95
RESULTS_DIR: Final[Path] = config.REPO_ROOT / "results" / "validation"


def outer_folds(groups: npt.NDArray[np.int_]) -> list[tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]]:
    """Leave-one-group-out: (train indices, test indices) for every split group."""
    return [(np.flatnonzero(groups != g), np.flatnonzero(groups == g)) for g in np.unique(groups)]


def _outer_fold(
    tables: dict[str, npt.NDArray[np.float64]],
    cohort: npt.NDArray[np.int_],
    groups: npt.NDArray[np.int_],
    stratum: npt.NDArray[np.str_],
    train: npt.NDArray[np.int_],
    test: npt.NDArray[np.int_],
    protocol: Protocol,
) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.float64], str]:
    """Run the full training procedure on ``train`` and predict ``test``."""
    train_tables = {key: x[train] for key, x in tables.items()}
    best, _ = models.select_candidate(train_tables, cohort[train], groups[train], stratum[train], protocol)
    pipeline = models.fit_candidate(train_tables[best.features], cohort[train], best)
    return test, pipeline.predict_proba(tables[best.features][test])[:, 1], best.key


def nested_oof(
    data: TrainingData,
    protocol: Protocol = PROTOCOL_1,
    cohort: npt.NDArray[np.int_] | None = None,
    stratum: npt.NDArray[np.str_] | None = None,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Out-of-fold probabilities of the complete procedure (selection included).

    ``cohort`` and ``stratum`` override the true labels (permutation test).

    Returns
    -------
    DataFrame indexed by subject_key with columns ``cohort, stratum,
    export_family, split_group, p, candidate``.
    """
    cohort = data.cohort if cohort is None else cohort
    stratum = data.stratum if stratum is None else stratum
    results = Parallel(n_jobs=n_jobs)(
        delayed(_outer_fold)(data.tables, cohort, data.groups, stratum, train, test, protocol)
        for train, test in outer_folds(data.groups)
    )
    p = np.empty(cohort.size)
    chosen = np.empty(cohort.size, dtype=object)
    for test, proba, key in results:
        p[test], chosen[test] = proba, key
    return pd.DataFrame(
        {
            "cohort": cohort,
            "stratum": stratum,
            "export_family": data.export_family,
            "split_group": data.groups,
            "p": p,
            "candidate": chosen,
        },
        index=pd.Index(data.subject_keys, name="subject_key"),
    )


def candidate_frequency(oof: pd.DataFrame) -> dict[str, int]:
    """How often each candidate was chosen, counted per outer fold (split group), not per subject."""
    return oof.groupby("split_group")["candidate"].first().value_counts().to_dict()


def metadata_oof(data: TrainingData) -> pd.DataFrame:
    """OOF probabilities of a logistic model that sees only export metadata.

    Same outer loop, no tuning (C = 1, balanced weights), trained on all
    cohorts. A diagnostic of confounding by recording format, never part of
    the solution.
    """
    p = np.empty(data.cohort.size)
    for train, test in outer_folds(data.groups):
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000)
        clf.fit(data.metadata[train], data.y[train])
        p[test] = clf.predict_proba(data.metadata[test])[:, 1]
    return pd.DataFrame(
        {"cohort": data.cohort, "stratum": data.stratum, "export_family": data.export_family, "p": p},
        index=pd.Index(data.subject_keys, name="subject_key"),
    )


def _group_indices(groups: npt.NDArray[np.int_], mask: npt.NDArray[np.bool_]) -> list[npt.NDArray[np.int_]]:
    return [np.flatnonzero(mask & (groups == g)) for g in np.unique(groups[mask])]


def auc_with_ci(
    y: npt.NDArray[np.int_],
    p: npt.NDArray[np.float64],
    groups: npt.NDArray[np.int_],
    n_boot: int = N_BOOTSTRAP,
    seed: int = config.RANDOM_STATE,
) -> dict[str, float]:
    """ROC-AUC with a class-stratified group bootstrap percentile CI."""
    rng = np.random.default_rng(seed)
    pos, neg = _group_indices(groups, y == 1), _group_indices(groups, y == 0)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate(
            [pos[i] for i in rng.integers(0, len(pos), len(pos))] + [neg[i] for i in rng.integers(0, len(neg), len(neg))]
        )
        boot[b] = roc_auc_score(y[idx], p[idx])
    alpha = (1.0 - CI_LEVEL) / 2.0
    return {
        "value": float(roc_auc_score(y, p)),
        "ci_low": float(np.quantile(boot, alpha)),
        "ci_high": float(np.quantile(boot, 1.0 - alpha)),
    }


def rate_with_ci(
    flags: npt.NDArray[np.bool_], groups: npt.NDArray[np.int_], n_boot: int = N_BOOTSTRAP, seed: int = config.RANDOM_STATE
) -> dict[str, float]:
    """Share of True with a group bootstrap percentile CI."""
    rng = np.random.default_rng(seed)
    idx_groups = _group_indices(groups, np.ones(flags.size, dtype=bool))
    boot = np.array(
        [
            flags[np.concatenate([idx_groups[i] for i in rng.integers(0, len(idx_groups), len(idx_groups))])].mean()
            for _ in range(n_boot)
        ]
    )
    alpha = (1.0 - CI_LEVEL) / 2.0
    return {
        "value": float(flags.mean()),
        "ci_low": float(np.quantile(boot, alpha)),
        "ci_high": float(np.quantile(boot, 1.0 - alpha)),
    }


def summarize(oof: pd.DataFrame, groups: npt.NDArray[np.int_]) -> dict[str, object]:
    """Metrics of pooled OOF probabilities; ``groups`` aligned with ``oof`` rows."""
    cohort, p, fam = oof["cohort"].to_numpy(), oof["p"].to_numpy(), oof["export_family"].to_numpy()
    thr = models.DECISION_THRESHOLD
    pair = (cohort == COHORT_PTSD) | (cohort == COHORT_CONTROL)
    y_pair = (cohort[pair] == COHORT_PTSD).astype(int)
    sens = float(np.mean(p[cohort == COHORT_PTSD] >= thr))
    spec_control = float(np.mean(p[cohort == COHORT_CONTROL] < thr))
    soma = cohort == COHORT_SOMATOFORM
    fpr_soma = rate_with_ci(p[soma] >= thr, groups[soma])
    auc = auc_with_ci(y_pair, p[pair], groups[pair])

    def auc_vs(mask_neg: npt.NDArray[np.bool_]) -> float | None:
        m = (cohort == COHORT_PTSD) | mask_neg
        return float(roc_auc_score(cohort[m] == COHORT_PTSD, p[m])) if mask_neg.any() else None

    by_format = {
        f"control_{f}": {
            "n": int(np.sum((cohort == COHORT_CONTROL) & (fam == f))),
            "fpr_at_0.5": float(np.mean(p[(cohort == COHORT_CONTROL) & (fam == f)] >= thr)),
            "auc_ptsd_vs_these": auc_vs((cohort == COHORT_CONTROL) & (fam == f)),
            "median_p": float(np.median(p[(cohort == COHORT_CONTROL) & (fam == f)])),
        }
        for f in ("A", "B", "C")
    }
    return {
        "n_subjects": int(cohort.size),
        "auc_ptsd_vs_control": auc,
        "balanced_accuracy": 0.5 * (sens + spec_control),
        "sensitivity": sens,
        "specificity_control": spec_control,
        "specificity_somatoform": {
            "value": 1.0 - fpr_soma["value"], "ci_low": 1.0 - fpr_soma["ci_high"], "ci_high": 1.0 - fpr_soma["ci_low"]
        },
        "auc_ptsd_vs_somatoform": auc_vs(soma),
        "brier_ptsd_vs_control": float(brier_score_loss(y_pair, p[pair])),
        "score_estimate": {
            "auc_points": 30.0 * max(0.0, (auc["value"] - 0.5) / 0.5),
            "specificity_points_somatoform_only": 20.0 * (1.0 - fpr_soma["value"]),
        },
        "by_control_format": by_format,
        "control_B_predictions": {k: float(v) for k, v in oof.loc[(cohort == COHORT_CONTROL) & (fam == "B"), "p"].items()},
        "median_p_by_cohort": {
            name: float(np.median(p[cohort == code])) for name, code in COHORT_CODES.items()
        },
    }


def permuted_cohort(
    cohort: npt.NDArray[np.int_], groups: npt.NDArray[np.int_], rng: np.random.Generator
) -> npt.NDArray[np.int_]:
    """Permute cohort labels between split groups; subjects of one group keep one label."""
    return permuted_labels(cohort, cohort.astype(str), groups, rng)[0]


def permuted_labels(
    cohort: npt.NDArray[np.int_], stratum: npt.NDArray[np.str_], groups: npt.NDArray[np.int_], rng: np.random.Generator
) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.str_]]:
    """Permute (cohort, stratum) pairs between split groups, keeping each pair intact."""
    uniq = np.unique(groups)
    first = np.array([np.flatnonzero(groups == g)[0] for g in uniq])
    order = rng.permutation(uniq.size)
    source = dict(zip(uniq, first[order]))
    idx = np.array([source[g] for g in groups])
    return cohort[idx], stratum[idx]


def permutation_test(
    data: TrainingData,
    observed_auc: float,
    n_perm: int,
    protocol: Protocol = PROTOCOL_1,
    seed: int = config.RANDOM_STATE,
    log: Path | None = None,
) -> dict[str, object]:
    """Null distribution of the nested OOF AUC (PTSD vs all controls) under permuted labels.

    Protocol 1 permutes cohorts (as in its registered run); protocol 2
    permutes (cohort, stratum) pairs between split groups.
    p-value = (1 + #{null AUC >= observed}) / (1 + n_perm).
    """
    rng = np.random.default_rng(seed)
    null: list[float] = []
    for i in range(n_perm):
        if protocol is PROTOCOL_1:
            cohort, stratum = permuted_cohort(data.cohort, data.groups, rng), data.stratum
        else:
            cohort, stratum = permuted_labels(data.cohort, data.stratum, data.groups, rng)
        oof = nested_oof(data, protocol, cohort=cohort, stratum=stratum)
        pair = (cohort == COHORT_PTSD) | (cohort == COHORT_CONTROL)
        null.append(float(roc_auc_score(cohort[pair] == COHORT_PTSD, oof["p"].to_numpy()[pair])))
        if log is not None:
            log.write_text(f"{i + 1}/{n_perm} permutations done\n", encoding="utf-8")
    null_arr = np.asarray(null)
    return {
        "observed_auc": observed_auc,
        "n_permutations": n_perm,
        "null_auc": null,
        "null_mean": float(null_arr.mean()),
        "null_q95": float(np.quantile(null_arr, 0.95)),
        "p_value": float((1 + np.sum(null_arr >= observed_auc)) / (1 + n_perm)),
    }


def summarize_strata(oof: pd.DataFrame, groups: npt.NDArray[np.int_]) -> dict[str, object]:
    """Protocol 2 report slices, fixed before the run.

    AUC with CI: PTSD vs all controls (task definition), vs controls of
    format B (original and supplement), vs the criterion negatives. P >= 0.5
    rates per stratum, with CI for somatoform and the supplement. The two
    original format-B controls are listed by name.
    """
    stratum, p = oof["stratum"].to_numpy(), oof["p"].to_numpy()
    thr = models.DECISION_THRESHOLD
    ptsd = stratum == models.STRATUM_PTSD

    def auc_ci(negative: Sequence[str]) -> dict[str, float]:
        mask = ptsd | np.isin(stratum, negative)
        return auc_with_ci(ptsd[mask].astype(int), p[mask], groups[mask])

    controls = [s for s in np.unique(stratum) if s.startswith("control_")]
    return {
        "n_by_stratum": {s: int(np.sum(stratum == s)) for s in np.unique(stratum)},
        "auc_ptsd_vs_all_controls": auc_ci(controls),
        "auc_ptsd_vs_control_B_all": auc_ci([models.STRATUM_CONTROL_B, models.STRATUM_CONTROL_B_SUPPLEMENT]),
        "auc_ptsd_vs_criterion_negatives": auc_ci(models.PROTOCOL2_AUC_NEGATIVES),
        "auc_ptsd_vs_stratum": {
            s: float(roc_auc_score(ptsd[ptsd | (stratum == s)], p[ptsd | (stratum == s)]))
            for s in np.unique(stratum)
            if s != models.STRATUM_PTSD
        },
        "sensitivity_ptsd": float(np.mean(p[ptsd] >= thr)),
        "rate_p_ge_0.5_by_stratum": {s: float(np.mean(p[stratum == s] >= thr)) for s in np.unique(stratum)},
        "specificity_somatoform": _specificity(p, stratum == models.STRATUM_SOMATOFORM, groups),
        "specificity_control_B_supplement": _specificity(p, stratum == models.STRATUM_CONTROL_B_SUPPLEMENT, groups),
        "control_B_original_predictions": {
            k: float(v) for k, v in oof.loc[stratum == models.STRATUM_CONTROL_B, "p"].items()
        },
        "median_p_by_stratum": {s: float(np.median(p[stratum == s])) for s in np.unique(stratum)},
        "candidate_frequency_per_outer_fold": candidate_frequency(oof) if "candidate" in oof else {},
    }


def _specificity(p: npt.NDArray[np.float64], mask: npt.NDArray[np.bool_], groups: npt.NDArray[np.int_]) -> dict[str, float]:
    if not mask.any():
        return {}
    fpr = rate_with_ci(p[mask] >= models.DECISION_THRESHOLD, groups[mask])
    return {"value": 1.0 - fpr["value"], "ci_low": 1.0 - fpr["ci_high"], "ci_high": 1.0 - fpr["ci_low"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Nested evaluation of the PTSD model")
    parser.add_argument("--protocol", choices=sorted(PROTOCOLS), default="protocol2")
    parser.add_argument("--n-perm", type=int, default=0, help="permutations of the whole procedure (0 = skip)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    protocol = PROTOCOLS[args.protocol]
    out = args.out or RESULTS_DIR / protocol.name
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    data = load_training_data(protocol=protocol)
    oof = nested_oof(data, protocol)
    oof.to_csv(out / "nested_oof.csv")
    meta = metadata_oof(data)
    meta.to_csv(out / "metadata_oof.csv")
    if protocol is PROTOCOL_1:
        metrics = summarize(oof, data.groups)
        metrics["candidate_frequency_per_outer_fold"] = candidate_frequency(oof)
        metrics["metadata_only_model"] = summarize(meta, data.groups)
        observed = metrics["auc_ptsd_vs_control"]["value"]
    else:
        metrics = summarize_strata(oof, data.groups)
        metrics["metadata_only_model"] = summarize_strata(meta, data.groups)
        observed = metrics["auc_ptsd_vs_all_controls"]["value"]
    metrics["runtime_s"] = round(time.time() - t0, 1)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{protocol.name}: done in {metrics['runtime_s']} s -> {out}")

    if args.n_perm:
        result = permutation_test(data, observed, args.n_perm, protocol, log=out / "permutation_progress.txt")
        (out / "permutation.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
        print(f"permutation p = {result['p_value']:.4f}")


if __name__ == "__main__":
    main()
