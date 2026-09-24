"""Validation mechanics: no group leakage, bootstrap behaviour, permutation, selection."""
import numpy as np
import pytest

from src import evaluation, models
from src.models import COHORT_CONTROL, COHORT_PTSD, COHORT_SOMATOFORM


def _toy(rng: np.random.Generator, n_groups: int = 60, signal: float = 1.5):
    """Subjects in groups of 1-2, three cohorts, one informative feature out of four."""
    sizes = rng.integers(1, 3, n_groups)
    groups = np.repeat(np.arange(n_groups), sizes)
    group_cohort = rng.choice([COHORT_CONTROL, COHORT_PTSD, COHORT_SOMATOFORM], n_groups, p=[0.45, 0.25, 0.30])
    cohort = group_cohort[groups]
    x = rng.normal(size=(groups.size, 4))
    x[:, 0] += signal * (cohort == COHORT_PTSD)
    return x, cohort, groups


def test_outer_folds_never_share_a_group() -> None:
    groups = np.array([0, 0, 1, 2, 2, 3])
    folds = evaluation.outer_folds(groups)

    assert len(folds) == 4
    for train, test in folds:
        assert not set(groups[train]) & set(groups[test])
        assert len(set(groups[test])) == 1
    assert sorted(np.concatenate([test for _, test in folds])) == list(range(groups.size))


def test_inner_splits_respect_groups() -> None:
    rng = np.random.default_rng(0)
    _, cohort, groups = _toy(rng)
    for train, test in models.inner_splits(cohort, groups):
        assert not set(groups[train]) & set(groups[test])


def test_bootstrap_ci_brackets_the_truth() -> None:
    rng = np.random.default_rng(1)
    groups = np.arange(200)
    y = (rng.random(200) < 0.3).astype(int)

    perfect = evaluation.auc_with_ci(y, y + 0.1 * rng.random(200), groups, n_boot=300)
    random = evaluation.auc_with_ci(y, rng.random(200), groups, n_boot=300)

    assert perfect["value"] == 1.0 and perfect["ci_low"] == pytest.approx(1.0)
    assert random["ci_low"] < 0.5 < random["ci_high"]


def test_permutation_keeps_groups_and_cohort_counts() -> None:
    rng = np.random.default_rng(2)
    _, cohort, groups = _toy(rng)

    permuted = evaluation.permuted_cohort(cohort, groups, np.random.default_rng(3))

    for g in np.unique(groups):
        assert len(set(permuted[groups == g])) == 1
    by_group = lambda c: sorted(c[np.unique(groups, return_index=True)[1]])  # noqa: E731
    assert by_group(permuted) == by_group(cohort)


def test_selection_score_formula() -> None:
    cohort = np.array([COHORT_PTSD, COHORT_PTSD, COHORT_CONTROL, COHORT_CONTROL, COHORT_SOMATOFORM, COHORT_SOMATOFORM])
    p = np.array([0.9, 0.8, 0.2, 0.1, 0.7, 0.3])  # AUC 1.0, one of two somatoform at >= 0.5

    assert models.selection_score(p, cohort) == pytest.approx(30.0 + 20.0 * 0.5)


def test_selection_is_deterministic_and_finds_signal() -> None:
    rng = np.random.default_rng(4)
    x, cohort, groups = _toy(rng, n_groups=80, signal=2.0)
    tables = {"native": x, "requantized": x + 0.01 * rng.normal(size=x.shape)}

    best1, scores1 = models.select_candidate(tables, cohort, groups)
    best2, scores2 = models.select_candidate(tables, cohort, groups, stratum=cohort.astype(str))

    assert best1 == best2 and scores1 == scores2
    assert scores1[best1.key] > 30.0  # well above chance: AUC points > 15 and specificity points > 15


def test_protocol2_score_uses_explicit_groups() -> None:
    stratum = np.array(["ptsd", "ptsd", "control_B", "control_B_supplement", "control_A", "somatoform", "control_C"])
    cohort = np.array([COHORT_PTSD, COHORT_PTSD, COHORT_CONTROL, COHORT_CONTROL, COHORT_CONTROL, COHORT_SOMATOFORM, COHORT_CONTROL])
    p = np.array([0.9, 0.8, 0.6, 0.1, 0.2, 0.3, 0.95])  # control_C ranks above PTSD but is outside the criterion

    score = models.selection_score_protocol2(p, cohort, stratum)

    # AUC over {A, B, B supplement, somatoform} = 1; FPR: A 0, B group 1/2, somatoform 0 -> mean 1/6
    assert score == pytest.approx(30.0 + 20.0 * (1.0 - 1.0 / 6.0))


def test_permuted_labels_keep_pairs_and_groups() -> None:
    rng = np.random.default_rng(5)
    _, cohort, groups = _toy(rng)
    stratum = np.where(cohort == COHORT_PTSD, "ptsd", np.where(cohort == COHORT_SOMATOFORM, "somatoform", "control_A"))

    c2, s2 = evaluation.permuted_labels(cohort, stratum, groups, np.random.default_rng(1))

    assert all(len(set(c2[groups == g])) == 1 for g in np.unique(groups))
    assert set(zip(c2, s2)) <= set(zip(cohort, stratum))
