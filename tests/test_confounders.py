"""Diagnostics of a trained procedure: contributions, masking, restricted protocols, fold AUCs."""
import numpy as np

from src import confounders, evaluation, models
from src.models import COHORT_CONTROL, COHORT_PTSD, COHORT_SOMATOFORM


def test_contributions_sum_to_logit() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 3))
    y = (x[:, 0] > 0).astype(int)
    model = models.fit_logistic(x, y, ("a", "b", "c"))
    x[0, 1] = np.nan

    contrib = confounders.feature_contributions(model, x)

    np.testing.assert_allclose(contrib.sum(axis=1) + model.intercept, model.decision_function(x))


def test_restricted_protocol_keeps_only_given_sets() -> None:
    restricted = models.restricted_protocol(models.PROTOCOL_2, ["rest_O1Fp"], "only_core")

    assert set(restricted.feature_sets) == {"rest_O1Fp"}
    assert {c.features for c in restricted.candidates} == {"rest_O1Fp"}
    assert len(restricted.candidates) == len(models.C_GRID) * len(models.NEGATIVES)


def _toy_data(rng: np.random.Generator) -> models.TrainingData:
    n_groups = 90
    groups = np.arange(n_groups)
    cohort = rng.choice([COHORT_CONTROL, COHORT_PTSD, COHORT_SOMATOFORM], n_groups, p=[0.45, 0.25, 0.30])
    names_of = {COHORT_CONTROL: "control_B_supplement", COHORT_PTSD: "ptsd", COHORT_SOMATOFORM: "somatoform"}
    stratum = np.array([names_of[c] for c in cohort])
    x = rng.normal(size=(n_groups, 3))
    x[:, 0] += 2.5 * (cohort == COHORT_PTSD)
    names = ("signal", "noise1", "noise2")
    return models.TrainingData(
        subject_keys=tuple(f"s{i}" for i in range(n_groups)), tables={"set": x}, feature_names={"set": names},
        cohort=cohort, groups=groups, stratum=stratum, export_family=np.array(["B"] * n_groups),
        metadata=np.zeros((n_groups, 5)), protocol="toy",
    )


def _toy_protocol() -> models.Protocol:
    return models.Protocol(
        name="toy", feature_sets={"set": ("native", ("signal", "noise1", "noise2"))},
        candidates=(models.Candidate(1.0, "control+somatoform", "set"),),
        score=models.selection_score_protocol2, include_rest_only=True, stratify_inner_by_stratum=True,
    )


def test_masking_the_informative_feature_removes_discrimination() -> None:
    data = _toy_data(np.random.default_rng(1))
    ptsd = data.cohort == COHORT_PTSD
    from sklearn.metrics import roc_auc_score

    full = evaluation.nested_oof(data, _toy_protocol(), n_jobs=1)["p"].to_numpy()
    masked = evaluation.nested_oof(data, _toy_protocol(), n_jobs=1, mask_at_test=["signal"])["p"].to_numpy()

    assert roc_auc_score(ptsd, full) > 0.85
    assert roc_auc_score(ptsd, masked) < 0.7


def test_fold_aucs_shape_and_range() -> None:
    data = _toy_data(np.random.default_rng(2))
    table = evaluation.fold_aucs(data, _toy_protocol(), {"vs_controls": ["control_B_supplement"]}, n_splits=3, n_repeats=2, n_jobs=1)

    assert len(table) == 6 and table["vs_controls"].between(0, 1).all()
