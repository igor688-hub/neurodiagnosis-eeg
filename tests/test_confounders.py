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


def test_age_prediction_recovers_linear_age_and_not_noise() -> None:
    rng = np.random.default_rng(3)
    n = 120
    age = rng.uniform(18, 60, n)
    x = rng.normal(size=(n, 5))
    x[:, 0] += (age - age.mean()) / 5.0
    x[5, 2] = np.nan
    groups = np.arange(n)

    pred, ref = confounders.age_prediction_oof(x, age, groups)
    noise_pred, _ = confounders.age_prediction_oof(rng.normal(size=(n, 5)), age, groups)

    assert np.corrcoef(pred, age)[0, 1] > 0.8
    assert np.mean(np.abs(pred - age)) < 0.6 * np.mean(np.abs(ref - age))
    assert np.mean((noise_pred - age) ** 2) > 0.95 * np.mean((ref - age) ** 2)


def test_schulte_times_drop_templated_and_incomplete_subjects(tmp_path) -> None:
    import pandas as pd

    rows = []
    for key, durations in {"g/full": [50, 40, 45, 42, 41], "g/templ": [41] * 5, "g/partial": [60, 50, None, 40, 45]}.items():
        rows.append({"subject_key": key, "stem": "T-П", "relpath": f"{key}/T-П.edf", "status": "ok", "active_duration_s": 61.0})
        for i, d in enumerate(durations, start=1):
            rows.append({"subject_key": key, "stem": f"T-{i}", "relpath": f"{key}/T-{i}.edf",
                         "status": "ok" if d is not None else "empty", "active_duration_s": d})

    out = confounders.schulte_times(pd.DataFrame(rows), tmp_path)

    assert out.loc["g/full", "trial1_s"] == 50 and out.loc["g/full", "total_s"] == 218
    assert np.isnan(out.loc["g/templ", "trial1_s"]) and np.isnan(out.loc["g/templ", "total_s"])
    assert out.loc["g/partial", "trial1_s"] == 60 and np.isnan(out.loc["g/partial", "total_s"])
    assert out.loc["g/partial", "n_trials"] == 4
