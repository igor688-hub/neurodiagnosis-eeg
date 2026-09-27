import numpy as np

from src import models, robustness


def _data(n: int = 12) -> models.TrainingData:
    stratum = np.array(["ptsd", "control_B_supplement", "control_A", "somatoform"] * (n // 4))
    cohort = np.where(stratum == "ptsd", models.COHORT_PTSD, np.where(stratum == "somatoform", models.COHORT_SOMATOFORM, models.COHORT_CONTROL))
    return models.TrainingData(
        subject_keys=tuple(f"s{i}" for i in range(n)), tables={"set": np.arange(n * 2, dtype=float).reshape(n, 2)},
        feature_names={"set": ("a", "b")}, cohort=cohort, groups=np.arange(n), stratum=stratum,
        export_family=np.array(["B"] * n), metadata=np.zeros((n, 5)), protocol="test",
    )


def test_subset_keeps_rows_aligned() -> None:
    data = _data()
    mask = data.stratum != "control_A"
    sub = robustness.subset(data, mask)

    assert len(sub.subject_keys) == mask.sum() == sub.tables["set"].shape[0] == sub.stratum.size
    assert "control_A" not in sub.stratum and sub.tables["set"][0, 0] == data.tables["set"][0, 0]


def test_permutation_moves_labels_only_inside_mask() -> None:
    data = _data(40)
    mask = np.isin(data.stratum, ["ptsd", "control_B_supplement"])
    cohort, stratum = robustness.permute_within(data, mask, np.random.default_rng(0))

    assert np.array_equal(stratum[~mask], data.stratum[~mask])
    assert sorted(stratum[mask]) == sorted(data.stratum[mask]) and not np.array_equal(stratum[mask], data.stratum[mask])
    assert np.array_equal(cohort == models.COHORT_PTSD, stratum == "ptsd")
