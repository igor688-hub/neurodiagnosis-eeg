from pathlib import Path

import numpy as np

from src import models


def _data(rng: np.random.Generator, n: int = 80, p: int = 5) -> tuple[np.ndarray, np.ndarray]:
    x = rng.normal(size=(n, p))
    y = (x[:, 0] + 0.5 * rng.normal(size=n) > 0.8).astype(int)
    x[rng.random(size=x.shape) < 0.1] = np.nan
    return x, y


def test_numpy_inference_matches_pipeline() -> None:
    rng = np.random.default_rng(0)
    x, y = _data(rng)
    names = tuple(f"f{i}" for i in range(x.shape[1]))

    pipeline = models.make_pipeline().fit(x, y)
    model = models.LogisticModel.from_pipeline(pipeline, names)

    np.testing.assert_allclose(model.predict_proba(x), pipeline.predict_proba(x)[:, 1], rtol=1e-10)


def test_json_roundtrip(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    x, y = _data(rng)
    model = models.fit_logistic(x, y, tuple(f"f{i}" for i in range(x.shape[1])), metadata={"note": "test"})

    model.to_json(tmp_path / "m.json")
    loaded = models.LogisticModel.from_json(tmp_path / "m.json")

    np.testing.assert_array_equal(loaded.predict_proba(x), model.predict_proba(x))
    assert loaded.metadata["note"] == "test" and loaded.feature_names == model.feature_names


def test_all_missing_features_give_finite_probability() -> None:
    rng = np.random.default_rng(2)
    x, y = _data(rng)
    model = models.fit_logistic(x, y, tuple(f"f{i}" for i in range(x.shape[1])))

    p = model.predict_proba(np.full((1, x.shape[1]), np.nan))

    assert np.isfinite(p).all() and 0.0 < p[0] < 1.0


def test_platt_calibration_roundtrip_and_monotonicity(tmp_path: Path) -> None:
    import dataclasses

    rng = np.random.default_rng(3)
    x, y = _data(rng)
    model = models.fit_logistic(x, y, tuple(f"f{i}" for i in range(x.shape[1])))
    raw = model.predict_proba_raw(x)
    calibration = models.fit_platt(raw, y)
    calibrated = dataclasses.replace(model, calibration=calibration)

    p = calibrated.predict_proba(x)
    np.testing.assert_allclose(p, models.apply_platt(raw, calibration), rtol=1e-9)
    assert calibration[0] > 0
    assert np.all(np.diff(p[np.argsort(raw)]) >= -1e-12)
    assert abs(p.mean() - y.mean()) < 0.02

    calibrated.to_json(tmp_path / "m.json")
    loaded = models.LogisticModel.from_json(tmp_path / "m.json")
    np.testing.assert_array_equal(loaded.predict_proba(x), p)


def test_balanced_platt_centres_threshold() -> None:
    rng = np.random.default_rng(4)
    y = (rng.random(400) < 0.12).astype(int)
    raw = 1.0 / (1.0 + np.exp(-(1.5 * (y - 0.5) + rng.normal(0, 1, 400))))

    plain = models.apply_platt(raw, models.fit_platt(raw, y))
    balanced = models.apply_platt(raw, models.fit_platt(raw, y, "balanced"))

    sens = lambda p: np.mean(p[y == 1] >= 0.5)  # noqa: E731
    assert sens(plain) < 0.4 < sens(balanced)
    assert 0.5 < sens(balanced) < 0.95 and np.mean(balanced[y == 0] < 0.5) > 0.5


def test_restricted_protocol_keeps_calibration_settings() -> None:
    restricted = models.restricted_protocol(models.PROTOCOL_7, ["rest_O1Fp"], "only_core")

    assert restricted.calibrate and restricted.calibration_class_weight == "balanced"
    assert restricted.score is models.PROTOCOL_7.score and set(restricted.feature_sets) == {"rest_O1Fp"}
    assert all(c.features == "rest_O1Fp" for c in restricted.candidates)


def test_equal_error_shift_balances_sensitivity_and_specificity() -> None:
    rng = np.random.default_rng(8)
    stratum = np.array(["ptsd"] * 40 + ["somatoform"] * 60 + ["control_A"] * 60)
    y = (stratum == "ptsd").astype(int)
    p = 1 / (1 + np.exp(-(1.2 * y + rng.normal(0, 1, y.size) + 0.8)))

    t = models.equal_error_shift(p, y, stratum)
    logit = np.log(p / (1 - p))
    sens, spec = np.mean(logit[y == 1] >= t), np.mean(logit[stratum == "somatoform"] < t)

    assert abs(sens - spec) < 0.05 and np.mean(p[y == 1] >= 0.5) > sens


def test_specificity_weight_changes_fit_and_key() -> None:
    rng = np.random.default_rng(9)
    stratum = np.array(["ptsd"] * 30 + ["somatoform"] * 30 + ["control_A"] * 30)
    cohort = np.where(stratum == "ptsd", models.COHORT_PTSD, np.where(stratum == "somatoform", models.COHORT_SOMATOFORM, models.COHORT_CONTROL))
    x = rng.normal(size=(90, 3)) + (stratum == "ptsd")[:, None]
    plain = models.Candidate(1.0, "control+somatoform", "set")
    weighted = models.Candidate(1.0, "control+somatoform", "set", 3.0)

    a = models.fit_candidate(x, cohort, plain, stratum).named_steps["clf"].coef_
    b = models.fit_candidate(x, cohort, weighted, stratum).named_steps["clf"].coef_

    assert plain.key.endswith("feat=set") and weighted.key.endswith("|w=3")
    assert not np.allclose(a, b)
