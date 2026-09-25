"""Portable logistic model: numpy inference equals the fitted scikit-learn pipeline."""
from pathlib import Path

import numpy as np

from src import models


def _data(rng: np.random.Generator, n: int = 80, p: int = 5) -> tuple[np.ndarray, np.ndarray]:
    x = rng.normal(size=(n, p))
    y = (x[:, 0] + 0.5 * rng.normal(size=n) > 0.8).astype(int)
    x[rng.random(size=x.shape) < 0.1] = np.nan  # missing features
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
    assert calibration[0] > 0  # monotone: ranking and AUC are preserved
    assert np.all(np.diff(p[np.argsort(raw)]) >= -1e-12)
    assert abs(p.mean() - y.mean()) < 0.02  # calibrated to the class mix of the fit

    calibrated.to_json(tmp_path / "m.json")
    loaded = models.LogisticModel.from_json(tmp_path / "m.json")
    np.testing.assert_array_equal(loaded.predict_proba(x), p)


def test_balanced_platt_centres_threshold() -> None:
    rng = np.random.default_rng(4)
    y = (rng.random(400) < 0.12).astype(int)  # rare positive class, as in the data
    raw = 1.0 / (1.0 + np.exp(-(1.5 * (y - 0.5) + rng.normal(0, 1, 400))))

    plain = models.apply_platt(raw, models.fit_platt(raw, y))
    balanced = models.apply_platt(raw, models.fit_platt(raw, y, "balanced"))

    sens = lambda p: np.mean(p[y == 1] >= 0.5)  # noqa: E731
    assert sens(plain) < 0.4 < sens(balanced)  # prevalence-calibrated: few positives above 0.5
    assert 0.5 < sens(balanced) < 0.95 and np.mean(balanced[y == 0] < 0.5) > 0.5
