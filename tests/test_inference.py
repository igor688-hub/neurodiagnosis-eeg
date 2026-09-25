"""End-to-end inference through model/model.py on synthetic subject folders."""
import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from src import config, features, models
from tests.test_dataset import _six_channel_signals, _spr, write_edf

MODEL_PY = config.REPO_ROOT / "model" / "model.py"


@pytest.fixture(scope="module")
def model_module():
    spec = importlib.util.spec_from_file_location("organizer_model", MODEL_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def toy_model() -> models.LogisticModel:
    p = len(features.FEATURE_NAMES)
    rng = np.random.default_rng(0)
    return models.LogisticModel(
        feature_names=features.FEATURE_NAMES,
        impute_values=np.zeros(p), mean=np.zeros(p), scale=np.ones(p),
        coef=rng.normal(0, 0.1, p), intercept=-1.0,
    )


def _subject(root: Path, name: str, stems: list[str], n_records: int = 30) -> Path:
    folder = root / name
    folder.mkdir()
    rng = np.random.default_rng(len(name))
    for stem in stems:
        signals = _six_channel_signals(rng, n_records)
        for label in config.CHANNELS:  # ~9 uV white noise: passes the 400 uV amplitude ceiling
            signals[label] = rng.normal(0, 150, n_records * 125).astype(np.int16)
        write_edf(folder / f"{stem}.edf", signals, _spr(signals))
    return folder


def test_run_handles_names_and_broken_input(tmp_path: Path, model_module, toy_model) -> None:
    data = tmp_path / "test_data"
    data.mkdir()
    _subject(data, "latin", ["T-П", "T-1", "T-2", "T-3", "T-4", "T-5"])
    _subject(data, "cyrillic", ["Т-П", "Т-1"])  # Cyrillic Т, only two records
    (_subject(data, "empty_files", []) / "T-П.edf").write_bytes(b"")
    (data / "no_files").mkdir()
    (data / "stray.txt").write_text("not a subject")

    out = tmp_path / "pred.csv"
    model_module.run(toy_model, data, out)

    with open(out, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    probs = {r["subject_id"]: float(r["ptsd_probability"]) for r in rows}
    assert set(probs) == {"latin", "cyrillic", "empty_files", "no_files", "stray.txt"}
    assert all(np.isfinite(p) and 0.0 <= p <= 1.0 for p in probs.values())
    # Cyrillic names are read: features differ from the all-missing fallback.
    assert probs["cyrillic"] != probs["no_files"]
    assert probs["empty_files"] == probs["no_files"] == probs["stray.txt"]


def test_save_load_roundtrip(tmp_path: Path, model_module, toy_model) -> None:
    model_module.save(toy_model, tmp_path / "weights")
    loaded = model_module.load(tmp_path / "weights")

    np.testing.assert_array_equal(loaded.coef, toy_model.coef)


@pytest.mark.skipif(not (MODEL_PY.parent / "weights" / "model.json").exists(), reason="weights not trained")
def test_committed_weights_match_feature_extractor(model_module) -> None:
    model = model_module.load(MODEL_PY.parent / "weights")

    assert set(model.feature_names) <= set(features.FEATURE_NAMES)
    assert np.isfinite(model.coef).all() and np.all(model.scale > 0)


@pytest.mark.skipif(not (config.DATA_DIR / config.GROUP_PTSD).is_dir(), reason="training data not downloaded")
def test_training_and_inference_features_are_identical() -> None:
    from src import dataset

    registry, _, subjects = dataset.scan_dataset()
    keep = subjects.index[~subjects["holdout"]]
    table = features.build_feature_table(registry[registry["subject_key"].isin(keep)])

    for subject_key in table.index:
        files = dataset.find_record_files(config.DATA_DIR / subject_key, strict=False)
        inferred = features.extract_subject_features(files)
        np.testing.assert_array_equal(
            np.array([inferred[name] for name in features.FEATURE_NAMES]), table.loc[subject_key].to_numpy(dtype=float)
        )


@pytest.mark.skipif(not (config.DATA_DIR / config.GROUP_PTSD).is_dir(), reason="training data not downloaded")
def test_task_features_identical_in_training_and_inference() -> None:
    from src import dataset, task_branch

    registry, _, _ = dataset.scan_dataset()
    keys = ["ПТСР/34PHG", "Норма/КС010_19", "Соматоформные/DFGH8", "Соматоформные/XYKT7"]
    table = task_branch.build_task_table(registry[registry["subject_key"].isin(keys)])

    for key in keys:
        inferred = task_branch.subject_task_features(dataset.find_record_files(config.DATA_DIR / key, strict=False))
        np.testing.assert_array_equal(
            np.array([inferred[c] for c in table.columns], dtype=float), table.loc[key].to_numpy(dtype=float)
        )
