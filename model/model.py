"""
PTSD probability prediction model.

Signatures of following functions should NOT be changed:
    load(path: Path) -> model
    predict(model, subject_dir: Path) -> float

Do NOT modify following functions:
    run(model, input_dir: Path, output_path: Path) -> None

Other functions may be changed as you wish.

Usage
-----
Inference with the committed weights (no training, no scikit-learn state)::

    from model.model import load, run
    model = load(Path("model/weights"))
    run(model, input_dir=Path("test_data"), output_path=Path("predictions.csv"))

Re-training from the raw training set (deterministic)::

    python model/model.py --data-dir data --out model/weights
"""
import argparse
import csv
import platform
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).parent
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:  # signal processing lives in <repo>/src
    sys.path.insert(0, str(REPO_ROOT))

from src import config, dataset, features, models  # noqa: E402
from src.models import LogisticModel  # noqa: E402

WEIGHTS_FILE = "model.json"


def train(data_dir: Path = config.DATA_DIR, *args: Any, **kwargs: Any) -> LogisticModel:
    """Trains model on every training subject of ``data_dir``.

    The procedure is the one evaluated by the nested cross-validation of
    ``src.evaluation``: candidate selection (C, negative class, preprocessing
    variant) by grouped inner cross-validation with the pre-declared
    criterion, then a fit of the chosen candidate on all training subjects.

    Controls aged ``config.HOLDOUT_MIN_AGE`` or older are the held-out ageing
    test and are excluded. Files whose recording condition is ambiguous
    (identical content under rest and task names) are excluded from that
    condition. No randomness is involved: re-training gives the same weights.
    """
    data = models.load_training_data(data_dir)
    best, scores = models.select_candidate(data.tables, data.cohort, data.groups)
    pipeline = models.fit_candidate(data.tables[best.features], data.cohort, best)
    trained_on = models.training_mask(data.cohort, best.negatives)
    metadata = {
        "candidate": best.key,
        "C": best.c,
        "negatives": best.negatives,
        "preprocessing_variant": best.features,
        "class_weight": models.BASELINE_CLASS_WEIGHT,
        "selection_scores": {key: round(value, 6) for key, value in scores.items()},
        "n_subjects_trained": int(trained_on.sum()),
        "n_ptsd": int(data.y.sum()),
        "python": platform.python_version(),
    }
    return LogisticModel.from_pipeline(pipeline, features.FEATURE_NAMES, metadata)


def predict(model, subject_dir: Path) -> float:
    """Predicts PTSD probability of the subject.

    Never raises for bad input: missing, empty or unreadable files are
    skipped, and a subject without usable data gets the prediction for
    all-missing features (training medians), which is finite.
    """
    subject_dir = Path(subject_dir)
    try:
        files = dataset.find_record_files(subject_dir, strict=False) if subject_dir.is_dir() else {}
        cfg = features.PREPROCESSING_VARIANTS[str(model.metadata.get("preprocessing_variant", "native"))]
        values = features.extract_subject_features(files, cfg)
    except Exception as err:  # any failure degrades to missing features, not to a crash
        warnings.warn(f"{subject_dir}: features unavailable ({err!r}); using training medians")
        values = {}
    x = np.array([[values.get(name, np.nan) for name in model.feature_names]], dtype=float)
    probability = float(model.predict_proba(x)[0])
    return float(np.clip(probability, 0.0, 1.0)) if np.isfinite(probability) else 0.5


def run(model, input_dir: Path , output_path: Path) -> None:
    """Runs model over all subjects in `input_dir` and saves results to `output_path`."""
    predictions = []
    for subject_dir in input_dir.iterdir():
        subject_id = subject_dir.name
        ptsd_probability = predict(model, subject_dir)
        predictions.append((subject_id, ptsd_probability))

    predictions.sort(key=lambda x: x[0])
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["subject_id", "ptsd_probability"])
        writer.writerows(predictions)


def save(model, path: Path):
    """Saves model to disk: ``path`` is a directory, weights go to ``path/model.json``."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.to_json(path / WEIGHTS_FILE)


def load(path: Path):
    """Loads model from disk: ``path`` is the weights directory or the JSON file itself."""
    path = Path(path)
    return LogisticModel.from_json(path / WEIGHTS_FILE if path.is_dir() else path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the PTSD model on the raw training set")
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--out", type=Path, default=ROOT / "weights")
    args = parser.parse_args()
    save(train(args.data_dir), args.out)
    print(f"saved {args.out / WEIGHTS_FILE}")
