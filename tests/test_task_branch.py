from pathlib import Path

import numpy as np

from src import task_branch
from src.preprocessing import EpochedRecord
from tests.test_dataset import _six_channel_signals, _spr, write_edf


def _write_subject(folder: Path, trial_seconds: list[int], rng: np.random.Generator) -> dict[str, Path]:
    folder.mkdir()
    files = {}
    for stem, seconds in [("T-П", 60), *zip(["T-1", "T-2", "T-3", "T-4", "T-5"], trial_seconds)]:
        signals = _six_channel_signals(rng, seconds)
        for ch in ("O1", "T3", "Fp1", "Fp2", "T4", "O2"):
            signals[ch] = rng.normal(0, 150, seconds * 125).astype(np.int16)
        files[stem] = folder / f"{stem}.edf"
        write_edf(files[stem], signals, _spr(signals))
    return files


def test_behaviour_features_and_templated_durations(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    real = task_branch.subject_task_features(_write_subject(tmp_path / "real", [50, 45, 44, 40, 41], rng))
    templated = task_branch.subject_task_features(_write_subject(tmp_path / "tpl", [41] * 5, rng))

    assert real["beh_log_duration_trial1"] == np.log10(50) and real["beh_templated"] == 0.0
    assert real["beh_log_duration_slope"] < 0 and real["beh_first_trial_excess"] > 0
    assert templated["beh_templated"] == 1.0 and np.isnan(templated["beh_log_duration_trial1"])
    assert np.isfinite(real["eeg_trial1_alpha_O1"]) and np.isfinite(real["eeg_slope_alpha_O1"])


def test_first_segment_is_time_ordered_prefix() -> None:
    windows = np.arange(20 * 6 * 500, dtype=float).reshape(20, 6, 500)
    epoched = EpochedRecord(windows, np.zeros((20, 6), dtype=np.uint8), 125.0)

    segment = task_branch.first_segment(epoched)

    assert segment.n_windows == task_branch.SEGMENT_WINDOWS
    np.testing.assert_array_equal(segment.windows, windows[:9])


def test_paired_auc_difference() -> None:
    rng = np.random.default_rng(1)
    y = (rng.random(300) < 0.3).astype(int)
    good = y + rng.normal(0, 0.5, 300)
    noise = rng.normal(size=300)

    diff = task_branch.paired_auc_difference(y, good, noise, np.arange(300), n_boot=300)
    same = task_branch.paired_auc_difference(y, good, good, np.arange(300), n_boot=300)

    assert diff["ci_low"] > 0.2
    assert same["value"] == same["ci_low"] == same["ci_high"] == 0.0
