"""Project-wide constants.

Every value here is either fixed by the task specification (channels, file
naming, groups) or read from the EDF headers of the training set (passband).
Analysis parameters (bands, window length, rejection thresholds) are added
by the modules that introduce them, together with their rationale.
"""
from pathlib import Path
from typing import Final

REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
DATA_DIR: Final[Path] = REPO_ROOT / "data"
CACHE_DIR: Final[Path] = REPO_ROOT / "cache"

# NeuroPlay-6C ring montage, ear-clip reference. Order in the EDF files is not
# alphabetical, so channels are always selected and reordered by name.
CHANNELS: Final[tuple[str, ...]] = ("O1", "T3", "Fp1", "Fp2", "T4", "O2")
N_CHANNELS: Final[int] = len(CHANNELS)

# Observed sampling rates are 123-127 Hz; every record is resampled to this rate.
TARGET_SFREQ: Final[float] = 125.0

# Hardware prefilter reported in all non-empty headers ("HP:2.0Hz LP:40.0Hz").
# Content outside this band is attenuated at acquisition and cannot be restored.
HEADER_HIGHPASS_HZ: Final[float] = 2.0
HEADER_LOWPASS_HZ: Final[float] = 40.0

# File stems: "T-П" is eyes-closed rest, "T-1".."T-5" are Schulte table trials
# (eyes open) recorded as separate sessions with unknown gaps between them.
REST_STEM: Final[str] = "T-П"
TASK_STEMS: Final[tuple[str, ...]] = ("T-1", "T-2", "T-3", "T-4", "T-5")
RECORD_STEMS: Final[tuple[str, ...]] = (REST_STEM, *TASK_STEMS)

# Top-level folder name -> cohort. The target is binary: PTSD vs any other cohort.
GROUP_CONTROL: Final[str] = "Норма"
GROUP_PTSD: Final[str] = "ПТСР"
GROUP_SOMATOFORM: Final[str] = "Соматоформные"
GROUPS: Final[tuple[str, ...]] = (GROUP_CONTROL, GROUP_PTSD, GROUP_SOMATOFORM)

RANDOM_STATE: Final[int] = 42
