from pathlib import Path
from typing import Final

REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
DATA_DIR: Final[Path] = REPO_ROOT / "data"
CACHE_DIR: Final[Path] = REPO_ROOT / "cache"

CHANNELS: Final[tuple[str, ...]] = ("O1", "T3", "Fp1", "Fp2", "T4", "O2")
N_CHANNELS: Final[int] = len(CHANNELS)

TARGET_SFREQ: Final[float] = 125.0

HEADER_HIGHPASS_HZ: Final[float] = 2.0
HEADER_LOWPASS_HZ: Final[float] = 40.0

REST_STEM: Final[str] = "T-П"
TASK_STEMS: Final[tuple[str, ...]] = ("T-1", "T-2", "T-3", "T-4", "T-5")
RECORD_STEMS: Final[tuple[str, ...]] = (REST_STEM, *TASK_STEMS)

GROUP_CONTROL: Final[str] = "Норма"
GROUP_PTSD: Final[str] = "ПТСР"
GROUP_SOMATOFORM: Final[str] = "Соматоформные"
GROUPS: Final[tuple[str, ...]] = (GROUP_CONTROL, GROUP_PTSD, GROUP_SOMATOFORM)

HOLDOUT_MIN_AGE: Final[float] = 65.0

RANDOM_STATE: Final[int] = 42
