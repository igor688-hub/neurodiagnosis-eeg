from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.signal import periodogram

from src import config
from src.dataset import EdfFormatError, resolve_ambiguous_records
from src.preprocessing import EpochedRecord, PreprocessingConfig, preprocess_file

EULER_GAMMA: Final[float] = 0.5772156649015329
LOG10_BIAS: Final[float] = EULER_GAMMA / np.log(10.0)
MIN_GOOD_WINDOWS: Final[int] = 5
MIN_TRIAL_WINDOWS: Final[int] = 3


def window_psd(
    windows: npt.NDArray[np.float64], sfreq: float
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Hann-tapered periodogram of every window."""
    return periodogram(windows, fs=sfreq, window="hann", detrend=False, scaling="density", axis=-1)


def log_mean_spectrum(psd: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Bias-corrected geometric mean over the first axis."""
    return 10.0 ** (np.log10(psd).mean(axis=0) + LOG10_BIAS)


def condition_spectrum(
    records: Sequence[EpochedRecord],
    n_channels: int = config.N_CHANNELS,
    n_samples: int = PreprocessingConfig().window_samples,
    sfreq: float = config.TARGET_SFREQ,
    min_windows: int = MIN_GOOD_WINDOWS,
    min_record_windows: int = 1,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Log-mean PSD of one condition pooled over records, per channel."""
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sfreq)
    per_record: list[npt.NDArray[np.float64]] = []
    n_good = np.zeros(n_channels, dtype=int)
    for epoched in records:
        if epoched.n_windows == 0:
            continue
        _, psd = window_psd(epoched.windows, epoched.sfreq)
        good = epoched.good
        mean_log = np.full((n_channels, freqs.size), np.nan)
        with np.errstate(divide="ignore"):
            for ch in range(n_channels):
                if good[:, ch].sum() >= max(1, min_record_windows):
                    mean_log[ch] = np.log10(psd[good[:, ch], ch]).mean(axis=0)
                    n_good[ch] += good[:, ch].sum()
        per_record.append(mean_log)
    spectrum = np.full((n_channels, freqs.size), np.nan)
    if per_record:
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            pooled = np.nanmean(np.stack(per_record), axis=0)
        enough = n_good >= min_windows
        spectrum[enough] = 10.0 ** (pooled[enough] + LOG10_BIAS)
    return freqs, spectrum


def record_spectrum(
    epoched: EpochedRecord, min_windows: int = MIN_GOOD_WINDOWS
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Log-mean PSD over the retained windows of one record, per channel."""
    n_windows, n_channels, n_samples = epoched.windows.shape
    return condition_spectrum([epoched], n_channels, n_samples, epoched.sfreq, min_windows)


@dataclass(frozen=True)
class SpectraTable:
    """Record spectra of many files, aligned with ``relpaths``."""

    relpaths: tuple[str, ...]
    freqs: npt.NDArray[np.float64]
    spectra: npt.NDArray[np.float64]
    n_good: npt.NDArray[np.int64]

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path, relpaths=np.array(self.relpaths), freqs=self.freqs, spectra=self.spectra, n_good=self.n_good
        )

    @classmethod
    def load(cls, path: Path) -> SpectraTable:
        with np.load(path) as f:
            return cls(tuple(f["relpaths"].tolist()), f["freqs"], f["spectra"], f["n_good"])


def compute_spectra(
    registry: pd.DataFrame, cfg: PreprocessingConfig = PreprocessingConfig(), data_dir: Path = config.DATA_DIR
) -> SpectraTable:
    """``record_spectrum`` for every readable file of the registry."""
    relpaths = tuple(registry.loc[registry["status"] == "ok", "relpath"])
    spectra, n_good = [], []
    freqs = np.fft.rfftfreq(cfg.window_samples, d=1.0 / cfg.target_sfreq)
    for relpath in relpaths:
        epoched = preprocess_file(data_dir / relpath, cfg)
        _, spectrum = record_spectrum(epoched)
        spectra.append(spectrum)
        n_good.append(epoched.good.sum(axis=0))
    return SpectraTable(relpaths, freqs, np.stack(spectra), np.stack(n_good).astype(np.int64))


BANDS: Final[dict[str, tuple[float, float]]] = {"theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0)}
TOTAL_BAND: Final[tuple[float, float]] = (4.0, 30.0)

BACKGROUND_FIT_HZ: Final[tuple[float, float]] = (3.0, 30.0)
BACKGROUND_EXCLUDE_HZ: Final[tuple[float, float]] = (7.0, 14.0)
ALPHA_SEARCH_HZ: Final[tuple[float, float]] = (7.0, 13.0)
MIN_PEAK_LOG10: Final[float] = 0.3
CENTROID_HALF_WIDTH_HZ: Final[float] = 1.0

OCCIPITAL: Final[tuple[str, ...]] = ("O1", "O2")

BANDS_20: Final[dict[str, tuple[float, float]]] = {"theta": (4.0, 8.0), "alpha": (8.0, 13.0), "lowbeta": (13.0, 20.0)}
TOTAL_BAND_20: Final[tuple[float, float]] = (4.0, 20.0)
BACKGROUND_FIT_20_HZ: Final[tuple[float, float]] = (3.0, 20.0)

PREPROCESSING_VARIANTS: Final[dict[str, PreprocessingConfig]] = {
    "native": PreprocessingConfig(),
    "requantized": PreprocessingConfig(requantize_step_uv=1.0),
}


def band_power(
    freqs: npt.NDArray[np.float64], spectrum: npt.NDArray[np.float64], band: tuple[float, float]
) -> npt.NDArray[np.float64]:
    """Integrated power in [lo, hi)."""
    lo, hi = band
    df = freqs[1] - freqs[0]
    return spectrum[..., (freqs >= lo) & (freqs < hi)].sum(axis=-1) * df


def log_relative_powers(
    freqs: npt.NDArray[np.float64],
    spectrum: npt.NDArray[np.float64],
    bands: Mapping[str, tuple[float, float]] = BANDS,
    total_band: tuple[float, float] = TOTAL_BAND,
) -> dict[str, npt.NDArray[np.float64]]:
    """log10 of band power over the power of ``total_band`` (default 4-30 Hz), per band."""
    total = band_power(freqs, spectrum, total_band)
    return {name: np.log10(band_power(freqs, spectrum, band) / total) for name, band in bands.items()}


def background_fit(
    freqs: npt.NDArray[np.float64],
    log_spectrum: npt.NDArray[np.float64],
    fit_range: tuple[float, float] = BACKGROUND_FIT_HZ,
) -> tuple[float, float]:
    """Least-squares line log10 S = b - chi * log10 f on ``fit_range`` (default 3-30 Hz) without 7-14 Hz."""
    fit = (freqs >= fit_range[0]) & (freqs <= fit_range[1])
    fit &= ~((freqs >= BACKGROUND_EXCLUDE_HZ[0]) & (freqs <= BACKGROUND_EXCLUDE_HZ[1]))
    if not np.isfinite(log_spectrum[fit]).all():
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(np.log10(freqs[fit]), log_spectrum[fit], 1)
    return float(-slope), float(intercept)


def alpha_peak(
    freqs: npt.NDArray[np.float64],
    log_spectrum: npt.NDArray[np.float64],
    fit_range: tuple[float, float] = BACKGROUND_FIT_HZ,
) -> tuple[float, float]:
    """Individual alpha frequency and peak height above the 1/f background."""
    if not np.isfinite(log_spectrum).all():
        return float("nan"), float("nan")
    chi, intercept = background_fit(freqs, log_spectrum, fit_range)
    slope = -chi
    search = np.flatnonzero((freqs >= ALPHA_SEARCH_HZ[0]) & (freqs <= ALPHA_SEARCH_HZ[1]))
    residual = log_spectrum[search] - (intercept + slope * np.log10(freqs[search]))
    peak = int(np.argmax(residual))
    prominence = float(residual[peak])
    if prominence < MIN_PEAK_LOG10:
        return float("nan"), prominence
    near = np.abs(freqs[search] - freqs[search][peak]) <= CENTROID_HALF_WIDTH_HZ
    weights = np.clip(residual, 0.0, None) * near
    return float(np.sum(freqs[search] * weights) / np.sum(weights)), prominence


def frontal_alpha_asymmetry(records: Sequence[EpochedRecord], min_windows: int = MIN_GOOD_WINDOWS) -> float:
    """Frontal alpha asymmetry FAA = ln P_alpha(Fp2) - ln P_alpha(Fp1)."""
    fp1, fp2 = config.CHANNELS.index("Fp1"), config.CHANNELS.index("Fp2")
    diffs: list[npt.NDArray[np.float64]] = []
    for epoched in records:
        if epoched.n_windows == 0:
            continue
        both = epoched.good[:, fp1] & epoched.good[:, fp2]
        if not both.any():
            continue
        freqs, psd = window_psd(epoched.windows[both][:, [fp1, fp2]], epoched.sfreq)
        alpha = band_power(freqs, psd, BANDS["alpha"])
        with np.errstate(divide="ignore", invalid="ignore"):
            diffs.append(np.log(alpha[:, 1]) - np.log(alpha[:, 0]))
    if not diffs:
        return float("nan")
    values = np.concatenate(diffs)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size >= min_windows else float("nan")


def _feature_names() -> tuple[str, ...]:
    names = [
        f"{condition}_relpow_{band}_{channel}"
        for condition in ("rest", "task")
        for band in BANDS
        for channel in config.CHANNELS
    ]
    protocol1 = (*names, "rest_iaf_occ", "rest_alpha_peak_occ", "alpha_reactivity_occ")
    protocol2 = (*(f"rest_exponent_{channel}" for channel in config.CHANNELS), "rest_iaf_O1", "rest_alpha_peak_O1")
    protocol3 = (
        *(f"rest_relpow20_{band}_{channel}" for band in BANDS_20 for channel in config.CHANNELS),
        *(f"rest_slope20_{channel}" for channel in config.CHANNELS),
        "rest_iaf20_O1",
        "rest_alpha_peak20_O1",
    )
    return (*protocol1, *protocol2, *protocol3, "rest_faa")


FEATURE_NAMES: Final[tuple[str, ...]] = _feature_names()
PROTOCOL1_FEATURES: Final[tuple[str, ...]] = FEATURE_NAMES[:39]


def subject_features(rest: Sequence[EpochedRecord], task: Sequence[EpochedRecord]) -> dict[str, float]:
    """EEG features of one subject."""
    occ = [config.CHANNELS.index(ch) for ch in OCCIPITAL]
    freqs, rest_s = condition_spectrum(rest)
    _, task_s = condition_spectrum(task, min_record_windows=MIN_TRIAL_WINDOWS)
    out: dict[str, float] = {}
    for condition, spectrum in (("rest", rest_s), ("task", task_s)):
        with np.errstate(invalid="ignore", divide="ignore"):
            rel = log_relative_powers(freqs, spectrum)
        for band, values in rel.items():
            for channel, value in zip(config.CHANNELS, values):
                out[f"{condition}_relpow_{band}_{channel}"] = float(value)

    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rest_occ = np.nanmean(np.log10(rest_s[occ]), axis=0)
        iaf, peak = alpha_peak(freqs, rest_occ)
        alpha = BANDS["alpha"]
        ratio = band_power(freqs, task_s[occ], alpha) / band_power(freqs, rest_s[occ], alpha)
        reactivity = np.nanmean(np.log10(ratio))
    out["rest_iaf_occ"] = iaf
    out["rest_alpha_peak_occ"] = peak
    out["alpha_reactivity_occ"] = float(reactivity)
    with np.errstate(invalid="ignore", divide="ignore"):
        log_rest = np.log10(rest_s)
    for channel, row in zip(config.CHANNELS, log_rest):
        out[f"rest_exponent_{channel}"] = background_fit(freqs, row)[0]
    out["rest_iaf_O1"], out["rest_alpha_peak_O1"] = alpha_peak(freqs, log_rest[config.CHANNELS.index("O1")])

    with np.errstate(invalid="ignore", divide="ignore"):
        rel20 = log_relative_powers(freqs, rest_s, BANDS_20, TOTAL_BAND_20)
    for band, values in rel20.items():
        for channel, value in zip(config.CHANNELS, values):
            out[f"rest_relpow20_{band}_{channel}"] = float(value)
    for channel, row in zip(config.CHANNELS, log_rest):
        out[f"rest_slope20_{channel}"] = background_fit(freqs, row, BACKGROUND_FIT_20_HZ)[0]
    out["rest_iaf20_O1"], out["rest_alpha_peak20_O1"] = alpha_peak(
        freqs, log_rest[config.CHANNELS.index("O1")], BACKGROUND_FIT_20_HZ
    )
    out["rest_faa"] = frontal_alpha_asymmetry(rest)
    return {name: out[name] for name in FEATURE_NAMES}


def _preprocess_many(paths: Sequence[Path], cfg: PreprocessingConfig) -> list[EpochedRecord]:
    """Preprocess readable files."""
    records: list[EpochedRecord] = []
    for path in paths:
        try:
            records.append(preprocess_file(path, cfg))
        except (EdfFormatError, OSError, ValueError):
            continue
    return records


def extract_subject_features(
    files: Mapping[str, Path], cfg: PreprocessingConfig = PreprocessingConfig()
) -> dict[str, float]:
    """Features of one subject from its record files keyed by canonical stem."""
    usable, _ = resolve_ambiguous_records(files)
    rest = [usable[stem] for stem in (config.REST_STEM,) if stem in usable]
    task = [usable[stem] for stem in config.TASK_STEMS if stem in usable]
    return subject_features(_preprocess_many(rest, cfg), _preprocess_many(task, cfg))


def build_feature_table(
    registry: pd.DataFrame, cfg: PreprocessingConfig = PreprocessingConfig(), data_dir: Path = config.DATA_DIR
) -> pd.DataFrame:
    """Feature matrix of every subject in ``registry``, indexed by ``subject_key``."""
    rows: dict[str, dict[str, float]] = {}
    for subject_key, recs in registry.groupby("subject_key", sort=True):
        usable = recs[recs["status"] == "ok"]
        files = {stem: data_dir / rel for stem, rel in zip(usable["stem"], usable["relpath"])}
        rows[subject_key] = extract_subject_features(files, cfg)
    return pd.DataFrame.from_dict(rows, orient="index")[list(FEATURE_NAMES)]
