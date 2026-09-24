"""Spectral estimation and EEG features.

Power spectral density of one record is estimated from the 4-s windows
produced by ``src.preprocessing``. Each retained window ``w`` gives a
Hann-tapered periodogram ``P_w(f)`` [uV^2/Hz]; the record spectrum is the
bias-corrected mean of the log-periodograms (a geometric-mean Welch estimate)::

    log10 S(f) = mean_w log10 P_w(f) + gamma / ln 10,    gamma = 0.5772 (Euler)

Why not the arithmetic mean or the median:

* For a Gaussian process, away from 0 Hz and Nyquist, ``P_w(f) / S(f)`` is
  approximately Exp(1) (chi-square with 2 degrees of freedom divided by 2).
  Then ``E[ln P_w] = ln S - gamma`` exactly, for every window. Expectation is
  linear, so the mean of ``n`` log-periodograms has the same bias ``-gamma``
  for any ``n`` and any correlation between overlapping windows; adding
  ``gamma / ln 10 = 0.2507`` removes it.
* The median of ``n`` Exp(1) variables has an expectation that depends on
  ``n`` (1.00 for n = 1, 0.83 for n = 3, 0.78 for n = 5, ln 2 = 0.69 only as
  n -> inf). The number of retained windows differs between cohorts because
  artifact rates differ, so a median would add a cohort-correlated bias to
  absolute power and to the aperiodic offset.
* The log domain limits the influence of a residual artifact that passed
  rejection: one window 100 times too strong shifts the arithmetic mean of 20
  windows 6-fold ((19 + 100) / 20) but the log-mean by 10^(2/20) = 1.26-fold.

Channels with fewer than ``MIN_GOOD_WINDOWS`` retained windows (fixed before
modelling: 5 windows = 12 s of data) get NaN, which is imputed later inside
the training pipeline.
"""
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
from src.dataset import EdfFormatError
from src.preprocessing import EpochedRecord, PreprocessingConfig, preprocess_file

EULER_GAMMA: Final[float] = 0.5772156649015329
LOG10_BIAS: Final[float] = EULER_GAMMA / np.log(10.0)  # 0.2507, bias of log10 Exp(1) with sign reversed
MIN_GOOD_WINDOWS: Final[int] = 5  # retained 4-s windows with 2-s step = 12 s of data


def window_psd(
    windows: npt.NDArray[np.float64], sfreq: float
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Hann-tapered periodogram of every window.

    Parameters
    ----------
    windows : shape (n_windows, n_channels, n_samples), uV, already detrended.

    Returns
    -------
    freqs : shape (n_samples // 2 + 1,), Hz; resolution sfreq / n_samples.
    psd : shape (n_windows, n_channels, n_freqs), uV^2 / Hz.
    """
    return periodogram(windows, fs=sfreq, window="hann", detrend=False, scaling="density", axis=-1)


def log_mean_spectrum(psd: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Bias-corrected geometric mean over the first axis.

    ``psd`` shape: (n_windows, ..., n_freqs), uV^2/Hz. Returns shape (..., n_freqs):
    ``10 ** (mean_w log10 P_w + gamma / ln 10)``.
    """
    return 10.0 ** (np.log10(psd).mean(axis=0) + LOG10_BIAS)


def condition_spectrum(
    records: Sequence[EpochedRecord],
    n_channels: int = config.N_CHANNELS,
    n_samples: int = PreprocessingConfig().window_samples,
    sfreq: float = config.TARGET_SFREQ,
    min_windows: int = MIN_GOOD_WINDOWS,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Log-mean PSD of one condition pooled over records, per channel.

    Every record (e.g. each Schulte trial) first gives the mean log10
    periodogram of its retained windows; records are then averaged with equal
    weight, so a slow trial does not dominate a fast one::

        log10 S(f) = mean_r [ mean_{w in r} log10 P_w(f) ] + gamma / ln 10

    Each inner mean is unbiased up to the same constant, so the result is
    unbiased for any window counts.

    Returns
    -------
    freqs : shape (n_freqs,), Hz.
    spectrum : shape (n_channels, n_freqs), uV^2 / Hz; NaN for channels with
        fewer than ``min_windows`` retained windows over all records.
    """
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sfreq)
    per_record: list[npt.NDArray[np.float64]] = []  # each shape: (n_channels, n_freqs)
    n_good = np.zeros(n_channels, dtype=int)
    for epoched in records:
        if epoched.n_windows == 0:
            continue
        _, psd = window_psd(epoched.windows, epoched.sfreq)  # shape: (n_windows, n_channels, n_freqs)
        good = epoched.good
        mean_log = np.full((n_channels, freqs.size), np.nan)
        with np.errstate(divide="ignore"):  # exact zeros at 0 Hz after detrend
            for ch in range(n_channels):
                if good[:, ch].any():
                    mean_log[ch] = np.log10(psd[good[:, ch], ch]).mean(axis=0)
        per_record.append(mean_log)
        n_good += good.sum(axis=0)
    spectrum = np.full((n_channels, freqs.size), np.nan)
    if per_record:
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # channels absent from every record
            pooled = np.nanmean(np.stack(per_record), axis=0)  # shape: (n_channels, n_freqs)
        enough = n_good >= min_windows
        spectrum[enough] = 10.0 ** (pooled[enough] + LOG10_BIAS)
    return freqs, spectrum


def record_spectrum(
    epoched: EpochedRecord, min_windows: int = MIN_GOOD_WINDOWS
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Log-mean PSD over the retained windows of one record, per channel.

    Returns
    -------
    freqs : shape (n_freqs,), Hz.
    spectrum : shape (n_channels, n_freqs), uV^2 / Hz; NaN for channels with
        fewer than ``min_windows`` retained windows.
    """
    n_windows, n_channels, n_samples = epoched.windows.shape
    return condition_spectrum([epoched], n_channels, n_samples, epoched.sfreq, min_windows)


@dataclass(frozen=True)
class SpectraTable:
    """Record spectra of many files, aligned with ``relpaths``."""

    relpaths: tuple[str, ...]
    freqs: npt.NDArray[np.float64]  # shape: (n_freqs,), Hz
    spectra: npt.NDArray[np.float64]  # shape: (n_files, n_channels, n_freqs), uV^2 / Hz
    n_good: npt.NDArray[np.int64]  # shape: (n_files, n_channels), retained windows

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


# ---------------------------------------------------------------------------
# Subject features (fixed from the literature before looking at group differences)
# ---------------------------------------------------------------------------

# Frequency bands in Hz, half-open [lo, hi). Delta is omitted: the hardware
# high-pass at 2 Hz attenuates it. Relative power is taken within 4-30 Hz,
# the union of the bands, so the three relative powers sum to one.
BANDS: Final[dict[str, tuple[float, float]]] = {"theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0)}
TOTAL_BAND: Final[tuple[float, float]] = (4.0, 30.0)

# Individual alpha peak on the occipital rest spectrum. The 1/f background is
# a straight line in log-log coordinates, fitted outside the alpha range.
BACKGROUND_FIT_HZ: Final[tuple[float, float]] = (3.0, 30.0)
BACKGROUND_EXCLUDE_HZ: Final[tuple[float, float]] = (7.0, 14.0)
ALPHA_SEARCH_HZ: Final[tuple[float, float]] = (7.0, 13.0)
# A log-periodogram bin has standard deviation (pi / sqrt 6) / ln 10 = 0.56;
# averaged over ~30 windows of two channels it is ~0.08, and the maximum of
# ~25 noise bins in 7-13 Hz reaches 0.15-0.2 with no oscillation at all. A
# peak must therefore exceed the background by 10^0.3 = 2x to count.
MIN_PEAK_LOG10: Final[float] = 0.3
CENTROID_HALF_WIDTH_HZ: Final[float] = 1.0

OCCIPITAL: Final[tuple[str, ...]] = ("O1", "O2")

# Preprocessing variants compared inside model selection: native quantization
# and all records re-quantized to the coarsest step of the data set (1 uV).
PREPROCESSING_VARIANTS: Final[dict[str, PreprocessingConfig]] = {
    "native": PreprocessingConfig(),
    "requantized": PreprocessingConfig(requantize_step_uv=1.0),
}


def band_power(
    freqs: npt.NDArray[np.float64], spectrum: npt.NDArray[np.float64], band: tuple[float, float]
) -> npt.NDArray[np.float64]:
    """Integrated power in [lo, hi): sum_f S(f) * df, uV^2. Shape: spectrum.shape[:-1]."""
    lo, hi = band
    df = freqs[1] - freqs[0]
    return spectrum[..., (freqs >= lo) & (freqs < hi)].sum(axis=-1) * df


def log_relative_powers(
    freqs: npt.NDArray[np.float64], spectrum: npt.NDArray[np.float64]
) -> dict[str, npt.NDArray[np.float64]]:
    """log10 of band power over 4-30 Hz power, per band.

    Invariant to a common gain x -> a*x: numerator and denominator both scale
    by a^2. Shape of each value: spectrum.shape[:-1].
    """
    total = band_power(freqs, spectrum, TOTAL_BAND)
    return {name: np.log10(band_power(freqs, spectrum, band) / total) for name, band in BANDS.items()}


def alpha_peak(freqs: npt.NDArray[np.float64], log_spectrum: npt.NDArray[np.float64]) -> tuple[float, float]:
    """Individual alpha frequency and peak height above the 1/f background.

    ``log_spectrum`` is log10 S(f), shape (n_freqs,). The background
    log10 S = b - chi * log10 f is fitted by least squares on 3-30 Hz
    excluding 7-14 Hz; the residual r(f) is searched for its maximum in
    7-13 Hz. The alpha frequency is the centroid of the positive residual
    within +-1 Hz of the maximum::

        IAF = sum f * max(r(f), 0) / sum max(r(f), 0)

    Returns
    -------
    (iaf_hz, prominence_log10). ``iaf_hz`` is NaN when the peak is lower than
    ``MIN_PEAK_LOG10``: no peak is invented at a default 10 Hz.
    """
    if not np.isfinite(log_spectrum).all():
        return float("nan"), float("nan")
    fit = (freqs >= BACKGROUND_FIT_HZ[0]) & (freqs <= BACKGROUND_FIT_HZ[1])
    fit &= ~((freqs >= BACKGROUND_EXCLUDE_HZ[0]) & (freqs <= BACKGROUND_EXCLUDE_HZ[1]))
    slope, intercept = np.polyfit(np.log10(freqs[fit]), log_spectrum[fit], 1)
    search = np.flatnonzero((freqs >= ALPHA_SEARCH_HZ[0]) & (freqs <= ALPHA_SEARCH_HZ[1]))
    residual = log_spectrum[search] - (intercept + slope * np.log10(freqs[search]))
    peak = int(np.argmax(residual))
    prominence = float(residual[peak])
    if prominence < MIN_PEAK_LOG10:
        return float("nan"), prominence
    near = np.abs(freqs[search] - freqs[search][peak]) <= CENTROID_HALF_WIDTH_HZ
    weights = np.clip(residual, 0.0, None) * near
    return float(np.sum(freqs[search] * weights) / np.sum(weights)), prominence


def _feature_names() -> tuple[str, ...]:
    names = [
        f"{condition}_relpow_{band}_{channel}"
        for condition in ("rest", "task")
        for band in BANDS
        for channel in config.CHANNELS
    ]
    return (*names, "rest_iaf_occ", "rest_alpha_peak_occ", "alpha_reactivity_occ")


FEATURE_NAMES: Final[tuple[str, ...]] = _feature_names()


def subject_features(rest: Sequence[EpochedRecord], task: Sequence[EpochedRecord]) -> dict[str, float]:
    """EEG features of one subject; missing values are NaN.

    Parameters
    ----------
    rest : preprocessed eyes-closed rest record(s); normally one.
    task : preprocessed Schulte trials; any subset of the five.

    Returns
    -------
    dict over ``FEATURE_NAMES``:

    * ``{rest,task}_relpow_{theta,alpha,beta}_{channel}``: log10 relative
      power (dimensionless), 36 values;
    * ``rest_iaf_occ``: individual alpha frequency on mean O1/O2, Hz;
    * ``rest_alpha_peak_occ``: alpha peak above the 1/f background, log10;
    * ``alpha_reactivity_occ``: log10(alpha power task / alpha power rest),
      mean over O1/O2; negative values mean alpha suppression in the task.
    """
    occ = [config.CHANNELS.index(ch) for ch in OCCIPITAL]
    freqs, rest_s = condition_spectrum(rest)
    _, task_s = condition_spectrum(task)
    out: dict[str, float] = {}
    for condition, spectrum in (("rest", rest_s), ("task", task_s)):
        with np.errstate(invalid="ignore", divide="ignore"):
            rel = log_relative_powers(freqs, spectrum)  # each shape: (n_channels,)
        for band, values in rel.items():
            for channel, value in zip(config.CHANNELS, values):
                out[f"{condition}_relpow_{band}_{channel}"] = float(value)

    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN occipital channels
        rest_occ = np.nanmean(np.log10(rest_s[occ]), axis=0)  # shape: (n_freqs,)
        iaf, peak = alpha_peak(freqs, rest_occ)
        alpha = BANDS["alpha"]
        ratio = band_power(freqs, task_s[occ], alpha) / band_power(freqs, rest_s[occ], alpha)  # shape: (2,)
        reactivity = np.nanmean(np.log10(ratio))
    out["rest_iaf_occ"] = iaf
    out["rest_alpha_peak_occ"] = peak
    out["alpha_reactivity_occ"] = float(reactivity)
    return {name: out[name] for name in FEATURE_NAMES}


def _preprocess_many(paths: Sequence[Path], cfg: PreprocessingConfig) -> list[EpochedRecord]:
    """Preprocess readable files; unreadable or empty files are skipped."""
    records: list[EpochedRecord] = []
    for path in paths:
        try:
            records.append(preprocess_file(path, cfg))
        except (EdfFormatError, OSError, ValueError):
            continue
    return records


def extract_subject_features(
    files: Mapping[str, Path], cfg: PreprocessingConfig = PreprocessingConfig(), exclude: frozenset[str] = frozenset()
) -> dict[str, float]:
    """Features of one subject from its record files keyed by canonical stem.

    ``exclude`` lists stems not to use (files whose condition is unknown).
    Missing, empty and unreadable files are skipped; a subject without any
    usable file gets all-NaN features.
    """
    rest = [files[stem] for stem in (config.REST_STEM,) if stem in files and stem not in exclude]
    task = [files[stem] for stem in config.TASK_STEMS if stem in files and stem not in exclude]
    return subject_features(_preprocess_many(rest, cfg), _preprocess_many(task, cfg))


def build_feature_table(
    registry: pd.DataFrame, cfg: PreprocessingConfig = PreprocessingConfig(), data_dir: Path = config.DATA_DIR
) -> pd.DataFrame:
    """Feature matrix of every subject in ``registry``, indexed by ``subject_key``.

    Files with ``condition_conflict`` are excluded from their condition, as
    fixed in the data audit. Columns follow ``FEATURE_NAMES``.
    """
    rows: dict[str, dict[str, float]] = {}
    for subject_key, recs in registry.groupby("subject_key", sort=True):
        usable = recs[recs["status"] == "ok"]
        files = {stem: data_dir / rel for stem, rel in zip(usable["stem"], usable["relpath"])}
        exclude = frozenset(usable.loc[usable["condition_conflict"], "stem"])
        rows[subject_key] = extract_subject_features(files, cfg, exclude)
    return pd.DataFrame.from_dict(rows, orient="index")[list(FEATURE_NAMES)]
