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

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.signal import periodogram

from src import config
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


def record_spectrum(
    epoched: EpochedRecord, min_windows: int = MIN_GOOD_WINDOWS
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Log-mean PSD over retained windows, per channel.

    Returns
    -------
    freqs : shape (n_freqs,), Hz.
    spectrum : shape (n_channels, n_freqs), uV^2 / Hz; NaN for channels with
        fewer than ``min_windows`` retained windows.
    """
    n_samples = epoched.windows.shape[2]
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / epoched.sfreq)
    spectrum = np.full((epoched.windows.shape[1], freqs.size), np.nan)
    if epoched.n_windows == 0:
        return freqs, spectrum
    _, psd = window_psd(epoched.windows, epoched.sfreq)  # shape: (n_windows, n_channels, n_freqs)
    good = epoched.good
    with np.errstate(divide="ignore"):  # exact zeros at 0 Hz after detrend
        for ch in range(psd.shape[1]):
            if good[:, ch].sum() >= min_windows:
                spectrum[ch] = log_mean_spectrum(psd[good[:, ch], ch])
    return freqs, spectrum


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
