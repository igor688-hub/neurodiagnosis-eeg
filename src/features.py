"""Spectral estimation and EEG features.

Power spectral density of one record is estimated with Welch's method on the
4-s windows produced by ``src.preprocessing``: each window is Hann-tapered,
its periodogram is computed, and the periodograms of the retained windows
are combined per channel by the median::

    P_w(f) = |sum_n h[n] x_w[n] exp(-2 pi i f n / fs)|^2 / (fs * sum_n h[n]^2)
    S(f)   = median_w P_w(f)                          [uV^2 / Hz]

The median is robust to residual artifacts that passed rejection. For a
Hann periodogram of Gaussian noise P_w(f) ~ S_true(f) * chi2_2 / 2, so the
median underestimates the mean by the constant factor ln 2; the factor is
identical for every record and frequency and cancels in relative power,
peak frequency and spectral slope.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.signal import periodogram

from src import config
from src.preprocessing import EpochedRecord, PreprocessingConfig, preprocess_file


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


def record_spectrum(epoched: EpochedRecord) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Median PSD over retained windows, per channel.

    Returns
    -------
    freqs : shape (n_freqs,), Hz.
    spectrum : shape (n_channels, n_freqs), uV^2 / Hz; NaN for channels
        without a single retained window.
    """
    n_samples = epoched.windows.shape[2]
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / epoched.sfreq)
    spectrum = np.full((epoched.windows.shape[1], freqs.size), np.nan)
    if epoched.n_windows == 0:
        return freqs, spectrum
    _, psd = window_psd(epoched.windows, epoched.sfreq)  # shape: (n_windows, n_channels, n_freqs)
    good = epoched.good
    for ch in range(psd.shape[1]):
        if good[:, ch].any():
            spectrum[ch] = np.median(psd[good[:, ch], ch], axis=0)
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
