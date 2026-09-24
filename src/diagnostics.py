"""Signal-plausibility diagnostics. Never used as model inputs.

Two hallmarks of scalp EEG recorded against a common ear reference:

1. Inter-channel correlation. All channels share the reference electrode and
   volume conduction spreads cortical sources, so neighbouring and homologous
   channels are positively correlated at zero lag (e.g. Fp1-Fp2 through eye
   activity, O1-O2 through occipital alpha).
2. Occipital alpha in eyes-closed rest (Berger effect): a spectral peak at
   8-13 Hz on O1/O2 that stands above its spectral neighbourhood.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from src import config
from src.features import record_spectrum
from src.preprocessing import EpochedRecord, PreprocessingConfig, preprocess_file

ALPHA_BAND_HZ: tuple[float, float] = (8.0, 13.0)
ALPHA_FLANKS_HZ: tuple[tuple[float, float], tuple[float, float]] = ((5.0, 7.0), (14.0, 17.0))


def interchannel_correlation(epoched: EpochedRecord) -> npt.NDArray[np.float64]:
    """Zero-lag Pearson correlation over windows retained on all channels.

    Returns
    -------
    shape (n_channels, n_channels); NaN if no window is retained on all channels.
    """
    windows = epoched.windows[epoched.good.all(axis=1)]  # shape: (n_kept, n_channels, n_samples)
    n_channels = epoched.windows.shape[1]
    if len(windows) == 0:
        return np.full((n_channels, n_channels), np.nan)
    x = np.concatenate(list(windows), axis=1)  # shape: (n_channels, n_kept * n_samples)
    return np.corrcoef(x)


def homologous_correlations(corr: npt.NDArray[np.float64]) -> dict[str, float]:
    """Correlation of the left-right pairs O1-O2, Fp1-Fp2, T3-T4."""
    ch = config.CHANNELS
    return {f"{a}-{b}": float(corr[ch.index(a), ch.index(b)]) for a, b in (("O1", "O2"), ("Fp1", "Fp2"), ("T3", "T4"))}


def alpha_prominence(freqs: npt.NDArray[np.float64], spectrum: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Height of the 8-13 Hz maximum above its flanks, in log10 units.

    prominence = max_{8-13 Hz} log10 S(f) - mean(mean_{5-7 Hz} log10 S, mean_{14-17 Hz} log10 S)

    A value of 0.3 means the alpha peak is 10^0.3 = 2 times its neighbourhood.
    ``spectrum`` shape: (..., n_freqs) in uV^2/Hz; returns shape (...,).
    """
    log_s = np.log10(spectrum)

    def band_mean(lo: float, hi: float) -> npt.NDArray[np.float64]:
        return log_s[..., (freqs >= lo) & (freqs <= hi)].mean(axis=-1)

    lo, hi = ALPHA_BAND_HZ
    peak = log_s[..., (freqs >= lo) & (freqs <= hi)].max(axis=-1)
    return peak - 0.5 * (band_mean(*ALPHA_FLANKS_HZ[0]) + band_mean(*ALPHA_FLANKS_HZ[1]))


def plausibility_table(
    registry: pd.DataFrame, cfg: PreprocessingConfig = PreprocessingConfig(), data_dir: Path = config.DATA_DIR
) -> pd.DataFrame:
    """Homologous-pair correlations and occipital alpha prominence per readable file.

    Returns
    -------
    DataFrame with columns ``relpath, O1-O2, Fp1-Fp2, T3-T4, alpha_prominence_occ``;
    the last is the mean over O1 and O2 in log10 units.
    """
    occipital = [config.CHANNELS.index("O1"), config.CHANNELS.index("O2")]
    rows: list[dict[str, object]] = []
    for relpath in registry.loc[registry["status"] == "ok", "relpath"]:
        epoched = preprocess_file(data_dir / relpath, cfg)
        freqs, spectrum = record_spectrum(epoched)
        rows.append(
            {
                "relpath": relpath,
                **homologous_correlations(interchannel_correlation(epoched)),
                "alpha_prominence_occ": float(np.nanmean(alpha_prominence(freqs, spectrum[occipital]))),
            }
        )
    return pd.DataFrame(rows)
