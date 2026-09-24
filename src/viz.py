"""Plotting helpers for the notebooks in ``results/``.

Colour encodes cohort identity only, in a fixed order taken from a palette
validated for colour-vision deficiency (all pairs of the first three slots).
Other factors (export format, condition) are encoded by line style.
"""
from __future__ import annotations

from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes

from src import config

COHORT_COLORS: Final[dict[str, str]] = {
    config.GROUP_CONTROL: "#2a78d6",
    config.GROUP_PTSD: "#eb6834",
    config.GROUP_SOMATOFORM: "#1baf7a",
}
TEXT_MUTED: Final[str] = "#52514e"
GRID: Final[str] = "#e4e3df"
LINE_STYLES: Final[tuple[str, ...]] = ("-", "--", ":")


def apply_style() -> None:
    """Thin marks, recessive grid and axes."""
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": TEXT_MUTED,
            "axes.labelcolor": "#0b0b0b",
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": TEXT_MUTED,
            "ytick.color": TEXT_MUTED,
            "lines.linewidth": 1.6,
            "legend.frameon": False,
            "font.size": 9,
        }
    )


def plot_traces(
    ax: Axes,
    data: npt.NDArray[np.float64],
    sfreq: float,
    channels: tuple[str, ...] = config.CHANNELS,
    spacing_uv: float = 100.0,
    color: str = "#0b0b0b",
) -> None:
    """Stacked channel traces. ``data`` shape: (n_channels, n_times), uV."""
    t = np.arange(data.shape[1]) / sfreq
    for i, trace in enumerate(data):
        ax.plot(t, trace - np.median(trace) - i * spacing_uv, color=color, lw=0.7)
    ax.set_yticks(-np.arange(len(channels)) * spacing_uv, channels)
    ax.set_xlabel("время, с")
    ax.grid(False)


def plot_median_spectra(
    ax: Axes,
    freqs: npt.NDArray[np.float64],
    spectra: npt.NDArray[np.float64],
    label: str,
    color: str,
    linestyle: str = "-",
    band: bool = True,
) -> None:
    """Median log10 PSD across records with interquartile band.

    ``spectra`` shape: (n_records, n_freqs), uV^2 / Hz; NaN rows are ignored.
    """
    log_psd = np.log10(spectra[~np.isnan(spectra).any(axis=1)])
    q25, q50, q75 = np.percentile(log_psd, [25, 50, 75], axis=0)
    ax.plot(freqs, q50, color=color, linestyle=linestyle, label=f"{label} (n={len(log_psd)})")
    if band:
        ax.fill_between(freqs, q25, q75, color=color, alpha=0.15, linewidth=0)
    ax.set_xlabel("частота, Гц")
    ax.set_ylabel("log₁₀ PSD, мкВ²/Гц")


def shade_unused_band(ax: Axes, fmin: float = config.HEADER_LOWPASS_HZ, fmax: float | None = None) -> None:
    """Grey band over frequencies excluded from features."""
    fmax = fmax if fmax is not None else ax.get_xlim()[1]
    ax.axvspan(fmin, fmax, color=GRID, alpha=0.6, linewidth=0, zorder=0)
