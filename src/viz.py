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
    """Stacked channel traces."""
    t = np.arange(data.shape[1]) / sfreq
    for i, trace in enumerate(data):
        ax.plot(t, trace - np.median(trace) - i * spacing_uv, color=color, lw=0.7)
    ax.set_yticks(-np.arange(len(channels)) * spacing_uv, channels)
    ax.set_xlabel("time, s")
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
    """Median log10 PSD across records with interquartile band."""
    log_psd = np.log10(spectra[~np.isnan(spectra).any(axis=1)])
    q25, q50, q75 = np.percentile(log_psd, [25, 50, 75], axis=0)
    ax.plot(freqs, q50, color=color, linestyle=linestyle, label=f"{label} (n={len(log_psd)})")
    if band:
        ax.fill_between(freqs, q25, q75, color=color, alpha=0.15, linewidth=0)
    ax.set_xlabel("frequency, Hz")
    ax.set_ylabel("log₁₀ PSD, µV²/Hz")


def shade_unused_band(ax: Axes, fmin: float = config.HEADER_LOWPASS_HZ, fmax: float | None = None) -> None:
    """Grey band over frequencies excluded from features."""
    fmax = fmax if fmax is not None else ax.get_xlim()[1]
    ax.axvspan(fmin, fmax, color=GRID, alpha=0.6, linewidth=0, zorder=0)


REJECT_COLORS: Final[tuple[str, ...]] = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
RETAINED_COLOR: Final[str] = "#f0efec"


def plot_rejection(ax: Axes, reject: npt.NDArray[np.uint8], step_s: float, channels: tuple[str, ...] = config.CHANNELS) -> None:
    """Rejection map of one record."""
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    from src.preprocessing import Reject

    reasons = list(Reject)
    code = np.zeros(reject.shape, dtype=int)
    for k, reason in enumerate(reasons, start=1):
        code[(code == 0) & ((reject & reason) > 0)] = k
    cmap = ListedColormap([RETAINED_COLOR, *REJECT_COLORS[: len(reasons)]])
    n_windows = reject.shape[0]
    ax.imshow(code.T, aspect="auto", cmap=cmap, vmin=-0.5, vmax=len(reasons) + 0.5, interpolation="nearest",
              extent=(-0.5 * step_s, (n_windows - 0.5) * step_s, len(channels) - 0.5, -0.5))
    ax.set_yticks(range(len(channels)), channels)
    ax.set_xlabel("window start, s")
    ax.grid(False)
    handles = [Patch(color=RETAINED_COLOR, label="retained")]
    handles += [Patch(color=c, label=r.name.lower()) for r, c in zip(reasons, REJECT_COLORS) if np.any(code == reasons.index(r) + 1)]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7)


def plot_spectrum_features(
    ax: Axes,
    freqs: npt.NDArray[np.float64],
    log_spectrum: npt.NDArray[np.float64],
    chi: float,
    offset: float,
    iaf: float,
    fmax: float = 40.0,
) -> None:
    """One channel spectrum with the fitted 1/f line, the alpha peak and the power bands."""
    from src.features import BANDS

    keep = (freqs >= 2.0) & (freqs <= fmax)
    ax.plot(freqs[keep], log_spectrum[keep], color="#0b0b0b", lw=1.2, label="spectrum")
    ax.plot(freqs[keep], offset - chi * np.log10(freqs[keep]), color=TEXT_MUTED, ls="--", lw=1.2,
            label=f"1/f background, slope χ = {chi:.2f}")
    for (name, (lo, hi)), shade in zip(BANDS.items(), ("#f4f3f0", "#e9e8e4", "#f4f3f0")):
        ax.axvspan(lo, hi, color=shade, zorder=0, lw=0)
        ax.text(0.5 * (lo + hi), 0.98, name, transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=7, color=TEXT_MUTED)
    if np.isfinite(iaf):
        ax.axvline(iaf, color=COHORT_COLORS[config.GROUP_PTSD], lw=1.2, label=f"alpha peak {iaf:.1f} Hz")
    ax.set_xlabel("frequency, Hz")
    ax.set_ylabel("log₁₀ PSD, µV²/Hz")
    ax.legend(loc="lower left", fontsize=7)


STRATUM_ORDER: Final[tuple[str, ...]] = (
    "ptsd", "control_A", "control_B", "control_C", "control_B_supplement", "control_ageing", "somatoform"
)
STRATUM_LABELS: Final[dict[str, str]] = {
    "ptsd": "PTSD",
    "control_A": "control A",
    "control_B": "control B",
    "control_C": "control C",
    "control_B_supplement": "suppl. controls",
    "control_ageing": "controls 65+",
    "somatoform": "somatoform",
}


def stratum_color(stratum: str) -> str:
    """Cohort colour of a stratum."""
    if stratum == "ptsd":
        return COHORT_COLORS[config.GROUP_PTSD]
    if stratum == "somatoform":
        return COHORT_COLORS[config.GROUP_SOMATOFORM]
    return COHORT_COLORS[config.GROUP_CONTROL]


def plot_by_stratum(
    ax: Axes,
    values: npt.NDArray[np.float64],
    stratum: npt.NDArray[np.str_],
    order: tuple[str, ...] = STRATUM_ORDER,
    seed: int = 0,
) -> None:
    """Jittered points per stratum (one point per subject) with the median as a black bar."""
    rng = np.random.default_rng(seed)
    present = [s for s in order if np.any(stratum == s)]
    for i, s in enumerate(present):
        v = values[(stratum == s) & np.isfinite(values)]
        ax.scatter(i + rng.uniform(-0.18, 0.18, v.size), v, s=9, color=stratum_color(s), alpha=0.75, lw=0)
        if v.size:
            ax.hlines(np.median(v), i - 0.3, i + 0.3, color="#0b0b0b", lw=1.6)
    ax.set_xticks(range(len(present)), [f"{STRATUM_LABELS.get(s, s)}\n(n={int(np.sum(stratum == s))})" for s in present],
                  fontsize=7)


def plot_roc(ax: Axes, y: npt.NDArray[np.int_], p: npt.NDArray[np.float64], label: str, color: str, ls: str = "-") -> None:
    """ROC curve of scores ``p`` for labels ``y`` (1 = PTSD)."""
    from sklearn.metrics import roc_auc_score, roc_curve

    fpr, tpr, _ = roc_curve(y, p)
    ax.plot(fpr, tpr, color=color, ls=ls, label=f"{label}: AUC {roc_auc_score(y, p):.2f}")
    ax.plot([0, 1], [0, 1], color=GRID, lw=1.0, zorder=0)
    ax.set_xlabel("1 − specificity")
    ax.set_ylabel("sensitivity")
    ax.set_aspect("equal")
