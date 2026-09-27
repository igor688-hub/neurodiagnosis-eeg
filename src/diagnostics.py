from __future__ import annotations

from collections.abc import Sequence
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
    """Zero-lag Pearson correlation over windows retained on all channels."""
    windows = epoched.windows[epoched.good.all(axis=1)]
    n_channels = epoched.windows.shape[1]
    if len(windows) == 0:
        return np.full((n_channels, n_channels), np.nan)
    x = np.concatenate(list(windows), axis=1)
    return np.corrcoef(x)


def homologous_correlations(corr: npt.NDArray[np.float64]) -> dict[str, float]:
    """Correlation of the left-right pairs O1-O2, Fp1-Fp2, T3-T4."""
    ch = config.CHANNELS
    return {f"{a}-{b}": float(corr[ch.index(a), ch.index(b)]) for a, b in (("O1", "O2"), ("Fp1", "Fp2"), ("T3", "T4"))}


def channel_rms_ratio(epoched: EpochedRecord, channel: str) -> float:
    """RMS of ``channel`` divided by the median RMS of the other channels."""
    good = epoched.good
    rms = np.array(
        [
            np.sqrt(np.median(epoched.windows[good[:, ch], ch].var(axis=1))) if good[:, ch].any() else np.nan
            for ch in range(epoched.windows.shape[1])
        ]
    )
    idx = epoched.channels.index(channel)
    return float(rms[idx] / np.nanmedian(np.delete(rms, idx)))


def alpha_prominence(freqs: npt.NDArray[np.float64], spectrum: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Height of the 8-13 Hz maximum above its flanks, in log10 units."""
    log_s = np.log10(spectrum)

    def band_mean(lo: float, hi: float) -> npt.NDArray[np.float64]:
        return log_s[..., (freqs >= lo) & (freqs <= hi)].mean(axis=-1)

    lo, hi = ALPHA_BAND_HZ
    peak = log_s[..., (freqs >= lo) & (freqs <= hi)].max(axis=-1)
    return peak - 0.5 * (band_mean(*ALPHA_FLANKS_HZ[0]) + band_mean(*ALPHA_FLANKS_HZ[1]))


def plausibility_table(
    registry: pd.DataFrame, cfg: PreprocessingConfig = PreprocessingConfig(), data_dir: Path = config.DATA_DIR
) -> pd.DataFrame:
    """Homologous-pair correlations and occipital alpha prominence per readable file."""
    occipital = [config.CHANNELS.index("O1"), config.CHANNELS.index("O2")]
    o2, fp2 = config.CHANNELS.index("O2"), config.CHANNELS.index("Fp2")
    rows: list[dict[str, object]] = []
    for relpath in registry.loc[registry["status"] == "ok", "relpath"]:
        epoched = preprocess_file(data_dir / relpath, cfg)
        freqs, spectrum = record_spectrum(epoched)
        corr = interchannel_correlation(epoched)
        rows.append(
            {
                "relpath": relpath,
                **homologous_correlations(corr),
                "O2-Fp2": float(corr[o2, fp2]),
                "alpha_prominence_occ": float(np.nanmean(alpha_prominence(freqs, spectrum[occipital]))),
                "t3_rms_ratio": channel_rms_ratio(epoched, "T3"),
            }
        )
    return pd.DataFrame(rows)


def synthetic_alpha_record(
    rng: np.random.Generator,
    duration_s: float,
    exponent: float,
    peak_log10: float,
    peak_hz: float = 10.0,
    peak_sd_hz: float = 0.5,
    rms_uv: float = 10.0,
    sfreq: float = config.TARGET_SFREQ,
) -> npt.NDArray[np.float64]:
    """Gaussian six-channel signal with PSD S(f) = f^(-exponent) * (1 + A g(f))^2."""
    n_times = int(duration_s * sfreq)
    freqs = np.fft.rfftfreq(n_times, d=1.0 / sfreq)
    amp_ratio = 10.0 ** (peak_log10 / 2.0) - 1.0
    bump = amp_ratio * np.exp(-0.5 * ((freqs - peak_hz) / peak_sd_hz) ** 2)
    safe = np.where(freqs > 0, freqs, 1.0)
    amplitude = np.where(freqs > 0, safe ** (-exponent / 2.0), 0.0) * (1.0 + bump)
    phases = rng.uniform(0.0, 2.0 * np.pi, (config.N_CHANNELS, freqs.size))
    x = np.fft.irfft(amplitude * np.exp(1j * phases), n=n_times)
    return rms_uv * x / x.std(axis=1, keepdims=True)


def alpha_detection_study(
    durations_s: Sequence[float] = (12.0, 30.0, 60.0),
    exponents: Sequence[float] = (1.0, 1.5, 2.0),
    peaks_log10: Sequence[float] = (0.0, 0.3, 0.5, 1.0),
    n_sim: int = 200,
    seed: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    """Measured peak height of ``features.alpha_peak`` on simulated recordings."""
    from src.dataset import EegRecord
    from src.features import alpha_peak
    from src.preprocessing import preprocess_record

    rng = np.random.default_rng(seed)
    occipital = [config.CHANNELS.index("O1"), config.CHANNELS.index("O2")]
    rows: list[tuple[float, float, float, float]] = []
    for duration in durations_s:
        for exponent in exponents:
            for true_peak in peaks_log10:
                for _ in range(n_sim):
                    data = synthetic_alpha_record(rng, duration, exponent, true_peak)
                    record = EegRecord(
                        data=data,
                        sfreq=config.TARGET_SFREQ,
                        channels=config.CHANNELS,
                        quantization_step_uv=np.full(config.N_CHANNELS, 0.061),
                        at_rail=np.zeros(data.shape, dtype=bool),
                    )
                    freqs, spectrum = record_spectrum(preprocess_record(record))
                    _, measured = alpha_peak(freqs, np.log10(spectrum[occipital]).mean(axis=0))
                    rows.append((duration, exponent, true_peak, measured))
    return pd.DataFrame(rows, columns=["duration_s", "exponent", "true_peak_log10", "measured"])


SLOPE_RANGES: dict[str, tuple[float, float, tuple[float, float] | None]] = {
    "3-30 w/o 7-14": (3.0, 30.0, (7.0, 14.0)),
    "4-25 w/o 7-14": (4.0, 25.0, (7.0, 14.0)),
    "14-30": (14.0, 30.0, None),
    "20-35": (20.0, 35.0, None),
}


def slope_fit(
    freqs: npt.NDArray[np.float64], log_spectrum: npt.NDArray[np.float64], lo: float, hi: float,
    exclude: tuple[float, float] | None,
) -> tuple[float, float]:
    """Exponent chi of log10 S = b - chi log10 f on [lo, hi] (minus ``exclude``) and fit RMSE in log10 units."""
    mask = (freqs >= lo) & (freqs <= hi)
    if exclude is not None:
        mask &= ~((freqs >= exclude[0]) & (freqs <= exclude[1]))
    y = log_spectrum[mask]
    if not np.isfinite(y).all():
        return float("nan"), float("nan")
    x = np.log10(freqs[mask])
    slope, intercept = np.polyfit(x, y, 1)
    return float(-slope), float(np.sqrt(np.mean((y - (intercept + slope * x)) ** 2)))


def slope_robustness_table(
    registry: pd.DataFrame, channels: Sequence[str], cfg: PreprocessingConfig = PreprocessingConfig(),
    data_dir: Path = config.DATA_DIR,
) -> pd.DataFrame:
    """Aperiodic exponent of rest records under several fit ranges, with fit RMSE."""
    from src.dataset import find_record_files, resolve_ambiguous_records
    from src.features import _preprocess_many

    rows: list[dict[str, object]] = []
    for relpath in registry.loc[(registry["status"] == "ok") & (registry["condition"] == "rest"), "relpath"]:
        path = data_dir / relpath
        usable, _ = resolve_ambiguous_records(find_record_files(path.parent, strict=False))
        records = _preprocess_many([usable[config.REST_STEM]], cfg) if config.REST_STEM in usable else []
        if not records:
            continue
        freqs, spectrum = record_spectrum(records[0])
        with np.errstate(divide="ignore", invalid="ignore"):
            log_s = np.log10(spectrum)
        for channel in channels:
            row = log_s[config.CHANNELS.index(channel)]
            for name, (lo, hi, excl) in SLOPE_RANGES.items():
                chi, rmse = slope_fit(freqs, row, lo, hi, excl)
                rows.append({"relpath": relpath, "channel": channel, "range": name, "exponent": chi, "rmse": rmse})
    return pd.DataFrame(rows)
