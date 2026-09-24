"""Signal-plausibility diagnostics. Never used as model inputs.

Two hallmarks of scalp EEG recorded against a common ear reference:

1. Inter-channel correlation. All channels share the reference electrode and
   volume conduction spreads cortical sources, so neighbouring and homologous
   channels are positively correlated at zero lag (e.g. Fp1-Fp2 through eye
   activity, O1-O2 through occipital alpha).
2. Occipital alpha in eyes-closed rest (Berger effect): a spectral peak at
   8-13 Hz on O1/O2 that stands above its spectral neighbourhood.

Their absence shows that a recording differs from typical EEG; it does not by
itself prove that the signal is not EEG.
"""
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


def channel_rms_ratio(epoched: EpochedRecord, channel: str) -> float:
    """RMS of ``channel`` divided by the median RMS of the other channels.

    RMS per channel is the square root of the median window variance over its
    retained windows. With the reference on the left ear, T3 (nearest to the
    reference) is expected below 1.
    """
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
    DataFrame with columns ``relpath, O1-O2, Fp1-Fp2, T3-T4, alpha_prominence_occ,
    t3_rms_ratio``; alpha prominence is the mean over O1 and O2 in log10 units.
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
    """Gaussian six-channel signal with PSD S(f) = f^(-exponent) * (1 + A g(f))^2.

    ``g`` is a Gaussian bump at ``peak_hz`` with SD ``peak_sd_hz``; A is set so
    that the spectral peak exceeds the power-law background by ``peak_log10``
    (0 = no oscillation). Channels are independent. Returns shape (6, n_times), uV.
    """
    n_times = int(duration_s * sfreq)
    freqs = np.fft.rfftfreq(n_times, d=1.0 / sfreq)
    amp_ratio = 10.0 ** (peak_log10 / 2.0) - 1.0  # (1 + A)^2 = 10^peak_log10
    bump = amp_ratio * np.exp(-0.5 * ((freqs - peak_hz) / peak_sd_hz) ** 2)
    safe = np.where(freqs > 0, freqs, 1.0)  # the 0 Hz bin is zeroed below
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
    """Measured peak height of ``features.alpha_peak`` on simulated recordings.

    Every simulated record goes through the real preprocessing and spectral
    estimation (low-pass, 4-s windows, rejection, log-mean spectrum, mean of
    O1 and O2). With ``peak_log10 = 0`` the share of heights above a threshold
    is the false-detection rate of that threshold; with a true peak it is the
    sensitivity. A power law is an idealised background: real spectra have
    knees and beta bumps, so the rates are indicative.

    Returns
    -------
    DataFrame with columns ``duration_s, exponent, true_peak_log10, measured``.
    """
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
