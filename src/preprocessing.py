"""Signal conditioning, windowing and per-channel artifact rejection.

Pipeline for one EDF record (native rate ``fs`` in 123-127 Hz)::

    x(t) [uV, fs]
      -> trim constant stretches at the record edges      (zero padding of the last EDF record)
      -> optional re-quantization to a common step        (experiment, off by default)
      -> polyphase resampling to 125 Hz                   (anti-aliased, rational ratio)
      -> zero-phase FIR low-pass at 40 Hz                  (common upper band edge)
      -> 4-s windows with 2-s step, linear detrend        (0.25 Hz Welch resolution later)
      -> rejection flags per (window, channel)

No high-pass is added: every header reports a hardware high-pass at 2 Hz,
and a second high-pass would only deepen the attenuation of the 2-4 Hz range.
Slow drift that survives the hardware filter is removed by the per-window
linear detrend.

Rejection is channel-wise: a blink on Fp1 removes that window from Fp1 only,
so occipital alpha of the same window is kept. Thresholds are fixed before
modelling and are the same for every cohort and export format.

A channel whose complete, non-flat signal equals another channel of the
same record is not an independent electrode. Equality does not tell which of
the two is genuine, so both are rejected in every window (``Reject.COPY``);
flat channels are left to the flat criterion. The rule uses only the record
itself and is identical in training and inference.

Saturation and signal dropouts are located on the native samples and dilated
by a guard interval before they are mapped onto windows. Resampling and the
low-pass are FIR filters that spread a discontinuity over their impulse
response (about +-0.2 s here); the guard of 0.5 s covers it, so no retained
window contains filter ringing from a rejected interval. The record is never
cut and re-joined inside, so no filter runs across an artificial splice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntFlag
from fractions import Fraction
from pathlib import Path
from typing import Final

import mne
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.ndimage import binary_dilation
from scipy.signal import detrend, resample_poly

from src import config
from src.dataset import EegRecord, constant_stretch_mask, edge_constant_samples, load_record

# Robust z-score scale: MAD * 1.4826 estimates the standard deviation of a normal sample.
MAD_TO_SIGMA: Final[float] = 1.4826


class Reject(IntFlag):
    """Reasons for rejecting one channel in one window; combined as bit flags."""

    FLAT = 1  # std below the noise level of a connected dry electrode
    RAIL = 2  # run of samples at the digital limits (ADC or export saturation)
    AMPLITUDE = 4  # peak-to-peak above a physiological ceiling
    VARIANCE = 8  # log-variance outlier relative to the same channel in the same record
    DROPOUT = 16  # all channels hold one value (lost signal), plus guard interval
    COPY = 32  # channel is a sample-by-sample copy of another channel in the record


@dataclass(frozen=True)
class PreprocessingConfig:
    """Parameters of ``preprocess_record``; the defaults are the pre-declared pipeline."""

    target_sfreq: float = config.TARGET_SFREQ  # Hz
    lowpass_hz: float | None = config.HEADER_LOWPASS_HZ  # Hz, -6 dB at 45 Hz; None only for diagnostics
    window_s: float = 4.0  # s, frequency resolution 1 / 4 s = 0.25 Hz
    step_s: float = 2.0  # s, 50 % overlap
    requantize_step_uv: float | None = None  # uV; None keeps the native quantization
    flat_std_uv: float = 0.5  # uV
    max_ptp_uv: float = 400.0  # uV, after low-pass and detrend
    min_rail_run: int = 3  # consecutive native samples at D_min or D_max
    min_constant_s: float = 0.1  # s, all-channel constant stretch treated as dropout
    guard_s: float = 0.5  # s, dilation of saturation and dropout intervals
    variance_z: float = 3.5  # robust z of log-variance, upper tail only
    min_windows_for_variance: int = 5  # the variance criterion needs a stable median

    @property
    def window_samples(self) -> int:
        return int(round(self.window_s * self.target_sfreq))

    @property
    def step_samples(self) -> int:
        return int(round(self.step_s * self.target_sfreq))


@dataclass(frozen=True)
class EpochedRecord:
    """Windows of one record at the target rate with per-channel rejection flags."""

    windows: npt.NDArray[np.float64]  # shape: (n_windows, n_channels, n_samples), unit: uV
    reject: npt.NDArray[np.uint8]  # shape: (n_windows, n_channels), ``Reject`` bit flags
    sfreq: float  # Hz
    channels: tuple[str, ...] = field(default=config.CHANNELS)

    @property
    def good(self) -> npt.NDArray[np.bool_]:
        """Mask of usable (window, channel) pairs. Shape: (n_windows, n_channels)."""
        return self.reject == 0

    @property
    def n_windows(self) -> int:
        return self.windows.shape[0]


def requantize(data: npt.NDArray[np.float64], step_uv: float) -> npt.NDArray[np.float64]:
    """Round to the grid ``step_uv * k``; the error is bounded by ``step_uv / 2``.

    Files already quantized with the same step on the same grid are unchanged.
    """
    return step_uv * np.round(data / step_uv)


def resample(data: npt.NDArray[np.float64], sfreq: float, target_sfreq: float) -> npt.NDArray[np.float64]:
    """Polyphase resampling by the rational factor ``target_sfreq / sfreq``.

    ``scipy.signal.resample_poly`` applies a Kaiser-windowed anti-aliasing FIR
    at the intermediate rate; for 126 -> 125 Hz the factor is up=125, down=126.
    Shape: (n_channels, n_times) -> (n_channels, ceil(n_times * up / down)).
    """
    ratio = Fraction(target_sfreq / sfreq).limit_denominator(1000)
    if ratio == 1:
        return data.copy()
    return resample_poly(data, ratio.numerator, ratio.denominator, axis=1, padtype="line")


def lowpass(data: npt.NDArray[np.float64], sfreq: float, h_freq: float) -> npt.NDArray[np.float64]:
    """Zero-phase windowed-sinc FIR low-pass (MNE defaults, transition 10 Hz at 40 Hz).

    Removes 50 Hz mains that is present in exports without a notch filter.
    Shape is preserved: (n_channels, n_times).
    """
    return mne.filter.filter_data(
        data, sfreq, l_freq=None, h_freq=h_freq, method="fir", fir_design="firwin", phase="zero", verbose="ERROR"
    )


def rail_run_mask(at_rail: npt.NDArray[np.bool_], min_run: int) -> npt.NDArray[np.bool_]:
    """Samples belonging to runs of at least ``min_run`` consecutive rail values.

    A single sample at the limit is expected when the export range is fitted to
    the extremes of the file; saturation produces plateaus.
    Shape is preserved: (n_channels, n_times).
    """
    n_times = at_rail.shape[1]
    mask = np.zeros_like(at_rail)
    if n_times < min_run:
        return mask
    starts = np.lib.stride_tricks.sliding_window_view(at_rail, min_run, axis=1).all(axis=2)
    for offset in range(min_run):
        mask[:, offset : offset + starts.shape[1]] |= starts
    return mask


def sliding_windows(data: npt.NDArray[np.float64], length: int, step: int) -> npt.NDArray[np.float64]:
    """Overlapping windows ``n_windows = floor((n_times - length) / step) + 1``.

    Shape: (n_channels, n_times) -> (n_windows, n_channels, length).
    """
    n_channels, n_times = data.shape
    if n_times < length:
        return np.empty((0, n_channels, length))
    view = np.lib.stride_tricks.sliding_window_view(data, length, axis=1)[:, ::step]
    return np.ascontiguousarray(view.transpose(1, 0, 2))


def copied_channels(data: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
    """Channels whose whole non-flat signal equals another channel. Shape: (n_channels,).

    ``data`` shape: (n_channels, n_times). Both members of an identical pair
    are marked; constant channels are ignored (their equality carries no
    information about copying).
    """
    n_channels = data.shape[0]
    varying = np.ptp(data, axis=1) > 0
    copied = np.zeros(n_channels, dtype=bool)
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            if varying[i] and varying[j] and np.array_equal(data[i], data[j]):
                copied[i] = copied[j] = True
    return copied


def dilate(mask: npt.NDArray[np.bool_], n_samples: int) -> npt.NDArray[np.bool_]:
    """Extend every True interval by ``n_samples`` on both sides along the last axis."""
    if n_samples <= 0 or not mask.any():
        return mask.copy()
    structure = np.ones((1,) * (mask.ndim - 1) + (2 * n_samples + 1,), dtype=bool)
    return binary_dilation(mask, structure=structure)


def _native_mask_to_windows(
    mask: npt.NDArray[np.bool_], n_windows: int, cfg: PreprocessingConfig, native_sfreq: float
) -> npt.NDArray[np.bool_]:
    """Windows overlapping any True native sample.

    ``mask`` shape: (n_rows, n_native_times); returns shape (n_windows, n_rows).
    """
    cumulative = np.concatenate([np.zeros((mask.shape[0], 1), int), np.cumsum(mask, axis=1)], axis=1)
    scale = native_sfreq / cfg.target_sfreq
    starts = np.arange(n_windows) * cfg.step_samples
    lo = np.minimum(np.floor(starts * scale).astype(int), mask.shape[1])
    hi = np.minimum(np.ceil((starts + cfg.window_samples) * scale).astype(int), mask.shape[1])
    return (cumulative[:, hi] - cumulative[:, lo] > 0).T


def _variance_outliers(windows: npt.NDArray[np.float64], already: npt.NDArray[np.bool_], cfg: PreprocessingConfig) -> npt.NDArray[np.bool_]:
    """Upper-tail robust z-score of log-variance within each channel of the record.

    z = (log var - median) / (1.4826 * MAD), median and MAD over windows not
    rejected by the absolute criteria. Shape: (n_windows, n_channels).
    """
    log_var = np.log(windows.var(axis=2) + np.finfo(float).tiny)  # shape: (n_windows, n_channels)
    flags = np.zeros_like(already)
    for ch in range(log_var.shape[1]):
        ref = log_var[~already[:, ch], ch]
        if ref.size < cfg.min_windows_for_variance:
            continue
        med = np.median(ref)
        mad = np.median(np.abs(ref - med))
        if mad == 0:
            continue
        flags[:, ch] = (log_var[:, ch] - med) / (MAD_TO_SIGMA * mad) > cfg.variance_z
    return flags


def preprocess_record(record: EegRecord, cfg: PreprocessingConfig = PreprocessingConfig()) -> EpochedRecord:
    """Condition, window and flag one record.

    Returns
    -------
    EpochedRecord with windows of shape (n_windows, 6, 500) at 125 Hz in uV.
    Records shorter than one window give ``n_windows == 0``.
    """
    copies = copied_channels(record.data)  # shape: (n_channels,)
    constant = constant_stretch_mask(record.data, max(2, round(cfg.min_constant_s * record.sfreq)))
    head, tail = edge_constant_samples(constant)
    keep = slice(head, record.data.shape[1] - tail)
    data, at_rail, constant = record.data[:, keep], record.at_rail[:, keep], constant[keep]

    if cfg.requantize_step_uv is not None:
        data = requantize(data, cfg.requantize_step_uv)
    data = resample(data, record.sfreq, cfg.target_sfreq)  # shape: (n_channels, n_times_125)
    if cfg.lowpass_hz is not None:
        data = lowpass(data, cfg.target_sfreq, cfg.lowpass_hz)

    windows = sliding_windows(data, cfg.window_samples, cfg.step_samples)  # shape: (n_windows, n_ch, n_samples)
    windows = detrend(windows, axis=2, type="linear") if len(windows) else windows
    n_windows, n_channels = windows.shape[:2]

    reject = np.zeros((n_windows, n_channels), dtype=np.uint8)
    if n_windows:
        reject[windows.std(axis=2) < cfg.flat_std_uv] |= np.uint8(Reject.FLAT)
        guard = round(cfg.guard_s * record.sfreq)
        rails = dilate(rail_run_mask(at_rail, cfg.min_rail_run), guard)  # shape: (n_channels, n_native)
        reject[_native_mask_to_windows(rails, n_windows, cfg, record.sfreq)] |= np.uint8(Reject.RAIL)
        dropout = dilate(constant[None, :], guard)  # shape: (1, n_native)
        reject[np.repeat(_native_mask_to_windows(dropout, n_windows, cfg, record.sfreq), n_channels, axis=1)] |= np.uint8(
            Reject.DROPOUT
        )
        reject[np.ptp(windows, axis=2) > cfg.max_ptp_uv] |= np.uint8(Reject.AMPLITUDE)
        reject[_variance_outliers(windows, reject > 0, cfg)] |= np.uint8(Reject.VARIANCE)
        reject[:, copies] |= np.uint8(Reject.COPY)

    return EpochedRecord(windows=windows, reject=reject, sfreq=cfg.target_sfreq, channels=record.channels)


def preprocess_file(path: Path, cfg: PreprocessingConfig = PreprocessingConfig()) -> EpochedRecord:
    """``load_record`` followed by ``preprocess_record``."""
    return preprocess_record(load_record(path), cfg)


def rejection_table(
    registry: pd.DataFrame, cfg: PreprocessingConfig = PreprocessingConfig(), data_dir: Path = config.DATA_DIR
) -> pd.DataFrame:
    """Quality-control summary for every readable file and channel.

    Returns
    -------
    Long DataFrame with one row per (file, channel): ``relpath, channel,
    n_windows, n_good`` and the fraction of windows carrying each ``Reject`` flag.
    """
    rows: list[dict[str, object]] = []
    for relpath in registry.loc[registry["status"] == "ok", "relpath"]:
        epoched = preprocess_file(data_dir / relpath, cfg)
        for ch_idx, channel in enumerate(epoched.channels):
            flags = epoched.reject[:, ch_idx]
            n = epoched.n_windows
            rows.append(
                {
                    "relpath": relpath,
                    "channel": channel,
                    "n_windows": n,
                    "n_good": int((flags == 0).sum()),
                    **{f"frac_{r.name.lower()}": float(((flags & r) > 0).mean()) if n else np.nan for r in Reject},
                }
            )
    return pd.DataFrame(rows)
