"""Bonus branch: metronome grid recovery and the deviant-locked response.

Stimulation (task specification, Appendix 1): an auditory metronome at
2.000 Hz plays during every record, every eighth beat is deviant. Stimulus
markers were not saved, and the metronome is started independently of the
recording, so the grid phase and the deviant position are unknown in every
file. Model of one record after band-pass filtering::

    x(t) = sum_j S(t - t_j) + sum_{j = d (mod 8)} D(t - t_j) + noise,   t_j = phi + 0.5 j  [s]

with S the response to any beat and D the extra response to the deviant.

**Cycle folding.** Eight beats last 4 s = 500 samples at 125 Hz, an integer
number, so the record is cut into consecutive 4-s cycles from its start and
averaged: ``y(t), t in [0, 4 s)``. Beats keep their positions from cycle to
cycle, noise averages out. The DFT of ``y`` has bins ``f_k = k / 4 Hz``:

* **grid subspace**, ``k = 0 (mod 8)`` (2, 4, 6 ... Hz): everything with
  period 0.5 s, i.e. the response to every beat;
* **deviant subspace**, ``k != 0 (mod 4)``: everything with period 4 s (or
  2 s) but not 1 s. Projecting ``y`` onto it gives
  ``3/4 * [D(t - tau) - 1/3 * sum_{m=1..3} D(t - tau - m)]``, tau = phi + 0.5 d:
  the deviant response minus the mean of the beats of the same parity that do
  not follow it (m = 1, 2, 3 s). Anything periodic with 1 s - e.g. a
  disturbance tied to the 1-s EDF data records - is removed exactly.

**Split halves.** Even cycles form half A, odd cycles half B: both halves span
the whole record, so drift and fatigue do not separate them. A quantity
estimated on one half is checked on the other; noise of the two halves is
independent, the metronome-locked signal is shared.

**Random control.** Circularly shifting one half by a random time in
[0, 4 s) keeps its spectrum, artifacts and number of cycles, but destroys its
alignment with the other half. Every test is recomputed on such surrogates,
including the search for the grid phase and the deviant position.

Waveforms are band-limited (1-20 Hz), so they are evaluated on a fine
circular grid of 2 ms by zero-padded inverse FFT; a circular shift by a
multiple of 2 ms on that grid is exact.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import json

import mne
import numpy as np
import numpy.typing as npt
import pandas as pd
from joblib import Parallel, delayed

from src import config
from src.dataset import (
    EegRecord,
    constant_stretch_mask,
    edge_constant_samples,
    load_record,
    resolve_ambiguous_records,
    scan_dataset,
)
from src.preprocessing import copied_channels, dilate, rail_run_mask, resample

BEAT_S: Final[float] = 0.5  # metronome period, s (2.000 Hz)
BEATS_PER_CYCLE: Final[int] = 8  # every eighth beat is deviant
CYCLE_S: Final[float] = BEAT_S * BEATS_PER_CYCLE  # 4 s
CYCLE_SAMPLES: Final[int] = int(round(CYCLE_S * config.TARGET_SFREQ))  # 500 at 125 Hz
N_BINS: Final[int] = CYCLE_SAMPLES // 2 + 1  # rfft bins of one cycle, f_k = k / 4 Hz
BIN_HZ: Final[float] = 1.0 / CYCLE_S  # 0.25 Hz

BAND_HZ: Final[tuple[float, float]] = (1.0, 20.0)  # analysis band; hardware high-pass is 2 Hz
MAX_PTP_UV: Final[float] = 150.0  # peak-to-peak limit of one channel in one 4-s cycle
GUARD_S: Final[float] = 0.5  # dilation of saturation and dropout, as in src.preprocessing
MIN_RAIL_RUN: Final[int] = 3
MIN_CYCLES_PER_HALF: Final[int] = 3

FRONTAL: Final[tuple[str, ...]] = ("Fp1", "Fp2")  # primary channel = mean of the two
GRID_TEST_HZ: Final[tuple[float, ...]] = (2.0, 4.0, 6.0, 8.0, 10.0)  # test 1
ODD_CONTROL_HZ: Final[tuple[float, ...]] = (1.0, 3.0, 5.0, 7.0, 9.0)  # 1-s periodic control of test 1
PHASE_HZ: Final[tuple[float, ...]] = (2.0, 4.0, 6.0)  # waveform whose minimum is taken as N1
N1_LATENCY_S: Final[float] = 0.100  # convention: beat onset = N1 minimum - 100 ms

FINE_HZ: Final[float] = 500.0  # evaluation grid of band-limited cycle waveforms (2 ms)
FINE_POINTS: Final[int] = int(round(CYCLE_S * FINE_HZ))  # 2000
BEAT_POINTS: Final[int] = int(round(BEAT_S * FINE_HZ))  # 250
SEARCH_WINDOW_S: Final[tuple[float, float]] = (0.05, 0.40)  # deviant search, after the beat
EPOCH_S: Final[tuple[float, float]] = (-0.10, 0.60)  # extracted curve
MMN_WINDOW_S: Final[tuple[float, float]] = (0.10, 0.25)
P3A_WINDOW_S: Final[tuple[float, float]] = (0.25, 0.40)
N_SURROGATES: Final[int] = 2000
# Split-half agreement: estimates closer than this count as agreeing. Chance level
# is 0.2 / 0.5 = 40 % for the beat onset and 0.2 / 4 = 5 % for the deviant onset;
# with a reliable beat onset and no deviant information the deviant onset agrees
# in onset_agreement / 8 of the records.
AGREEMENT_S: Final[float] = 0.1
# The deviant subspace holds 3/4 of the deviant response D (see module docstring);
# curves are multiplied by 4/3 so that they estimate D itself, in uV.
DEVIANT_GAIN: Final[float] = 4.0 / 3.0


# ---------------------------------------------------------------------------
# Continuous signal and cycle spectra of one record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContinuousRecord:
    """Band-passed six-channel signal at 125 Hz with a per-sample bad mask."""

    data: npt.NDArray[np.float64]  # shape: (n_channels, n_times), uV
    bad: npt.NDArray[np.bool_]  # shape: (n_channels, n_times); saturation, dropout or copied channel
    channels: tuple[str, ...] = config.CHANNELS


def continuous_record(record: EegRecord, band_hz: tuple[float, float] = BAND_HZ) -> ContinuousRecord:
    """Trim constant edges, resample to 125 Hz, zero-phase FIR band-pass.

    Unlike ``src.preprocessing`` the record is not windowed: the metronome
    grid needs the uninterrupted time axis. Saturation runs and all-channel
    dropouts are located on native samples, dilated by ``GUARD_S`` and mapped
    to the 125-Hz axis; copied channels are bad everywhere.
    """
    copies = copied_channels(record.data)  # shape: (n_channels,)
    constant = constant_stretch_mask(record.data, max(2, round(0.1 * record.sfreq)))
    head, tail = edge_constant_samples(constant)
    keep = slice(head, record.data.shape[1] - tail)
    native, at_rail, constant = record.data[:, keep], record.at_rail[:, keep], constant[keep]

    guard = round(GUARD_S * record.sfreq)
    bad_native = dilate(rail_run_mask(at_rail, MIN_RAIL_RUN), guard) | dilate(constant[None, :], guard)

    data = resample(native, record.sfreq, config.TARGET_SFREQ)  # shape: (n_channels, n_times)
    data = mne.filter.filter_data(
        data, config.TARGET_SFREQ, band_hz[0], band_hz[1], method="fir", phase="zero", verbose="ERROR"
    )
    n_times = data.shape[1]
    native_idx = np.minimum(
        np.floor(np.arange(n_times) * record.sfreq / config.TARGET_SFREQ).astype(int), native.shape[1] - 1
    )
    bad = bad_native[:, native_idx] if native.shape[1] else np.ones_like(data, dtype=bool)
    bad = bad | copies[:, None]
    return ContinuousRecord(data=data, bad=bad, channels=record.channels)


@dataclass(frozen=True)
class CycleSpectra:
    """Spectra of the cycle averages of the two halves of one record."""

    spectra: npt.NDArray[np.complex128]  # shape: (2, n_channels, N_BINS); halves A (even cycles), B (odd)
    n_cycles: npt.NDArray[np.int_]  # shape: (2, n_channels); retained cycles per half and channel
    channels: tuple[str, ...] = config.CHANNELS

    def valid(self, channels: Sequence[str], min_cycles: int = MIN_CYCLES_PER_HALF) -> bool:
        idx = [self.channels.index(ch) for ch in channels]
        return bool((self.n_cycles[:, idx] >= min_cycles).all())

    def channel_mean(self, channels: Sequence[str]) -> npt.NDArray[np.complex128]:
        """Spectrum of the mean of ``channels``. Shape: (2, N_BINS)."""
        idx = [self.channels.index(ch) for ch in channels]
        return self.spectra[:, idx].mean(axis=1)


def cycle_spectra(cont: ContinuousRecord, max_ptp_uv: float = MAX_PTP_UV) -> CycleSpectra:
    """Fold into 4-s cycles, reject cycles per channel, average each half, FFT.

    A cycle is retained for a channel if its peak-to-peak is at most
    ``max_ptp_uv`` and it has no bad sample. Channels of a half without a
    retained cycle get NaN spectra and zero count.
    """
    n_channels, n_times = cont.data.shape
    n_cycles = n_times // CYCLE_SAMPLES
    spectra = np.full((2, n_channels, N_BINS), np.nan + 0j)
    counts = np.zeros((2, n_channels), dtype=int)
    if n_cycles == 0:
        return CycleSpectra(spectra, counts, cont.channels)
    span = n_cycles * CYCLE_SAMPLES
    cycles = cont.data[:, :span].reshape(n_channels, n_cycles, CYCLE_SAMPLES)  # (n_ch, n_cyc, 500)
    bad = cont.bad[:, :span].reshape(n_channels, n_cycles, CYCLE_SAMPLES).any(axis=2)  # (n_ch, n_cyc)
    good = (np.ptp(cycles, axis=2) <= max_ptp_uv) & ~bad  # (n_ch, n_cyc)
    for half in (0, 1):
        sel = slice(half, None, 2)
        weight = good[:, sel].astype(float)  # (n_ch, n_half)
        count = weight.sum(axis=1)
        mean = (cycles[:, sel] * weight[:, :, None]).sum(axis=1) / np.where(count > 0, count, 1.0)[:, None]
        spectra[half] = np.where(count[:, None] > 0, np.fft.rfft(mean, axis=1), np.nan)
        counts[half] = count
    return CycleSpectra(spectra, counts, cont.channels)


def record_spectra_from(record: EegRecord) -> CycleSpectra:
    """``continuous_record`` -> ``cycle_spectra``."""
    return cycle_spectra(continuous_record(record))


def record_cycle_spectra(path: Path) -> CycleSpectra:
    """Cycle spectra of one EDF file."""
    return record_spectra_from(load_record(path))


# ---------------------------------------------------------------------------
# Subspaces and band-limited waveforms on the fine circular grid
# ---------------------------------------------------------------------------

_K: Final[npt.NDArray[np.int_]] = np.arange(N_BINS)
_F: Final[npt.NDArray[np.float64]] = _K * BIN_HZ
_IN_BAND: Final[npt.NDArray[np.bool_]] = (_F >= BAND_HZ[0]) & (_F <= BAND_HZ[1])
GRID_MASK: Final[npt.NDArray[np.bool_]] = (_K % BEATS_PER_CYCLE == 0) & _IN_BAND  # period 0.5 s
DEVIANT_MASK: Final[npt.NDArray[np.bool_]] = (_K % (BEATS_PER_CYCLE // 2) != 0) & _IN_BAND  # not 1-s periodic


def frequency_mask(freqs_hz: Sequence[float]) -> npt.NDArray[np.bool_]:
    """Bins at exactly the given frequencies (multiples of 0.25 Hz). Shape: (N_BINS,)."""
    mask = np.zeros(N_BINS, dtype=bool)
    mask[np.rint(np.asarray(freqs_hz) / BIN_HZ).astype(int)] = True
    return mask


def fine_waveform(spectrum: npt.NDArray[np.complex128], mask: npt.NDArray[np.bool_]) -> npt.NDArray[np.float64]:
    """Cycle waveform of the masked bins on the 2-ms circular grid.

    Zero-padded inverse FFT = exact band-limited interpolation of the 500-sample
    cycle. Shape: (..., N_BINS) -> (..., FINE_POINTS), uV.
    """
    padded = np.zeros(spectrum.shape[:-1] + (FINE_POINTS // 2 + 1,), dtype=complex)
    padded[..., :N_BINS] = np.where(mask, spectrum, 0.0)
    return np.fft.irfft(padded, n=FINE_POINTS, axis=-1) * (FINE_POINTS / CYCLE_SAMPLES)


def roll_fine(waveforms: npt.NDArray[np.float64], shifts: npt.NDArray[np.int_]) -> npt.NDArray[np.float64]:
    """Delay each row by its own number of fine samples: out[i, t] = in[i, t - shift_i]."""
    idx = (np.arange(FINE_POINTS)[None, :] - np.asarray(shifts)[:, None]) % FINE_POINTS
    return np.take_along_axis(waveforms, idx, axis=1)


def _points(window_s: tuple[float, float]) -> npt.NDArray[np.int_]:
    return np.arange(int(round(window_s[0] * FINE_HZ)), int(round(window_s[1] * FINE_HZ)))


SEARCH_POINTS: Final[npt.NDArray[np.int_]] = _points(SEARCH_WINDOW_S)
EPOCH_POINTS: Final[npt.NDArray[np.int_]] = _points(EPOCH_S)
EPOCH_TIMES_S: Final[npt.NDArray[np.float64]] = EPOCH_POINTS / FINE_HZ


# ---------------------------------------------------------------------------
# Recovery of the grid phase and the deviant position (vectorised over records)
# ---------------------------------------------------------------------------


def grid_onsets(phase_waveforms: npt.NDArray[np.float64]) -> npt.NDArray[np.int_]:
    """Beat onset within the 0.5-s period, in fine samples.

    ``phase_waveforms`` shape (n, FINE_POINTS): the waveform of the 2, 4, 6 Hz
    bins, periodic with 0.5 s. Its minimum is taken as the auditory N1 and the
    onset is placed ``N1_LATENCY_S`` earlier. Returns shape (n,), values in
    [0, BEAT_POINTS).
    """
    trough = np.argmin(phase_waveforms[:, :BEAT_POINTS], axis=1)
    return (trough - int(round(N1_LATENCY_S * FINE_HZ))) % BEAT_POINTS


def deviant_onsets(deviant_waveforms: npt.NDArray[np.float64], onsets: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
    """Onset of the deviant beat within the 4-s cycle, in fine samples.

    Candidates are the eight beats ``onset + 0.5 d``; the chosen one maximises
    the energy of the deviant-subspace waveform in ``SEARCH_WINDOW_S`` after
    it (no assumption on the polarity). Shapes: (n, FINE_POINTS), (n,) -> (n,).
    """
    n = deviant_waveforms.shape[0]
    candidates = onsets[:, None] + BEAT_POINTS * np.arange(BEATS_PER_CYCLE)[None, :]  # (n, 8)
    idx = (candidates[:, :, None] + SEARCH_POINTS[None, None, :]) % FINE_POINTS  # (n, 8, n_search)
    values = np.take_along_axis(deviant_waveforms, idx.reshape(n, -1), axis=1).reshape(idx.shape)
    best = np.argmax((values**2).sum(axis=2), axis=1)  # (n,)
    return candidates[np.arange(n), best] % FINE_POINTS


def epochs_at(waveforms: npt.NDArray[np.float64], onsets: npt.NDArray[np.int_]) -> npt.NDArray[np.float64]:
    """Segments ``EPOCH_S`` around each onset. Shapes: (n, [c,] FINE_POINTS), (n,) -> (n, [c,] n_epoch)."""
    idx = (onsets[:, None] + EPOCH_POINTS[None, :]) % FINE_POINTS  # (n, n_epoch)
    if waveforms.ndim == 3:
        idx = np.broadcast_to(idx[:, None, :], waveforms.shape[:2] + idx.shape[1:])
    return np.take_along_axis(waveforms, idx, axis=-1)


def half_correlation(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Pearson-type correlation without centring of two zero-mean waveforms. (n, T), (n, T) -> (n,)."""
    num = (a * b).sum(axis=1)
    den = np.sqrt((a**2).sum(axis=1) * (b**2).sum(axis=1))
    return num / np.where(den > 0, den, np.nan)


@dataclass(frozen=True)
class HalfWaveforms:
    """Fine-grid waveforms of the two halves of n records; axis 0 of each array is the half."""

    grid_test: npt.NDArray[np.float64]  # shape: (2, n, FINE_POINTS); 2-10 Hz at the primary channel
    odd_control: npt.NDArray[np.float64]  # shape: (2, n, FINE_POINTS); 1, 3 ... 9 Hz
    phase: npt.NDArray[np.float64]  # shape: (2, n, FINE_POINTS); 2, 4, 6 Hz
    deviant: npt.NDArray[np.float64]  # shape: (2, n, FINE_POINTS); deviant subspace, primary channel
    deviant_channels: npt.NDArray[np.float64]  # shape: (2, n, n_channels, FINE_POINTS)
    grid_channels: npt.NDArray[np.float64]  # shape: (2, n, n_channels, FINE_POINTS); grid subspace


def half_waveforms(records: Sequence[CycleSpectra], primary: Sequence[str] = FRONTAL) -> HalfWaveforms:
    """Waveforms used by the tests, for records already checked with ``CycleSpectra.valid``."""
    primary_spec = np.stack([r.channel_mean(primary) for r in records], axis=1)  # (2, n, N_BINS)
    all_spec = np.stack([r.spectra for r in records], axis=1)  # (2, n, n_ch, N_BINS)
    return HalfWaveforms(
        grid_test=fine_waveform(primary_spec, frequency_mask(GRID_TEST_HZ)),
        odd_control=fine_waveform(primary_spec, frequency_mask(ODD_CONTROL_HZ)),
        phase=fine_waveform(primary_spec, frequency_mask(PHASE_HZ)),
        deviant=fine_waveform(primary_spec, DEVIANT_MASK),
        deviant_channels=fine_waveform(all_spec, DEVIANT_MASK),
        grid_channels=fine_waveform(all_spec, GRID_MASK),
    )


def recover(w: HalfWaveforms, shift_b: npt.NDArray[np.int_] | None = None) -> dict[str, npt.NDArray[np.float64]]:
    """Tests 1-3 on n records; ``shift_b`` (n,) circularly delays half B (random control).

    Returns per-record arrays: ``grid_r``, ``odd_r``, ``deviant_r`` (half
    correlations of tests 1, 2 and the 1-s control), ``onset_a``, ``onset_b``,
    ``tau_a``, ``tau_b`` (fine samples), ``curve`` (n, n_epoch) - estimate of
    the deviant response D at the primary channel, each half aligned by the
    other half's estimate and the two averaged.
    """
    n = w.deviant.shape[1]
    shift_b = np.zeros(n, dtype=int) if shift_b is None else np.asarray(shift_b)

    def b(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return roll_fine(x, shift_b)

    phase_a, phase_b = w.phase[0], b(w.phase[1])
    dev_a, dev_b = w.deviant[0], b(w.deviant[1])
    onset_a, onset_b = grid_onsets(phase_a), grid_onsets(phase_b)
    tau_a, tau_b = deviant_onsets(dev_a, onset_a), deviant_onsets(dev_b, onset_b)
    curve = 0.5 * DEVIANT_GAIN * (epochs_at(dev_b, tau_a) + epochs_at(dev_a, tau_b))  # (n, n_epoch), uV
    return {
        "grid_r": half_correlation(w.grid_test[0], b(w.grid_test[1])),
        "odd_r": half_correlation(w.odd_control[0], b(w.odd_control[1])),
        "deviant_r": half_correlation(dev_a, dev_b),
        "onset_a": onset_a,
        "onset_b": onset_b,
        "tau_a": tau_a,
        "tau_b": tau_b,
        "curve": curve,
    }


def aligned_channel_curves(w: HalfWaveforms, out: dict[str, npt.NDArray[np.float64]]) -> dict[str, npt.NDArray[np.float64]]:
    """All-channel curves of the observed data, each half aligned by the other half's estimate.

    ``deviant``: deviant subspace at the deviant onset; ``grid``: grid subspace
    (response to every beat) at the beat onset. Shapes: (n, n_channels, n_epoch).
    """
    deviant = 0.5 * DEVIANT_GAIN * (
        epochs_at(w.deviant_channels[1], out["tau_a"]) + epochs_at(w.deviant_channels[0], out["tau_b"])
    )
    grid = 0.5 * (epochs_at(w.grid_channels[1], out["onset_a"]) + epochs_at(w.grid_channels[0], out["onset_b"]))
    return {"deviant": deviant, "grid": grid}


def window_mean(curves: npt.NDArray[np.float64], window_s: tuple[float, float]) -> npt.NDArray[np.float64]:
    """Mean over ``window_s`` of curves on ``EPOCH_TIMES_S``. Shape: (..., n_epoch) -> (...)."""
    sel = (EPOCH_TIMES_S >= window_s[0]) & (EPOCH_TIMES_S < window_s[1])
    return curves[..., sel].mean(axis=-1)


def group_statistics(out: dict[str, npt.NDArray[np.float64]], subject: npt.NDArray[np.int_]) -> dict[str, float]:
    """Group statistics with equal weight per subject (records of a subject averaged first)."""
    def per_subject(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return _subject_means(values, subject)

    curve = per_subject(out["curve"]).mean(axis=0)  # (n_epoch,)
    sel = (EPOCH_TIMES_S >= SEARCH_WINDOW_S[0]) & (EPOCH_TIMES_S < SEARCH_WINDOW_S[1])
    distance = np.abs(out["tau_a"] - out["tau_b"]) % FINE_POINTS
    distance = np.minimum(distance, FINE_POINTS - distance) / FINE_HZ  # s, circular on 4 s
    beat = np.abs(out["onset_a"] - out["onset_b"]) % BEAT_POINTS
    beat = np.minimum(beat, BEAT_POINTS - beat) / FINE_HZ  # s, circular on 0.5 s
    return {
        "grid_r": float(per_subject(out["grid_r"]).mean()),
        "odd_r": float(per_subject(out["odd_r"]).mean()),
        "deviant_r": float(per_subject(out["deviant_r"]).mean()),
        "curve_energy": float((curve[sel] ** 2).mean()),
        "mmn_mean": float(window_mean(curve, MMN_WINDOW_S)),
        "p3a_mean": float(window_mean(curve, P3A_WINDOW_S)),
        "onset_agreement": float(per_subject((beat < AGREEMENT_S).astype(float)).mean()),
        "tau_agreement": float(per_subject((distance < AGREEMENT_S).astype(float)).mean()),
    }


# Direction of the effect under the alternative: +1 larger, -1 smaller than the null.
TEST_DIRECTION: Final[dict[str, int]] = {
    "grid_r": 1,
    "odd_r": 1,
    "deviant_r": 1,
    "curve_energy": 1,
    "mmn_mean": -1,
    "p3a_mean": 1,
    "onset_agreement": 1,
    "tau_agreement": 1,
}


def random_control(
    w: HalfWaveforms, subject: npt.NDArray[np.int_], n_surrogates: int = N_SURROGATES, seed: int = config.RANDOM_STATE
) -> tuple[dict[str, npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
    """Group statistics of ``n_surrogates`` random controls and their primary-channel curves.

    In every surrogate half B of each record is delayed by an independent
    uniform random time on the 2-ms grid of [0, 4 s), then the whole
    procedure - grid phase, deviant search, extraction - is repeated.
    Returns ({statistic: (n_surrogates,)}, curves (n_surrogates, n_epoch)).
    """
    rng = np.random.default_rng(seed)
    n = w.deviant.shape[1]
    stats: dict[str, list[float]] = {key: [] for key in TEST_DIRECTION}
    curves = []
    for _ in range(n_surrogates):
        out = recover(w, rng.integers(0, FINE_POINTS, size=n))
        for key, value in group_statistics(out, subject).items():
            stats[key].append(value)
        curves.append(_subject_means(out["curve"], subject).mean(axis=0))
    return {key: np.array(values) for key, values in stats.items()}, np.array(curves)


def p_value(observed: float, null: npt.NDArray[np.float64], direction: int) -> float:
    """One-sided permutation p = (1 + #{null at least as extreme}) / (1 + n)."""
    extreme = null >= observed if direction > 0 else null <= observed
    return float((1 + extreme.sum()) / (1 + null.size))


# ---------------------------------------------------------------------------
# Pilot on the training set
# ---------------------------------------------------------------------------

RESULTS_DIR: Final[Path] = config.REPO_ROOT / "results" / "validation" / "metronome_pilot"
EEG_TYPICAL_FAMILIES: Final[tuple[str, ...]] = ("A", "B")  # format C lacks typical EEG features (docs/DATA_AUDIT.md)
CHANNEL_SURROGATES: Final[int] = 500  # random controls of the descriptive per-channel tests


def analysis_records(registry: pd.DataFrame, condition: str, data_dir: Path = config.DATA_DIR) -> pd.DataFrame:
    """Readable files of formats A and B in ``condition`` ("rest" or "task").

    Ambiguous records are removed by the within-subject policy of
    ``dataset.resolve_ambiguous_records``; of files with identical digital
    content only the first (sorted by path) is kept, so a recording shared by
    two subject folders counts once.
    """
    ok = registry[(registry["status"] == "ok") & registry["export_family"].isin(EEG_TYPICAL_FAMILIES)]
    usable_paths: set[Path] = set()
    for _, recs in ok.groupby("subject_key"):
        files = {stem: data_dir / rel for stem, rel in zip(recs["stem"], recs["relpath"])}
        usable, _ = resolve_ambiguous_records(files)
        usable_paths |= set(usable.values())
    keep = ok["relpath"].map(lambda rel: data_dir / rel in usable_paths) & (ok["condition"] == condition)
    return ok[keep].sort_values("relpath").drop_duplicates("content_hash").reset_index(drop=True)


def stratum(row: pd.Series) -> str:
    """Evaluation stratum of a registry row joined with the subject table (reporting only)."""
    if row["group"] == config.GROUP_PTSD:
        return "ptsd"
    if row["group"] == config.GROUP_SOMATOFORM:
        return "somatoform"
    if row["holdout"]:
        return "control_ageing"
    return f"control_{row['export_family']}" + ("" if row["has_task_files"] else "_supplement")


def channel_half_correlations(
    w: HalfWaveforms, shift_b: npt.NDArray[np.int_] | None = None
) -> dict[str, npt.NDArray[np.float64]]:
    """Tests 1 and 2 on every channel (descriptive topography). Shapes: (n, n_channels)."""
    n, n_channels = w.deviant_channels.shape[1:3]
    shift_b = np.zeros(n, dtype=int) if shift_b is None else np.asarray(shift_b)
    out = {}
    for name, both in (("grid_r", w.grid_channels), ("deviant_r", w.deviant_channels)):
        out[name] = np.stack(
            [half_correlation(both[0][:, c], roll_fine(both[1][:, c], shift_b)) for c in range(n_channels)], axis=1
        )
    return out


def _subject_means(values: npt.NDArray[np.float64], subject: npt.NDArray[np.int_]) -> npt.NDArray[np.float64]:
    """Average records of each subject, ignoring NaN. Shape: (n_records, ...) -> (n_subjects, ...)."""
    labels, inverse = np.unique(subject, return_inverse=True)
    finite = np.isfinite(values)
    total = np.zeros((labels.size,) + values.shape[1:])
    count = np.zeros_like(total)
    np.add.at(total, inverse, np.where(finite, values, 0.0))
    np.add.at(count, inverse, finite.astype(float))
    return np.where(count > 0, total / np.where(count > 0, count, 1.0), np.nan)


def condition_report(
    rows: pd.DataFrame, spectra: Sequence[CycleSpectra], n_surrogates: int, seed: int
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Tests 1-3 with random control, per-channel topography, group curves, per-record estimates."""
    valid = np.array([s.valid(FRONTAL) for s in spectra])
    rows = rows[valid].reset_index(drop=True)
    spectra = [s for s, v in zip(spectra, valid) if v]
    subject = pd.factorize(rows["subject_key"])[0]
    w = half_waveforms(spectra)
    out = recover(w)
    observed = group_statistics(out, subject)
    null, null_curves = random_control(w, subject, n_surrogates, seed)

    # Descriptive per-channel results use the records where that channel has enough cycles.
    channel_ok = np.stack([(s.n_cycles >= MIN_CYCLES_PER_HALF).all(axis=0) for s in spectra])  # (n, n_ch)

    def channel_group(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.nanmean(_subject_means(np.where(channel_ok, values, np.nan), subject), axis=0)

    rng = np.random.default_rng(seed + 1)
    obs_ch = {k: channel_group(v) for k, v in channel_half_correlations(w).items()}
    null_ch: dict[str, list[npt.NDArray[np.float64]]] = {k: [] for k in obs_ch}
    for _ in range(CHANNEL_SURROGATES):
        shifted = channel_half_correlations(w, rng.integers(0, FINE_POINTS, size=len(rows)))
        for key, value in shifted.items():
            null_ch[key].append(channel_group(value))
    channels = {
        ch: {
            "n_records": int(channel_ok[:, c].sum()),
            **{
                key: {
                    "observed": round(float(obs_ch[key][c]), 4),
                    "p": round(p_value(float(obs_ch[key][c]), np.array(null_ch[key])[:, c], 1), 4),
                }
                for key in obs_ch
            },
        }
        for c, ch in enumerate(config.CHANNELS)
    }
    fp = [config.CHANNELS.index(ch) for ch in FRONTAL]
    report = {
        "n_records": int(len(rows)),
        "n_subjects": int(np.unique(subject).size),
        "records_by_stratum": {str(k): int(v) for k, v in rows["stratum"].value_counts().sort_index().items()},
        "fp_cycles_per_half_median": float(np.median([s.n_cycles[:, fp].min() for s in spectra])),
        "observed": {k: round(v, 5) for k, v in observed.items()},
        "null_mean": {k: round(float(v.mean()), 5) for k, v in null.items()},
        "null_q05": {k: round(float(np.quantile(v, 0.05)), 5) for k, v in null.items()},
        "null_q95": {k: round(float(np.quantile(v, 0.95)), 5) for k, v in null.items()},
        "p": {k: round(p_value(observed[k], null[k], TEST_DIRECTION[k]), 4) for k in observed},
        "tau_agreement_expected_from_onset_only": round(observed["onset_agreement"] / BEATS_PER_CYCLE, 5),
        "channels": channels,
    }

    frames = []
    for kind, values in aligned_channel_curves(w, out).items():  # (n, n_ch, n_epoch)
        per_subject = _subject_means(np.where(channel_ok[:, :, None], values, np.nan), subject)
        n_subjects = np.isfinite(per_subject).sum(axis=0)
        mean = np.nanmean(per_subject, axis=0)
        sem = np.nanstd(per_subject, axis=0, ddof=1) / np.sqrt(n_subjects)
        for c, ch in enumerate(config.CHANNELS):
            frames.append(
                pd.DataFrame({"kind": kind, "channel": ch, "time_s": EPOCH_TIMES_S, "mean_uv": mean[c], "sem_uv": sem[c]})
            )
    frames.append(
        pd.DataFrame(
            {
                "kind": "deviant",
                "channel": "Fp",
                "time_s": EPOCH_TIMES_S,
                "mean_uv": _subject_means(out["curve"], subject).mean(axis=0),
                "null_q025_uv": np.quantile(null_curves, 0.025, axis=0),
                "null_q975_uv": np.quantile(null_curves, 0.975, axis=0),
            }
        )
    )
    per_record = rows[["subject_key", "relpath", "stratum", "export_family"]].assign(
        fp_cycles_a=[int(s.n_cycles[0, fp].min()) for s in spectra],
        fp_cycles_b=[int(s.n_cycles[1, fp].min()) for s in spectra],
        grid_r=out["grid_r"],
        deviant_r=out["deviant_r"],
        onset_a_s=out["onset_a"] / FINE_HZ,
        onset_b_s=out["onset_b"] / FINE_HZ,
        deviant_a_s=out["tau_a"] / FINE_HZ,
        deviant_b_s=out["tau_b"] / FINE_HZ,
    )
    return report, pd.concat(frames, ignore_index=True), per_record


def main(n_surrogates: int = N_SURROGATES, n_jobs: int = -1) -> None:
    """Pilot: rest (primary) and Schulte trials (secondary); writes ``RESULTS_DIR``."""
    registry, _, subjects = scan_dataset()
    registry = registry.join(subjects[["has_task_files"]], on="subject_key")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, object] = {
        "procedure": "docs/DATA_AUDIT.md, section 'Бонус: метроном, MMN/P3a', pilot",
        "n_surrogates": n_surrogates,
        "primary_channel": "mean of " + ", ".join(FRONTAL),
    }
    for condition, seed in (("rest", config.RANDOM_STATE), ("task", config.RANDOM_STATE + 100)):
        rows = analysis_records(registry, condition)
        rows["stratum"] = rows.apply(stratum, axis=1)
        spectra = Parallel(n_jobs=n_jobs)(delayed(record_cycle_spectra)(config.DATA_DIR / rel) for rel in rows["relpath"])
        report, curves, per_record = condition_report(rows, spectra, n_surrogates, seed)
        metrics[condition] = report
        curves.to_csv(RESULTS_DIR / f"curves_{condition}.csv", index=False)
        per_record.to_csv(RESULTS_DIR / f"records_{condition}.csv", index=False)
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
