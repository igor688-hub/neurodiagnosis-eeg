"""Hybrid validation of the metronome branch: known responses in real EEG noise.

The pilot (``src.metronome``) found the response to every beat but not the
deviant response. This module measures what the same procedure can detect:
realistic beat and deviant responses with known timing are added to real
resting records of the training set, and the pilot tests are run on them.

**Noise.** The 155 resting records of the pilot, read as if their sampling rate
were ``20/21`` of the header value: after the usual resampling to 125 Hz the
record is 5 % longer and every frequency is 5 % lower. The genuine metronome
response then has a period of 0.525 s (4.2 s for the deviant) and does not
accumulate over 4-s cycles; exact coincidences of its harmonics with cycle
bins fall on integer frequencies only, which the deviant subspace excludes.
Artifacts, alpha, missing channels and rejected cycles are those of the real
records. A random start offset of the cycle folding (0-499 samples) makes
the noise differ between replicates.

**Signal.** ERP CORE grand averages (``src.erp_core``) on our six channels:
the standard response 0-400 ms as the response to every beat, the deviant
minus standard difference -200..800 ms times ``a`` at the deviant beat. Both
pass the emulated hardware high-pass (causal Butterworth, order 4, 2 Hz; the
roll-off of our own spectra below 2 Hz fits order 4 best). The signal is
exactly periodic with 4 s, so it is built on one cycle in the frequency
domain, where time shifts and the filter are exact::

    Y_sig(f_k) = H(f_k) * [ b * B(f_k) * sum_j e^{-2 pi i f_k t_j} + a * D(f_k) * e^{-2 pi i f_k tau} ]

and is added to the cycle spectra of both halves (every retained cycle holds
the same signal). ``b`` is calibrated so that test 1 reproduces the beat
response strength observed in the pilot.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd
from joblib import Parallel, delayed
from scipy.signal.windows import hann, tukey

from src import config, erp_core, metronome
from src.dataset import EegRecord, load_record

STRETCH: Final[Fraction] = Fraction(21, 20)  # time scale of the noise records
HIGHPASS_ORDER: Final[int] = 4
AMPLITUDES: Final[tuple[float, ...]] = (0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0)  # multiples of the ERP CORE difference
N_REPLICATES: Final[int] = 20
NULL_SURROGATES: Final[int] = 300
VARIANT_BANDS: Final[dict[str, tuple[float, float]]] = {"pilot_1_20Hz": metronome.BAND_HZ, "mmn_2_8Hz": (2.0, 8.0)}
TARGET_GRID_R: Final[float] = 0.163  # test 1 observed in the pilot (results/validation/metronome_pilot)
CALIBRATION_SCALES: Final[tuple[float, ...]] = tuple(float(x) for x in np.geomspace(0.1, 3.2, 11))
CALIBRATION_REPLICATES: Final[int] = 5
BEAT_WINDOW_S: Final[tuple[float, float]] = (0.0, 0.4)  # standard response used for one beat
BEAT_TAPER_S: Final[float] = 0.1  # Hann half-window at the end of the beat template
DEVIANT_WINDOW_S: Final[tuple[float, float]] = (-0.2, 0.8)
DEVIANT_TUKEY_ALPHA: Final[float] = 0.2  # 10 % taper at each end
PHASE_TOLERANCE_S: Final[float] = 0.05
CURVE_AMPLITUDE: Final[float] = 4.0  # amplitude at which recovered and injected curves are compared
RESULTS_DIR: Final[Path] = config.REPO_ROOT / "results" / "validation" / "metronome_validation"

_F: Final[npt.NDArray[np.float64]] = np.arange(metronome.N_BINS) * metronome.BIN_HZ


@dataclass(frozen=True)
class Templates:
    """Single-response templates at 125 Hz on our six channels, before the hardware high-pass."""

    beat: npt.NDArray[np.float64]  # shape: (n_channels, n_beat), starts at the beat onset
    deviant: npt.NDArray[np.float64]  # shape: (n_channels, n_deviant), starts DEVIANT_WINDOW_S[0] before onset
    deviant_start_s: float


def templates_from(erpsets: list[erp_core.ErpSet]) -> Templates:
    """Grand-average ERP CORE standard and deviant-minus-standard waves, resampled and tapered."""
    sites = list(erp_core.CHANNEL_MAP.values())
    idx = [erpsets[0].channels.index(site) for site in sites]
    standard = np.mean([e.bins[erp_core.STANDARD_BIN][idx] for e in erpsets], axis=0)  # (6, n_t) at 256 Hz
    deviant = np.mean([e.bins[erp_core.DEVIANT_BIN][idx] for e in erpsets], axis=0)
    t0, sfreq = erpsets[0].times_s[0], erpsets[0].sfreq
    standard_125 = erp_core.to_125hz(standard, sfreq)
    difference_125 = erp_core.to_125hz(deviant - standard, sfreq)
    times = t0 + np.arange(standard_125.shape[1]) / config.TARGET_SFREQ

    beat_sel = (times >= BEAT_WINDOW_S[0]) & (times < BEAT_WINDOW_S[1])
    beat = standard_125[:, beat_sel]
    n_taper = int(round(BEAT_TAPER_S * config.TARGET_SFREQ))
    taper = np.ones(beat.shape[1])
    taper[-n_taper:] = hann(2 * n_taper)[n_taper:]
    beat = (beat - beat[:, :1]) * taper  # starts at zero, ends at zero

    dev_sel = (times >= DEVIANT_WINDOW_S[0]) & (times < DEVIANT_WINDOW_S[1])
    deviant_t = difference_125[:, dev_sel] * tukey(int(dev_sel.sum()), DEVIANT_TUKEY_ALPHA)
    return Templates(beat=beat, deviant=deviant_t, deviant_start_s=float(times[dev_sel][0]))


def template_spectrum(template: npt.NDArray[np.float64], start_s: float) -> npt.NDArray[np.complex128]:
    """Cycle spectrum of one response starting at ``start_s`` relative to an onset at t = 0.

    Shape: (n_channels, n) -> (n_channels, N_BINS); circular on the 4-s cycle.
    """
    padded = np.zeros((template.shape[0], metronome.CYCLE_SAMPLES))
    padded[:, : template.shape[1]] = template
    return np.fft.rfft(padded, axis=1) * np.exp(-2j * np.pi * _F * start_s)


def onset_phasors(onsets_s: npt.NDArray[np.float64]) -> npt.NDArray[np.complex128]:
    """Sum of time-shift factors of the onsets. Shape: (n_onsets,) -> (N_BINS,)."""
    return np.exp(-2j * np.pi * _F[None, :] * np.asarray(onsets_s)[:, None]).sum(axis=0)


@dataclass(frozen=True)
class SignalModel:
    """Filtered spectra of one beat response and one deviant response at onset 0."""

    beat: npt.NDArray[np.complex128]  # shape: (n_channels, N_BINS)
    deviant: npt.NDArray[np.complex128]  # shape: (n_channels, N_BINS)

    @classmethod
    def from_templates(cls, templates: Templates, highpass_order: int = HIGHPASS_ORDER) -> "SignalModel":
        response = erp_core.highpass_response(_F, highpass_order)
        return cls(
            beat=template_spectrum(templates.beat, 0.0) * response,
            deviant=template_spectrum(templates.deviant, templates.deviant_start_s) * response,
        )

    def spectrum(self, phase_s: float, deviant: int, beat_scale: float, amplitude: float) -> npt.NDArray[np.complex128]:
        """Cycle spectrum of beats at ``phase_s + 0.5 j`` and the deviant at beat ``deviant``."""
        beats = phase_s + metronome.BEAT_S * np.arange(metronome.BEATS_PER_CYCLE)
        tau = phase_s + metronome.BEAT_S * deviant
        return beat_scale * self.beat * onset_phasors(beats) + amplitude * self.deviant * onset_phasors(np.array([tau]))


def stretch(record: EegRecord) -> metronome.ContinuousRecord:
    """Continuous record read at ``1 / STRETCH`` of its sampling rate: time runs 5 % slower."""
    return metronome.continuous_record(dataclasses.replace(record, sfreq=record.sfreq / float(STRETCH)))


def stretched_continuous(path: Path) -> metronome.ContinuousRecord:
    return stretch(load_record(path))


def fold_from(cont: metronome.ContinuousRecord, offset: int) -> metronome.CycleSpectra:
    """Cycle spectra with the folding started ``offset`` samples into the record."""
    return metronome.cycle_spectra(metronome.ContinuousRecord(cont.data[:, offset:], cont.bad[:, offset:], cont.channels))


@dataclass(frozen=True)
class HybridData:
    records: list[metronome.CycleSpectra]
    phase_s: npt.NDArray[np.float64]  # shape: (n,), true beat onset within 0.5 s
    deviant: npt.NDArray[np.int_]  # shape: (n,), true deviant beat 0..7
    subject: npt.NDArray[np.int_]  # shape: (n,)


def hybrid(
    noise: list[metronome.CycleSpectra], model: SignalModel, beat_scale: float, amplitude: float, rng: np.random.Generator
) -> HybridData:
    """Add a signal with random phase and deviant position to every noise record."""
    phase = rng.uniform(0.0, metronome.BEAT_S, size=len(noise))
    deviant = rng.integers(0, metronome.BEATS_PER_CYCLE, size=len(noise))
    records = [
        metronome.CycleSpectra(r.spectra + model.spectrum(p, d, beat_scale, amplitude)[None], r.n_cycles, r.channels)
        for r, p, d in zip(noise, phase, deviant)
    ]
    return HybridData(records, phase, deviant, np.arange(len(noise)))


def fold_noise(conts: list[metronome.ContinuousRecord], rng: np.random.Generator) -> list[metronome.CycleSpectra]:
    """Fold every record from a random offset; keep records valid on the primary channel."""
    folded = [fold_from(c, int(rng.integers(0, metronome.CYCLE_SAMPLES))) for c in conts]
    return [f for f in folded if f.valid(metronome.FRONTAL)]


def circular_error(estimate_s: npt.NDArray[np.float64], truth_s: npt.NDArray[np.float64], period_s: float) -> npt.NDArray[np.float64]:
    """Signed error wrapped to [-period/2, period/2)."""
    return (estimate_s - truth_s + period_s / 2) % period_s - period_s / 2


def evaluate(data: HybridData, band_hz: tuple[float, float]) -> tuple[dict[str, float], metronome.HalfWaveforms, dict]:
    """Pilot statistics plus recovery accuracy against the known truth."""
    w = metronome.half_waveforms(data.records, deviant_band_hz=band_hz)
    out = metronome.recover(w)
    stats = metronome.group_statistics(out, data.subject)
    phase_error = circular_error(out["onset_a"] / metronome.FINE_HZ, data.phase_s, metronome.BEAT_S)
    bias = float(np.angle(np.mean(np.exp(2j * np.pi * phase_error / metronome.BEAT_S))) / (2 * np.pi) * metronome.BEAT_S)
    tau_true = data.phase_s + metronome.BEAT_S * data.deviant
    tau_error = circular_error(out["tau_a"] / metronome.FINE_HZ, tau_true, metronome.CYCLE_S)
    stats.update(
        phase_bias_s=bias,
        phase_within_50ms=float(np.mean(np.abs(phase_error) <= PHASE_TOLERANCE_S)),
        phase_within_50ms_of_bias=float(np.mean(np.abs(circular_error(phase_error, bias, metronome.BEAT_S)) <= PHASE_TOLERANCE_S)),
        deviant_found=float(np.mean(np.abs(tau_error) <= metronome.AGREEMENT_S)),
    )
    return stats, w, out


def calibrate_beat_scale(
    conts: list[metronome.ContinuousRecord], model: SignalModel, seed: int
) -> tuple[float, list[dict[str, float]]]:
    """Beat scale at which the mean test-1 statistic equals ``TARGET_GRID_R`` (a = 0, log interpolation)."""
    rng = np.random.default_rng(seed)
    rows = []
    for scale in CALIBRATION_SCALES:
        values = []
        for _ in range(CALIBRATION_REPLICATES):
            data = hybrid(fold_noise(conts, rng), model, scale, 0.0, rng)
            values.append(evaluate(data, metronome.BAND_HZ)[0]["grid_r"])
        rows.append({"beat_scale": scale, "grid_r": float(np.mean(values))})
    grid_r = np.array([r["grid_r"] for r in rows])
    scale = float(np.exp(np.interp(TARGET_GRID_R, grid_r, np.log(CALIBRATION_SCALES))))
    return scale, rows


def expected_curve(model: SignalModel, amplitude: float, band_hz: tuple[float, float]) -> npt.NDArray[np.float64]:
    """Deviant curve the extraction would give with perfect alignment. Shape: (n_channels, n_epoch)."""
    spectrum = amplitude * model.deviant  # deviant at onset 0
    fine = metronome.fine_waveform(spectrum, metronome.deviant_mask(band_hz))  # (n_ch, FINE_POINTS)
    return metronome.DEVIANT_GAIN * metronome.epochs_at(fine[None], np.array([0]))[0]


def power_analysis(
    conts: list[metronome.ContinuousRecord], model: SignalModel, beat_scale: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Replicates for every amplitude and variant; identical hybrid data for both variants (paired)."""
    rows, nulls, curves = [], [], []
    for a_idx, amplitude in enumerate(AMPLITUDES):
        for rep in range(N_REPLICATES):
            rng = np.random.default_rng([seed, a_idx, rep])
            data = hybrid(fold_noise(conts, rng), model, beat_scale, amplitude, rng)
            for variant, band in VARIANT_BANDS.items():
                stats, w, out = evaluate(data, band)
                rows.append({"amplitude": amplitude, "replicate": rep, "variant": variant, "n_records": len(data.records), **stats})
                if rep == 0:
                    null, _ = metronome.random_control(w, data.subject, NULL_SURROGATES, seed + a_idx)
                    nulls.append(
                        {"amplitude": amplitude, "variant": variant}
                        | {f"{k}_q95": float(np.quantile(v, 0.95)) for k, v in null.items()}
                    )
                if amplitude == CURVE_AMPLITUDE:
                    recovered = np.nanmean(metronome.aligned_channel_curves(w, out)["deviant"], axis=0)  # (n_ch, n_epoch)
                    for c, ch in enumerate(config.CHANNELS):
                        curves.append(
                            pd.DataFrame(
                                {
                                    "variant": variant,
                                    "replicate": rep,
                                    "channel": ch,
                                    "time_s": metronome.EPOCH_TIMES_S,
                                    "recovered_uv": recovered[c],
                                    "injected_uv": expected_curve(model, amplitude, band)[c],
                                }
                            )
                        )
    return pd.DataFrame(rows), pd.DataFrame(nulls), pd.concat(curves, ignore_index=True)


def summarize_power(replicates: pd.DataFrame, nulls: pd.DataFrame, microvolts_per_unit: float) -> pd.DataFrame:
    """Power of tests 2 and 3 and recovery accuracy per amplitude and variant."""
    merged = replicates.merge(nulls, on=["amplitude", "variant"])
    merged["test2_detected"] = merged["deviant_r"] > merged["deviant_r_q95"]
    merged["test3_detected"] = merged["curve_energy"] > merged["curve_energy_q95"]
    table = (
        merged.groupby(["variant", "amplitude"])
        .agg(
            test2_power=("test2_detected", "mean"),
            test3_power=("test3_detected", "mean"),
            deviant_r=("deviant_r", "mean"),
            grid_r=("grid_r", "mean"),
            deviant_found=("deviant_found", "mean"),
            phase_within_50ms=("phase_within_50ms", "mean"),
            phase_within_50ms_of_bias=("phase_within_50ms_of_bias", "mean"),
            phase_bias_s=("phase_bias_s", "mean"),
            n_records=("n_records", "mean"),
        )
        .reset_index()
    )
    table.insert(2, "difference_rms_uv", table["amplitude"] * microvolts_per_unit)
    return table


def main(n_jobs: int = -1, seed: int = config.RANDOM_STATE) -> None:
    """ERP CORE summary, beat calibration and power analysis; writes ``RESULTS_DIR``."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    erpsets = erp_core.load_all()
    erp_metrics, erp_curves = erp_core.summary(erpsets)
    erp_curves.to_csv(RESULTS_DIR / "erpcore_curves.csv", index=False)

    templates = templates_from(erpsets)
    model = SignalModel.from_templates(templates)
    pilot = pd.read_csv(metronome.RESULTS_DIR / "records_rest.csv")
    conts = Parallel(n_jobs=n_jobs)(delayed(stretched_continuous)(config.DATA_DIR / rel) for rel in pilot["relpath"])

    beat_scale, calibration = calibrate_beat_scale(conts, model, seed)
    replicates, nulls, curves = power_analysis(conts, model, beat_scale, seed)
    fp = [config.CHANNELS.index(ch) for ch in metronome.FRONTAL]
    unit_rms = float(np.sqrt(np.mean(expected_curve(model, 1.0, metronome.BAND_HZ)[fp].mean(axis=0) ** 2)))
    table = summarize_power(replicates, nulls, unit_rms)

    replicates.to_csv(RESULTS_DIR / "hybrid_replicates.csv", index=False)
    nulls.to_csv(RESULTS_DIR / "hybrid_null_q95.csv", index=False)
    table.to_csv(RESULTS_DIR / "hybrid_power.csv", index=False)
    curves.to_csv(RESULTS_DIR / "hybrid_curves.csv", index=False)
    power_by_variant = {
        variant: float(table[(table["variant"] == variant) & (table["amplitude"] > 0)]["test2_power"].mean())
        for variant in VARIANT_BANDS
    }
    best, other = sorted(power_by_variant, key=power_by_variant.get, reverse=True)
    chosen = best if power_by_variant[best] - power_by_variant[other] >= 0.05 else "pilot_1_20Hz"
    metrics = {
        "procedure": "docs/DATA_AUDIT.md, 'Проверка на открытых данных и гибридная симуляция'",
        "erp_core": erp_metrics,
        "stretch": str(STRETCH),
        "highpass_order": HIGHPASS_ORDER,
        "beat_scale": beat_scale,
        "beat_calibration": calibration,
        "difference_rms_uv_at_fp_per_unit_amplitude": unit_rms,
        "mean_test2_power_a_ge_1": power_by_variant,
        "confirmatory_variant": chosen,
    }
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
