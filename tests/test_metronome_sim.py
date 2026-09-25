"""Hybrid validation: signal construction, emulated high-pass, time stretch, recovery of known truth."""
import numpy as np

from src import config, erp_core, metronome, metronome_sim
from src.dataset import EegRecord

FS = config.TARGET_SFREQ


def _gauss_templates() -> metronome_sim.Templates:
    t = np.arange(50) / FS
    beat = -3.0 * np.exp(-0.5 * ((t - 0.10) / 0.02) ** 2)
    td = np.arange(-25, 100) / FS
    deviant = -4.0 * np.exp(-0.5 * ((td - 0.17) / 0.03) ** 2)
    return metronome_sim.Templates(np.tile(beat, (6, 1)), np.tile(deviant, (6, 1)), float(td[0]))


def test_highpass_response_is_butterworth() -> None:
    h = erp_core.highpass_response(np.array([0.5, 2.0, 20.0]), 4)
    assert abs(abs(h[1]) - 2**-0.5) < 1e-9
    assert abs(h[0]) < 0.01 and abs(abs(h[2]) - 1.0) < 1e-3


def test_signal_model_places_responses_at_their_onsets() -> None:
    model = metronome_sim.SignalModel.from_templates(_gauss_templates(), highpass_order=0)
    spectrum = model.spectrum(phase_s=0.2, deviant=3, beat_scale=0.0, amplitude=1.0)
    fine = metronome.fine_waveform(spectrum, np.ones(metronome.N_BINS, dtype=bool))[2]
    t = np.arange(metronome.FINE_POINTS) / metronome.FINE_HZ
    assert abs(t[np.argmin(fine)] - (0.2 + 0.5 * 3 + 0.17)) < 0.004


def _noise_record(rng: np.random.Generator, n_cycles: int = 14, periodic: np.ndarray | None = None) -> metronome_sim.NoiseRecord:
    cycles = rng.normal(0.0, 5.0, (6, n_cycles, metronome.CYCLE_SAMPLES))
    if periodic is not None:
        cycles += periodic[None, None, :]
    return metronome_sim.NoiseRecord(cycles, np.ones((6, n_cycles), dtype=bool))


def test_plus_minus_cancels_what_repeats_in_every_cycle() -> None:
    rng = np.random.default_rng(3)
    periodic = 50.0 * np.sin(2 * np.pi * 2.25 * np.arange(metronome.CYCLE_SAMPLES) / FS)  # deviant-subspace bin
    with_signal = metronome_sim.plus_minus(_noise_record(np.random.default_rng(4), periodic=periodic), np.random.default_rng(5))
    without = metronome_sim.plus_minus(_noise_record(np.random.default_rng(4)), np.random.default_rng(5))
    assert np.allclose(with_signal.spectra, without.spectra)


def test_plus_minus_halves_are_independent_with_half_average_noise() -> None:
    rng = np.random.default_rng(6)
    correlations, variances = [], []
    for _ in range(300):
        halves = metronome_sim.plus_minus(_noise_record(rng), rng)
        a, b = np.fft.irfft(halves.spectra[:, 2], n=metronome.CYCLE_SAMPLES, axis=1)
        correlations.append(np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b)))
        variances.append(a.var())
    assert abs(np.mean(correlations)) < 0.01
    # 14 cycles dealt into groups of 4, 4, 3, 3: half A stands for a mean of 8 cycles
    assert abs(np.mean(variances) - 5.0**2 / 8) < 0.1 * 5.0**2 / 8
    assert halves.n_cycles[0, 2] == 8 and halves.n_cycles[1, 2] == 6


def test_pipeline_recovers_injected_deviant_in_noise() -> None:
    rng = np.random.default_rng(1)
    model = metronome_sim.SignalModel.from_templates(_gauss_templates(), highpass_order=0)
    noise = []
    for _ in range(30):
        spectra = np.fft.rfft(rng.normal(0.0, 1.0, (2, 6, metronome.CYCLE_SAMPLES)), axis=2)
        noise.append(metronome.CycleSpectra(spectra, np.full((2, 6), 7), config.CHANNELS))

    data = metronome_sim.hybrid(noise, model, beat_scale=1.0, amplitude=1.0, rng=rng)
    stats, _, _ = metronome_sim.evaluate(data, metronome.BAND_HZ)

    assert stats["phase_within_50ms"] > 0.9 and abs(stats["phase_bias_s"]) < 0.01
    assert stats["deviant_found"] > 0.9 and stats["deviant_r"] > 0.3


def test_highpass_order_fit_recovers_order_and_slope() -> None:
    freqs = np.arange(1, 200) / 16.0
    log_psd = 1.5 - 1.2 * np.log10(freqs) + 2 * np.log10(np.abs(erp_core.highpass_response(freqs, 3)))

    fit = metronome_sim.highpass_order_fit(freqs, log_psd)

    assert fit["ssr"].idxmin() == 3
    assert fit.loc[3, "ssr"] < 1e-12 and abs(fit.loc[3, "chi"] - 1.2) < 1e-9
