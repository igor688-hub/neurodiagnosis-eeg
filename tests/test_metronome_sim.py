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


def test_stretch_slows_the_record() -> None:
    t = np.arange(int(40 * FS)) / FS
    data = np.tile(20.0 * np.sin(2 * np.pi * 10.0 * t), (6, 1)) + np.random.default_rng(0).normal(0, 1, (6, t.size))
    record = EegRecord(data, FS, config.CHANNELS, np.full(6, 0.061), np.zeros(data.shape, dtype=bool))

    cont = metronome_sim.stretch(record)

    assert abs(cont.data.shape[1] - 1.05 * t.size) <= 2
    spectrum = np.abs(np.fft.rfft(cont.data[0]))
    freqs = np.fft.rfftfreq(cont.data.shape[1], 1 / FS)
    assert abs(freqs[np.argmax(spectrum)] - 10.0 / 1.05) < 0.05


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
