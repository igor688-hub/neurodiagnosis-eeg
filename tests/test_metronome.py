"""Metronome branch: subspaces, grid and deviant recovery on synthetic records, random control."""
import numpy as np

from src import config, metronome
from src.dataset import EegRecord

FS = config.TARGET_SFREQ


def _gauss(t: np.ndarray, centre: float, width: float) -> np.ndarray:
    return np.exp(-0.5 * ((t - centre) / width) ** 2)


def _synthetic_record(
    seconds: float, onset_s: float, deviant: int, rng: np.random.Generator, noise_uv: float = 2.0, gain: float = 1.0
) -> EegRecord:
    """N1-like trough 100 ms after every beat on Fp1/Fp2, extra negativity 170 ms after deviants."""
    t = np.arange(int(seconds * FS)) / FS
    beats = onset_s + metronome.BEAT_S * np.arange(int(seconds / metronome.BEAT_S) + 2)
    signal = np.zeros_like(t)
    for j, tb in enumerate(beats):
        signal -= 6.0 * gain * _gauss(t, tb + 0.100, 0.02)
        if j % metronome.BEATS_PER_CYCLE == deviant:
            signal -= 4.0 * gain * _gauss(t, tb + 0.170, 0.03)
    data = rng.normal(0.0, noise_uv, (6, t.size))
    fp = [config.CHANNELS.index(ch) for ch in metronome.FRONTAL]
    data[fp] += signal
    return EegRecord(data, FS, config.CHANNELS, np.full(6, 0.061), np.zeros(data.shape, dtype=bool))


def _circular_gauss(t: np.ndarray, centre: float, width: float) -> np.ndarray:
    lag = (t - centre + metronome.CYCLE_S / 2) % metronome.CYCLE_S - metronome.CYCLE_S / 2
    return np.exp(-0.5 * (lag / width) ** 2)


def test_subspaces_split_beat_and_deviant_parts() -> None:
    t = np.arange(metronome.CYCLE_SAMPLES) / FS
    beat = sum(_circular_gauss(t, 0.1 + 0.5 * j, 0.03) for j in range(8))  # period 0.5 s
    one_second = sum(_circular_gauss(t, 0.3 + 1.0 * j, 0.03) for j in range(4))  # period 1 s
    deviant = _circular_gauss(t, 1.7, 0.03)
    spectrum = np.fft.rfft(beat + one_second + deviant)[None, :]

    dev = metronome.fine_waveform(spectrum, metronome.DEVIANT_MASK)[0]
    fine_t = np.arange(metronome.FINE_POINTS) / metronome.FINE_HZ

    # The deviant subspace keeps 3/4 of the in-band deviant and nothing of the 0.5-s or 1-s periodic parts.
    in_band = (np.arange(metronome.N_BINS) * metronome.BIN_HZ >= metronome.BAND_HZ[0]) & (
        np.arange(metronome.N_BINS) * metronome.BIN_HZ <= metronome.BAND_HZ[1]
    )
    band_limited = metronome.fine_waveform(np.fft.rfft(deviant)[None, :], in_band)[0]
    second = int(metronome.FINE_HZ)
    echoes = sum(np.roll(band_limited, m * second) for m in (1, 2, 3)) / 3.0
    assert abs(fine_t[np.argmax(dev)] - 1.7) < 0.004
    assert np.allclose(dev * metronome.DEVIANT_GAIN, band_limited - echoes, atol=1e-9)
    only_periodic = np.fft.rfft(beat + one_second)[None, :]
    assert np.abs(metronome.fine_waveform(only_periodic, metronome.DEVIANT_MASK)).max() < 1e-9


def test_roll_fine_is_a_delay() -> None:
    wave = np.sin(2 * np.pi * 3 * np.arange(metronome.FINE_POINTS) / metronome.FINE_HZ)[None, :]
    shifted = metronome.roll_fine(wave, np.array([25]))
    assert np.allclose(shifted[0, 25:], wave[0, :-25])


def test_cycle_spectra_rejects_high_amplitude_cycles() -> None:
    rng = np.random.default_rng(1)
    record = _synthetic_record(40.0, 0.2, 3, rng)
    data = record.data.copy()
    data[0, 100:110] += 500.0  # artifact inside the first cycle of O1
    cont = metronome.continuous_record(EegRecord(data, FS, config.CHANNELS, record.quantization_step_uv, record.at_rail))

    spectra = metronome.cycle_spectra(cont)

    assert spectra.n_cycles[0, 0] == spectra.n_cycles[0, 1] - 1  # O1 lost one even cycle, T3 did not
    assert spectra.valid(metronome.FRONTAL)


def test_recovers_grid_phase_and_deviant_position() -> None:
    rng = np.random.default_rng(2)
    truth = [(0.13, 5), (0.41, 0), (0.02, 7)]
    records = [metronome.record_spectra_from(_synthetic_record(60.0, onset, dev, rng)) for onset, dev in truth]
    w = metronome.half_waveforms(records)

    out = metronome.recover(w)

    for i, (onset, dev) in enumerate(truth):
        expected_tau = (onset + metronome.BEAT_S * dev) * metronome.FINE_HZ
        assert abs(out["onset_a"][i] - onset * metronome.FINE_HZ) <= 3  # within 6 ms
        assert abs(out["tau_a"][i] - expected_tau) <= 3 and abs(out["tau_b"][i] - expected_tau) <= 3
    curve = out["curve"].mean(axis=0)
    assert metronome.window_mean(curve, metronome.MMN_WINDOW_S) < -1.0  # extra negativity recovered


def test_random_control_separates_signal_from_noise() -> None:
    rng = np.random.default_rng(3)
    signal = [metronome.record_spectra_from(_synthetic_record(60.0, rng.uniform(0, 0.5), int(rng.integers(8)), rng, 8.0)) for _ in range(12)]
    noise = [metronome.record_spectra_from(_synthetic_record(60.0, 0.1, 0, rng, 8.0, gain=0.0)) for _ in range(12)]
    subjects = np.arange(12)

    for records, should_detect in ((signal, True), (noise, False)):
        w = metronome.half_waveforms(records)
        observed = metronome.group_statistics(metronome.recover(w), subjects)
        null, _ = metronome.random_control(w, subjects, n_surrogates=199)
        p = {key: metronome.p_value(observed[key], null[key], metronome.TEST_DIRECTION[key]) for key in ("grid_r", "deviant_r")}
        if should_detect:
            assert p["grid_r"] <= 0.01 and p["deviant_r"] <= 0.01
        else:
            assert p["grid_r"] > 0.01 and p["deviant_r"] > 0.01
