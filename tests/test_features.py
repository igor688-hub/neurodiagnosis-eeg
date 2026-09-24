"""Spectral estimator: bias independent of the number of windows."""
import numpy as np
import pytest

from src import features
from src.preprocessing import EpochedRecord


@pytest.mark.parametrize("n_windows", [1, 3, 5, 30])
def test_log_mean_spectrum_unbiased_for_any_window_count(n_windows: int) -> None:
    rng = np.random.default_rng(n_windows)
    sfreq, sigma, n_rep = 125.0, 10.0, 400
    true_log_psd = np.log10(2 * sigma**2 / sfreq)  # one-sided PSD of white noise, uV^2/Hz
    windows = rng.normal(0.0, sigma, (n_rep, n_windows, 1, 500))

    freqs, psd = features.window_psd(windows, sfreq)  # shape: (n_rep, n_windows, 1, n_freqs)
    band = (freqs >= 2) & (freqs <= 40)
    estimate = np.log10(10.0 ** (np.log10(psd).mean(axis=1) + features.LOG10_BIAS))[..., band]
    median = np.log10(np.median(psd, axis=1))[..., band]

    assert estimate.mean() == pytest.approx(true_log_psd, abs=0.01)
    if n_windows in (3, 30):  # the median's bias depends on n: -0.08 log10 at n=3, -0.15 at n=30
        assert abs(median.mean() - true_log_psd) > 0.05


def test_record_spectrum_requires_minimum_windows() -> None:
    rng = np.random.default_rng(0)
    windows = rng.normal(size=(8, 6, 500))
    reject = np.zeros((8, 6), dtype=np.uint8)
    reject[:4, 2] = 1  # Fp1 keeps 4 windows, below the minimum of 5

    freqs, spectrum = features.record_spectrum(EpochedRecord(windows, reject, 125.0))

    assert np.isnan(spectrum[2]).all()
    assert np.isfinite(spectrum[[0, 1, 3, 4, 5]][:, freqs > 0]).all()


def _pink_with_alpha(rng: np.random.Generator, n_windows: int, peak_hz: float | None, gain: float = 1.0) -> np.ndarray:
    """Windows (n_windows, 6, 500) of 1/f noise plus an optional alpha oscillation."""
    freqs = np.fft.rfftfreq(500, d=1 / 125.0)
    amplitude = np.where(freqs > 0, 1.0 / np.sqrt(np.maximum(freqs, 1e-3)), 0.0)
    phases = rng.uniform(0, 2 * np.pi, (n_windows, 6, freqs.size))
    x = np.fft.irfft(amplitude * np.exp(1j * phases), n=500) * 200.0
    if peak_hz is not None:
        t = np.arange(500) / 125.0
        x += 0.8 * np.sin(2 * np.pi * peak_hz * t + rng.uniform(0, 2 * np.pi, (n_windows, 6, 1)))
    return gain * x


def test_relative_power_is_gain_invariant() -> None:
    rng = np.random.default_rng(1)
    windows = _pink_with_alpha(rng, 20, 10.0)
    ok = np.zeros((20, 6), dtype=np.uint8)

    freqs, s1 = features.record_spectrum(EpochedRecord(windows, ok, 125.0))
    _, s2 = features.record_spectrum(EpochedRecord(7.0 * windows, ok, 125.0))

    r1, r2 = features.log_relative_powers(freqs, s1), features.log_relative_powers(freqs, s2)
    for band in features.BANDS:
        np.testing.assert_allclose(r1[band], r2[band], atol=1e-12)
    np.testing.assert_allclose(sum(10.0 ** r1[b] for b in features.BANDS), 1.0)  # bands partition 4-30 Hz


def test_alpha_peak_found_or_left_missing() -> None:
    rng = np.random.default_rng(2)
    ok = np.zeros((25, 6), dtype=np.uint8)
    for peak_hz, expect_peak in ((9.5, True), (None, False)):
        freqs, s = features.record_spectrum(EpochedRecord(_pink_with_alpha(rng, 25, peak_hz), ok, 125.0))
        iaf, prominence = features.alpha_peak(freqs, np.log10(s[[0, 5]]).mean(axis=0))
        if expect_peak:
            assert iaf == pytest.approx(9.5, abs=0.25) and prominence > features.MIN_PEAK_LOG10
        else:
            assert np.isnan(iaf) and prominence < features.MIN_PEAK_LOG10


def test_trials_are_weighted_equally() -> None:
    rng = np.random.default_rng(3)
    short = EpochedRecord(rng.normal(0, 1, (5, 6, 500)), np.zeros((5, 6), dtype=np.uint8), 125.0)
    long = EpochedRecord(rng.normal(0, 3, (50, 6, 500)), np.zeros((50, 6), dtype=np.uint8), 125.0)

    freqs, pooled = features.condition_spectrum([short, long])
    _, s_short = features.record_spectrum(short)
    _, s_long = features.record_spectrum(long)

    band = (freqs >= 4) & (freqs <= 30)
    expected = 0.5 * (np.log10(s_short) + np.log10(s_long))  # equal weight in the log domain
    np.testing.assert_allclose(np.log10(pooled)[:, band], expected[:, band])


def test_subject_without_data_gets_all_nan() -> None:
    values = features.subject_features([], [])

    assert tuple(values) == features.FEATURE_NAMES
    assert all(np.isnan(v) for v in values.values())
