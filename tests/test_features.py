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
