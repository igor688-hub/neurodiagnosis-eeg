"""Signal-plausibility diagnostics on synthetic data."""
import numpy as np
import pytest

from src import config, diagnostics
from src.preprocessing import EpochedRecord


def test_alpha_prominence_of_peak_and_flat_spectrum() -> None:
    freqs = np.arange(0.0, 62.75, 0.25)
    flat = np.ones_like(freqs)
    peaked = flat.copy()
    peaked[(freqs >= 9.5) & (freqs <= 10.5)] = 10.0  # 10x peak at 10 Hz

    assert diagnostics.alpha_prominence(freqs, flat) == 0.0
    assert np.isclose(diagnostics.alpha_prominence(freqs, peaked), 1.0)
    assert diagnostics.alpha_prominence(freqs, np.stack([flat, peaked])).shape == (2,)


def test_interchannel_correlation_ignores_rejected_windows() -> None:
    rng = np.random.default_rng(0)
    common = rng.normal(size=(10, 1, 500))
    windows = common + 0.5 * rng.normal(size=(10, config.N_CHANNELS, 500))  # shared reference signal
    reject = np.zeros((10, config.N_CHANNELS), dtype=np.uint8)
    reject[0, 2] = 1
    windows[0, 2] = 1e4 * rng.normal(size=500)  # artifact in a rejected window

    corr = diagnostics.interchannel_correlation(EpochedRecord(windows, reject, 125.0))

    off_diagonal = corr[~np.eye(config.N_CHANNELS, dtype=bool)]
    assert np.all(off_diagonal > 0.7)  # 1 / (1 + 0.25) = 0.8 expected
    assert set(diagnostics.homologous_correlations(corr)) == {"O1-O2", "Fp1-Fp2", "T3-T4"}


def test_channel_rms_ratio() -> None:
    rng = np.random.default_rng(1)
    windows = rng.normal(size=(12, config.N_CHANNELS, 500))
    windows[:, config.CHANNELS.index("T3")] *= 3.0

    epoched = EpochedRecord(windows, np.zeros((12, config.N_CHANNELS), dtype=np.uint8), 125.0)

    assert diagnostics.channel_rms_ratio(epoched, "T3") == pytest.approx(3.0, rel=0.05)


def test_alpha_threshold_on_synthetic_recordings() -> None:
    from src import features

    study = diagnostics.alpha_detection_study(
        durations_s=(60.0,), exponents=(1.5,), peaks_log10=(0.0, 1.0), n_sim=15, seed=11
    )
    null = study.loc[study["true_peak_log10"] == 0.0, "measured"]
    strong = study.loc[study["true_peak_log10"] == 1.0, "measured"]

    assert (null < features.MIN_PEAK_LOG10).all()  # no false peak in 60 s of 1/f noise
    assert (strong >= features.MIN_PEAK_LOG10).all()
