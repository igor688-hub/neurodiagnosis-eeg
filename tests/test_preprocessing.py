import numpy as np
import numpy.typing as npt
import pytest
from scipy.signal import welch

from src import config, preprocessing
from src.dataset import EegRecord, load_record
from src.preprocessing import PreprocessingConfig, Reject

needs_data = pytest.mark.skipif(
    not (config.DATA_DIR / config.GROUP_SOMATOFORM).is_dir(), reason="training data not downloaded"
)


def _sine(freq: float, sfreq: float, duration: float, amp: float = 10.0) -> npt.NDArray[np.float64]:
    t = np.arange(int(duration * sfreq)) / sfreq
    return amp * np.sin(2 * np.pi * freq * t)


def _record(data: npt.NDArray[np.float64], sfreq: float = 125.0) -> EegRecord:
    return EegRecord(
        data=data,
        sfreq=sfreq,
        channels=config.CHANNELS,
        quantization_step_uv=np.full(data.shape[0], 0.061),
        at_rail=np.zeros(data.shape, dtype=bool),
    )


def _background(rng: np.random.Generator, n_times: int, std: float = 10.0) -> npt.NDArray[np.float64]:
    return rng.normal(0.0, std, (config.N_CHANNELS, n_times))


def test_requantize_error_bounded_and_idempotent() -> None:
    x = np.random.default_rng(0).normal(0, 20, (6, 1000))
    q = preprocessing.requantize(x, 1.0)

    assert np.abs(q - x).max() <= 0.5
    np.testing.assert_array_equal(q, np.round(q))
    np.testing.assert_array_equal(preprocessing.requantize(q, 1.0), q)


@pytest.mark.parametrize("sfreq", [123.0, 124.0, 126.0, 127.0])
def test_resampling_keeps_frequency_and_amplitude(sfreq: float) -> None:
    x = np.tile(_sine(10.0, sfreq, 20.0), (6, 1))
    y = preprocessing.resample(x, sfreq, 125.0)

    assert abs(y.shape[1] - x.shape[1] * 125.0 / sfreq) <= 1
    f, pxx = welch(y[0], fs=125.0, nperseg=1000)
    assert f[np.argmax(pxx)] == pytest.approx(10.0, abs=0.125)
    core = y[0, 125:-125]
    assert np.sqrt(2) * core.std() == pytest.approx(10.0, rel=0.01)


def test_lowpass_passes_alpha_and_removes_mains() -> None:
    alpha = np.tile(_sine(10.0, 125.0, 20.0), (6, 1))
    mains = np.tile(_sine(50.0, 125.0, 20.0), (6, 1))

    alpha_out = preprocessing.lowpass(alpha, 125.0, 40.0)[:, 250:-250]
    mains_out = preprocessing.lowpass(mains, 125.0, 40.0)[:, 250:-250]

    assert alpha_out.std() == pytest.approx(alpha[:, 250:-250].std(), rel=0.01)
    assert 20 * np.log10(mains_out.std() / mains.std()) < -40.0


def test_window_count_and_short_records() -> None:
    cfg = PreprocessingConfig()
    x = np.zeros((6, 8 * 125))

    assert preprocessing.sliding_windows(x, cfg.window_samples, cfg.step_samples).shape == (3, 6, 500)
    assert preprocessing.sliding_windows(x[:, :375], cfg.window_samples, cfg.step_samples).shape == (0, 6, 500)


def test_rail_runs_ignore_single_extreme_samples() -> None:
    at_rail = np.zeros((1, 20), dtype=bool)
    at_rail[0, 3] = True
    at_rail[0, 10:14] = True

    mask = preprocessing.rail_run_mask(at_rail, min_run=3)

    assert not mask[0, 3]
    assert mask[0, 10:14].all() and mask.sum() == 4


def test_rejection_flags_are_channel_specific() -> None:
    rng = np.random.default_rng(1)
    n_times = 60 * 125
    data = _background(rng, n_times)
    data[1] = 0.1 * rng.normal(size=n_times)
    data[2, 10 * 125 : 11 * 125] += 600 * np.hanning(125)
    record = _record(data)
    record.at_rail[4, 30 * 125 : 30 * 125 + 5] = True

    epoched = preprocessing.preprocess_record(record)
    flags = epoched.reject

    assert epoched.windows.shape == (29, 6, 500)
    assert (flags[:, 1] & Reject.FLAT).all()
    hit_fp1 = (flags[:, 2] & Reject.AMPLITUDE) > 0
    assert set(np.flatnonzero(hit_fp1)) == {4, 5}
    hit_t4 = (flags[:, 4] & Reject.RAIL) > 0
    assert set(np.flatnonzero(hit_t4)) == {13, 14, 15}
    assert (flags[:, [0, 3, 5]] == 0).mean() > 0.9


def test_variance_outlier_within_record() -> None:
    rng = np.random.default_rng(2)
    data = _background(rng, 60 * 125)
    data[0, 40 * 125 : 44 * 125] *= 8.0

    flags = preprocessing.preprocess_record(_record(data)).reject

    hit = np.flatnonzero((flags[:, 0] & Reject.VARIANCE) > 0)
    assert set(hit) == {19, 20, 21}
    assert not (flags[20, 0] & Reject.AMPLITUDE)


@needs_data
def test_corrupted_temporal_channels_are_rejected() -> None:
    path = config.DATA_DIR / config.GROUP_SOMATOFORM / "ABXZ5" / "T-1.edf"
    epoched = preprocessing.preprocess_record(load_record(path))
    good = epoched.good.mean(axis=0)

    t3, t4 = config.CHANNELS.index("T3"), config.CHANNELS.index("T4")
    assert good[t3] < 0.5 and good[t4] < 0.5


def _touched_windows(clean: preprocessing.EpochedRecord, dirty: preprocessing.EpochedRecord) -> npt.NDArray[np.bool_]:
    return np.abs(dirty.windows - clean.windows).max(axis=2) > 1e-6


@pytest.mark.parametrize("sfreq", [125.0, 126.0])
def test_guard_covers_filter_spread_of_saturation(sfreq: float) -> None:
    rng = np.random.default_rng(3)
    n_times = int(40 * sfreq)
    clean_data = _background(rng, n_times)
    dirty_data = clean_data.copy()
    a, b = int(20.0 * sfreq), int(20.3 * sfreq)
    dirty_data[3, a:b] = 2000.0
    dirty = _record(dirty_data, sfreq)
    dirty.at_rail[3, a:b] = True

    clean_ep = preprocessing.preprocess_record(_record(clean_data, sfreq))
    dirty_ep = preprocessing.preprocess_record(dirty)

    touched = _touched_windows(clean_ep, dirty_ep)
    assert touched[:, 3].any() and not touched[:, [0, 1, 2, 4, 5]].any()
    assert ((dirty_ep.reject[touched] & Reject.RAIL) > 0).all()


def test_dropout_rejected_on_all_channels_with_guard() -> None:
    rng = np.random.default_rng(4)
    clean_data = _background(rng, 40 * 125)
    dirty_data = clean_data.copy()
    dirty_data[:, 15 * 125 : 16 * 125] = 0.0

    clean_ep = preprocessing.preprocess_record(_record(clean_data))
    dirty_ep = preprocessing.preprocess_record(_record(dirty_data))

    touched = _touched_windows(clean_ep, dirty_ep)
    assert touched.any()
    assert ((dirty_ep.reject[touched] & Reject.DROPOUT) > 0).all()
    assert not (dirty_ep.reject[:5] & Reject.DROPOUT).any()


def test_trailing_zero_padding_is_trimmed() -> None:
    rng = np.random.default_rng(5)
    data = _background(rng, 21 * 125)
    data[:, -125:] = 0.0

    epoched = preprocessing.preprocess_record(_record(data))

    assert epoched.n_windows == 9
    assert not (epoched.reject & Reject.DROPOUT).any()
    assert (np.abs(epoched.windows[-1, :, -10:]) > 0).any(axis=1).all()


def test_copied_channels_rejected_but_shared_zeros_ignored() -> None:
    rng = np.random.default_rng(8)
    data = _background(rng, 30 * 125)
    data[4] = data[1]
    data[0, : 10 * 125] = 0.0
    data[5, : 10 * 125] = 0.0

    flags = preprocessing.preprocess_record(_record(data)).reject

    copy = (flags & Reject.COPY) > 0
    assert copy[:, [1, 4]].all()
    assert not copy[:, [0, 2, 3, 5]].any()
    np.testing.assert_array_equal(preprocessing.copied_channels(np.zeros((3, 10))), np.zeros(3, dtype=bool))
