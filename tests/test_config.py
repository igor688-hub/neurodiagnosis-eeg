from src import config


def test_channels_are_the_six_device_leads() -> None:
    assert config.CHANNELS == ("O1", "T3", "Fp1", "Fp2", "T4", "O2")
    assert len(set(config.CHANNELS)) == config.N_CHANNELS == 6


def test_header_passband_below_nyquist() -> None:
    nyquist = config.TARGET_SFREQ / 2.0
    assert 0.0 < config.HEADER_HIGHPASS_HZ < config.HEADER_LOWPASS_HZ < nyquist


def test_six_records_per_subject() -> None:
    assert len(config.RECORD_STEMS) == len(set(config.RECORD_STEMS)) == 6
