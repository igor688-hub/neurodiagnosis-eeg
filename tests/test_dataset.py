"""EDF reader, registry and duplicate grouping.

Synthetic EDF files test the reader against known values; tests marked
``needs_data`` check the local training set against the audited figures.
"""
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from src import config, dataset

needs_data = pytest.mark.skipif(
    not (config.DATA_DIR / config.GROUP_PTSD).is_dir(), reason="training data not downloaded"
)


def write_edf(
    path: Path,
    signals: dict[str, npt.NDArray[np.int16]],
    samples_per_record: dict[str, int],
    physical_range: tuple[float, float] = (-2000.0, 2000.0),
    n_records_header: int | None = None,
) -> None:
    """Write a minimal EDF file with 1-s records and common scaling."""
    labels = list(signals)
    ns = len(labels)
    n_records = len(next(iter(signals.values()))) // samples_per_record[labels[0]]

    def fmt(value: object, width: int) -> bytes:
        return str(value).ljust(width)[:width].encode("latin-1")

    fixed = (
        fmt("0", 8) + fmt("", 80) + fmt("", 80) + fmt("01.01.26", 8) + fmt("00.00.00", 8)
        + fmt(256 * (ns + 1), 8) + fmt("", 44)
        + fmt(n_records if n_records_header is None else n_records_header, 8) + fmt(1, 8) + fmt(ns, 4)
    )
    fields = [
        (16, labels), (80, [""] * ns), (8, ["uV"] * ns),
        (8, [physical_range[0]] * ns), (8, [physical_range[1]] * ns),
        (8, [-32768] * ns), (8, [32767] * ns),
        (80, ["HP:2.0Hz LP:40.0Hz"] * ns), (8, [samples_per_record[lb] for lb in labels]), (32, [""] * ns),
    ]
    signal_header = b"".join(fmt(v, width) for width, values in fields for v in values)
    records = [
        np.asarray(signals[lb][r * samples_per_record[lb] : (r + 1) * samples_per_record[lb]], dtype="<i2")
        for r in range(n_records)
        for lb in labels
    ]
    path.write_bytes(fixed + signal_header + b"".join(rec.tobytes() for rec in records))


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


def _six_channel_signals(
    rng: np.random.Generator, n_records: int, spr: int = 125
) -> dict[str, npt.NDArray[np.int16]]:
    # Non-alphabetical order as in the device export, plus an annotation channel.
    order = ("O1", "T3", "Fp1", "Fp2", "T4", "O2")
    signals = {ch: rng.integers(-3000, 3000, n_records * spr).astype(np.int16) for ch in order}
    signals["EDF Annotations"] = np.zeros(n_records * 10, dtype=np.int16)
    return signals


def _spr(signals: dict[str, npt.NDArray[np.int16]], spr: int = 125) -> dict[str, int]:
    return {lb: (10 if lb == "EDF Annotations" else spr) for lb in signals}


def test_digital_samples_roundtrip_in_requested_order(tmp_path: Path, rng: np.random.Generator) -> None:
    signals = _six_channel_signals(rng, n_records=4)
    path = tmp_path / "rec.edf"
    write_edf(path, signals, _spr(signals))

    header = dataset.read_edf_header(path)
    channels = ("O2", "Fp1", "T3")
    digital = dataset.read_digital(header, channels)

    assert digital.shape == (3, 4 * 125)
    for row, ch in zip(digital, channels):
        np.testing.assert_array_equal(row, signals[ch])
    assert header.sfreq() == 125.0


def test_scaling_to_microvolts(tmp_path: Path, rng: np.random.Generator) -> None:
    signals = _six_channel_signals(rng, n_records=2)
    path = tmp_path / "rec.edf"
    write_edf(path, signals, _spr(signals), physical_range=(-2000.0, 2000.0))

    record = dataset.load_record(path)
    step = 4000.0 / 65535.0  # uV per digital unit
    expected = -2000.0 + (signals["O1"].astype(float) + 32768.0) * step

    np.testing.assert_allclose(record.data[0], expected)
    np.testing.assert_allclose(record.quantization_step_uv, step)
    assert record.channels == config.CHANNELS


def test_negative_record_count_is_derived_from_file_size(tmp_path: Path, rng: np.random.Generator) -> None:
    signals = _six_channel_signals(rng, n_records=3)
    path = tmp_path / "rec.edf"
    write_edf(path, signals, _spr(signals), n_records_header=-1)

    header = dataset.read_edf_header(path)

    assert header.n_records_header == -1
    assert header.n_records == 3
    assert header.trailing_bytes == 0


def test_empty_file_and_missing_channel_raise(tmp_path: Path, rng: np.random.Generator) -> None:
    empty = tmp_path / "empty.edf"
    empty.write_bytes(b"")
    with pytest.raises(dataset.EdfFormatError):
        dataset.read_edf_header(empty)

    signals = _six_channel_signals(rng, n_records=1)
    del signals["T4"]
    path = tmp_path / "rec.edf"
    write_edf(path, signals, _spr(signals))
    with pytest.raises(dataset.EdfFormatError, match="T4"):
        dataset.load_record(path)


def test_record_hashes_skip_flat_blocks(rng: np.random.Generator) -> None:
    digital = rng.integers(-100, 100, (6, 3 * 125)).astype(np.int16)
    digital[:, 125:250] = 0  # second record is flat on every channel

    hashes = dataset.record_hashes(digital, samples_per_record=125)

    assert len(hashes) == 2


def test_connected_components_merge_transitively() -> None:
    nodes = ["a", "b", "c", "d", "e"]
    component = dataset.connected_components(nodes, [("b", "c"), ("c", "e")])

    assert component["b"] == component["c"] == component["e"]
    assert len({component["a"], component["b"], component["d"]}) == 3
    assert component == dataset.connected_components(nodes, [("e", "c"), ("c", "b")])


def test_links_detect_identical_and_cropped_copies() -> None:
    registry = pd.DataFrame(
        {
            "relpath": ["G/s1/T-П.edf", "G/s2/T-1.edf", "G/s3/T-П.edf", "G/s4/T-П.edf"],
            "subject_key": ["G/s1", "G/s2", "G/s3", "G/s4"],
            "status": ["ok"] * 4,
            "content_hash": ["h1", "h1", "h2", "h3"],
            "record_hashes": [("r1", "r2", "r3"), ("r1", "r2", "r3"), ("r2", "r3"), ("r3", "r9")],
        }
    )

    links = dataset.find_duplicate_links(registry)
    kinds = {(a, b): k for a, b, k in zip(links["subject_a"], links["subject_b"], links["kind"])}

    assert kinds[("G/s1", "G/s2")] == "identical"
    assert kinds[("G/s1", "G/s3")] == "overlap"  # s3 is a cropped copy, 2 shared records
    assert all("G/s4" not in pair for pair in kinds)  # one shared record is below threshold


@needs_data
def test_training_set_matches_audit() -> None:
    registry, links, subjects = dataset.scan_dataset()
    registry, subjects = registry[~registry["holdout"]], subjects[~subjects["holdout"]]
    ok = registry[registry["status"] == "ok"]

    assert len(registry) == 996 and len(ok) == 991
    assert subjects.groupby("group").size().to_dict() == {
        config.GROUP_CONTROL: 67, config.GROUP_PTSD: 25, config.GROUP_SOMATOFORM: 74
    }
    assert set(registry.loc[registry["status"] == "empty", "subject_id"]) == {"XCFU5"}
    assert (ok["n_records_header"] == -1).sum() == 9
    assert (registry["duplicate_set"] >= 0).sum() == 51
    assert registry.loc[registry["duplicate_set"] >= 0, "duplicate_set"].nunique() == 25
    assert subjects["split_group"].nunique() == 158
    assert (ok["edge_constant_s"] >= 0.1).groupby(ok["export_family"]).sum().to_dict() == {"A": 0, "B": 150, "C": 0}
    conflict = registry[registry["condition_conflict"]]
    assert len(conflict) == 11
    assert set(conflict.loc[conflict["condition"] == "rest", "subject_id"]) == {"DFGH", "GHRD3", "ZILO6", "TMVN4", "XYKT7"}
    # Split groups never mix cohorts, so cohort labels stay well defined per group.
    assert (subjects.groupby("split_group")["label"].nunique() == 1).all()


@needs_data
def test_reader_matches_mne() -> None:
    mne = pytest.importorskip("mne")
    path = config.DATA_DIR / config.GROUP_PTSD / "HJIK" / "T-3.edf"  # header declares n_records = -1

    ours = dataset.load_record(path)
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR").pick(list(config.CHANNELS))

    assert ours.sfreq == raw.info["sfreq"]
    np.testing.assert_allclose(ours.data, raw.get_data() * 1e6, atol=1e-6)


def test_stems_match_cyrillic_lookalikes(tmp_path: Path) -> None:
    for name in ("Т-П.edf", "T-1.EDF", "Т-2.edf", "notes.txt"):  # Cyrillic Т in the first and third
        (tmp_path / name).write_bytes(b"")

    files = dataset.find_record_files(tmp_path)

    assert set(files) == {"T-П", "T-1", "T-2"}
    assert files["T-П"].name == "Т-П.edf"


def test_two_files_for_one_stem_raise(tmp_path: Path) -> None:
    (tmp_path / "T-1.edf").write_bytes(b"")
    (tmp_path / "Т-1.edf").write_bytes(b"")  # Cyrillic Т
    with pytest.raises(dataset.EdfFormatError, match="T-1"):
        dataset.find_record_files(tmp_path)


def test_constant_stretches_and_zero_padding() -> None:
    data = np.random.default_rng(0).integers(-50, 50, (6, 300)).astype(np.int16)
    data[:, 100:130] = 7  # 30 samples held on all channels: dropout
    data[0, 200:260] = 3  # one channel flat only: not a dropout
    data[:, -40:] = 0  # zero padding of the last record

    mask = dataset.constant_stretch_mask(data, min_samples=12)

    assert mask[100:130].all() and not mask[99] and not mask[130]
    assert not mask[200:260].any()
    assert dataset.edge_constant_samples(mask) == (0, 40)


@needs_data
@pytest.mark.parametrize("sfreq", [123.0, 124.0, 125.0, 126.0, 127.0])
def test_reader_matches_mne_on_format_c(sfreq: float) -> None:
    mne = pytest.importorskip("mne")
    registry = dataset.build_registry()
    path = config.DATA_DIR / registry.query("export_family == 'C' and sfreq == @sfreq")["relpath"].iloc[0]

    ours = dataset.load_record(path)
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR").pick(list(config.CHANNELS))

    assert ours.sfreq == raw.info["sfreq"] == sfreq
    np.testing.assert_allclose(ours.data, raw.get_data() * 1e6, atol=1e-6)


@needs_data
def test_ageing_holdout_is_independent() -> None:
    registry, links, subjects = dataset.scan_dataset()
    holdout = subjects[subjects["holdout"]]
    files = registry[registry["holdout"] & (registry["status"] == "ok")]

    assert len(holdout) == 23 and set(holdout["group"]) == {config.GROUP_CONTROL}
    assert holdout["age"].between(65, 70).all()
    assert len(files) == 46 and set(files["stem"]) == {"T-П", "T-1"}
    touched = links["subject_a"].isin(holdout.index) | links["subject_b"].isin(holdout.index)
    assert not touched.any()  # no shared recording with any other subject


def test_ambiguous_records_policy(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    signals = {stem: _six_channel_signals(rng, n_records=3) for stem in ("T-П", "T-1", "T-2", "T-3")}
    signals["T-2"] = signals["T-П"]  # rest saved again as trial 2: condition unknown
    signals["T-3"] = signals["T-1"]  # trial 1 saved twice
    files = {}
    for stem, sig in signals.items():
        files[stem] = tmp_path / f"{stem}.edf"
        write_edf(files[stem], sig, _spr(sig))

    usable, dropped = dataset.resolve_ambiguous_records(files)

    assert set(usable) == {"T-1"}
    assert set(dropped) == {"T-П", "T-2", "T-3"} and "T-1" in dropped["T-3"]


@pytest.mark.parametrize(
    ("name", "age"), [("КС010_19", 19.0), ("КС207м_36", 36.0), ("КС238_18м", 18.0), ("A003_68", 68.0)]
)
def test_age_parsing_with_letter_marks(name: str, age: float) -> None:
    assert dataset.parse_age(config.GROUP_CONTROL, name) == age
    assert np.isnan(dataset.parse_age(config.GROUP_PTSD, name))
