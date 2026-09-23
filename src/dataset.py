"""EDF reading, subject registry and duplicate-aware grouping.

The EDF reader is implemented directly on the byte layout of the format
(https://www.edfplus.info/specs/edf.html) instead of delegating to a generic
reader, for three reasons specific to this data set:

1. Export scaling differs between cohorts and is label-correlated, so the
   quantization step ``(P_max - P_min) / (D_max - D_min)`` must be exposed.
2. Nine files declare ``n_records = -1`` in the header; the record count is
   therefore always derived from the file size.
3. Duplicate detection operates on the raw digital samples, which is exact
   and independent of the physical scaling of each export.

Conversion from digital values ``d`` to physical values ``x`` follows the EDF
specification::

    x = P_min + (d - D_min) * (P_max - P_min) / (D_max - D_min)

All physical signals returned by this module are in microvolts.
"""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from src import config

EDF_BLOCK_BYTES: Final[int] = 256
EDF_SAMPLE_DTYPE: Final[np.dtype] = np.dtype("<i2")

# (offset, width) of every per-signal header field, in bytes per signal.
# Fields are stored field-major: all labels, then all transducers, and so on.
_SIGNAL_FIELDS: Final[dict[str, tuple[int, int]]] = {
    "label": (0, 16),
    "transducer": (16, 80),
    "physical_dim": (96, 8),
    "physical_min": (104, 8),
    "physical_max": (112, 8),
    "digital_min": (120, 8),
    "digital_max": (128, 8),
    "prefilter": (136, 80),
    "samples_per_record": (216, 8),
}

_UNIT_TO_MICROVOLT: Final[dict[str, float]] = {"uv": 1.0, "µv": 1.0, "mv": 1e3, "v": 1e6}

# Two files are linked as overlapping when they share at least this many
# identical, non-degenerate data records (one record = 1 s of 6-channel data).
MIN_SHARED_RECORDS: Final[int] = 2
# A record takes part in overlap detection only if at least this many channels
# are non-constant within it; flat or saturated blocks are not informative.
MIN_ACTIVE_CHANNELS: Final[int] = 3

_CONTROL_AGE_PATTERN: Final[re.Pattern[str]] = re.compile(r"_(\d{2})$")


class EdfFormatError(ValueError):
    """Raised when a file cannot be interpreted as an EDF recording."""


@dataclass(frozen=True)
class EdfHeader:
    """Parsed EDF header with the record count derived from the file size."""

    path: Path
    header_bytes: int
    n_records_header: int
    n_records: int
    trailing_bytes: int
    record_duration: float
    labels: tuple[str, ...]
    physical_dim: tuple[str, ...]
    prefilter: tuple[str, ...]
    physical_min: npt.NDArray[np.float64]  # shape: (n_signals,)
    physical_max: npt.NDArray[np.float64]  # shape: (n_signals,)
    digital_min: npt.NDArray[np.float64]  # shape: (n_signals,)
    digital_max: npt.NDArray[np.float64]  # shape: (n_signals,)
    samples_per_record: npt.NDArray[np.int64]  # shape: (n_signals,)

    @property
    def n_signals(self) -> int:
        return len(self.labels)

    def channel_indices(self, channels: tuple[str, ...] = config.CHANNELS) -> npt.NDArray[np.int64]:
        """Indices of ``channels`` in the file, in the requested order.

        Raises
        ------
        EdfFormatError
            If a channel is absent or the channels have different sampling rates.
        """
        normalized = [_normalize_label(label) for label in self.labels]
        missing = [ch for ch in channels if ch.lower() not in normalized]
        if missing:
            raise EdfFormatError(f"{self.path}: channels {missing} not found in {self.labels}")
        idx = np.array([normalized.index(ch.lower()) for ch in channels], dtype=np.int64)
        if np.unique(self.samples_per_record[idx]).size != 1:
            raise EdfFormatError(f"{self.path}: channels {channels} have unequal sampling rates")
        return idx

    def sfreq(self, channels: tuple[str, ...] = config.CHANNELS) -> float:
        """Sampling rate of ``channels`` in Hz."""
        spr = self.samples_per_record[self.channel_indices(channels)[0]]
        return float(spr) / self.record_duration

    def quantization_step_uv(self, channels: tuple[str, ...] = config.CHANNELS) -> npt.NDArray[np.float64]:
        """Physical value of one digital unit, in microvolts. Shape: (n_channels,)."""
        idx = self.channel_indices(channels)
        gain = (self.physical_max[idx] - self.physical_min[idx]) / (self.digital_max[idx] - self.digital_min[idx])
        return gain * np.array([_unit_factor(self.physical_dim[i]) for i in idx])


@dataclass(frozen=True)
class EegRecord:
    """Six-channel EEG of one EDF file in physical units."""

    data: npt.NDArray[np.float64]  # shape: (n_channels, n_times), unit: uV
    sfreq: float  # Hz
    channels: tuple[str, ...]
    quantization_step_uv: npt.NDArray[np.float64]  # shape: (n_channels,)

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return self.data.shape[1] / self.sfreq


def _normalize_label(label: str) -> str:
    label = label.strip().lower()
    return label[4:].strip() if label.startswith("eeg ") else label


def _unit_factor(dim: str) -> float:
    try:
        return _UNIT_TO_MICROVOLT[dim.strip().lower()]
    except KeyError as err:
        raise EdfFormatError(f"unsupported physical dimension {dim!r}") from err


def read_edf_header(path: Path) -> EdfHeader:
    """Parse the fixed and per-signal EDF header of ``path``.

    The number of data records is computed as
    ``(file_size - header_bytes) // (2 * sum(samples_per_record))``; the
    header value is kept separately in ``n_records_header``.
    """
    path = Path(path)
    file_size = path.stat().st_size
    if file_size < EDF_BLOCK_BYTES:
        raise EdfFormatError(f"{path}: file is empty or truncated ({file_size} bytes)")

    with path.open("rb") as f:
        fixed = f.read(EDF_BLOCK_BYTES)
        try:
            n_signals = int(fixed[252:256])
        except ValueError as err:
            raise EdfFormatError(f"{path}: invalid signal count") from err
        signal_block = f.read(EDF_BLOCK_BYTES * n_signals)
    if len(signal_block) != EDF_BLOCK_BYTES * n_signals:
        raise EdfFormatError(f"{path}: truncated signal header")

    def field(name: str) -> list[str]:
        offset, width = _SIGNAL_FIELDS[name]
        start = offset * n_signals
        return [
            signal_block[start + i * width : start + (i + 1) * width].decode("latin-1").strip()
            for i in range(n_signals)
        ]

    def numeric(name: str) -> npt.NDArray[np.float64]:
        return np.array([float(v) for v in field(name)], dtype=np.float64)

    header_bytes = int(fixed[184:192])
    spr = numeric("samples_per_record").astype(np.int64)
    record_bytes = int(spr.sum()) * EDF_SAMPLE_DTYPE.itemsize
    n_records, trailing = divmod(file_size - header_bytes, record_bytes)

    return EdfHeader(
        path=path,
        header_bytes=header_bytes,
        n_records_header=int(fixed[236:244]),
        n_records=n_records,
        trailing_bytes=trailing,
        record_duration=float(fixed[244:252]),
        labels=tuple(field("label")),
        physical_dim=tuple(field("physical_dim")),
        prefilter=tuple(field("prefilter")),
        physical_min=numeric("physical_min"),
        physical_max=numeric("physical_max"),
        digital_min=numeric("digital_min"),
        digital_max=numeric("digital_max"),
        samples_per_record=spr,
    )


def read_digital(header: EdfHeader, channels: tuple[str, ...] = config.CHANNELS) -> npt.NDArray[np.int16]:
    """Raw digital samples of ``channels``, ordered as requested.

    Returns
    -------
    ndarray of int16, shape (n_channels, n_records * samples_per_record)
    """
    idx = header.channel_indices(channels)
    total_spr = int(header.samples_per_record.sum())
    raw = np.fromfile(
        header.path, dtype=EDF_SAMPLE_DTYPE, count=header.n_records * total_spr, offset=header.header_bytes
    ).reshape(header.n_records, total_spr)  # shape: (n_records, total_spr)
    starts = np.concatenate([[0], np.cumsum(header.samples_per_record)[:-1]])
    spr = int(header.samples_per_record[idx[0]])
    blocks = [raw[:, starts[i] : starts[i] + spr].reshape(-1) for i in idx]
    return np.stack(blocks).astype(np.int16, copy=False)  # shape: (n_channels, n_times)


def digital_to_microvolts(
    digital: npt.NDArray[np.int16], header: EdfHeader, channels: tuple[str, ...] = config.CHANNELS
) -> npt.NDArray[np.float64]:
    """Apply the EDF linear scaling and convert to microvolts.

    Shape is preserved: (n_channels, n_times).
    """
    idx = header.channel_indices(channels)
    gain = (header.physical_max[idx] - header.physical_min[idx]) / (header.digital_max[idx] - header.digital_min[idx])
    offset = header.physical_min[idx] - gain * header.digital_min[idx]
    unit = np.array([_unit_factor(header.physical_dim[i]) for i in idx])
    return (gain[:, None] * digital + offset[:, None]) * unit[:, None]


def load_record(path: Path, channels: tuple[str, ...] = config.CHANNELS) -> EegRecord:
    """Read one EDF file as a six-channel microvolt array at its native rate."""
    header = read_edf_header(path)
    if header.n_records == 0:
        raise EdfFormatError(f"{path}: no complete data records")
    digital = read_digital(header, channels)
    return EegRecord(
        data=digital_to_microvolts(digital, header, channels),
        sfreq=header.sfreq(channels),
        channels=channels,
        quantization_step_uv=header.quantization_step_uv(channels),
    )


def content_hash(digital: npt.NDArray[np.int16]) -> str:
    """SHA-1 of the digital samples, a fingerprint of the recorded signal."""
    return hashlib.sha1(np.ascontiguousarray(digital).tobytes()).hexdigest()


def record_hashes(digital: npt.NDArray[np.int16], samples_per_record: int) -> tuple[str, ...]:
    """SHA-1 of every informative 1-record block, used to find partial overlaps.

    Blocks with fewer than ``MIN_ACTIVE_CHANNELS`` non-constant channels are
    skipped: flat or clipped segments coincide between unrelated files.
    """
    n_channels, n_times = digital.shape
    n_records = n_times // samples_per_record
    blocks = digital[:, : n_records * samples_per_record].reshape(n_channels, n_records, samples_per_record)
    blocks = blocks.transpose(1, 0, 2)  # shape: (n_records, n_channels, samples_per_record)
    active = (np.ptp(blocks, axis=2) > 0).sum(axis=1)  # shape: (n_records,)
    return tuple(
        hashlib.sha1(np.ascontiguousarray(block).tobytes()).hexdigest()
        for block, n_active in zip(blocks, active)
        if n_active >= MIN_ACTIVE_CHANNELS
    )


def parse_age(group: str, subject_id: str) -> float:
    """Age in years encoded in control folder names (``<code>_<age>``), else NaN."""
    if group != config.GROUP_CONTROL:
        return float("nan")
    match = _CONTROL_AGE_PATTERN.search(subject_id)
    return float(match.group(1)) if match else float("nan")


def export_family(quant_step_uv: float, physical_max: float) -> str:
    """Export format of a file, a diagnostic variable that never enters the model.

    ``"A"``: range +-32768 uV, step 1 uV (controls, all somatoform);
    ``"B"``: range +-2000 uV, step 0.061 uV (all PTSD, two controls);
    ``"C"``: range fitted to each file, step < 0.03 uV, 123-127 Hz (controls).
    """
    if np.isnan(quant_step_uv):
        return ""
    if quant_step_uv == 1.0:
        return "A"
    if physical_max == 2000.0:
        return "B"
    return "C"


def _describe_file(path: Path) -> dict[str, object]:
    """Header metadata, fingerprints and status of one expected EDF file."""
    if not path.exists():
        return {"status": "missing"}
    try:
        header = read_edf_header(path)
        digital = read_digital(header)
    except EdfFormatError:
        return {"status": "empty" if path.stat().st_size == 0 else "error", "file_bytes": path.stat().st_size}

    idx = header.channel_indices()
    step = header.quantization_step_uv()
    spr = int(header.samples_per_record[idx[0]])
    prefilter = header.prefilter[idx[0]]
    return {
        "status": "ok" if header.n_records > 0 else "empty",
        "file_bytes": path.stat().st_size,
        "sfreq": header.sfreq(),
        "n_times": digital.shape[1],
        "duration_s": digital.shape[1] / header.sfreq(),
        "n_records_header": header.n_records_header,
        "n_records": header.n_records,
        "trailing_bytes": header.trailing_bytes,
        "physical_dim": header.physical_dim[idx[0]],
        "physical_min": float(header.physical_min[idx].min()),
        "physical_max": float(header.physical_max[idx].max()),
        "digital_min": float(header.digital_min[idx].min()),
        "digital_max": float(header.digital_max[idx].max()),
        "uniform_scaling": bool(np.ptp(step) == 0),
        "quant_step_uv": float(step.max()),
        "prefilter": prefilter,
        "notch_50": "BS:50" in prefilter.replace(" ", ""),
        "content_hash": content_hash(digital),
        "record_hashes": record_hashes(digital, spr),
    }


def build_registry(data_dir: Path = config.DATA_DIR) -> pd.DataFrame:
    """One row per expected EDF file of every subject in ``data_dir``.

    ``data_dir`` must contain the cohort folders listed in ``config.GROUPS``.
    Files absent from disk get ``status == "missing"``, zero-length files
    ``"empty"``, unreadable files ``"error"``.

    Returns
    -------
    DataFrame with columns ``group, subject_id, subject_key, label, age, stem,
    condition, trial, relpath, status`` followed by header metadata and the
    fingerprints ``content_hash`` and ``record_hashes``.
    """
    rows: list[dict[str, object]] = []
    for group in config.GROUPS:
        group_dir = data_dir / group
        if not group_dir.is_dir():
            continue
        for subject_dir in sorted(p for p in group_dir.iterdir() if p.is_dir()):
            for stem in config.RECORD_STEMS:
                path = subject_dir / f"{stem}.edf"
                is_rest = stem == config.REST_STEM
                rows.append(
                    {
                        "group": group,
                        "subject_id": subject_dir.name,
                        "subject_key": f"{group}/{subject_dir.name}",
                        "label": int(group == config.GROUP_PTSD),
                        "age": parse_age(group, subject_dir.name),
                        "stem": stem,
                        "condition": "rest" if is_rest else "task",
                        "trial": 0 if is_rest else config.TASK_STEMS.index(stem) + 1,
                        "relpath": path.relative_to(data_dir).as_posix(),
                        **_describe_file(path),
                    }
                )
    registry = pd.DataFrame(rows)
    registry["export_family"] = [
        export_family(step, pmax) for step, pmax in zip(registry["quant_step_uv"], registry["physical_max"])
    ]
    return registry


def find_duplicate_links(registry: pd.DataFrame) -> pd.DataFrame:
    """Pairs of files that share signal content.

    ``kind == "identical"``: equal digital content of the six channels.
    ``kind == "overlap"``: different content but at least
    ``MIN_SHARED_RECORDS`` identical informative 1-s records (e.g. one file is
    a cropped copy of another). Overlaps that are not aligned to EDF record
    boundaries are not detected.

    Returns
    -------
    DataFrame with columns ``relpath_a, relpath_b, subject_a, subject_b, kind,
    n_shared_records``.
    """
    ok = registry[registry["status"] == "ok"]
    subject_of = dict(zip(ok["relpath"], ok["subject_key"]))
    content_of = dict(zip(ok["relpath"], ok["content_hash"]))

    files_by_record: dict[str, set[str]] = defaultdict(set)
    for relpath, hashes in zip(ok["relpath"], ok["record_hashes"]):
        for h in hashes:
            files_by_record[h].add(relpath)
    shared: dict[tuple[str, str], int] = defaultdict(int)
    for files in files_by_record.values():
        for a, b in combinations(sorted(files), 2):
            shared[(a, b)] += 1

    pairs: dict[tuple[str, str], str] = {}
    for files in ok.groupby("content_hash")["relpath"].apply(sorted):
        for a, b in combinations(files, 2):
            pairs[(a, b)] = "identical"
    for (a, b), n in shared.items():
        if n >= MIN_SHARED_RECORDS and (a, b) not in pairs and content_of[a] != content_of[b]:
            pairs[(a, b)] = "overlap"

    columns = ["relpath_a", "relpath_b", "subject_a", "subject_b", "kind", "n_shared_records"]
    links = [(a, b, subject_of[a], subject_of[b], kind, shared.get((a, b), 0)) for (a, b), kind in pairs.items()]
    return pd.DataFrame(links, columns=columns).sort_values(["relpath_a", "relpath_b"], ignore_index=True)


def connected_components(nodes: list[str], edges: list[tuple[str, str]]) -> dict[str, int]:
    """Label connected components of an undirected graph (union-find).

    Component ids are consecutive integers ordered by the smallest node name
    in each component, so the labelling is deterministic.
    """
    parent = {node: node for node in nodes}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for a, b in edges:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)

    roots = sorted({find(node) for node in nodes})
    root_id = {root: i for i, root in enumerate(roots)}
    return {node: root_id[find(node)] for node in nodes}


def assign_groups(registry: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    """Subject table with the independence group used for every data split.

    Subjects linked by any shared recording are merged into one group, the
    minimal unit that can be placed on one side of a train/test split without
    leaking a copy of the same signal to the other side.

    Returns
    -------
    DataFrame indexed by ``subject_key`` with columns ``group, subject_id,
    label, age, split_group, group_size``.
    """
    subjects = (
        registry.groupby("subject_key", sort=True)[["group", "subject_id", "label", "age"]].first().copy()
    )
    edges = [(a, b) for a, b in zip(links["subject_a"], links["subject_b"]) if a != b]
    component = connected_components(list(subjects.index), edges)
    subjects["split_group"] = subjects.index.map(component)
    subjects["group_size"] = subjects.groupby("split_group")["label"].transform("size")
    return subjects


def mark_duplicate_files(registry: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    """Add ``duplicate_set``: shared id for files linked by content, -1 otherwise."""
    relpaths = list(registry["relpath"])
    edges = list(zip(links["relpath_a"], links["relpath_b"]))
    component = connected_components(relpaths, edges)
    sizes = pd.Series(component).value_counts()
    out = registry.copy()
    out["duplicate_set"] = [component[r] if sizes[component[r]] > 1 else -1 for r in relpaths]
    return out


def scan_dataset(data_dir: Path = config.DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Registry of files, duplicate links and subject table with split groups."""
    registry = build_registry(data_dir)
    links = find_duplicate_links(registry)
    registry = mark_duplicate_files(registry, links)
    subjects = assign_groups(registry, links)
    return registry, links, subjects


if __name__ == "__main__":
    registry, links, subjects = scan_dataset()
    config.CACHE_DIR.mkdir(exist_ok=True)
    registry.drop(columns="record_hashes").to_csv(config.CACHE_DIR / "registry.csv", index=False)
    links.to_csv(config.CACHE_DIR / "duplicate_links.csv", index=False)
    subjects.to_csv(config.CACHE_DIR / "subjects.csv")
    print(f"files: {len(registry)}, subjects: {len(subjects)}, split groups: {subjects['split_group'].nunique()}")
    print(f"duplicate links: {len(links)}, files in duplicate sets: {(registry['duplicate_set'] >= 0).sum()}")
