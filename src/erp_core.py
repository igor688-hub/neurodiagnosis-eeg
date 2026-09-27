from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.io
from scipy.signal import butter, lfilter, resample_poly

from src import config

OSF_API: Final[str] = "https://api.osf.io/v2/nodes/5q4xs/files/osfstorage"
ALL_DATA_FOLDER: Final[str] = "5f248e8db084f6011bc9da61"
ERPSET_SUFFIX: Final[str] = "_MMN_erp_ar_lpfilt.erp"
DEFAULT_DIR: Final[Path] = config.DATA_DIR / "external" / "erp_core_mmn"

CHANNEL_MAP: Final[dict[str, str]] = {"O1": "O1", "T3": "C5", "Fp1": "FP1", "Fp2": "FP2", "T4": "C6", "O2": "O2"}
HARDWARE_HIGHPASS_HZ: Final[float] = config.HEADER_HIGHPASS_HZ
MMN_WINDOW_S: Final[tuple[float, float]] = (0.125, 0.225)
DEVIANT_BIN: Final[str] = "Deviants"
STANDARD_BIN: Final[str] = "Standards, Preceded by a Standard"


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def list_osf_folder(folder_id: str) -> list[dict]:
    """All entries of an OSF storage folder (follows pagination)."""
    url: str | None = f"{OSF_API}/{folder_id}/?page[size]=100"
    entries: list[dict] = []
    while url:
        page = _get_json(url)
        entries += page["data"]
        url = page["links"].get("next")
    return entries


def download_erpsets(dest: Path = DEFAULT_DIR) -> list[Path]:
    """Download the 40 low-pass filtered subject ERPsets (~0.18 MB each)."""
    dest.mkdir(parents=True, exist_ok=True)
    paths = []
    for folder in list_osf_folder(ALL_DATA_FOLDER):
        name = folder["attributes"]["name"]
        if folder["attributes"]["kind"] != "folder" or not name.isdigit():
            continue
        for entry in list_osf_folder(folder["id"]):
            attrs = entry["attributes"]
            if attrs["kind"] != "file" or not attrs["name"].endswith(ERPSET_SUFFIX):
                continue
            path = dest / attrs["name"]
            md5 = attrs["extra"]["hashes"]["md5"]
            if not (path.exists() and hashlib.md5(path.read_bytes()).hexdigest() == md5):
                with urllib.request.urlopen(entry["links"]["download"], timeout=120) as response:
                    content = response.read()
                if hashlib.md5(content).hexdigest() != md5:
                    raise OSError(f"{attrs['name']}: MD5 mismatch")
                path.write_bytes(content)
            paths.append(path)
    return sorted(paths, key=lambda p: int(p.name.split("_")[0]))


@dataclass(frozen=True)
class ErpSet:
    """Averaged ERPs of one subject."""

    times_s: npt.NDArray[np.float64]
    channels: tuple[str, ...]
    bins: dict[str, npt.NDArray[np.float64]]
    sfreq: float

    def channel(self, bin_name: str, name: str) -> npt.NDArray[np.float64]:
        return self.bins[bin_name][self.channels.index(name)]


def load_erpset(path: Path) -> ErpSet:
    """Read an ERPLAB ``.erp`` file (MATLAB struct ``ERP``)."""
    erp = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)["ERP"]
    labels = tuple(str(loc.labels).strip() for loc in np.atleast_1d(erp.chanlocs))
    data = np.asarray(erp.bindata, dtype=float)
    if data.ndim == 2:
        data = data[:, :, None]
    descr = [str(d).strip() for d in np.atleast_1d(erp.bindescr)]
    return ErpSet(
        times_s=np.asarray(erp.times, dtype=float) / 1000.0,
        channels=labels,
        bins={d: data[:, :, i] for i, d in enumerate(descr)},
        sfreq=float(erp.srate),
    )


def causal_highpass(
    x: npt.NDArray[np.float64], sfreq: float, order: int, cutoff_hz: float = HARDWARE_HIGHPASS_HZ
) -> npt.NDArray[np.float64]:
    """Causal Butterworth high-pass along the last axis, zero input assumed before the first sample."""
    b, a = butter(order, cutoff_hz, btype="highpass", fs=sfreq)
    return lfilter(b, a, x, axis=-1)


def highpass_response(freqs_hz: npt.NDArray[np.float64], order: int, cutoff_hz: float = HARDWARE_HIGHPASS_HZ):
    """Complex response of the analogue Butterworth high-pass (steady state of periodic signals)."""
    if order == 0:
        return np.ones_like(freqs_hz, dtype=complex)
    b, a = butter(order, 2 * np.pi * cutoff_hz, btype="highpass", analog=True)
    s = 2j * np.pi * np.asarray(freqs_hz, dtype=float)
    return np.polyval(b, s) / np.polyval(a, s)


def to_125hz(x: npt.NDArray[np.float64], sfreq: float) -> npt.NDArray[np.float64]:
    """Polyphase resampling along the last axis, 256 -> 125 Hz."""
    from fractions import Fraction

    ratio = Fraction(config.TARGET_SFREQ / sfreq).limit_denominator(1000)
    return resample_poly(x, ratio.numerator, ratio.denominator, axis=-1, padtype="line")


def neuroplay_view(erp: ErpSet, highpass_order: int) -> ErpSet:
    """ERP on our six channels (nearest ERP CORE sites) after the emulated hardware high-pass."""
    idx = [erp.channels.index(src) for src in CHANNEL_MAP.values()]
    bins = {}
    for name, data in erp.bins.items():
        x = data[idx]
        bins[name] = causal_highpass(x, erp.sfreq, highpass_order) if highpass_order else x
    return ErpSet(erp.times_s, tuple(CHANNEL_MAP), bins, erp.sfreq)


def window_mean(times_s: npt.NDArray[np.float64], x: npt.NDArray[np.float64], window_s: tuple[float, float]) -> npt.NDArray[np.float64]:
    """Mean over a time window along the last axis."""
    sel = (times_s >= window_s[0]) & (times_s <= window_s[1])
    return x[..., sel].mean(axis=-1)


def load_all(directory: Path = DEFAULT_DIR) -> list[ErpSet]:
    """All downloaded subject ERPsets, in subject order."""
    paths = sorted(directory.glob(f"*{ERPSET_SUFFIX}"), key=lambda p: int(p.name.split("_")[0]))
    if not paths:
        raise FileNotFoundError(f"no ERP CORE files in {directory}; run `python -m src.erp_core`")
    return [load_erpset(p) for p in paths]


def site_average(erpsets: list[ErpSet], sites: tuple[str, ...], highpass_order: int) -> dict[str, npt.NDArray[np.float64]]:
    """Per-subject deviant, standard and difference waves averaged over ``sites``."""
    out: dict[str, list[npt.NDArray[np.float64]]] = {"deviant": [], "standard": []}
    for erp in erpsets:
        idx = [erp.channels.index(site) for site in sites]
        for key, bin_name in (("deviant", DEVIANT_BIN), ("standard", STANDARD_BIN)):
            x = erp.bins[bin_name][idx].mean(axis=0)
            out[key].append(causal_highpass(x, erp.sfreq, highpass_order) if highpass_order else x)
    result = {key: np.array(values) for key, values in out.items()}
    result["difference"] = result["deviant"] - result["standard"]
    return result


VIEWS: Final[tuple[tuple[str, tuple[str, ...], int], ...]] = (
    ("FCz", ("FCz",), 0),
    ("Fp", ("FP1", "FP2"), 0),
    ("Fp HP1", ("FP1", "FP2"), 1),
    ("Fp HP2", ("FP1", "FP2"), 2),
    ("Fp HP4", ("FP1", "FP2"), 4),
    ("C5/C6 HP4", ("C5", "C6"), 4),
    ("O1/O2 HP4", ("O1", "O2"), 4),
)


def summary(erpsets: list[ErpSet]) -> tuple[dict[str, dict[str, float]], pd.DataFrame]:
    """MMN measures and grand-average curves for every view in ``VIEWS``."""
    times = erpsets[0].times_s
    metrics: dict[str, dict[str, float]] = {}
    frames = []
    for label, sites, order in VIEWS:
        waves = site_average(erpsets, sites, order)
        diff = waves["difference"]
        mmn = window_mean(times, diff, MMN_WINDOW_S)
        grand = diff.mean(axis=0)
        search = (times >= 0.05) & (times <= 0.40)
        extreme = np.argmax(np.abs(np.where(search, grand, 0.0)))
        n1_sel = (times >= 0.05) & (times <= 0.20)
        std_grand = waves["standard"].mean(axis=0)
        metrics[label] = {
            "mmn_window_mean_uv": round(float(mmn.mean()), 3),
            "mmn_window_sd_uv": round(float(mmn.std(ddof=1)), 3),
            "mmn_window_share_negative": round(float(np.mean(mmn < 0)), 3),
            "difference_extremum_uv": round(float(grand[extreme]), 3),
            "difference_extremum_latency_s": round(float(times[extreme]), 4),
            "difference_rms_0_600ms_uv": round(float(np.sqrt(np.mean(grand[(times >= 0) & (times <= 0.6)] ** 2))), 3),
            "standard_n1_uv": round(float(std_grand[n1_sel].min()), 3),
            "standard_n1_latency_s": round(float(times[n1_sel][np.argmin(std_grand[n1_sel])]), 4),
        }
        for key, values in waves.items():
            frames.append(
                pd.DataFrame(
                    {
                        "view": label,
                        "wave": key,
                        "time_s": times,
                        "mean_uv": values.mean(axis=0),
                        "sem_uv": values.std(axis=0, ddof=1) / np.sqrt(values.shape[0]),
                    }
                )
            )
    return metrics, pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    files = download_erpsets()
    print(f"{len(files)} ERPsets in {DEFAULT_DIR}")
