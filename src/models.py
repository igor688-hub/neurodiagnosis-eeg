from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config, dataset, features

BASELINE_C: Final[float] = 0.1
BASELINE_CLASS_WEIGHT: Final[str] = "balanced"


def make_pipeline(c: float = BASELINE_C, class_weight: str | None = BASELINE_CLASS_WEIGHT) -> Pipeline:
    """Unfitted pipeline."""
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=c, class_weight=class_weight, max_iter=5000)),
        ]
    )


@dataclass(frozen=True)
class LogisticModel:
    """Fitted pipeline as plain parameters."""

    feature_names: tuple[str, ...]
    impute_values: npt.NDArray[np.float64]
    mean: npt.NDArray[np.float64]
    scale: npt.NDArray[np.float64]
    coef: npt.NDArray[np.float64]
    intercept: float
    metadata: dict[str, object] = field(default_factory=dict)
    calibration: tuple[float, float] | None = None

    def decision_function(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Log-odds of the fitted classifier."""
        x = np.where(np.isnan(x), self.impute_values, x)
        z = (x - self.mean) / self.scale
        return z @ self.coef + self.intercept

    def predict_proba_raw(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Output of the class-weighted classifier before calibration, shape (n_subjects,)."""
        return 1.0 / (1.0 + np.exp(-self.decision_function(x)))

    def predict_proba(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Probability of the PTSD cohort, shape (n_subjects,)."""
        logit = self.decision_function(x)
        if self.calibration is not None:
            a, b = self.calibration
            logit = a * logit + b
        return 1.0 / (1.0 + np.exp(-logit))

    @classmethod
    def from_pipeline(
        cls, pipeline: Pipeline, feature_names: tuple[str, ...], metadata: dict[str, object] | None = None
    ) -> LogisticModel:
        imputer: SimpleImputer = pipeline.named_steps["impute"]
        scaler: StandardScaler = pipeline.named_steps["scale"]
        clf: LogisticRegression = pipeline.named_steps["clf"]
        return cls(
            feature_names=tuple(feature_names),
            impute_values=imputer.statistics_.astype(float),
            mean=scaler.mean_.astype(float),
            scale=scaler.scale_.astype(float),
            coef=clf.coef_.ravel().astype(float),
            intercept=float(clf.intercept_[0]),
            metadata=dict(metadata or {}),
        )

    def to_json(self, path: Path) -> None:
        payload = asdict(self)
        for key in ("impute_values", "mean", "scale", "coef"):
            payload[key] = getattr(self, key).tolist()
        payload["feature_names"] = list(self.feature_names)
        payload["calibration"] = None if self.calibration is None else list(self.calibration)
        path.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> LogisticModel:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            feature_names=tuple(payload["feature_names"]),
            impute_values=np.asarray(payload["impute_values"], dtype=float),
            mean=np.asarray(payload["mean"], dtype=float),
            scale=np.asarray(payload["scale"], dtype=float),
            coef=np.asarray(payload["coef"], dtype=float),
            intercept=float(payload["intercept"]),
            metadata=payload.get("metadata", {}),
            calibration=None if payload.get("calibration") is None else tuple(payload["calibration"]),
        )


def fit_logistic(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.int_],
    feature_names: tuple[str, ...],
    c: float = BASELINE_C,
    class_weight: str | None = BASELINE_CLASS_WEIGHT,
    metadata: dict[str, object] | None = None,
) -> LogisticModel:
    """Fit the pipeline on all given subjects."""
    pipeline = make_pipeline(c, class_weight).fit(x, y)
    meta = {"C": c, "class_weight": class_weight, "random_state": config.RANDOM_STATE, **(metadata or {})}
    return LogisticModel.from_pipeline(pipeline, feature_names, meta)


COHORT_CONTROL: Final[int] = 0
COHORT_PTSD: Final[int] = 1
COHORT_SOMATOFORM: Final[int] = 2
COHORT_CODES: Final[dict[str, int]] = {
    config.GROUP_CONTROL: COHORT_CONTROL,
    config.GROUP_PTSD: COHORT_PTSD,
    config.GROUP_SOMATOFORM: COHORT_SOMATOFORM,
}

STRATUM_PTSD: Final[str] = "ptsd"
STRATUM_SOMATOFORM: Final[str] = "somatoform"
STRATUM_CONTROL_A: Final[str] = "control_A"
STRATUM_CONTROL_B: Final[str] = "control_B"
STRATUM_CONTROL_B_SUPPLEMENT: Final[str] = "control_B_supplement"
STRATUM_CONTROL_C: Final[str] = "control_C"
STRATUM_CONTROL_AGEING: Final[str] = "control_ageing"

C_GRID: Final[tuple[float, ...]] = (0.01, 0.03, 0.1, 0.3, 1.0)
NEGATIVES: Final[tuple[str, ...]] = ("control+somatoform", "control")
FEATURE_VARIANTS: Final[tuple[str, ...]] = ("native", "requantized")
INNER_SPLITS: Final[int] = 5
DECISION_THRESHOLD: Final[float] = 0.5


@dataclass(frozen=True)
class Candidate:
    """One configuration of the training procedure."""

    c: float
    negatives: str
    features: str

    @property
    def key(self) -> str:
        return f"C={self.c}|neg={self.negatives}|feat={self.features}"


CANDIDATES: Final[tuple[Candidate, ...]] = tuple(
    Candidate(c, negatives, variant) for variant in FEATURE_VARIANTS for negatives in NEGATIVES for c in C_GRID
)


def _auc(p: npt.NDArray[np.float64], positive: npt.NDArray[np.bool_], negative: npt.NDArray[np.bool_]) -> float:
    mask = positive | negative
    return float(roc_auc_score(positive[mask], p[mask]))


def selection_score(
    p: npt.NDArray[np.float64], cohort: npt.NDArray[np.int_], stratum: npt.NDArray[np.str_] | None = None
) -> float:
    """Protocol 1 criterion, mirroring the objective part of the scoring rules."""
    auc = _auc(p, cohort == COHORT_PTSD, cohort == COHORT_CONTROL)
    soma = cohort == COHORT_SOMATOFORM
    fpr = float(np.mean(p[soma] >= DECISION_THRESHOLD)) if soma.any() else 0.0
    return 30.0 * max(0.0, (auc - 0.5) / 0.5) + 20.0 * (1.0 - fpr)


PROTOCOL2_AUC_NEGATIVES: Final[tuple[str, ...]] = (
    STRATUM_CONTROL_A,
    STRATUM_CONTROL_B,
    STRATUM_CONTROL_B_SUPPLEMENT,
    STRATUM_SOMATOFORM,
)
PROTOCOL2_SPECIFICITY_GROUPS: Final[tuple[tuple[str, ...], ...]] = (
    (STRATUM_CONTROL_A,),
    (STRATUM_CONTROL_B, STRATUM_CONTROL_B_SUPPLEMENT),
    (STRATUM_SOMATOFORM,),
)


def selection_score_protocol2(
    p: npt.NDArray[np.float64], cohort: npt.NDArray[np.int_], stratum: npt.NDArray[np.str_] | None = None
) -> float:
    """Protocol 2 criterion with explicit groups."""
    assert stratum is not None, "protocol 2 needs strata"
    ptsd = stratum == STRATUM_PTSD
    auc = _auc(p, ptsd, np.isin(stratum, PROTOCOL2_AUC_NEGATIVES))
    fprs = [
        float(np.mean(p[np.isin(stratum, group)] >= DECISION_THRESHOLD))
        for group in PROTOCOL2_SPECIFICITY_GROUPS
        if np.isin(stratum, group).any()
    ]
    return 30.0 * max(0.0, (auc - 0.5) / 0.5) + 20.0 * (1.0 - float(np.mean(fprs)))


def _names(prefix_bands: str, channels: Sequence[str]) -> tuple[str, ...]:
    from src.features import BANDS

    return tuple(f"{prefix_bands}_{band}_{ch}" for band in BANDS for ch in channels)


PROTOCOL2_CORE_CHANNELS: Final[tuple[str, ...]] = ("O1", "Fp1", "Fp2")
PROTOCOL2_TEMPORAL_CHANNELS: Final[tuple[str, ...]] = ("T3", "T4")
_P2_ALPHA: Final[tuple[str, ...]] = ("rest_iaf_O1", "rest_alpha_peak_O1")


def _protocol2_feature_sets() -> dict[str, tuple[str, ...]]:
    core, temporal = PROTOCOL2_CORE_CHANNELS, PROTOCOL2_CORE_CHANNELS + PROTOCOL2_TEMPORAL_CHANNELS
    return {
        "rest_O1Fp": (*_names("rest_relpow", core), *_P2_ALPHA),
        "rest_O1Fp_exp": (*_names("rest_relpow", core), *_P2_ALPHA, *(f"rest_exponent_{ch}" for ch in core)),
        "rest_O1FpT": (*_names("rest_relpow", temporal), *_P2_ALPHA),
        "rest_O1FpT_exp": (*_names("rest_relpow", temporal), *_P2_ALPHA, *(f"rest_exponent_{ch}" for ch in temporal)),
    }


@dataclass(frozen=True)
class Protocol:
    """A pre-registered training procedure."""

    name: str
    feature_sets: dict[str, tuple[str, tuple[str, ...]]]
    candidates: tuple[Candidate, ...]
    score: Callable[[npt.NDArray[np.float64], npt.NDArray[np.int_], npt.NDArray[np.str_] | None], float]
    include_rest_only: bool
    stratify_inner_by_stratum: bool
    include_ageing: bool = False
    calibrate: bool = False
    calibration_class_weight: str | None = None


PROTOCOL_1: Final[Protocol] = Protocol(
    name="protocol1",
    feature_sets={variant: (variant, features.PROTOCOL1_FEATURES) for variant in FEATURE_VARIANTS},
    candidates=CANDIDATES,
    score=selection_score,
    include_rest_only=False,
    stratify_inner_by_stratum=False,
)

PROTOCOL_2: Final[Protocol] = Protocol(
    name="protocol2",
    feature_sets={key: ("native", names) for key, names in _protocol2_feature_sets().items()},
    candidates=tuple(
        Candidate(c, negatives, key) for key in _protocol2_feature_sets() for negatives in NEGATIVES for c in C_GRID
    ),
    score=selection_score_protocol2,
    include_rest_only=True,
    stratify_inner_by_stratum=True,
)
PROTOCOLS: Final[dict[str, Protocol]] = {PROTOCOL_1.name: PROTOCOL_1, PROTOCOL_2.name: PROTOCOL_2}


def restricted_protocol(protocol: Protocol, feature_sets: Sequence[str], name: str) -> Protocol:
    """The same procedure limited to some feature tables (descriptive ablations only)."""
    return replace(
        protocol,
        name=name,
        feature_sets={key: protocol.feature_sets[key] for key in feature_sets},
        candidates=tuple(c for c in protocol.candidates if c.features in feature_sets),
    )


PROTOCOL3_CHANNELS: Final[tuple[str, ...]] = ("O1", "Fp1", "Fp2")
PROTOCOL3_AUC_NEGATIVES: Final[tuple[str, ...]] = (
    STRATUM_CONTROL_A,
    STRATUM_CONTROL_B,
    STRATUM_CONTROL_B_SUPPLEMENT,
)
PROTOCOL3_SPECIFICITY_GROUPS: Final[tuple[tuple[str, ...], ...]] = (
    (STRATUM_CONTROL_AGEING,),
    (STRATUM_SOMATOFORM,),
)


def selection_score_protocol3(
    p: npt.NDArray[np.float64], cohort: npt.NDArray[np.int_], stratum: npt.NDArray[np.str_] | None = None
) -> float:
    """Protocol 3 criterion following the structure of the objective scoring."""
    assert stratum is not None, "protocol 3 needs strata"
    ptsd = stratum == STRATUM_PTSD
    auc = _auc(p, ptsd, np.isin(stratum, PROTOCOL3_AUC_NEGATIVES))
    fprs = [
        float(np.mean(p[np.isin(stratum, group)] >= DECISION_THRESHOLD))
        for group in PROTOCOL3_SPECIFICITY_GROUPS
        if np.isin(stratum, group).any()
    ]
    return 30.0 * max(0.0, (auc - 0.5) / 0.5) + 20.0 * (1.0 - float(np.mean(fprs)))


def _protocol3_feature_sets() -> dict[str, tuple[str, ...]]:
    from src.features import BANDS_20

    rel = tuple(f"rest_relpow20_{band}_{ch}" for band in BANDS_20 for ch in PROTOCOL3_CHANNELS)
    alpha = ("rest_iaf20_O1", "rest_alpha_peak20_O1")
    slope = tuple(f"rest_slope20_{ch}" for ch in PROTOCOL3_CHANNELS)
    return {"rest20": (*rel, *alpha), "rest20_slope": (*rel, *alpha, *slope)}


PROTOCOL_3: Final[Protocol] = Protocol(
    name="protocol3",
    feature_sets={key: ("native", names) for key, names in _protocol3_feature_sets().items()},
    candidates=tuple(
        Candidate(c, negatives, key) for key in _protocol3_feature_sets() for negatives in NEGATIVES for c in C_GRID
    ),
    score=selection_score_protocol3,
    include_rest_only=True,
    stratify_inner_by_stratum=True,
    include_ageing=True,
    calibrate=True,
)
PROTOCOLS[PROTOCOL_3.name] = PROTOCOL_3


PLATT_C: Final[float] = 1e6


def fit_platt(
    p_raw: npt.NDArray[np.float64], y: npt.NDArray[np.int_], class_weight: str | None = None
) -> tuple[float, float]:
    """Platt scaling."""
    eps = 1e-6
    p = np.clip(p_raw, eps, 1.0 - eps)
    logit = np.log(p / (1.0 - p))[:, None]
    clf = LogisticRegression(C=PLATT_C, class_weight=class_weight, max_iter=5000).fit(logit, y)
    return float(clf.coef_[0, 0]), float(clf.intercept_[0])


def apply_platt(p_raw: npt.NDArray[np.float64], calibration: tuple[float, float]) -> npt.NDArray[np.float64]:
    """Calibrated probability sigmoid(a * logit(p_raw) + b)."""
    eps = 1e-12
    p = np.clip(p_raw, eps, 1.0 - eps)
    a, b = calibration
    return 1.0 / (1.0 + np.exp(-(a * np.log(p / (1.0 - p)) + b)))


def training_mask(cohort: npt.NDArray[np.int_], negatives: str) -> npt.NDArray[np.bool_]:
    """Subjects a candidate is trained on."""
    if negatives == "control":
        return cohort != COHORT_SOMATOFORM
    return np.ones(cohort.shape, dtype=bool)


def fit_candidate(
    x: npt.NDArray[np.float64], cohort: npt.NDArray[np.int_], candidate: Candidate
) -> Pipeline:
    """Fit the pipeline of ``candidate`` on the subjects it trains on."""
    mask = training_mask(cohort, candidate.negatives)
    return make_pipeline(candidate.c).fit(x[mask], (cohort[mask] == COHORT_PTSD).astype(int))


def inner_splits(labels: npt.NDArray[np.generic], groups: npt.NDArray[np.int_], seed: int = config.RANDOM_STATE):
    """Group-aware folds stratified by ``labels`` (cohort or stratum), shared by all candidates."""
    codes = np.unique(labels, return_inverse=True)[1]
    cv = StratifiedGroupKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=seed)
    return list(cv.split(np.zeros(codes.size), codes, groups))


def select_candidate(
    tables: Mapping[str, npt.NDArray[np.float64]],
    cohort: npt.NDArray[np.int_],
    groups: npt.NDArray[np.int_],
    stratum: npt.NDArray[np.str_] | None = None,
    protocol: Protocol = PROTOCOL_1,
    oof_out: dict[str, npt.NDArray[np.float64]] | None = None,
) -> tuple[Candidate, dict[str, float]]:
    """Choose a candidate by inner group cross-validation on the given subjects only."""
    folds = inner_splits(stratum if protocol.stratify_inner_by_stratum else cohort, groups)
    scores: dict[str, float] = {}
    for candidate in protocol.candidates:
        x = tables[candidate.features]
        oof = np.empty(cohort.size)
        for train, test in folds:
            oof[test] = fit_candidate(x[train], cohort[train], candidate).predict_proba(x[test])[:, 1]
        scores[candidate.key] = protocol.score(oof, cohort, stratum)
        if oof_out is not None:
            oof_out[candidate.key] = oof
    candidates = protocol.candidates
    order = sorted(range(len(candidates)), key=lambda i: (-round(scores[candidates[i].key], 10), candidates[i].c, i))
    return candidates[order[0]], scores


@dataclass(frozen=True)
class TrainingData:
    """Training subjects (hold-out excluded) with aligned feature tables and labels."""

    subject_keys: tuple[str, ...]
    tables: dict[str, npt.NDArray[np.float64]]
    feature_names: dict[str, tuple[str, ...]]
    cohort: npt.NDArray[np.int_]
    groups: npt.NDArray[np.int_]
    stratum: npt.NDArray[np.str_]
    export_family: npt.NDArray[np.str_]
    metadata: npt.NDArray[np.float64]
    protocol: str

    @property
    def y(self) -> npt.NDArray[np.int_]:
        return (self.cohort == COHORT_PTSD).astype(int)


def _export_descriptors(registry: pd.DataFrame, subject_keys: Sequence[str]) -> npt.NDArray[np.float64]:
    """Per subject."""
    ok = registry[registry["status"] == "ok"].groupby("subject_key")
    fam = ok["export_family"].first().reindex(subject_keys)
    notch = ok["notch_50"].mean().reindex(subject_keys)
    at_125 = ok["sfreq"].apply(lambda s: float(np.mean(s == config.TARGET_SFREQ))).reindex(subject_keys)
    return np.column_stack([fam == "A", fam == "B", fam == "C", notch, at_125]).astype(float)


def _strata(subjects: pd.DataFrame, family: pd.Series) -> npt.NDArray[np.str_]:
    out = []
    for key, row in subjects.iterrows():
        if row["group"] == config.GROUP_PTSD:
            out.append(STRATUM_PTSD)
        elif row["holdout"]:
            out.append(STRATUM_CONTROL_AGEING)
        elif row["group"] == config.GROUP_SOMATOFORM:
            out.append(STRATUM_SOMATOFORM)
        elif not row["has_task_files"]:
            out.append(STRATUM_CONTROL_B_SUPPLEMENT if family[key] == "B" else f"control_{family[key]}_supplement")
        else:
            out.append(f"control_{family[key]}")
    return np.array(out)


def load_training_data(data_dir: Path = config.DATA_DIR, protocol: Protocol = PROTOCOL_1) -> TrainingData:
    """Registry, split groups, strata and the feature tables of ``protocol``."""
    registry, _, subjects = dataset.scan_dataset(data_dir)
    keep_mask = (~subjects["holdout"] | protocol.include_ageing) & (
        subjects["has_task_files"] | protocol.include_rest_only
    )
    keep = subjects.index[keep_mask]
    registry, subjects = registry[registry["subject_key"].isin(keep)], subjects.loc[keep]
    variants = sorted({variant for variant, _ in protocol.feature_sets.values()})
    full = {
        variant: features.build_feature_table(registry, features.PREPROCESSING_VARIANTS[variant], data_dir)
        for variant in variants
    }
    keys = tuple(full[variants[0]].index)
    family = registry[registry["status"] == "ok"].groupby("subject_key")["export_family"].first().reindex(keys)
    return TrainingData(
        subject_keys=keys,
        tables={
            key: full[variant].loc[list(keys), list(names)].to_numpy(dtype=float)
            for key, (variant, names) in protocol.feature_sets.items()
        },
        feature_names={key: names for key, (_, names) in protocol.feature_sets.items()},
        cohort=subjects.loc[list(keys), "group"].map(COHORT_CODES).to_numpy(dtype=int),
        groups=subjects.loc[list(keys), "split_group"].to_numpy(dtype=int),
        stratum=_strata(subjects.loc[list(keys)], family),
        export_family=family.to_numpy(dtype=str),
        metadata=_export_descriptors(registry, keys),
        protocol=protocol.name,
    )


def selection_score_protocol6(
    p: npt.NDArray[np.float64], cohort: npt.NDArray[np.int_], stratum: npt.NDArray[np.str_] | None = None
) -> float:
    """Threshold-free criterion with the weights of the objective scoring."""
    assert stratum is not None, "protocol 6 needs strata"
    ptsd = stratum == STRATUM_PTSD
    young_controls = np.char.startswith(stratum.astype(str), "control_") & (stratum != STRATUM_CONTROL_AGEING)
    auc_controls = _auc(p, ptsd, young_controls)
    auc_spec = _auc(p, ptsd, np.isin(stratum, [STRATUM_CONTROL_AGEING, STRATUM_SOMATOFORM]))
    return 30.0 * max(0.0, (auc_controls - 0.5) / 0.5) + 20.0 * max(0.0, (auc_spec - 0.5) / 0.5)


PROTOCOL_6: Final[Protocol] = Protocol(
    name="protocol6",
    feature_sets=PROTOCOL_2.feature_sets,
    candidates=PROTOCOL_2.candidates,
    score=selection_score_protocol6,
    include_rest_only=True,
    stratify_inner_by_stratum=True,
    include_ageing=True,
    calibrate=True,
)
PROTOCOLS[PROTOCOL_6.name] = PROTOCOL_6


PROTOCOL_7: Final[Protocol] = Protocol(
    name="protocol7",
    feature_sets={key: (variant, (*names, "rest_faa")) for key, (variant, names) in PROTOCOL_2.feature_sets.items()},
    candidates=PROTOCOL_2.candidates,
    score=selection_score_protocol6,
    include_rest_only=True,
    stratify_inner_by_stratum=True,
    include_ageing=True,
    calibrate=True,
    calibration_class_weight="balanced",
)
PROTOCOLS[PROTOCOL_7.name] = PROTOCOL_7
