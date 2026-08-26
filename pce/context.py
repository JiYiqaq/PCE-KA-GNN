"""Leakage-free preprocessing for device and processing context."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


MISSING_CATEGORY_INDEX = 0
UNKNOWN_CATEGORY_INDEX = 1
MISSING_CATEGORY_TOKEN = "__MISSING__"
UNKNOWN_CATEGORY_TOKEN = "__UNKNOWN__"

NUMERIC_SOURCE_COLUMNS: Tuple[str, ...] = (
    "homo_d",
    "lumo_d",
    "homo_a",
    "lumo_a",
    "active_layer_thickness",
    "annealing_temp",
    "d_a_ratio",
    "additive_ratio",
)
NUMERIC_FEATURES: Tuple[str, ...] = (
    "homo_d",
    "lumo_d",
    "homo_a",
    "lumo_a",
    "active_layer_thickness",
    "annealing_temp",
    "donor_acceptor_log_ratio",
    "additive_percentage",
)
CATEGORICAL_FEATURES: Tuple[str, ...] = (
    "device_type",
    "etl_canonical",
    "htl_canonical",
    "solvent_canonical",
    "additive_canonical",
)

_NUMBER = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))"
_RATIO_RE = re.compile(rf"^\s*{_NUMBER}\s*:\s*{_NUMBER}", re.IGNORECASE)
_PERCENT_RE = re.compile(rf"{_NUMBER}\s*(?:vol\s*)?%", re.IGNORECASE)


def parse_donor_acceptor_log_ratio(value: Any) -> float:
    """Parse a donor:acceptor mass ratio and return log(donor / acceptor)."""
    if not isinstance(value, str):
        return math.nan
    match = _RATIO_RE.match(value)
    if match is None:
        return math.nan
    donor, acceptor = (float(match.group(1)), float(match.group(2)))
    if not math.isfinite(donor) or not math.isfinite(acceptor) or donor <= 0 or acceptor <= 0:
        return math.nan
    return math.log(donor / acceptor)


def parse_additive_percentage(value: Any) -> float:
    """Parse an explicitly percentage-valued additive amount."""
    if not isinstance(value, str):
        return math.nan
    match = _PERCENT_RE.search(value)
    if match is None:
        return math.nan
    percentage = float(match.group(1))
    return percentage if math.isfinite(percentage) and percentage >= 0 else math.nan


def _numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    values: Dict[str, pd.Series] = {}
    for feature in NUMERIC_FEATURES[:6]:
        source = frame[feature] if feature in frame else pd.Series(np.nan, index=frame.index)
        values[feature] = pd.to_numeric(source, errors="coerce")

    ratio_source = frame["d_a_ratio"] if "d_a_ratio" in frame else pd.Series(None, index=frame.index)
    additive_source = (
        frame["additive_ratio"] if "additive_ratio" in frame else pd.Series(None, index=frame.index)
    )
    values["donor_acceptor_log_ratio"] = ratio_source.map(parse_donor_acceptor_log_ratio)
    values["additive_percentage"] = additive_source.map(parse_additive_percentage)
    return pd.DataFrame(values, index=frame.index, dtype=np.float64)


def _category_value(value: Any) -> str:
    if pd.isna(value) or not str(value).strip():
        return MISSING_CATEGORY_TOKEN
    return str(value).strip()


class ContextPreprocessor:
    """Fit robust numeric statistics and categorical vocabularies on training rows only."""

    VERSION = 1

    def __init__(self, min_category_frequency: int = 2):
        if min_category_frequency < 1:
            raise ValueError("min_category_frequency must be at least 1")
        self.min_category_frequency = int(min_category_frequency)
        self.numeric_statistics: Dict[str, Dict[str, float]] = {}
        self.category_vocabularies: Dict[str, Dict[str, int]] = {}
        self._is_fitted = False

    @property
    def category_sizes(self) -> Tuple[int, ...]:
        self._require_fitted()
        return tuple(len(self.category_vocabularies[name]) for name in CATEGORICAL_FEATURES)

    @property
    def numeric_output_dim(self) -> int:
        return len(NUMERIC_FEATURES) * 2

    def fit(self, frame: pd.DataFrame) -> "ContextPreprocessor":
        numeric = _numeric_frame(frame)
        self.numeric_statistics = {}
        for feature in NUMERIC_FEATURES:
            finite = numeric.loc[np.isfinite(numeric[feature]), feature].to_numpy(dtype=np.float64)
            if finite.size:
                lower, median, upper = np.quantile(finite, [0.01, 0.5, 0.99])
                q25, q75 = np.quantile(finite, [0.25, 0.75])
                scale = float(q75 - q25)
                if not math.isfinite(scale) or scale < 1e-12:
                    scale = 1.0
            else:
                lower = median = upper = 0.0
                scale = 1.0
            self.numeric_statistics[feature] = {
                "lower": float(lower),
                "median": float(median),
                "upper": float(upper),
                "scale": float(scale),
            }

        self.category_vocabularies = {}
        for feature in CATEGORICAL_FEATURES:
            source = frame[feature] if feature in frame else pd.Series(None, index=frame.index)
            counts = Counter(_category_value(value) for value in source)
            retained = sorted(
                value
                for value, count in counts.items()
                if value != MISSING_CATEGORY_TOKEN and count >= self.min_category_frequency
            )
            vocabulary = {
                MISSING_CATEGORY_TOKEN: MISSING_CATEGORY_INDEX,
                UNKNOWN_CATEGORY_TOKEN: UNKNOWN_CATEGORY_INDEX,
            }
            vocabulary.update({value: index + 2 for index, value in enumerate(retained)})
            self.category_vocabularies[feature] = vocabulary

        self._is_fitted = True
        return self

    def transform(self, frame: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        self._require_fitted()
        numeric = _numeric_frame(frame)
        scaled_columns = []
        mask_columns = []
        for feature in NUMERIC_FEATURES:
            raw = numeric[feature].to_numpy(dtype=np.float64)
            present = np.isfinite(raw)
            stats = self.numeric_statistics[feature]
            imputed = np.where(present, raw, stats["median"])
            clipped = np.clip(imputed, stats["lower"], stats["upper"])
            scaled_columns.append(((clipped - stats["median"]) / stats["scale"]).astype(np.float32))
            mask_columns.append(present.astype(np.float32))
        numeric_array = np.stack((*scaled_columns, *mask_columns), axis=1)

        category_columns = []
        for feature in CATEGORICAL_FEATURES:
            source = frame[feature] if feature in frame else pd.Series(None, index=frame.index)
            vocabulary = self.category_vocabularies[feature]
            category_columns.append(
                np.asarray(
                    [vocabulary.get(_category_value(value), UNKNOWN_CATEGORY_INDEX) for value in source],
                    dtype=np.int64,
                )
            )
        category_array = np.stack(category_columns, axis=1)
        return torch.from_numpy(numeric_array), torch.from_numpy(category_array)

    def fit_transform(self, frame: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.fit(frame).transform(frame)

    def to_dict(self) -> Dict[str, Any]:
        self._require_fitted()
        return {
            "version": self.VERSION,
            "min_category_frequency": self.min_category_frequency,
            "numeric_statistics": self.numeric_statistics,
            "category_vocabularies": self.category_vocabularies,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ContextPreprocessor":
        if int(payload.get("version", -1)) != cls.VERSION:
            raise ValueError("Unsupported context preprocessor version")
        instance = cls(min_category_frequency=int(payload["min_category_frequency"]))
        instance.numeric_statistics = {
            str(feature): {str(key): float(value) for key, value in statistics.items()}
            for feature, statistics in payload["numeric_statistics"].items()
        }
        instance.category_vocabularies = {
            str(feature): {str(value): int(index) for value, index in vocabulary.items()}
            for feature, vocabulary in payload["category_vocabularies"].items()
        }
        instance._is_fitted = True
        return instance

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("ContextPreprocessor must be fitted before use")
