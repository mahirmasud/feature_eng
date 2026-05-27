"""
src/profiling/profiler.py
──────────────────────────
Schema inference, column-type classification, data quality
metrics, and profiling report generation.

Column categories
─────────────────
  id          – likely a unique identifier
  numeric     – continuous numbers
  categorical – low-cardinality strings / ints
  boolean     – binary flag
  text        – free-form text (high avg token count)
  datetime    – date/time values
  constant    – single unique value
  url         – URLs / emails / IPs
  geospatial  – lat/lon pairs (detected by column name)
  unknown     – anything else
"""

from __future__ import annotations

import re
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.config import PipelineConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)
warnings.filterwarnings("ignore")

# ── regex patterns ────────────────────────────────────────────
_RE_URL = re.compile(r"https?://|www\.", re.I)
_RE_EMAIL = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")
_RE_IP = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
_RE_LATLON = re.compile(r"lat|lon|latitude|longitude|coord", re.I)


# ──────────────────────────────────────────────────────────────
class DataProfiler:
    """
    Analyse a DataFrame and return a rich profile dictionary
    consumed by downstream stages.
    """

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.td = cfg.type_detection

    # ── public ────────────────────────────────────────────────

    def profile(
        self, df: pd.DataFrame, target_col: Optional[str] = None
    ) -> Dict[str, Any]:
        logger.info(f"Profiling {df.shape[0]} rows × {df.shape[1]} columns")

        col_types = self._classify_columns(df)
        quality = self._quality_metrics(df)
        corr = self._correlation_summary(df, col_types)
        suggested_target = target_col or self._suggest_target(df, col_types)

        profile = {
            "shape": list(df.shape),
            "column_types": col_types,
            "quality": quality,
            "correlation_summary": corr,
            "suggested_target": suggested_target,
            "datetime_columns": [c for c, t in col_types.items() if t == "datetime"],
            "id_columns": [c for c, t in col_types.items() if t == "id"],
            "text_columns": [c for c, t in col_types.items() if t == "text"],
            "numeric_columns": [c for c, t in col_types.items() if t == "numeric"],
            "categorical_columns": [c for c, t in col_types.items() if t == "categorical"],
            "boolean_columns": [c for c, t in col_types.items() if t == "boolean"],
            "constant_columns": [c for c, t in col_types.items() if t == "constant"],
        }

        self._log_summary(profile)
        return profile

    # ── column type classification ─────────────────────────────

    def _classify_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        types: Dict[str, str] = {}
        for col in df.columns:
            types[col] = self._classify_col(df[col])
        return types

    def _classify_col(self, series: pd.Series) -> str:
        s = series.dropna()
        n = len(s)
        if n == 0:
            return "constant"

        n_unique = s.nunique()
        unique_ratio = n_unique / n

        # Constant
        if n_unique <= 1:
            return "constant"

        # Datetime (attempt parse if object)
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        if series.dtype == "object":
            dt = self._try_parse_datetime(s)
            if dt is not None:
                return "datetime"

        # Boolean
        if self._is_boolean(s):
            return "boolean"

        # URL / email / IP (string pattern)
        if series.dtype == "object":
            sample = s.head(50).astype(str)
            if sample.str.match(_RE_URL).mean() > 0.5:
                return "url"
            if sample.str.match(_RE_EMAIL).mean() > 0.5:
                return "url"
            if sample.str.match(_RE_IP).mean() > 0.5:
                return "url"

        # Geospatial (column name heuristic)
        if _RE_LATLON.search(series.name):
            return "geospatial"

        # ID column
        if unique_ratio >= self.td.id_max_cardinality_ratio and n_unique > 100:
            return "id"

        # Numeric
        if pd.api.types.is_numeric_dtype(series):
            return "numeric"

        # Free-text
        if series.dtype == "object":
            avg_tokens = s.astype(str).str.split().str.len().mean()
            if avg_tokens >= self.td.text_avg_token_threshold:
                return "text"

        # Categorical (string or int with low cardinality)
        if n_unique <= self.td.cat_max_cardinality:
            return "categorical"

        # High-cardinality string → still categorical but flagged
        if series.dtype == "object":
            return "categorical"

        return "numeric"

    def _try_parse_datetime(self, s: pd.Series) -> Optional[pd.Series]:
        for fmt in self.td.datetime_patterns:
            try:
                parsed = pd.to_datetime(s, format=fmt, errors="raise")
                return parsed
            except Exception:
                continue
        try:
            # infer_datetime_format removed in pandas 2.2 – use format="mixed"
            parsed = pd.to_datetime(s, format="mixed", errors="raise")
            return parsed
        except Exception:
            pass
        try:
            parsed = pd.to_datetime(s, errors="coerce")
            if parsed.notna().mean() > 0.8:   # >80% parsed → treat as datetime
                return parsed
        except Exception:
            pass
        return None

    def _is_boolean(self, s: pd.Series) -> bool:
        vals = set(s.unique())
        for pair in self.td.bool_values:
            if vals <= {str(v).lower() for v in pair} | {v for v in pair}:
                return True
        return False

    # ── quality metrics ───────────────────────────────────────

    def _quality_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        total = len(df)
        metrics: Dict[str, Any] = {
            "total_rows": total,
            "total_columns": len(df.columns),
            "duplicate_rows": int(df.duplicated().sum()),
            "duplicate_pct": round(df.duplicated().mean() * 100, 2),
        }

        col_stats = {}
        for col in df.columns:
            s = df[col]
            missing = s.isna().sum()
            col_stats[col] = {
                "dtype": str(s.dtype),
                "missing_count": int(missing),
                "missing_pct": round(missing / total * 100, 2) if total else 0,
                "unique_count": int(s.nunique()),
                "unique_pct": round(s.nunique() / total * 100, 2) if total else 0,
            }
            # Numeric extras
            if pd.api.types.is_numeric_dtype(s):
                ns = s.dropna()
                if len(ns) > 1:
                    col_stats[col].update({
                        "mean": round(float(ns.mean()), 4),
                        "std": round(float(ns.std()), 4),
                        "min": round(float(ns.min()), 4),
                        "max": round(float(ns.max()), 4),
                        "skewness": round(float(stats.skew(ns)), 4),
                        "kurtosis": round(float(stats.kurtosis(ns)), 4),
                    })
        metrics["column_stats"] = col_stats
        return metrics

    def _correlation_summary(
        self, df: pd.DataFrame, col_types: Dict[str, str]
    ) -> Dict[str, Any]:
        num_cols = [c for c, t in col_types.items() if t == "numeric"]
        if len(num_cols) < 2:
            return {}
        try:
            corr = df[num_cols].corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            high_pairs = [
                {"col1": c1, "col2": c2, "corr": round(float(upper.loc[c1, c2]), 4)}
                for c1 in upper.index
                for c2 in upper.columns
                if pd.notna(upper.loc[c1, c2]) and upper.loc[c1, c2] > 0.85
            ]
            return {"high_correlation_pairs": high_pairs[:50]}
        except Exception:
            return {}

    # ── target suggestion ─────────────────────────────────────

    def _suggest_target(
        self, df: pd.DataFrame, col_types: Dict[str, str]
    ) -> Optional[str]:
        candidates = {
            "target", "label", "y", "outcome", "class",
            "survived", "churn", "fraud", "default", "response",
        }
        for col in df.columns:
            if col.lower() in candidates:
                return col
        # Last numeric column as fallback
        num_cols = [c for c, t in col_types.items() if t in ("numeric", "categorical", "boolean")]
        return num_cols[-1] if num_cols else None

    # ── logging ───────────────────────────────────────────────

    def _log_summary(self, profile: dict) -> None:
        ct = profile["column_types"]
        from collections import Counter
        dist = Counter(ct.values())
        logger.info("Column type distribution:")
        for t, n in sorted(dist.items()):
            logger.info(f"  {t:<15} {n}")
        q = profile["quality"]
        logger.info(
            f"Quality: {q['duplicate_rows']} duplicate rows "
            f"({q['duplicate_pct']}%)"
        )
        if profile.get("suggested_target"):
            logger.info(f"Suggested target: '{profile['suggested_target']}'")