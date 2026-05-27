"""
src/cleaning/cleaner.py
────────────────────────
Adaptive data cleaning:
  - Drop columns with too many missing values
  - Impute remaining nulls (strategy per column type)
  - Remove duplicate rows
  - Fix infinite values
  - Clip outliers (IQR or z-score)
  - Parse / normalise datetime columns
  - Clean text columns
  - Convert boolean-like columns to 0/1
  - Remove constant / near-constant columns
"""

from __future__ import annotations

import re
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from src.utils.config import PipelineConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)
warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────
class DataCleaner:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.cc = cfg.cleaning
        self.td = cfg.type_detection

    # ── public ────────────────────────────────────────────────

    def clean(self, df: pd.DataFrame, profile: Dict[str, Any]) -> pd.DataFrame:
        logger.info(f"Cleaning: input shape {df.shape}")
        df = df.copy()

        col_types = profile.get("column_types", {})
        const_cols = profile.get("constant_columns", [])

        df = self._drop_high_missing(df)
        df = self._drop_constant(df, const_cols)
        df = self._strip_whitespace(df, col_types)
        df = self._fix_infinites(df)
        df = self._parse_datetimes(df, profile.get("datetime_columns", []))
        df = self._normalise_booleans(df, profile.get("boolean_columns", []))
        df = self._clean_text(df, profile.get("text_columns", []))
        df = self._impute(df, col_types)
        df = self._clip_outliers(df, col_types)
        if self.cc.drop_duplicates:
            before = len(df)
            df.drop_duplicates(inplace=True)
            df.reset_index(drop=True, inplace=True)
            logger.info(f"Removed {before - len(df)} duplicate rows")

        logger.info(f"Cleaning: output shape {df.shape}")
        return df

    # ── stages ────────────────────────────────────────────────

    def _drop_high_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        thr = self.cc.drop_missing_threshold
        miss_pct = df.isna().mean()
        drop_cols = miss_pct[miss_pct > thr].index.tolist()
        if drop_cols:
            logger.info(f"Dropping {len(drop_cols)} cols with >{thr*100:.0f}% missing: {drop_cols}")
            df.drop(columns=drop_cols, inplace=True)
        return df

    def _drop_constant(self, df: pd.DataFrame, const_cols: List[str]) -> pd.DataFrame:
        existing = [c for c in const_cols if c in df.columns]
        if existing and self.cc.remove_constant_cols:
            logger.info(f"Dropping {len(existing)} constant columns")
            df.drop(columns=existing, inplace=True)
        return df

    def _strip_whitespace(
        self, df: pd.DataFrame, col_types: Dict[str, str]
    ) -> pd.DataFrame:
        if not self.cc.whitespace_strip:
            return df
        str_cols = df.select_dtypes(include="object").columns
        for col in str_cols:
            df[col] = df[col].str.strip()
        return df

    def _fix_infinites(self, df: pd.DataFrame) -> pd.DataFrame:
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        return df

    def _parse_datetimes(self, df: pd.DataFrame, dt_cols: List[str]) -> pd.DataFrame:
        for col in dt_cols:
            if col not in df.columns:
                continue
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                try:
                    # pandas 2.0+: use format="mixed" instead of infer_datetime_format
                    df[col] = pd.to_datetime(df[col], format="mixed", errors="coerce")
                    logger.debug(f"Parsed datetime: {col}")
                except Exception:
                    try:
                        df[col] = pd.to_datetime(df[col], errors="coerce")
                        logger.debug(f"Parsed datetime (fallback): {col}")
                    except Exception:
                        pass
        return df

    def _normalise_booleans(
        self, df: pd.DataFrame, bool_cols: List[str]
    ) -> pd.DataFrame:
        for col in bool_cols:
            if col not in df.columns:
                continue
            s = df[col].astype(str).str.lower().str.strip()
            true_vals = {"true", "1", "yes", "y"}
            df[col] = s.map(lambda x: 1 if x in true_vals else (0 if pd.notna(x) and x != "nan" else np.nan))
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def _clean_text(self, df: pd.DataFrame, text_cols: List[str]) -> pd.DataFrame:
        for col in text_cols:
            if col not in df.columns:
                continue
            df[col] = (
                df[col]
                .astype(str)
                .str.lower()
                .str.replace(r"<[^>]+>", " ", regex=True)   # strip HTML
                .str.replace(r"[^\w\s]", " ", regex=True)    # punctuation
                .str.replace(r"\s+", " ", regex=True)        # collapse spaces
                .str.strip()
                .replace("nan", np.nan)
            )
            df[col].fillna(self.cc.text_impute_value, inplace=True)
        return df

    def _impute(
        self,
        df: pd.DataFrame,
        col_types: Dict[str, str]
    ) -> pd.DataFrame:

        import pandas as pd
        import numpy as np
        from pandas.api.types import (
            is_numeric_dtype,
            is_bool_dtype,
            is_datetime64_any_dtype,
            is_timedelta64_dtype,
            is_object_dtype,
            is_string_dtype,
        )

        logger.debug("Starting universal imputation")

        for col in df.columns:

            s = df[col]

            # Skip columns without missing values
            if not s.isna().any():
                continue

            try:

                # -------------------------------------------------
                # NUMERIC
                # -------------------------------------------------
                if is_numeric_dtype(s):

                    strategy = getattr(
                        self.cc,
                        "numeric_impute_strategy",
                        "median"
                    )

                    if strategy == "mean":
                        fill_value = s.mean()

                    elif strategy == "most_frequent":
                        mode = s.mode(dropna=True)
                        fill_value = mode.iloc[0] if not mode.empty else 0

                    else:
                        # median default
                        fill_value = s.median()

                    if pd.isna(fill_value):
                        fill_value = 0

                    df[col] = s.fillna(fill_value)

                # -------------------------------------------------
                # BOOLEAN
                # -------------------------------------------------
                elif is_bool_dtype(s):

                    mode = s.mode(dropna=True)

                    fill_value = (
                        mode.iloc[0]
                        if not mode.empty
                        else False
                    )

                    df[col] = s.fillna(fill_value)

                # -------------------------------------------------
                # DATETIME
                # -------------------------------------------------
                elif is_datetime64_any_dtype(s):

                    non_null = s.dropna()

                    if len(non_null) > 0:
                        fill_value = non_null.mode().iloc[0]
                    else:
                        fill_value = pd.Timestamp("1970-01-01")

                    df[col] = s.fillna(fill_value)

                # -------------------------------------------------
                # TIMEDELTA
                # -------------------------------------------------
                elif is_timedelta64_dtype(s):

                    fill_value = pd.Timedelta(0)

                    df[col] = s.fillna(fill_value)

                # -------------------------------------------------
                # STRING / OBJECT / CATEGORICAL
                # -------------------------------------------------
                elif (
                    is_string_dtype(s)
                    or is_object_dtype(s)
                    or str(s.dtype) == "category"
                ):

                    mode = s.mode(dropna=True)

                    fill_value = (
                        mode.iloc[0]
                        if not mode.empty
                        else "unknown"
                    )

                    df[col] = s.fillna(fill_value)

                # -------------------------------------------------
                # FALLBACK
                # -------------------------------------------------
                else:

                    df[col] = s.fillna("unknown")

                logger.debug(f"Imputed column: {col}")

            except Exception as e:

                logger.warning(
                    f"Imputation failed for column '{col}': {e}"
                )

                # Ultimate fallback
                try:
                    if is_numeric_dtype(s):
                        df[col] = s.fillna(0)
                    else:
                        df[col] = s.astype(str).fillna("unknown")
                except Exception:
                    pass

        logger.debug("Universal imputation completed")

        return df

    def _clip_outliers(
        self,
        df: pd.DataFrame,
        col_types: Dict[str, str]
    ) -> pd.DataFrame:

        import pandas as pd
        import numpy as np
        from pandas.api.types import is_numeric_dtype

        method = self.cc.outlier_method

        if method == "none":
            return df

        factor = self.cc.outlier_clip_factor

        # Use profiler numeric columns only
        num_cols = [
            c for c, t in col_types.items()
            if t == "numeric" and c in df.columns
        ]

        for col in num_cols:

            try:

                # -------------------------------------------------
                # SAFE COLUMN EXTRACTION
                # -------------------------------------------------
                s = df[col]

                # Extra protection against bad profiling
                if not is_numeric_dtype(s):

                    # Try forced numeric conversion
                    s = pd.to_numeric(s, errors="coerce")

                    # If conversion failed entirely -> skip
                    if s.dropna().empty:
                        logger.debug(
                            f"Skipping non-numeric column: {col}"
                        )
                        continue

                else:
                    # normalize nullable/arrow numeric types
                    s = pd.to_numeric(s, errors="coerce")

                s = s.dropna()

                # Too small for stable statistics
                if len(s) < 10:
                    logger.debug(
                        f"Skipping small column: {col}"
                    )
                    continue

                # -------------------------------------------------
                # IQR METHOD
                # -------------------------------------------------
                if method == "iqr":

                    q1 = s.quantile(0.25)
                    q3 = s.quantile(0.75)

                    iqr = q3 - q1

                    # Avoid zero/invalid IQR
                    if pd.isna(iqr) or iqr == 0:
                        logger.debug(
                            f"Skipping zero-IQR column: {col}"
                        )
                        continue

                    lo = q1 - factor * iqr
                    hi = q3 + factor * iqr

                # -------------------------------------------------
                # Z-SCORE METHOD
                # -------------------------------------------------
                elif method == "zscore":

                    mean = s.mean()
                    std = s.std()

                    # Avoid divide-by-zero behavior
                    if pd.isna(std) or std == 0:
                        logger.debug(
                            f"Skipping zero-std column: {col}"
                        )
                        continue

                    lo = mean - factor * std
                    hi = mean + factor * std

                else:
                    continue

                # -------------------------------------------------
                # APPLY CLIPPING
                # -------------------------------------------------
                clipped = pd.to_numeric(
                    df[col],
                    errors="coerce"
                ).clip(lower=lo, upper=hi)

                df[col] = clipped

                logger.debug(
                    f"Clipped outliers for column: {col}"
                )

            except Exception as e:

                logger.warning(
                    f"Outlier clipping failed for '{col}': {e}"
                )

                continue

        return df