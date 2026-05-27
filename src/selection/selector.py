"""
src/selection/selector.py
──────────────────────────
Prune the generated feature matrix:

1. Remove non-numeric / unsupported columns (encode categoricals first)
2. Remove duplicate columns
3. Remove zero/near-zero variance columns
4. Remove highly correlated feature pairs (keep one)
5. Rank by mutual information, keep top-K
6. Drop leaky ID-like columns
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    VarianceThreshold,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.preprocessing import LabelEncoder

from src.utils.config import PipelineConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)
warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────
class FeatureSelector:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.sc = cfg.selection

    # ── public ────────────────────────────────────────────────

    def select(
        self,
        df: pd.DataFrame,
        target_col: Optional[str],
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        logger.info(f"Feature selection: input {df.shape}")

        feat_cols = [c for c in df.columns if c != target_col]
        target_series = df[target_col] if target_col and target_col in df.columns else None

        # 1 – encode all categoricals to numeric for selection
        df_num, encoders = self._encode(df[feat_cols])

        # 2 – drop duplicate columns
        df_num = self._drop_duplicates(df_num)

        # 3 – variance threshold
        df_num = self._variance_filter(df_num)

        # 4 – correlation filter
        df_num, dropped_corr = self._correlation_filter(df_num)

        # 5 – mutual information ranking
        if target_series is not None and len(df_num.columns) > self.sc.mutual_info_top_k:
            df_num, mi_scores = self._mi_filter(df_num, target_series.reindex(df_num.index))
        else:
            mi_scores = {}

        # 6 – drop leaky / ID-like columns
        if self.sc.drop_leaky_id_cols:
            df_num = self._drop_leaky(df_num)

        # Reassemble with target
        selected = list(df_num.columns)
        result = df[selected].copy()
        if target_series is not None:
            result[target_col] = target_series.values

        meta = {
            "selected_features": selected,
            "n_selected": len(selected),
            "dropped_high_correlation": dropped_corr,
            "mi_scores": {k: round(float(v), 6) for k, v in list(mi_scores.items())[:50]},
        }

        logger.info(f"Feature selection: kept {len(selected)} features")
        return result, meta

    # ── stages ────────────────────────────────────────────────

    def _encode(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
        df = df.copy()
        encoders: Dict[str, LabelEncoder] = {}
        for col in df.select_dtypes(include=["object", "category"]).columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        # Drop datetimes (can't encode easily for MI)
        dt_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
        if dt_cols:
            df.drop(columns=dt_cols, inplace=True)
        # Fill any remaining NaN
        df.fillna(0, inplace=True)
        return df, encoders

    def _drop_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.sc.remove_duplicates:
            return df
        before = df.shape[1]
        df = df.T.drop_duplicates().T
        dropped = before - df.shape[1]
        if dropped:
            logger.info(f"Removed {dropped} duplicate feature columns")
        return df

    def _variance_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        before = df.shape[1]
        try:
            sel = VarianceThreshold(threshold=self.sc.variance_threshold)
            arr = sel.fit_transform(df)
            kept_cols = df.columns[sel.get_support()].tolist()
            df = pd.DataFrame(arr, columns=kept_cols, index=df.index)
        except Exception as e:
            logger.warning(f"Variance threshold failed: {e}")
        dropped = before - df.shape[1]
        if dropped:
            logger.info(f"Variance threshold removed {dropped} features")
        return df

    def _correlation_filter(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[str]]:
        thr = self.sc.correlation_threshold
        dropped: List[str] = []
        try:
            corr_matrix = df.corr().abs()
            upper = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            to_drop = [
                col for col in upper.columns if any(upper[col] > thr)
            ]
            dropped = to_drop
            df = df.drop(columns=to_drop)
            if to_drop:
                logger.info(
                    f"Correlation filter removed {len(to_drop)} features "
                    f"(threshold={thr})"
                )
        except Exception as e:
            logger.warning(f"Correlation filter failed: {e}")
        return df, dropped

    def _mi_filter(
        self, df: pd.DataFrame, target: pd.Series
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        k = self.sc.mutual_info_top_k
        try:
            # Decide classification vs regression
            n_unique = target.nunique()
            if n_unique <= 20 or target.dtype == "object":
                mi_fn = mutual_info_classif
                t = LabelEncoder().fit_transform(target.astype(str))
            else:
                mi_fn = mutual_info_regression
                t = target.values.astype(float)

            mi = mi_fn(df, t, random_state=self.cfg.random_seed)
            mi_scores = dict(zip(df.columns, mi))
            top_cols = sorted(mi_scores, key=mi_scores.get, reverse=True)[:k]
            df = df[top_cols]
            logger.info(f"Mutual information kept top {k} features")
            return df, mi_scores
        except Exception as e:
            logger.warning(f"MI filter failed: {e}")
            return df, {}

    def _drop_leaky(self, df: pd.DataFrame) -> pd.DataFrame:
        leaky_patterns = [r"_id$", r"^id_", r"^index$", r"^row_id$", r"__row_id"]
        import re
        to_drop = [
            col for col in df.columns
            if any(re.search(p, col.lower()) for p in leaky_patterns)
        ]
        if to_drop:
            logger.info(f"Dropping potentially leaky ID columns: {to_drop}")
            df.drop(columns=to_drop, inplace=True)
        return df
